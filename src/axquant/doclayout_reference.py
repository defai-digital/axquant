"""Reference decoder for development AXDocLayout AXQ packs (FP32 inference).

Packing is reduced precision; execution reconstructs FP32 weights on CPU.
MX formats require Apple Silicon MLX to decode. No native FP4 execution.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


def read_state(directory: str | Path) -> dict[str, Any]:
    """Verify the complete pack and reconstruct its exact state dictionary."""
    import numpy as np
    from safetensors.numpy import load_file

    root = Path(directory).resolve()
    manifest = json.loads((root / "axq-manifest.json").read_text())
    if manifest["schema_version"] not in {
        "axq-doclayout.development/1",
        "axq-doclayout.development/2",
    }:
        raise ValueError("Unsupported AXQ DocLayout schema")
    mode = manifest["format"]
    if mode not in {"nvfp4", "mxfp4", "mxfp8"}:
        raise ValueError("Unsupported physical format")
    cache = root.parent.parent
    # Canonical: AXDocLayout-M-v{1.0,1.1}-AXQ-{NVFP4,MXFP4,MXFP8}. The AX-/-CUDA/
    # -MLX/-W4A16 forms were briefly produced by a 2.3 development branch and are
    # still accepted so already-downloaded caches keep loading.
    canonical_names = {
        f"models--AutomatosX--AXDocLayout-M-v{ver}-AXQ-{fmt}"
        for ver in ["1.0", "1.1"]
        for fmt in ["NVFP4", "MXFP4", "MXFP8"]
    }
    retired_2_3_names = {
        f"models--AutomatosX--AX-AXDocLayout-M-v{ver}-CUDA-AXQ-NVFP4-W4A16"
        for ver in ["1.0", "1.1"]
    } | {
        f"models--AutomatosX--AX-AXDocLayout-M-v{ver}-MLX-AXQ-{fmt}"
        for ver in ["1.0", "1.1"]
        for fmt in ["MXFP4", "MXFP8"]
    }
    hub_snapshot = (
        root.parent.name == "snapshots"
        and re.fullmatch(r"[0-9a-f]{40}", root.name) is not None
        and cache.name in (canonical_names | retired_2_3_names)
    )
    for name, digest in manifest["files"].items():
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("Invalid pack filename")
        path = (root / name).resolve()
        cache_blob = hub_snapshot and path.parent == cache / "blobs"
        if (not path.is_relative_to(root) and not cache_blob) or not path.is_file():
            raise ValueError("Invalid pack file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pack digest mismatch: {name}")
    schema_module = (
        "axquant.schema.doclayout" if __package__ == "axquant" else "axq_doclayout_schema"
    )
    schema = importlib.import_module(schema_module)
    schema.DocLayoutManifest.model_validate(manifest)
    tensors = load_file(str(root / "model.safetensors"))
    state = {}
    consumed = set()
    for name, entry in manifest["tensors"].items():
        shape = tuple(entry["shape"])
        if entry["method"] == "preserved":
            value = tensors[name]
            consumed.add(name)
        elif entry["method"] == mode:
            keys = [name + ".axq_packed", name + ".axq_scales"]
            packed, scales = (tensors[k] for k in keys)
            consumed.update(keys)
            if mode == "nvfp4":
                from axquant.nvfp4 import Nvfp4Tensor, dequantize_nvfp4

                value = dequantize_nvfp4(
                    Nvfp4Tensor(
                        packed,
                        scales,
                        entry["global_scale"],
                        (shape[0], entry["padded_columns"]),
                        "numpy-reference",
                    )
                )
            else:
                mx: Any = importlib.import_module("mlx.core")

                value = np.array(
                    mx.dequantize(
                        mx.array(packed),
                        mx.array(scales),
                        group_size=32,
                        bits=4 if mode == "mxfp4" else 8,
                        mode=mode,
                    ).astype(mx.float32)
                )
            value = value[:, : int(np.prod(shape[1:]))].reshape(shape)
        else:
            raise ValueError("Tensor method differs from pack format")
        if value.shape != shape or str(value.dtype) != entry["dtype"]:
            raise ValueError("Decoded tensor shape or dtype mismatch")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            # Invalid proposal anchors are +inf in the pinned source checkpoint.
            source_anchor = entry["method"] == "preserved" and name == "decoder.anchors"
            if not source_anchor or np.isnan(value).any() or np.isneginf(value).any():
                raise ValueError("Nonfinite decoded weight")
        state[name] = value
    if consumed != set(tensors):
        raise ValueError("Unaccounted pack tensors")
    return state


def load(directory: str | Path) -> Any:
    """Load a verified pack into the bundled PyTorch CPU architecture."""
    import torch

    root = Path(directory).resolve()
    state = read_state(root)
    manifest = json.loads((root / "axq-manifest.json").read_text())
    code_root = root / "inference" / "src" / "ax_doclayout"
    if not (code_root / "_torch_rtdetrv2.py").is_file():
        raise ValueError("Missing bound inference architecture")
    for code_file in code_root.rglob("*.py"):
        if str(code_file.relative_to(root)) not in manifest["files"]:
            raise ValueError("Unbound inference source")
    sys.path.insert(0, str(root / "inference" / "src"))
    architecture = importlib.import_module("ax_doclayout._torch_rtdetrv2")
    model = architecture.build_rtdetrv2_r50vd_m()
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()}, strict=True)
    return model.eval()


def predict(model: Any, rgb: Any, *, page_index: int = 0) -> list[dict[str, Any]]:
    """Return original-raster elements using the source preprocessing contract."""
    from dataclasses import asdict

    import torch

    PageImage = importlib.import_module("ax_doclayout.contract").PageImage
    elements_from_letterbox_detections = importlib.import_module(
        "ax_doclayout.postprocess"
    ).elements_from_letterbox_detections
    preprocess = importlib.import_module("ax_doclayout.preprocess")

    page = PageImage(rgb=rgb, page_index=page_index)
    letterbox = preprocess.letterbox_page(page)
    with torch.inference_mode():
        labels, boxes, scores = model(
            torch.from_numpy(preprocess.letterbox_nchw(letterbox)),
            torch.from_numpy(preprocess.orig_target_sizes(letterbox)),
        )
    elements = elements_from_letterbox_detections(
        labels=labels.numpy(),
        boxes=boxes.numpy(),
        scores=scores.numpy(),
        page=page,
        letterbox=letterbox,
        threshold=0.30,
    )
    return [asdict(element) for element in elements]
