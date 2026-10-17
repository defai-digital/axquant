"""Source-bound RT-DETRv2 development storage export with FP32 reference inference."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError, BackendUnavailableError, PlanningError
from axquant.factory import require_factory_host
from axquant.nvfp4 import quantize_nvfp4_reference
from axquant.schema.doclayout import (
    DocLayoutFormat,
    DocLayoutManifest,
    DocLayoutRuntimeCompatibility,
)
from axquant.serde import write_data

SOURCES: dict[str, dict[str, str]] = {
    "v1.0": {
        "name": "AutomatosX/AXDocLayout-M-v1.0",
        "revision": "69ddc0da17a7fc35dfde29b06adbb83a75284ae1",
        "weight_sha256": "4d9ee36db616f3f4ef293588e4258b5f6f42d7a84097a5afb520779076045bf2",
        "metadata_bound_sha256": "281ebfdf4cb0445beed016682485d2fe449252d28c66371c57bfb6adc95136f0",
    },
    "v1.1": {
        "name": "AutomatosX/AXDocLayout-M-v1.1",
        "revision": "2c419a0da355ee5e979cfeb0808f596aae6e896e",
        "weight_sha256": "3b97ba1c0b1b0816f45e79a76bb2fd495e026d2394471b76ce1a442baa2989c4",
        "metadata_bound_sha256": "ad2f432b1d947c03c1b9ca19e0eefa7828c7f6c8eaef59e60d49303ca14e2393",
    },
}

# Backwards-compatible defaults: the v1.0 binding remains the admitted source for any
# importer that still references these module-level names. New callers should pass
# ``source_version`` to ``convert_doclayout`` instead.
DEFAULT_SOURCE_VERSION = "v1.0"
SOURCE = SOURCES[DEFAULT_SOURCE_VERSION]["name"]
REVISION = SOURCES[DEFAULT_SOURCE_VERSION]["revision"]
SOURCE_DIGEST = SOURCES[DEFAULT_SOURCE_VERSION]["weight_sha256"]
SOURCE_METADATA_BOUND = SOURCES[DEFAULT_SOURCE_VERSION]["metadata_bound_sha256"]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_data(path, payload)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible(name: str, weight: Any) -> bool:
    protected = ("score_head", "bbox_head", "query_pos_head", "denoising_class_embed")
    return (
        str(weight.dtype) == "float32"
        and weight.ndim in {2, 4}
        and (name.endswith(".weight") or name.endswith(".in_proj_weight"))
        and not any(token in name for token in protected)
    )


def convert_doclayout(
    model: str | Path,
    output: str | Path,
    *,
    q_mode: DocLayoutFormat,
    revision: str,
    allow_unmeasured: bool = False,
    source_version: str = DEFAULT_SOURCE_VERSION,
) -> Path:
    if source_version not in SOURCES:
        raise ArtifactError(
            f"Unknown DocLayout source_version {source_version!r}; "
            f"admitted versions: {sorted(SOURCES)}"
        )
    binding = SOURCES[source_version]
    if not allow_unmeasured:
        raise PlanningError("DocLayout development export requires --allow-unmeasured")
    if q_mode not in {"nvfp4", "mxfp4", "mxfp8"}:
        raise PlanningError("Unsupported DocLayout physical format")
    if revision != binding["revision"]:
        raise ArtifactError(
            f"DocLayout source_version={source_version} requires the admitted immutable "
            f"source revision {binding['revision']}; got {revision}"
        )
    try:
        import numpy as np
    except ImportError as exc:
        raise BackendUnavailableError(
            "DocLayout packing requires NumPy; install axquant[doclayout]"
        ) from exc

    if q_mode != "nvfp4":
        try:
            mx: Any = importlib.import_module("mlx.core")
        except ImportError as exc:
            raise BackendUnavailableError(
                "MX DocLayout packing requires Apple Silicon MLX"
            ) from exc
    mode = q_mode
    from safetensors.numpy import load_file, save_file

    require_factory_host(socket.gethostname())
    source = Path(model).expanduser().resolve()
    verify_source(source, binding["metadata_bound_sha256"])
    if digest(source / "model.safetensors") != binding["weight_sha256"]:
        raise ValueError("Pinned source weights mismatch")
    # DocLayout packs are a custom storage schema, not language-model packs: they
    # run no native low-precision runtime and decode to FP32 on CPU. They therefore
    # use the AXDocLayout naming convention from the operator guide
    # (docs/guides/doclayout.md) and sync_hf_collections.py -- no AX- prefix
    # (the base name already carries AX), no CUDA/MLX lane tag, no activation
    # suffix. See "AXDocLayout-M-v1.0-AXQ-NVFP4" in sync_hf_collections.py.
    name = f"AXDocLayout-M-{source_version}-AXQ-{mode.upper()}"
    target = Path(output).expanduser().resolve()
    if target == source or source.is_relative_to(target) or target.is_relative_to(source):
        raise ArtifactError("Output cannot overlap source")
    stage = target.with_name(target.name + ".staging")
    if target.exists() or stage.exists():
        raise FileExistsError("Refusing to replace an existing pack or staging directory")
    stage.mkdir(parents=True)
    state = load_file(str(source / "model.safetensors"))
    packed: dict[str, Any] = {}
    entries: dict[str, Any] = {}
    quantized_parameters = 0
    max_error = 0.0
    for key, weight in state.items():
        entry: dict[str, Any] = {"shape": list(weight.shape), "dtype": str(weight.dtype)}
        if not eligible(key, weight):
            packed[key] = weight
            entry["method"] = "preserved"
        else:
            columns = int(np.prod(weight.shape[1:]))
            group = 16 if mode == "nvfp4" else 32
            padded = ((columns + group - 1) // group) * group
            matrix = np.pad(
                weight.reshape(weight.shape[0], columns), ((0, 0), (0, padded - columns))
            )
            entry.update(method=mode, padded_columns=padded, group_size=group)
            if mode == "nvfp4":
                from axquant.nvfp4 import dequantize_nvfp4

                q = quantize_nvfp4_reference(matrix)
                values, scales = q.packed, q.scale_bytes
                entry["global_scale"] = q.global_scale
                restored = dequantize_nvfp4(q)
            else:
                bits = 4 if mode == "mxfp4" else 8
                values_mx, scales_mx = mx.quantize(
                    mx.array(matrix),
                    group_size=32,
                    bits=bits,
                    mode=mode,
                )
                values, scales = np.array(values_mx), np.array(scales_mx)
                restored = np.array(
                    mx.dequantize(
                        values_mx,
                        scales_mx,
                        group_size=32,
                        bits=bits,
                        mode=mode,
                    ).astype(mx.float32)
                )
            packed[key + ".axq_packed"] = values
            packed[key + ".axq_scales"] = scales
            error = float(np.max(np.abs(restored[:, :columns].reshape(weight.shape) - weight)))
            if not np.isfinite(error):
                raise ValueError("Nonfinite reconstruction")
            entry["max_absolute_error"] = error
            max_error = max(max_error, error)
            quantized_parameters += weight.size
        entries[key] = entry
    save_file(packed, str(stage / "model.safetensors"))
    for filename in ["LICENSE", "NOTICE", "taxonomy.json", "preprocessor_config.json"]:
        shutil.copy2(source / filename, stage / filename)
    shutil.copytree(
        source / "inference" / "src",
        stage / "inference" / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for filename in ["LICENSE", "NOTICE"]:
        shutil.copy2(source / "inference" / filename, stage / "inference" / filename)
    shutil.copy2(Path(__file__).with_name("doclayout_reference.py"), stage / "axq_doclayout.py")
    shutil.copy2(
        Path(__file__).parent / "schema" / "doclayout.py", stage / "axq_doclayout_schema.py"
    )
    write_json(
        stage / "config.json",
        {
            "architecture": "rtdetrv2_r50vd_m",
            "format": mode,
            "loader": "axq_doclayout.load",
            "execution": "dequantize-to-fp32-pytorch-cpu",
            "source_model": binding["name"],
            "source_revision": binding["revision"],
            "source_version": source_version,
        },
    )
    write_json(
        stage / "runtime-compatibility.json",
        DocLayoutRuntimeCompatibility().model_dump(),
    )
    files = {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    manifest = {
        "schema_version": "axq-doclayout.development/2",
        "format": mode,
        "source_model": binding["name"],
        "source_revision": binding["revision"],
        "source_version": source_version,
        "source_weight_sha256": binding["weight_sha256"],
        "status": "development",
        "quality_certified": False,
        "native_runtime_verified": False,
        "quantized_parameters": quantized_parameters,
        "total_parameters_and_buffers": sum(w.size for w in state.values()),
        "max_absolute_weight_error": max_error,
        "files": files,
        "tensors": entries,
    }
    DocLayoutManifest.model_validate(manifest)
    write_json(stage / "axq-manifest.json", manifest)
    from axquant.doclayout_reference import read_state

    read_state(stage)
    os.replace(stage, target)
    print(
        json.dumps(
            {
                "pack": name,
                "bytes": (target / "model.safetensors").stat().st_size,
                "quantized_parameters": quantized_parameters,
            }
        ),
        flush=True,
    )
    return target


def verify_source(source: Path, expected_bound: str | None = None) -> None:
    """Bind inference source and metadata to the admitted immutable Hub snapshot.

    ``expected_bound`` defaults to the v1.0 binding; v1.1 callers pass the v1.1 bound.
    """
    if expected_bound is None:
        expected_bound = SOURCES[DEFAULT_SOURCE_VERSION]["metadata_bound_sha256"]
    names = ["config.json", "taxonomy.json", "preprocessor_config.json", "LICENSE", "NOTICE"]
    names += [
        str(p.relative_to(source))
        for p in sorted((source / "inference").rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and (p.suffix in {".py", ".typed"} or p.name in {"LICENSE", "NOTICE"})
    ]
    files = {name: digest(source / name) for name in names}
    bound = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if bound != expected_bound:
        raise ArtifactError(
            "DocLayout source metadata or inference code differs from admitted revision"
        )
