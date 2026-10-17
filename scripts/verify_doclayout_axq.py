#!/usr/bin/env python3
"""Full-weight reference smoke and optional immutable Hub publication verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from run_doclayout_axq import BASE, _pack_dir
from safetensors.numpy import load_file

from axquant import __version__
from axquant.doclayout import REVISION, SOURCE, digest, write_json
from axquant.doclayout_reference import load, predict, read_state


def pages():
    for name in ["blank", "text", "table"]:
        image = Image.new("RGB", (768, 1024), "white")
        draw = ImageDraw.Draw(image)
        if name != "blank":
            draw.text((60, 60), "AX DocLayout Reference Smoke", fill="black", font_size=28)
            for row in range(24):
                draw.text(
                    (60, 140 + row * 28),
                    f"Section {row + 1}: deterministic document layout sample.",
                    fill="black",
                    font_size=18,
                )
        if name == "table":
            draw.rectangle((50, 350, 710, 800), fill="white", outline="black", width=2)
            for x in range(50, 711, 110):
                draw.line((x, 350, x, 800), fill="black", width=2)
            for y in range(350, 801, 50):
                draw.line((50, y, 710, y), fill="black", width=2)
        yield name, np.array(image)


def smoke() -> None:
    sys.path.insert(0, str(BASE / "source" / "inference" / "src"))
    from ax_doclayout._torch_rtdetrv2 import build_rtdetrv2_r50vd_m
    from ax_doclayout.contract import PageImage
    from ax_doclayout.preprocess import letterbox_nchw, letterbox_page, orig_target_sizes

    torch.set_num_threads(8)
    torch.manual_seed(0)
    source = load_file(str(BASE / "source" / "model.safetensors"))
    baseline = build_rtdetrv2_r50vd_m()
    baseline.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in source.items()}, strict=True
    )
    baseline.eval()
    baseline_outputs = {}
    baseline_counts = {}
    for name, rgb in pages():
        page = letterbox_page(PageImage(rgb=rgb, page_index=0))
        inputs = (torch.from_numpy(letterbox_nchw(page)), torch.from_numpy(orig_target_sizes(page)))
        with torch.inference_mode():
            output = baseline(*inputs)
        baseline_outputs[name] = tuple(t.numpy() for t in output)
        baseline_counts[name] = len(predict(baseline, rgb))
    for mode in ["nvfp4", "mxfp4", "mxfp8"]:
        pack = _pack_dir(mode)
        manifest = json.loads((pack / "axq-manifest.json").read_text())
        state = read_state(pack)
        if set(source) != set(state):
            raise ValueError("State keys mismatch")
        for key, entry in manifest["tensors"].items():
            if entry["method"] == "preserved":
                np.testing.assert_array_equal(state[key], source[key])
        model = load(pack)
        results, raw = [], {}
        for name, rgb in pages():
            page = letterbox_page(PageImage(rgb=rgb, page_index=0))
            with torch.inference_mode():
                output = model(
                    torch.from_numpy(letterbox_nchw(page)),
                    torch.from_numpy(orig_target_sizes(page)),
                )
            labels, boxes, scores = (t.numpy() for t in output)
            assert labels.shape == (1, 300) and boxes.shape == (1, 300, 4)
            assert scores.shape == (1, 300)
            assert np.isfinite(boxes).all() and np.isfinite(scores).all()
            assert ((scores >= 0) & (scores <= 1)).all()
            assert ((labels >= 0) & (labels <= 11)).all()
            elements = predict(model, rgb)
            json.dumps(elements, allow_nan=False)
            results.append(
                {
                    "input": name,
                    "detections_at_0_30": len(elements),
                    "finite_outputs": True,
                    "baseline_detections_at_0_30": baseline_counts[name],
                    "quality_parity_asserted": False,
                }
            )
            for label, arr, reference in zip(
                ["labels", "boxes", "scores"],
                [labels, boxes, scores],
                baseline_outputs[name],
                strict=True,
            ):
                raw[f"{name}_{label}"] = arr
                raw[f"{name}_fp32_{label}"] = reference
            print(json.dumps({"format": mode, **results[-1]}), flush=True)
        np.savez_compressed(pack / "reference-smoke-raw.npz", **raw)
        write_json(
            pack / "reference-smoke.json",
            {
                "schema_version": "axq-doclayout.reference-smoke/1",
                "execution": "dequantize-to-fp32-pytorch-cpu",
                "torch": torch.__version__,
                "os": platform.mac_ver()[0],
                "hardware": "Mac Studio M2 Ultra 192 GB",
                "inputs": "deterministic synthetic blank/text/table pages; no labeled holdout",
                "quality_certified": False,
                "native_runtime_verified": False,
                "strict_state_load": True,
                "preserved_tensors_identical": True,
                "cases": results,
            },
        )
        for filename in ["reference-smoke.json", "reference-smoke-raw.npz"]:
            manifest["files"][filename] = digest(pack / filename)
        write_json(pack / "axq-manifest.json", manifest)


def publish() -> None:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi(token=True)
    receipts = []
    for mode in ["nvfp4", "mxfp4", "mxfp8"]:
        name = _pack_dir(mode).name
        pack = _pack_dir(mode)
        manifest = json.loads((pack / "axq-manifest.json").read_text())
        smoke_record = json.loads((pack / "reference-smoke.json").read_text())
        assert len(smoke_record["cases"]) == 3
        read_state(pack)
        repo = "AutomatosX/" + name
        weight_bytes = (pack / "model.safetensors").stat().st_size
        (pack / "README.md").write_text(
            f"""---
license: agpl-3.0
library_name: ax-doclayout
pipeline_tag: object-detection
tags:
- document-layout
- rtdetrv2
- axq
- {mode}
- development
base_model: {SOURCE}
---

# {name}

Development **{mode.upper()} storage** derivative of [{SOURCE}](https://huggingface.co/{SOURCE}/tree/{REVISION}).
The reference loader decodes weights to FP32 and runs PyTorch on CPU.
This is not a native low-precision GPU/MLX inference model. No speed, runtime-memory,
accuracy parity, or certification claim is made. No labeled quality evaluation was run.

Source: `{REVISION}`. Source FP32 weight SHA-256: `{manifest["source_weight_sha256"]}`.
Quantized weights: **{weight_bytes:,} bytes** versus 146,759,800 source bytes.
AXQuant {__version__} codecs quantize {manifest["quantized_parameters"]:,}
of 36,683,526 parameters/buffers.
Convolution kernels flatten all input/kernel dimensions per output channel, then zero-pad
each row to a block boundary. The manifest records original shapes and padding.
"""
            + (
                "NVFP4 uses E2M1 weights, E4M3 block scales (16 weights/block), "
                "and an inverse FP32 global scale.\n"
                if mode == "nvfp4"
                else (
                    f"MLX {mode.upper()} uses "
                    f"{'E2M1' if mode == 'mxfp4' else 'E4M3'} weights "
                    "and E8M0 scales (32 weights/block).\n"
                )
            )
            + f"""
Biases, norms, anchors, detection class/bounding-box heads, query-position heads,
and denoising embeddings remain byte-identical to source tensor values.
FP32 PyTorch strict state loading and finite-output detection pass on three synthetic
pages (blank/text/table); see `reference-smoke.json` and the raw baseline/candidate outputs.
These checks do not establish document detection quality.
The initial 4-bit exports return one detection on the blank page where the
FP32 baseline returns none. All formats change some detection counts on the
synthetic text/table pages. Do not assume FP32 accuracy parity.

## Reference use

Python 3.12/3.13. Install `torch`, `numpy`, `pillow`, `safetensors`,
`huggingface_hub`, and `axquant[doclayout]=={__version__}`.
"""
            + (
                "MX decoding additionally needs `mlx==0.32.3` on Apple Silicon.\n"
                if mode != "nvfp4"
                else ""
            )
            + f'''
```python
import sys
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, snapshot_download

repo = "{repo}"
revision = HfApi().model_info(repo).sha  # record this immutable revision
path = snapshot_download(repo, revision=revision)
sys.path.insert(0, path)
import axq_doclayout

model = axq_doclayout.load(path)
rgb = np.asarray(Image.open("page.png").convert("RGB"), dtype=np.uint8)
elements = axq_doclayout.predict(model, rgb)
print(elements)
```

Outputs are original-raster boxes and labels for the source's 11 document-layout classes,
using its 1024-pixel letterbox preprocessing and score threshold 0.30.
The return is a list of elements, not the source's complete detection envelope.
The loader verifies manifest-bound payload SHA-256 digests and rejects mismatches.

The original `AXDocLayout.from_pretrained`, ONNX Runtime, Transformers, MLX-LM,
MLX-VLM, oMLX, MTPLX, AX Engine and TensorRT cannot load this custom storage schema.
Do not point those loaders at this pack. See `runtime-compatibility.json`.
No ONNX graph or native CUDA kernel is included. FP32 reconstruction requires full
FP32 model memory. This model detects regions; it does not perform OCR or parse PDFs.

## License and attribution

Weights and bundled AXDocLayout inference source retain AGPL-3.0-only, LICENSE and NOTICE.
The RT-DETRv2 architecture attribution and pinned upstream reference remain in the
bundled inference source NOTICE. AXQuant helper code is MIT; see AXQUANT_LICENSE.
No training data or training code is included. Full tensor layout and file bindings
are in `axq-manifest.json` (custom `axq-doclayout.development/1` storage schema,
not a standard AXQuant language-model manifest or certificate).
'''
        )
        shutil_license = Path(__file__).resolve().parents[1] / "LICENSE"
        (pack / "AXQUANT_LICENSE").write_bytes(shutil_license.read_bytes())
        manifest["files"]["AXQUANT_LICENSE"] = digest(pack / "AXQUANT_LICENSE")
        write_json(pack / "axq-manifest.json", manifest)
        parent_commit = None
        if api.repo_exists(repo):
            parent_commit = api.model_info(repo).sha
            existing = json.loads(
                Path(
                    hf_hub_download(
                        repo,
                        "axq-manifest.json",
                        revision=parent_commit,
                        token=True,
                    )
                ).read_text()
            )
            for key in [
                "source_model",
                "source_revision",
                "source_weight_sha256",
                "format",
                "tensors",
            ]:
                if existing[key] != manifest[key]:
                    raise ValueError("Refusing to replace an unrelated existing pack")
            if existing["files"]["model.safetensors"] != manifest["files"]["model.safetensors"]:
                raise ValueError("Existing pack has different weight bytes")
        else:
            api.create_repo(repo, exist_ok=False, private=False)
        commit = api.upload_folder(
            repo_id=repo,
            folder_path=pack,
            allow_patterns=[*manifest["files"], "axq-manifest.json", "README.md"],
            commit_message="Publish AXQ development storage pack and reference checks",
            parent_commit=parent_commit,
        )
        revision = commit.oid
        downloaded = Path(snapshot_download(repo, revision=revision, token=True))
        for filename in [*manifest["files"], "axq-manifest.json", "README.md"]:
            if digest(downloaded / filename) != digest(pack / filename):
                raise ValueError("Published payload differs from local verified pack")
        read_state(downloaded)
        receipts.append(
            {
                "repo": repo,
                "revision": revision,
                "weight_bytes": weight_bytes,
                "all_file_digests_verified": True,
                "manifest_sha256": hashlib.sha256(
                    (pack / "axq-manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
        write_json(BASE / "publication-receipt.json", {"publications": receipts})
        print(json.dumps(receipts[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["smoke", "publish"])
    args = parser.parse_args()
    if args.action == "smoke":
        smoke()
    else:
        publish()
