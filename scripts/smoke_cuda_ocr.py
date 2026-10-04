#!/usr/bin/env python3
"""Run a development OCR smoke with supported prefill and explicit FP4 execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    parser.add_argument("--require-native-fp4", action="store_true")
    args = parser.parse_args()
    source = Path(args.model)
    config = json.loads((source / "config.json").read_text())
    processors = {
        "DeepseekOCR2ForCausalLM": "deepseek_ocr",
        "UnlimitedOCRForCausalLM": "unlimited_ocr",
    }
    architecture = config.get("architectures", [""])[0]
    if architecture not in processors:
        raise ValueError("This development smoke supports official OCR2/Unlimited-OCR only")
    quant = config.get("quantization_config", {})
    activation = quant.get("config_groups", {}).get("nvfp4", {}).get("input_activations")
    if args.require_native_fp4 and (not activation or activation.get("num_bits") != 4):
        raise ValueError("Native FP4 execution requires a calibrated W4A4 checkpoint")
    # Official image build metadata are not runtime environment settings.
    for name in ("VLLM_BUILD_COMMIT", "VLLM_BUILD_PIPELINE", "VLLM_BUILD_URL", "VLLM_IMAGE_TAG"):
        os.environ.pop(name, None)
    import torch
    from PIL import Image
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=str(source),
        model_impl="vllm",
        dtype="bfloat16",
        trust_remote_code=False,
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        enforce_eager=True,
        gpu_memory_utilization=args.memory_fraction,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        limit_mm_per_prompt={"image": 1},
        skip_mm_profiling=True,
        mm_processor_cache_gb=0,
        mm_encoder_attn_backend="TORCH_SDPA",
        attention_config={"backend": "TRITON_ATTN"},
        compilation_config={"custom_ops": ["none"]},
        enable_prefix_caching=False,
        generation_config="vllm",
        kernel_config={"linear_backend": "cutlass", "moe_backend": "cutlass"}
        if args.require_native_fp4
        else {},
        logits_processors=[
            f"vllm.model_executor.models.{processors[architecture]}:NGramPerReqLogitsProcessor"
        ],
    )
    image = Path(args.image)
    expected = ["AXQuant NVFP4", "Invoice 12345", "Total USD 42.50"]
    result = engine.generate(
        [
            {
                "prompt": "<image>\nFree OCR.",
                "multi_modal_data": {"image": Image.open(image).convert("RGB")},
            }
        ],
        SamplingParams(
            temperature=0,
            max_tokens=128,
            min_tokens=16,
            skip_special_tokens=False,
            extra_args={"ngram_size": 35, "window_size": 128},
        ),
    )[0].outputs[0]
    if not all(line in result.text for line in expected):
        raise RuntimeError(f"OCR smoke did not recognize every expected line: {result.text!r}")
    write_result(
        Path(args.output),
        {
            "status": "development-smoke",
            "quality_certified": False,
            "scope": "One generated English page; no broad OCR quality or speed certification",
            "runtime": "vllm",
            "runtime_version": importlib.metadata.version("vllm"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "chunked_prefill": True,
            "native_fp4_required": args.require_native_fp4,
            "requested_backends": {"linear": "cutlass", "moe": "cutlass"}
            if args.require_native_fp4
            else {"linear": "auto", "moe": "auto"},
            "input_image_sha256": file_sha256(image),
            "config_sha256": file_sha256(source / "config.json"),
            "minimum_tokens": 16,
            "generated_text": result.text,
            "generated_token_count": len(result.token_ids),
            "finish_reason": result.finish_reason,
            "expected_lines_present": True,
        },
    )
    print(result.text, flush=True)


if __name__ == "__main__":
    main()
