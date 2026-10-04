#!/usr/bin/env python3
"""Run a development OCR or Qwen3-VL smoke with supported prefill and explicit FP4 execution."""

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


def require_smoke_text(text: str) -> None:
    expected = ["AXQuant NVFP4", "Invoice 12345", "Total USD 42.50"]
    if not all(line in text for line in expected):
        raise RuntimeError(f"OCR smoke did not recognize every expected line: {text!r}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(lines.count(line) >= 3 for line in set(lines)):
        raise RuntimeError("OCR smoke contains repetitive output outside the test-page contents")


def require_runtime_layout(config: dict[str, Any]) -> None:
    if config.get("architectures") != ["UnlimitedOCRForCausalLM"]:
        return
    import re

    ignored = (config.get("quantization_config") or {}).get("ignore") or []
    if any(re.fullmatch(r"model\.layers\.\d+\.self_attn\.[qkvo]_proj", name) for name in ignored):
        raise ValueError(
            "vLLM 0.25.1 Unlimited-OCR MHA lacks Linear prefixes; per-layer preserved "
            "attention is unsupported. Use a plan with MLP protection instead."
        )


def require_kernel_coverage(plan: dict[str, Any], record: dict[str, Any]) -> None:
    import re

    if plan.get("schema_version") != "axquant.cuda-w4a4-plan.v1":
        raise ValueError("Native verification requires an AXQuant W4A4 plan")
    linear: set[str] = set()
    experts: set[str] = set()
    for item in plan["weight_plan"]["allocations"]:
        if item["method"] != "nvfp4":
            continue
        name = item["tensor_name"].removesuffix(".weight")
        expert = re.fullmatch(r"(.+\.experts)\.\d+\.(?:gate_proj|up_proj|down_proj)", name)
        if expert:
            experts.add(expert.group(1))
            continue
        for projection, fused in (
            ("q_proj", "qkv_proj"),
            ("k_proj", "qkv_proj"),
            ("v_proj", "qkv_proj"),
            ("gate_proj", "gate_up_proj"),
            ("up_proj", "gate_up_proj"),
        ):
            if name.endswith("." + projection):
                name = name.removesuffix(projection) + fused
                break
        linear.add(name)
    if sum(record["linear_kernels"].values()) != len(linear):
        raise RuntimeError("Actual native FP4 Linear coverage differs from the allocation plan")
    if sum(record["moe_backends"].values()) != len(experts):
        raise RuntimeError("Actual native FP4 MoE coverage differs from the allocation plan")


def inspect_native_fp4(model: Any) -> dict[str, Any]:
    linear: dict[str, int] = {}
    moe: dict[str, int] = {}
    native_moe = {
        "VLLM_CUTLASS",
        "FLASHINFER_CUTLASS",
        "FLASHINFER_TRTLLM",
        "FLASHINFER_CUTEDSL",
        "FLASHINFER_CUTEDSL_BATCHED",
        "FLASHINFER_B12X",
    }
    for _, module in model.named_modules():
        scheme = getattr(module, "scheme", None)
        if type(scheme).__name__ == "CompressedTensorsW4A4Fp4":
            kernel = type(scheme.kernel).__name__
            if scheme.use_a16 or kernel != "CutlassNvFp4LinearKernel":
                raise RuntimeError(f"Native FP4 Linear check rejected {kernel}")
            linear[kernel] = linear.get(kernel, 0) + 1
        method = getattr(module, "quant_method", None)
        backend = getattr(method, "nvfp4_backend", None)
        if backend is not None:
            name = backend.name
            if name not in native_moe:
                raise RuntimeError(f"Native FP4 MoE check rejected {name}")
            moe[name] = moe.get(name, 0) + 1
    if not linear:
        raise RuntimeError("No verified native FP4 Linear layers were found")
    return {"linear_kernels": linear, "moe_backends": moe}


class SmokeWorkerExtension:
    def axquant_native_kernels(self: Any) -> dict[str, Any]:
        return inspect_native_fp4(self.get_model())


def image_prompt(engine: Any, architecture: str) -> str:
    if architecture == "Qwen3VLForConditionalGeneration":
        return engine.get_tokenizer().apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "<|vision_start|><|image_pad|><|vision_end|>"
                    "Read all text in this image. Return only the exact text, one line per line.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return "<image>\nFree OCR."


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
        "Qwen3VLForConditionalGeneration": None,
    }
    architecture = config.get("architectures", [""])[0]
    if architecture not in processors:
        raise ValueError(
            "This development smoke supports official OCR2, Unlimited-OCR and Qwen3-VL"
        )
    require_runtime_layout(config)
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
        worker_extension_cls=f"{Path(__file__).stem}.SmokeWorkerExtension",
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
        kv_cache_memory_bytes=(512 if architecture == "Qwen3VLForConditionalGeneration" else 128)
        * 1024
        * 1024,
        limit_mm_per_prompt={"image": 1},
        skip_mm_profiling=True,
        mm_processor_cache_gb=0,
        mm_encoder_attn_backend="TORCH_SDPA",
        attention_config={"backend": "TRITON_ATTN"},
        compilation_config={"custom_ops": ["none"]},
        enable_prefix_caching=False,
        generation_config="vllm",
        kernel_config={"linear_backend": "cutlass", "moe_backend": "auto"}
        if args.require_native_fp4
        else {},
        logits_processors=[
            f"vllm.model_executor.models.{processors[architecture]}:NGramPerReqLogitsProcessor"
        ]
        if processors[architecture]
        else [],
    )
    kernels = engine.collective_rpc("axquant_native_kernels") if args.require_native_fp4 else []
    if args.require_native_fp4 and len(kernels) != 1:
        raise RuntimeError("Native kernel verification requires one worker")
    if args.require_native_fp4:
        plan = json.loads((source / "axquant_cuda_plan.json").read_text())
        require_kernel_coverage(plan, kernels[0])
    image = Path(args.image)
    result = engine.generate(
        [
            {
                "prompt": image_prompt(engine, architecture),
                "multi_modal_data": {"image": Image.open(image).convert("RGB")},
            }
        ],
        SamplingParams(
            temperature=0,
            max_tokens=128,
            min_tokens=16,
            skip_special_tokens=False,
            extra_args={"ngram_size": 35, "window_size": 128} if processors[architecture] else {},
        ),
    )[0].outputs[0]
    require_smoke_text(result.text)
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
            "verified_kernels": kernels,
            "requested_backends": {"linear": "cutlass", "moe": "auto"}
            if args.require_native_fp4
            else {"linear": "auto", "moe": "auto"},
            "input_image_sha256": file_sha256(image),
            "config_sha256": file_sha256(source / "config.json"),
            "minimum_tokens": 16,
            "generated_text": result.text,
            "generated_token_count": len(result.token_ids),
            "finish_reason": result.finish_reason,
            "expected_lines_present": True,
            "repetitive_output_detected": False,
        },
    )
    print(result.text, flush=True)


if __name__ == "__main__":
    main()
