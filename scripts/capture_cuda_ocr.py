#!/usr/bin/env python3
"""Capture source-bound BF16 OCR and Qwen3-VL inputs for development NVFP4 W4A4 export."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any

from axquant.cuda import _verify_source, plan_cuda_nvfp4_w4a4
from axquant.schema.cuda import CudaFileDigest, CudaQuantizationPlan
from axquant.schema.cuda_activation import CudaActivationCalibration
from axquant.serde import file_sha256, load_model, read_data, stable_sha256, write_data


def runtime_source_name(runtime_name: str, selected: dict[str, Any]) -> str:
    qwen_prefix = "language_model.model."
    if runtime_name.startswith(qwen_prefix) and any(
        name.startswith("model.language_model.") for name in selected
    ):
        return "model.language_model." + runtime_name.removeprefix(qwen_prefix)
    name = runtime_name.removeprefix("language_model.")
    if name.startswith("model.") and any(key.startswith("layers.") for key in selected):
        return name.removeprefix("model.")
    return name


def install_hooks(model: Any, source: str, allocations: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    selected = {item["tensor_name"]: item for item in allocations if item["method"] == "nvfp4"}
    stats: dict[str, dict[str, Any]] = {}
    handles: list[Any] = []
    cache: dict[str, Any] = {}

    def record(name: str, inputs: Any) -> None:
        if inputs.shape[-1] != selected[name]["shape"][1]:
            raise RuntimeError(f"Calibration shape differs for {name}")
        if not torch.isfinite(inputs).all():
            raise RuntimeError(f"Non-finite calibration input for {name}")
        maximum = float(inputs.abs().amax().float().item())
        count = inputs.numel() // inputs.shape[-1]
        old = stats.get(name)
        stats[name] = {
            "tensor_name": name,
            "input_columns": inputs.shape[-1],
            "sample_count": count + (old["sample_count"] if old else 0),
            "absolute_maximum": max(maximum, old["absolute_maximum"] if old else 0),
        }

    def weight(name: str, device: Any) -> Any:
        if name not in cache:
            with safe_open(str(Path(source) / selected[name]["source_file"]), framework="pt") as sf:
                cache[name] = sf.get_tensor(name).to(device=device)
        return cache[name]

    def make_hook(names: list[str], expert_names: list[str]):
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            inputs = args[0] if args else kwargs["hidden_states"]
            for name in names:
                record(name, inputs)
            if not expert_names:
                return
            # Replay every source expert on the observed BF16 hidden states.
            # This covers rare/unrouted experts without inventing measurements.
            for gate_name in expert_names:
                up_name = gate_name.removesuffix("gate_proj.weight") + "up_proj.weight"
                down_name = gate_name.removesuffix("gate_proj.weight") + "down_proj.weight"
                record(gate_name, inputs)
                record(up_name, inputs)
                gate_weight = weight(gate_name, inputs.device)
                up_weight = weight(up_name, inputs.device)
                flat = inputs.reshape(-1, inputs.shape[-1])
                for chunk in flat.split(64):
                    gate = torch.nn.functional.linear(chunk, gate_weight)
                    up = torch.nn.functional.linear(chunk, up_weight)
                    intermediate = torch.nn.functional.silu(gate) * up
                    record(down_name, intermediate)

        return hook

    matched: set[str] = set()
    for runtime_name, module in model.named_modules():
        module_name = runtime_source_name(runtime_name, selected)
        direct = module_name + ".weight"
        names = [direct] if direct in selected else []
        for fused_name, source_names in (
            ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
            ("gate_up_proj", ("gate_proj", "up_proj")),
        ):
            if module_name.endswith("." + fused_name):
                prefix = module_name.removesuffix(fused_name)
                names.extend(
                    prefix + item + ".weight"
                    for item in source_names
                    if prefix + item + ".weight" in selected
                )
        expert_names = [
            name
            for name in selected
            if name.startswith(module_name + ".")
            and name.endswith(".gate_proj.weight")
            and module_name.endswith(".experts")
        ]
        if names or expert_names:
            handles.append(
                module.register_forward_pre_hook(make_hook(names, expert_names), with_kwargs=True)
            )
            matched.update(names)
            for name in expert_names:
                matched.update(
                    [
                        name,
                        name.removesuffix("gate_proj.weight") + "up_proj.weight",
                        name.removesuffix("gate_proj.weight") + "down_proj.weight",
                    ]
                )
    missing = set(selected) - matched
    if missing:
        sample = [name for name, _ in model.named_modules() if ".layers.0." in name][:20]
        raise RuntimeError(
            f"Calibration runtime module coverage incomplete: {sorted(missing)[:8]}; "
            f"runtime={sample}"
        )
    model.axquant_activation_statistics = stats
    model.axquant_activation_handles = handles
    model.axquant_activation_source_cache = cache
    return {"matched_tensors": len(matched), "hooks": len(handles)}


def collect_statistics(model: Any) -> list[dict[str, Any]]:
    result = list(model.axquant_activation_statistics.values())
    for handle in model.axquant_activation_handles:
        handle.remove()
    model.axquant_activation_source_cache.clear()
    return result


class CalibrationWorkerExtension:
    """Expose named RPCs with primitive arguments; callable pickle is unnecessary."""

    def axquant_install(
        self: Any, source: str, allocations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return install_hooks(self.get_model(), source, allocations)

    def axquant_collect(self: Any) -> list[dict[str, Any]]:
        return collect_statistics(self.get_model())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    args = parser.parse_args()
    source = Path(args.model).resolve()
    plan = load_model(args.plan, CudaQuantizationPlan)
    _verify_source(source, plan)
    image_path = Path(args.image)
    image_digest = file_sha256(image_path)
    config = read_data(source / "config.json")
    architecture = config.get("architectures", [""])[0]
    processors = {
        "DeepseekOCR2ForCausalLM": "deepseek_ocr",
        "UnlimitedOCRForCausalLM": "unlimited_ocr",
        "Qwen3VLForConditionalGeneration": None,
    }
    if architecture not in processors:
        raise ValueError(
            "This development capture script supports official OCR2, Unlimited-OCR and Qwen3-VL"
        )
    from PIL import Image
    from vllm import LLM, SamplingParams

    engine = LLM(
        worker_extension_cls="capture_cuda_ocr.CalibrationWorkerExtension",
        model=str(source),
        model_impl="vllm",
        dtype=plan.activation_dtype,
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
        logits_processors=[
            f"vllm.model_executor.models.{processors[architecture]}:NGramPerReqLogitsProcessor"
        ]
        if processors[architecture]
        else [],
    )

    from smoke_cuda_ocr import image_prompt

    print(
        engine.collective_rpc(
            "axquant_install",
            args=(str(source), [item.model_dump() for item in plan.allocations]),
        ),
        flush=True,
    )
    outputs = engine.generate(
        [
            {
                "prompt": image_prompt(engine, architecture),
                "multi_modal_data": {"image": Image.open(image_path).convert("RGB")},
            }
        ],
        SamplingParams(
            temperature=0,
            max_tokens=128,
            min_tokens=16,
            skip_special_tokens=False,
            extra_args={"ngram_size": 35, "window_size": 128} if processors[architecture] else {},
        ),
    )
    print(outputs[0].outputs[0].text, flush=True)
    workers = engine.collective_rpc("axquant_collect")
    if len(workers) != 1:
        raise ValueError("Development capture requires one worker for complete local coverage")
    if file_sha256(image_path) != image_digest:
        raise ValueError("Calibration image changed during capture")
    _verify_source(source, plan)
    capture = CudaActivationCalibration(
        weight_plan_sha256=stable_sha256(plan),
        input_files=[
            CudaFileDigest(
                path=image_path.name, sha256=image_digest, size_bytes=image_path.stat().st_size
            )
        ],
        runtime="vllm",
        runtime_version=importlib.metadata.version("vllm"),
        source_precision=plan.activation_dtype,
        statistics=workers[0],
    )
    plan_cuda_nvfp4_w4a4(plan, capture)
    write_data(args.output, capture)
    print(f"Verified complete calibration: {len(capture.statistics)} tensors", flush=True)


if __name__ == "__main__":
    main()
