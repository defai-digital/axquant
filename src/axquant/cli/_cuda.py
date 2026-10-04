"""Explicit experimental CUDA commands, separate from the MLX CLI contract."""

from __future__ import annotations

import argparse

from axquant.serde import load_model, write_data


def add_cuda_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    for name in ("plan-cuda", "convert-cuda", "quantize-cuda"):
        parser = subparsers.add_parser(name, help="Experimental native CUDA NVFP4/FP8, no AWQ")
        parser.add_argument("model", help="Local unquantized Safetensors checkpoint")
        parser.add_argument("--output", required=True)
        parser.add_argument("--q-mode", choices=("nvfp4", "fp8"), default="nvfp4")
        parser.add_argument("--allow-unmeasured", action="store_true")
        if name != "convert-cuda":
            parser.add_argument(
                "--model-id", help="Portable source identity; defaults to folder name"
            )
            parser.add_argument(
                "--revision", help="Optional immutable 40-character source revision"
            )
            parser.add_argument(
                "--keep", action="append", default=[], help="Preserve a tensor glob"
            )
        if name == "convert-cuda":
            parser.add_argument("--plan", required=True)
        if name != "plan-cuda":
            parser.add_argument(
                "--device", default="cuda", help="cuda, cuda:N, or explicit cpu oracle"
            )
            parser.add_argument("--rows-per-chunk", type=int, default=256)


def run_cuda_command(args: argparse.Namespace) -> int:
    import structlog

    from axquant.cuda import convert_cuda_nvfp4, plan_cuda_nvfp4
    from axquant.cuda_fp8 import convert_cuda_fp8, plan_cuda_fp8
    from axquant.schema.cuda import CudaQuantizationPlan
    from axquant.schema.cuda_fp8 import CudaFp8QuantizationPlan

    if args.command == "convert-cuda":
        if args.q_mode == "fp8":
            fp8_plan = load_model(args.plan, CudaFp8QuantizationPlan)
        else:
            plan = load_model(args.plan, CudaQuantizationPlan)
    else:
        parameters = dict(
            model_id=args.model_id,
            revision=args.revision,
            keep_patterns=args.keep,
            allow_unmeasured=args.allow_unmeasured,
        )
        if args.q_mode == "fp8":
            fp8_plan = plan_cuda_fp8(args.model, **parameters)
        else:
            plan = plan_cuda_nvfp4(args.model, **parameters)
    if args.command == "plan-cuda":
        write_data(args.output, fp8_plan if args.q_mode == "fp8" else plan)
        structlog.get_logger().info("cuda_plan_created", output=args.output, evidence="unmeasured")
        return 0
    if args.q_mode == "fp8":
        fp8_manifest = convert_cuda_fp8(
            args.model,
            fp8_plan,
            args.output,
            device=args.device,
            rows_per_chunk=args.rows_per_chunk,
            allow_unmeasured=args.allow_unmeasured,
        )
        structlog.get_logger().info(
            "cuda_fp8_pack_created",
            output=args.output,
            backend=fp8_manifest.backend,
            status=fp8_manifest.status,
            quantized_tensors=len(fp8_manifest.quantized_tensors),
        )
        return 0
    manifest = convert_cuda_nvfp4(
        args.model,
        plan,
        args.output,
        device=args.device,
        rows_per_chunk=args.rows_per_chunk,
        allow_unmeasured=args.allow_unmeasured,
    )
    structlog.get_logger().info(
        "cuda_nvfp4_pack_created",
        output=args.output,
        backend=manifest.backend,
        status=manifest.status,
        quantized_tensors=len(manifest.quantized_tensors),
    )
    return 0
