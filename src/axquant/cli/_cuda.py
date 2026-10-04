"""Explicit experimental CUDA commands, separate from the MLX CLI contract."""

from __future__ import annotations

import argparse

from axquant.serde import load_model, write_data


def add_cuda_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    for name in ("plan-cuda", "convert-cuda", "quantize-cuda"):
        parser = subparsers.add_parser(name, help="Experimental native NVFP4, no AWQ")
        parser.add_argument("model", help="Local unquantized Safetensors checkpoint")
        parser.add_argument("--output", required=True)
        parser.add_argument("--q-mode", choices=("nvfp4",), default="nvfp4")
        parser.add_argument("--allow-unmeasured", action="store_true")
        if name != "convert-cuda":
            parser.add_argument("--activation-bits", type=int, choices=(4, 16), default=16)
            parser.add_argument(
                "--activation-calibration", help="Exact-source-bound capture required for W4A4"
            )
            parser.add_argument(
                "--model-id", help="Portable source identity; defaults to folder name"
            )
            parser.add_argument(
                "--revision", help="Optional immutable 40-character source revision"
            )
            parser.add_argument(
                "--keep", action="append", default=[], help="Preserve a tensor glob"
            )
            parser.add_argument(
                "--embedding-protection",
                action="store_true",
                help="Preserve Qwen3 embedding attention and first/last two MLP blocks",
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

    from axquant.cuda import (
        convert_cuda_nvfp4,
        convert_cuda_nvfp4_w4a4,
        plan_cuda_nvfp4,
        plan_cuda_nvfp4_w4a4,
    )
    from axquant.errors import PlanningError
    from axquant.schema.cuda import CudaPackManifest, CudaQuantizationPlan
    from axquant.schema.cuda_activation import (
        CudaActivationCalibration,
        CudaW4A4PackManifest,
        CudaW4A4Plan,
    )
    from axquant.serde import read_data

    plan: CudaQuantizationPlan | CudaW4A4Plan
    if args.command == "convert-cuda":
        version = read_data(args.plan).get("schema_version")
        plan = (
            load_model(args.plan, CudaW4A4Plan)
            if version == "axquant.cuda-w4a4-plan.v1"
            else load_model(args.plan, CudaQuantizationPlan)
        )
    else:
        if args.activation_bits == 4 and not args.activation_calibration:
            raise PlanningError("NVFP4 W4A4 requires --activation-calibration")
        if args.activation_bits == 16 and args.activation_calibration:
            raise PlanningError("--activation-calibration requires --activation-bits 4")
        plan = plan_cuda_nvfp4(
            args.model,
            model_id=args.model_id,
            revision=args.revision,
            keep_patterns=args.keep,
            embedding_protection=args.embedding_protection,
            allow_unmeasured=args.allow_unmeasured,
        )
        if args.activation_bits == 4:
            calibration = load_model(args.activation_calibration, CudaActivationCalibration)
            plan = plan_cuda_nvfp4_w4a4(plan, calibration)
    if args.command == "plan-cuda":
        write_data(args.output, plan)
        structlog.get_logger().info("cuda_plan_created", output=args.output, evidence="unmeasured")
        return 0
    manifest: CudaPackManifest | CudaW4A4PackManifest
    if isinstance(plan, CudaW4A4Plan):
        manifest = convert_cuda_nvfp4_w4a4(
            args.model,
            plan,
            args.output,
            device=args.device,
            rows_per_chunk=args.rows_per_chunk,
            allow_unmeasured=args.allow_unmeasured,
        )
    else:
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
