from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import structlog
from pydantic import ValidationError

from axquant.analyzer import architecture_prior_report
from axquant.architectures.registry import support_matrix
from axquant.benchmark import (
    parse_runtime_env_items,
    result_to_evaluation_bundle,
    run_benchmark,
    run_mtp_ab,
    run_mtp_diagnostics,
)
from axquant.calibration import prepare_calibration
from axquant.converter import convert_model
from axquant.errors import AxquantError, PlanningError
from axquant.feasibility import ArtifactTarget, assess_feasibility, feasibility_markdown
from axquant.inspector import inspect_model
from axquant.logging import configure_logging
from axquant.manual import manual_quantization_plan
from axquant.naming import model_name
from axquant.planner import allocate_kv_cache, allocate_kv_cache_measured, plan_quantization
from axquant.profiles import implemented_profiles, thresholds_for
from axquant.publisher import publish_model
from axquant.quantize import DEVELOPMENT_NOTE, quick_convert
from axquant.recipes import export_recipe_bundle
from axquant.reporting import plan_markdown, prepare_publication, validation_markdown
from axquant.reproduction import verify_reproduction
from axquant.runtime import check_ax_engine, check_mlx_lm_generation, check_mlx_lm_static
from axquant.schema import (
    BaselineKind,
    BenchmarkConfig,
    CalibrationManifest,
    EvaluationBundle,
    HardwareProfile,
    Inventory,
    KvSensitivityReport,
    ManualPlanRecipe,
    ModelIdentity,
    MtpPolicy,
    MtpSidecarLayout,
    PlanRequest,
    ProfileName,
    QualityEvaluationResult,
    QuantizationPlan,
    QuantMethod,
    RuntimeName,
    SensitivityReport,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data, write_text
from axquant.validator import validate_evaluations


def _load_matching_quality_evaluation(
    path: str | Path | None,
    model: ModelIdentity,
) -> QualityEvaluationResult | None:
    if path is None:
        return None
    result = load_model(path, QualityEvaluationResult)
    if result.model != model:
        raise ValueError("quality evaluation model does not match benchmark model")
    return result


def _bits(value: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if normalized == "bf16":
            parsed.append(16)
            continue
        try:
            parsed.append(int(normalized))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid precision {item!r}") from exc
    result = tuple(sorted(set(parsed)))
    if not result:
        raise argparse.ArgumentTypeError("at least one precision is required")
    return result


def _domains(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one calibration domain is required")
    return result


def _methods(value: str) -> tuple[QuantMethod, ...]:
    parsed: list[QuantMethod] = []
    for item in value.split(","):
        try:
            parsed.append(QuantMethod(item.strip().lower()))
        except ValueError as exc:
            choices = ", ".join(method.value for method in QuantMethod)
            raise argparse.ArgumentTypeError(
                f"quantization method must be one of {choices}"
            ) from exc
    result = tuple(sorted(set(parsed), key=lambda method: method.value))
    if not result:
        raise argparse.ArgumentTypeError("at least one quantization method is required")
    return result


def _probe_methods(value: str) -> tuple[QuantMethod, ...]:
    result = _methods(value)
    unsupported = set(result) - {QuantMethod.AFFINE, QuantMethod.DWQ}
    if unsupported:
        names = ", ".join(sorted(method.value for method in unsupported))
        raise argparse.ArgumentTypeError(f"measured probing cannot execute methods: {names}")
    return result


def _profile(value: str) -> ProfileName:
    try:
        profile = ProfileName(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in implemented_profiles())
        raise argparse.ArgumentTypeError(f"profile must be one of {choices}") from exc
    if profile not in implemented_profiles():
        choices = ", ".join(item.value for item in implemented_profiles())
        raise argparse.ArgumentTypeError(f"profile must be one of {choices}")
    return profile


def _output_json(path: str | Path, default_name: str) -> Path:
    output = Path(path).expanduser()
    return output if output.suffix.lower() == ".json" else output / default_name


def _iso_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if (
            not separator
            or not name
            or not raw_path
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in name)
        ):
            raise ValueError("exception evidence must use lowercase-name=path")
        if name in result:
            raise ValueError(f"duplicate exception evidence name: {name}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axquant",
        description="Plan, convert, and validate supported LLM checkpoints for MLX and AX Engine",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    feasibility_parser = subparsers.add_parser("feasibility")
    feasibility_parser.add_argument("--reference-4bit", required=True)
    feasibility_parser.add_argument("--reference-4bit-id")
    feasibility_parser.add_argument("--reference-4bit-revision")
    feasibility_parser.add_argument("--reference-6bit", required=True)
    feasibility_parser.add_argument("--reference-6bit-id")
    feasibility_parser.add_argument("--reference-6bit-revision")
    feasibility_parser.add_argument("--mixed-baseline")
    feasibility_parser.add_argument("--mixed-baseline-id")
    feasibility_parser.add_argument("--mixed-baseline-revision")
    feasibility_parser.add_argument("--source-bf16")
    feasibility_parser.add_argument("--source-bf16-id")
    feasibility_parser.add_argument("--source-bf16-revision")
    feasibility_parser.add_argument("--run-runtime-checks", action="store_true")
    feasibility_parser.add_argument("--ax-engine", default="ax-engine")
    feasibility_parser.add_argument("--require-ready", action="store_true")
    feasibility_parser.add_argument("--output", default="feasibility_report.json")
    feasibility_parser.add_argument(
        "--markdown-output",
        default="feasibility_report.md",
    )

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--model", required=True)
    inspect_parser.add_argument("--model-id")
    inspect_parser.add_argument("--revision")
    inspect_parser.add_argument("--output", default="architecture_report.json")
    inspect_parser.add_argument("--allow-download", action="store_true")
    inspect_parser.add_argument("--allow-quantized", action="store_true")

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--model", required=True)
    calibrate_parser.add_argument("--model-id")
    calibrate_parser.add_argument("--revision")
    calibrate_parser.add_argument("--tokenizer-revision")
    calibrate_parser.add_argument("--dataset", required=True)
    calibrate_parser.add_argument(
        "--profile",
        type=_profile,
        default=ProfileName.AGENT_CODING,
    )
    calibrate_parser.add_argument("--domains", type=_domains, default=["general"])
    calibrate_parser.add_argument("--max-seq-length", type=int, default=2048)
    calibrate_parser.add_argument("--seed", type=int, default=0)
    calibrate_parser.add_argument("--attest-calibration-eval-separation", action="store_true")
    calibrate_parser.add_argument("--manifest-only", action="store_true")
    calibrate_parser.add_argument("--output", default="calibration-cache")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--model", required=True)
    analyze_parser.add_argument("--model-id")
    analyze_parser.add_argument("--revision")
    analyze_parser.add_argument(
        "--profile",
        type=_profile,
        default=ProfileName.AGENT_CODING,
    )
    analyze_parser.add_argument("--bits", type=_bits, default=(4, 6, 8, 16))
    analyze_parser.add_argument(
        "--methods",
        type=_probe_methods,
        default=(QuantMethod.AFFINE,),
    )
    analyze_parser.add_argument("--group-size", type=int, default=64)
    analyze_parser.add_argument("--calibration")
    analyze_parser.add_argument("--base-sensitivity")
    analyze_parser.add_argument("--target-tensor", action="append", default=[])
    analyze_parser.add_argument("--token-budget", type=int, default=2048)
    analyze_parser.add_argument("--replay-batch-size", type=int, default=1)
    analyze_parser.add_argument("--metric-positions", type=int, default=32)
    analyze_parser.add_argument("--long-context-min-tokens", type=int, default=1024)
    analyze_parser.add_argument("--warmup-replays", type=int, default=1)
    analyze_parser.add_argument("--state")
    analyze_parser.add_argument("--output", default="sensitivity_map.json")
    analyze_parser.add_argument("--allow-download", action="store_true")

    analyze_kv_parser = subparsers.add_parser(
        "analyze-kv",
        help="Measure per-layer KV-cache sensitivity over a tokenized calibration cache",
    )
    analyze_kv_parser.add_argument("--model", required=True)
    analyze_kv_parser.add_argument("--model-id")
    analyze_kv_parser.add_argument("--revision")
    analyze_kv_parser.add_argument(
        "--profile",
        type=_profile,
        default=ProfileName.AGENT_CODING,
    )
    analyze_kv_parser.add_argument("--calibration", required=True)
    analyze_kv_parser.add_argument("--bits", type=_bits, default=(4, 6, 8))
    analyze_kv_parser.add_argument("--group-size", type=int, default=64)
    analyze_kv_parser.add_argument("--token-budget", type=int, default=2048)
    analyze_kv_parser.add_argument("--metric-positions", type=int, default=32)
    analyze_kv_parser.add_argument("--output", default="kv_sensitivity.json")

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--sensitivity", "--analysis", dest="analysis", required=True)
    plan_parser.add_argument("--target-bpw", type=float, default=4.8)
    plan_parser.add_argument("--bits", type=_bits, default=(4, 6, 8, 16))
    plan_parser.add_argument(
        "--methods",
        type=_methods,
        default=(QuantMethod.AFFINE, QuantMethod.DWQ, QuantMethod.BF16),
    )
    plan_parser.add_argument("--group-size", type=int, default=64)
    plan_parser.add_argument("--minimum-quality", type=float, default=0.98)
    plan_parser.add_argument("--minimum-mtp-retention", type=float, default=0.95)
    plan_parser.add_argument("--minimum-mtp-speedup", type=float, default=1.20)
    plan_parser.add_argument("--max-size-ratio", type=float, default=1.10)
    plan_parser.add_argument("--candidates", type=int, default=1)
    plan_parser.add_argument(
        "--mode",
        choices=["balanced", "quality", "low-memory", "speed"],
        default="balanced",
    )
    plan_parser.add_argument("--seed", type=int, default=0)
    plan_parser.add_argument(
        "--mtp",
        choices=["protected", "adaptive", "disabled"],
        default="protected",
    )
    plan_parser.add_argument("--mtp-bits", type=_bits, default=(8, 16))
    plan_parser.add_argument("--mtp-min-bits", type=int, default=8)
    plan_parser.add_argument("--allow-unmeasured", action="store_true")
    plan_parser.add_argument("--kv-cache", choices=["off", "prior", "measured"], default="off")
    plan_parser.add_argument("--kv-default-bits", type=int, default=4)
    plan_parser.add_argument("--kv-analysis", help="KV sensitivity report for --kv-cache measured")
    plan_parser.add_argument("--kv-max-kl", type=float, default=0.005)
    plan_parser.add_argument("--output", default="quantization-plans")

    manual_plan_parser = subparsers.add_parser("plan-manual")
    manual_plan_parser.add_argument("--inventory", required=True)
    manual_plan_parser.add_argument("--recipe", required=True)
    manual_plan_parser.add_argument("--output", default="manual-plan.json")
    manual_plan_parser.add_argument("--markdown-output")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--model", required=True)
    convert_parser.add_argument("--revision")
    convert_parser.add_argument("--plan", required=True)
    convert_parser.add_argument("--mtp-sidecar")
    convert_parser.add_argument(
        "--mtp-layout",
        choices=tuple(layout.value for layout in MtpSidecarLayout),
        default=MtpSidecarLayout.BYTE_PRESERVED.value,
        help="External MTP handling; transformed Qwen 3.6 layout requires explicit opt-in",
    )
    convert_parser.add_argument("--calibration-manifest")
    convert_parser.add_argument(
        "--kv-sensitivity",
        help="KV sensitivity report bound by a measured KV-cache plan (AXQ-025)",
    )
    convert_parser.add_argument("--allow-unmeasured", action="store_true")
    convert_parser.add_argument(
        "--ax-engine-manifest",
        choices=["required", "if-available", "skip"],
        default="required",
    )
    convert_parser.add_argument("--ax-engine-bench", default="ax-engine-bench")
    convert_parser.add_argument("--output", required=True)

    quantize_parser = subparsers.add_parser(
        "quantize",
        help="One-command development conversion: inspect, plan from priors, convert",
    )
    quantize_parser.add_argument("--model", required=True)
    quantize_parser.add_argument("--model-id")
    quantize_parser.add_argument("--revision")
    quantize_parser.add_argument("--output", required=True)
    quantize_parser.add_argument(
        "--profile",
        choices=[profile.value for profile in implemented_profiles()],
        default=ProfileName.GENERAL.value,
    )
    quantize_parser.add_argument("--target-bpw", type=float, default=4.8)
    quantize_parser.add_argument("--kv-cache", choices=["off", "prior"], default="off")
    quantize_parser.add_argument(
        "--recipe",
        help="Recipe bundle file or directory; replaces prior-based planning",
    )
    quantize_parser.add_argument("--calibration-manifest")
    quantize_parser.add_argument("--mtp-sidecar")
    quantize_parser.add_argument(
        "--runtime-smoke",
        choices=["none", "mlx-lm", "ax-engine"],
        default="none",
    )
    quantize_parser.add_argument("--ax-engine", default="ax-engine")
    quantize_parser.add_argument("--mlx-lm", default="mlx_lm.generate")
    quantize_parser.add_argument(
        "--ax-engine-manifest",
        choices=["required", "if-available", "skip"],
        default="if-available",
    )
    quantize_parser.add_argument("--json", dest="json_output")

    recipe_export_parser = subparsers.add_parser(
        "recipe-export",
        help="Export a plan as a checksummed recipe bundle",
    )
    recipe_export_parser.add_argument("--plan", required=True)
    recipe_export_parser.add_argument("--bundle-id", required=True)
    recipe_export_parser.add_argument("--output-dir", required=True)
    recipe_export_parser.add_argument(
        "--lineage",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="Digest of a producing evidence artifact; repeatable",
    )
    recipe_export_parser.add_argument("--note", action="append", default=[])

    support_matrix_parser = subparsers.add_parser(
        "support-matrix",
        help="List every registered model family with its declared support tier",
    )
    support_matrix_parser.add_argument("--output", help="Optional JSON output path")

    head_to_head_parser = subparsers.add_parser(
        "head-to-head",
        help="Render the public comparison page from a bound benchmark evidence index",
    )
    head_to_head_parser.add_argument("--benchmark-index", required=True)
    head_to_head_parser.add_argument("--title")
    head_to_head_parser.add_argument("--output", default="head-to-head.md")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--reference-evaluation", required=True)
    validate_parser.add_argument("--candidate-direct-evaluation", required=True)
    validate_parser.add_argument("--candidate-evaluation", required=True)
    validate_parser.add_argument("--calibration-manifest")
    validate_parser.add_argument("--size-reference")
    validate_parser.add_argument("--candidate-size")
    validate_parser.add_argument("--plan")
    validate_parser.add_argument("--release-exception")
    validate_parser.add_argument(
        "--exception-evidence",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    validate_parser.add_argument(
        "--profile",
        type=_profile,
        default=ProfileName.AGENT_CODING,
    )
    validate_parser.add_argument("--output", default="benchmark_report.json")

    size_parser = subparsers.add_parser("size-evidence")
    size_parser.add_argument("--artifact-manifest")
    size_parser.add_argument("--feasibility-report")
    size_parser.add_argument("--model-id")
    size_parser.add_argument("--revision")
    size_parser.add_argument("--output", default="artifact_size_evidence.json")

    exception_parser = subparsers.add_parser("release-exception")
    exception_parser.add_argument("--exception-id", required=True)
    exception_parser.add_argument("--plan", required=True)
    exception_parser.add_argument("--candidate-size", required=True)
    exception_parser.add_argument("--size-reference", required=True)
    exception_parser.add_argument("--tradeoff-evidence", required=True)
    exception_parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    exception_parser.add_argument("--max-weight-size-ratio", type=float, default=1.10)
    exception_parser.add_argument("--minimum-measured-bpw", type=float, default=4.3)
    exception_parser.add_argument("--maximum-measured-bpw", type=float, default=4.8)
    exception_parser.add_argument("--measured-tradeoff", required=True)
    exception_parser.add_argument("--owner", required=True)
    exception_parser.add_argument("--approved-by", required=True)
    exception_parser.add_argument("--approval-reference", required=True)
    exception_parser.add_argument("--approved-at", type=_iso_datetime, required=True)
    exception_parser.add_argument("--expires-at", type=_iso_datetime, required=True)
    exception_parser.add_argument("--output", default="release_exception.json")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--plan", required=True)
    report_parser.add_argument("--validation")
    report_parser.add_argument("--output", default="benchmark_report.md")

    prepare_parser = subparsers.add_parser("publish-prepare")
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--repo", required=True)
    prepare_parser.add_argument("--validation-index", required=True)
    prepare_parser.add_argument("--hardware-registry", required=True)
    prepare_parser.add_argument("--pareto-report", required=True)

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--model", required=True)
    publish_parser.add_argument("--repo", required=True)
    publish_parser.add_argument("--validation-index", required=True)
    publish_parser.add_argument("--hardware-registry", required=True)
    publish_parser.add_argument("--pareto-report", required=True)
    publish_parser.add_argument("--release-audit")
    publish_parser.add_argument("--release-audit-request")
    publish_parser.add_argument("--private", action="store_true")
    publish_parser.add_argument("--yes", action="store_true")

    reproduction_parser = subparsers.add_parser("verify-reproduction")
    reproduction_parser.add_argument("--recipe", required=True)
    reproduction_parser.add_argument("--artifact", required=True)
    reproduction_parser.add_argument("--output", default="reproduction_verification.json")

    name_parser = subparsers.add_parser("name")
    name_parser.add_argument("--base", required=True)
    name_parser.add_argument("--target-class", default="4bit")
    name_parser.add_argument("--owner", default="AutomatosX")
    name_parser.add_argument("--mtp-suffix", action="store_true")
    name_parser.add_argument("--no-mlx", action="store_true")

    runtime_parser = subparsers.add_parser("runtime-check")
    runtime_parser.add_argument("--model", required=True)
    runtime_parser.add_argument("--model-id")
    runtime_parser.add_argument("--revision")
    runtime_parser.add_argument(
        "--runtime",
        choices=[runtime.value for runtime in RuntimeName],
        default=RuntimeName.AX_ENGINE.value,
    )
    runtime_parser.add_argument("--ax-engine", default="ax-engine")
    runtime_parser.add_argument("--mlx-lm", default="mlx_lm.generate")
    runtime_parser.add_argument("--static-only", action="store_true")
    runtime_parser.add_argument("--output", default="runtime_check.json")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--model", required=True)
    benchmark_parser.add_argument("--model-id")
    benchmark_parser.add_argument("--revision")
    benchmark_parser.add_argument("--prompts", required=True)
    benchmark_parser.add_argument("--workload", default="agent-coding")
    benchmark_parser.add_argument("--mtp", action="store_true", default=False)
    benchmark_parser.add_argument(
        "--baseline-kind",
        default="candidate",
        choices=[
            "bf16",
            "uniform-4bit",
            "uniform-6bit",
            "mixed-precision",
            "awq",
            "dwq",
            "axquant-mtp-off",
            "axquant-mtp-on",
            "candidate",
        ],
    )
    benchmark_parser.add_argument("--trials", type=int, default=5)
    benchmark_parser.add_argument("--warmup", type=int, default=2)
    benchmark_parser.add_argument("--max-tokens", type=int, default=512)
    benchmark_parser.add_argument("--temperature", type=float, default=0.0)
    benchmark_parser.add_argument("--draft-depth", type=int)
    benchmark_parser.add_argument("--power-mode")
    benchmark_parser.add_argument("--quantizer")
    benchmark_parser.add_argument("--quantizer-version")
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument("--timeout", type=float, default=300.0)
    benchmark_parser.add_argument("--ax-engine", default="ax-engine-bench")
    benchmark_parser.add_argument(
        "--runtime-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Allowlisted AX Engine env control (repeatable), e.g. "
        "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL=0",
    )
    benchmark_parser.add_argument("--output", default="evaluation_bundle.json")
    benchmark_parser.add_argument("--log-dir")
    benchmark_parser.add_argument("--quality-evaluation")

    benchmark_ab_parser = subparsers.add_parser("benchmark-ab")
    benchmark_ab_parser.add_argument("--model", required=True)
    benchmark_ab_parser.add_argument("--model-id")
    benchmark_ab_parser.add_argument("--revision")
    benchmark_ab_parser.add_argument("--prompts", required=True)
    benchmark_ab_parser.add_argument("--workload", default="agent-coding")
    benchmark_ab_parser.add_argument(
        "--direct-baseline-kind",
        default="axquant-mtp-off",
        choices=[
            "bf16",
            "uniform-4bit",
            "uniform-6bit",
            "mixed-precision",
            "awq",
            "dwq",
            "axquant-mtp-off",
            "axquant-mtp-on",
            "candidate",
        ],
    )
    benchmark_ab_parser.add_argument(
        "--mtp-baseline-kind",
        default="axquant-mtp-on",
        choices=[
            "bf16",
            "uniform-4bit",
            "uniform-6bit",
            "mixed-precision",
            "awq",
            "dwq",
            "axquant-mtp-off",
            "axquant-mtp-on",
            "candidate",
        ],
    )
    benchmark_ab_parser.add_argument("--trials", type=int, default=5)
    benchmark_ab_parser.add_argument("--warmup", type=int, default=2)
    benchmark_ab_parser.add_argument("--max-tokens", type=int, default=512)
    benchmark_ab_parser.add_argument("--temperature", type=float, default=0.0)
    benchmark_ab_parser.add_argument("--draft-depth", type=int)
    benchmark_ab_parser.add_argument(
        "--minimum-speedup",
        type=float,
        default=1.20,
        help="Fail closed unless MTP median throughput is at least this multiple of direct",
    )
    benchmark_ab_parser.add_argument(
        "--record-failed-speedup",
        action="store_true",
        help=(
            "Write complete A/B evidence when only the speed gate fails, then return status 1; "
            "exactness and matched-control invariants remain fail-closed"
        ),
    )
    benchmark_ab_parser.add_argument("--power-mode")
    benchmark_ab_parser.add_argument("--quantizer")
    benchmark_ab_parser.add_argument("--quantizer-version")
    benchmark_ab_parser.add_argument("--seed", type=int, default=0)
    benchmark_ab_parser.add_argument("--timeout", type=float, default=300.0)
    benchmark_ab_parser.add_argument("--ax-engine", default="ax-engine-bench")
    benchmark_ab_parser.add_argument(
        "--runtime-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Allowlisted AX Engine env control applied to both arms (repeatable)",
    )
    benchmark_ab_parser.add_argument("--output-dir", default="benchmark-ab")
    benchmark_ab_parser.add_argument("--quality-evaluation")

    mtp_diagnose_parser = subparsers.add_parser(
        "mtp-diagnose",
        help="Run M2 kill-switch matrix (soft exactness) and write a diagnostic report",
    )
    mtp_diagnose_parser.add_argument("--model", required=True)
    mtp_diagnose_parser.add_argument("--model-id")
    mtp_diagnose_parser.add_argument("--revision")
    mtp_diagnose_parser.add_argument("--prompts", required=True)
    mtp_diagnose_parser.add_argument("--workload", default="agent-coding")
    mtp_diagnose_parser.add_argument("--trials", type=int, default=5)
    mtp_diagnose_parser.add_argument("--warmup", type=int, default=2)
    mtp_diagnose_parser.add_argument("--max-tokens", type=int, default=64)
    mtp_diagnose_parser.add_argument("--temperature", type=float, default=0.0)
    mtp_diagnose_parser.add_argument("--draft-depth", type=int, default=1)
    mtp_diagnose_parser.add_argument("--power-mode")
    mtp_diagnose_parser.add_argument("--quantizer")
    mtp_diagnose_parser.add_argument("--quantizer-version")
    mtp_diagnose_parser.add_argument("--seed", type=int, default=0)
    mtp_diagnose_parser.add_argument("--timeout", type=float, default=300.0)
    mtp_diagnose_parser.add_argument("--ax-engine", default="ax-engine-bench")
    mtp_diagnose_parser.add_argument(
        "--profile",
        action="append",
        default=[],
        dest="profiles",
        help="Diagnostic profile name (repeatable). Default: baseline, "
        "disable-post-input-metal, disable-la-decode-metal",
    )
    mtp_diagnose_parser.add_argument(
        "--runtime-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra allowlisted env applied to every profile (repeatable)",
    )
    mtp_diagnose_parser.add_argument(
        "--minimum-mtp-speedup",
        type=float,
        default=1.20,
        help="Release speedup floor used for release_ready scoring (default 1.20)",
    )
    mtp_diagnose_parser.add_argument("--output-dir", default="mtp-diagnose")
    mtp_diagnose_parser.add_argument("--output", default="mtp_diagnostic_report.json")

    quality_parser = subparsers.add_parser("evaluate-quality")
    quality_parser.add_argument("--model", required=True)
    quality_parser.add_argument("--model-id")
    quality_parser.add_argument("--revision")
    quality_parser.add_argument("--dataset", required=True)
    quality_parser.add_argument("--max-seq-length", type=int, default=2048)
    quality_parser.add_argument("--max-tokens", type=int, default=256)
    quality_parser.add_argument("--max-samples", type=int)
    quality_parser.add_argument("--seed", type=int, default=0)
    quality_parser.add_argument("--output", default="quality_evaluation.json")

    quality_compare_parser = subparsers.add_parser("compare-quality")
    quality_compare_parser.add_argument("--reference", required=True)
    quality_compare_parser.add_argument("--candidate", required=True)
    quality_compare_parser.add_argument("--output", default="quality_comparison.json")

    tokenize_parser = subparsers.add_parser("tokenize-calibration")
    tokenize_parser.add_argument("--model", required=True)
    tokenize_parser.add_argument("--model-id")
    tokenize_parser.add_argument("--revision")
    tokenize_parser.add_argument("--dataset", required=True)
    tokenize_parser.add_argument(
        "--profile",
        type=_profile,
        default=ProfileName.AGENT_CODING,
    )
    tokenize_parser.add_argument("--max-seq-length", type=int, default=2048)
    tokenize_parser.add_argument("--seed", type=int, default=0)
    tokenize_parser.add_argument("--domains", type=_domains, default=["general"])
    suite_parser = subparsers.add_parser("prepare-suite")
    suite_parser.add_argument("--output-dir", required=True)
    suite_parser.add_argument("--seed", type=int, default=20260728)
    tokenize_parser.add_argument("--tokenizer-revision")
    tokenize_parser.add_argument("--attest-calibration-eval-separation", action="store_true")
    tokenize_parser.add_argument("--output", default="calibration-cache")

    refine_parser = subparsers.add_parser("refine")
    refine_parser.add_argument("--sensitivity", "--analysis", dest="analysis", required=True)
    refine_parser.add_argument("--target-bpw", type=float, default=4.5)
    refine_parser.add_argument("--bits", type=_bits, default=(4, 6, 8, 16))
    refine_parser.add_argument("--group-size", type=int, default=64)
    refine_parser.add_argument("--top-n", type=int, default=3)
    refine_parser.add_argument("--max-iterations", type=int, default=10)
    refine_parser.add_argument("--eval-budget", type=int, default=50)
    refine_parser.add_argument("--wall-clock", type=float, default=86400.0)
    refine_parser.add_argument("--convergence", type=float, default=0.001)
    refine_parser.add_argument("--swap-radius", type=int, default=5)
    refine_parser.add_argument("--seed", type=int, default=0)
    refine_parser.add_argument("--allow-unmeasured", action="store_true")
    refine_parser.add_argument("--output", default="refinement_result.json")
    refine_select_parser = subparsers.add_parser("refine-select")
    refine_select_parser.add_argument("--refinement", required=True)
    refine_select_parser.add_argument("--measurements", required=True)
    refine_select_parser.add_argument("--output", default="refinement_selected.json")
    refine_measure_parser = subparsers.add_parser("refine-measure")
    refine_measure_parser.add_argument("--refinement", required=True)
    refine_measure_parser.add_argument("--candidate-id", required=True)
    refine_measure_parser.add_argument("--measurement-id")
    refine_measure_parser.add_argument("--artifact-manifest", required=True)
    refine_measure_parser.add_argument("--quality-comparison", required=True)
    refine_measure_parser.add_argument("--validation", required=True)
    refine_measure_parser.add_argument("--existing")
    refine_measure_parser.add_argument("--output", default="refinement_measurements.json")
    refine_export_parser = subparsers.add_parser("refine-export")
    refine_export_parser.add_argument("--refinement", required=True)
    refine_export_parser.add_argument("--output-dir", required=True)
    refine_run_parser = subparsers.add_parser("refine-run")
    refine_run_parser.add_argument("--request", required=True)
    refine_run_parser.add_argument("--output-dir", required=True)
    refine_run_parser.add_argument("--execute", action="store_true")
    pareto_parser = subparsers.add_parser("pareto")
    pareto_parser.add_argument("--measurements", required=True)
    pareto_parser.add_argument("--output", default="pareto_report.json")
    hardware_registry_parser = subparsers.add_parser("hardware-registry")
    hardware_registry_parser.add_argument("--request", required=True)
    hardware_registry_parser.add_argument("--output", default="hardware_profile_registry.json")
    compatibility_parser = subparsers.add_parser("compatibility-matrix")
    compatibility_parser.add_argument("--request", required=True)
    compatibility_parser.add_argument("--output", default="compatibility_matrix.json")
    evidence_parser = subparsers.add_parser("benchmark-index")
    evidence_parser.add_argument("--request", required=True)
    evidence_parser.add_argument("--output", default="benchmark_evidence_index.json")
    validation_index_parser = subparsers.add_parser("validation-index")
    validation_index_parser.add_argument("--request", required=True)
    validation_index_parser.add_argument("--output", default="release_validation_index.json")
    release_audit_parser = subparsers.add_parser("release-audit")
    release_audit_parser.add_argument("--request", required=True)
    release_audit_parser.add_argument("--output", default="release_audit.json")

    return parser


def _run(args: argparse.Namespace) -> int:
    log = structlog.get_logger()
    if args.command == "feasibility":
        report = assess_feasibility(
            reference_4bit=ArtifactTarget(
                model=args.reference_4bit,
                kind=BaselineKind.UNIFORM_4BIT,
                model_id=args.reference_4bit_id,
                revision=args.reference_4bit_revision,
            ),
            reference_6bit=ArtifactTarget(
                model=args.reference_6bit,
                kind=BaselineKind.UNIFORM_6BIT,
                model_id=args.reference_6bit_id,
                revision=args.reference_6bit_revision,
            ),
            mixed_baseline=(
                ArtifactTarget(
                    model=args.mixed_baseline,
                    kind=BaselineKind.MIXED_PRECISION,
                    model_id=args.mixed_baseline_id,
                    revision=args.mixed_baseline_revision,
                )
                if args.mixed_baseline
                else None
            ),
            source_bf16=(
                ArtifactTarget(
                    model=args.source_bf16,
                    kind=BaselineKind.BF16_SOURCE,
                    model_id=args.source_bf16_id,
                    revision=args.source_bf16_revision,
                )
                if args.source_bf16
                else None
            ),
            run_runtime_checks=args.run_runtime_checks,
            ax_engine=args.ax_engine,
        )
        write_data(args.output, report)
        write_text(args.markdown_output, feasibility_markdown(report))
        log.info(
            "feasibility_completed",
            output=str(args.output),
            markdown_output=str(args.markdown_output),
            status=report.status,
        )
        if report.status == "blocked":
            return 1
        if args.require_ready and report.status != "ready-for-conversion":
            return 1
        return 0

    if args.command == "inspect":
        inventory = inspect_model(
            args.model,
            model_id=args.model_id,
            revision=args.revision,
            allow_download=args.allow_download,
            allow_quantized=args.allow_quantized,
        )
        write_data(args.output, inventory)
        log.info(
            "inspection_completed",
            output=str(args.output),
            tensors=len(inventory.tensors),
            mtp=inventory.mtp_present,
        )
        return 0

    if args.command == "calibrate":
        model_path = Path(args.model).expanduser()
        model_identity = ModelIdentity(
            model_id=args.model_id or args.model,
            revision=args.revision,
            local_path=str(model_path.resolve()) if model_path.is_dir() else None,
        )
        manifest = prepare_calibration(
            model=model_identity,
            dataset=args.dataset,
            output_dir=args.output,
            profile=args.profile,
            domains=args.domains,
            sequence_length=args.max_seq_length,
            random_seed=args.seed,
            separation_attested=args.attest_calibration_eval_separation,
        )
        tokenized_manifest = None
        if not args.manifest_only:
            from axquant.activation_cache import tokenize_calibration

            tokenized_manifest = tokenize_calibration(
                model=model_identity,
                dataset_path=args.dataset,
                output_dir=args.output,
                profile=args.profile,
                sequence_length=args.max_seq_length,
                random_seed=args.seed,
                tokenizer_revision=args.tokenizer_revision,
                calibration_manifest_sha256=stable_sha256(
                    manifest.model_dump(mode="json", exclude={"created_at"})
                ),
                separation_attested=args.attest_calibration_eval_separation,
                domains=args.domains,
            )
        log.info(
            "calibration_created",
            output=str(Path(args.output) / "calibration_manifest.json"),
            samples=manifest.samples,
            tokenized=tokenized_manifest is not None,
            tokens=tokenized_manifest.total_tokens if tokenized_manifest is not None else 0,
        )
        return 0

    if args.command == "analyze":
        if (args.base_sensitivity or args.target_tensor) and not args.calibration:
            raise ValueError(
                "--base-sensitivity/--target-tensor require measured --calibration probing"
            )
        inventory = inspect_model(
            args.model,
            model_id=args.model_id,
            revision=args.revision,
            allow_download=args.allow_download,
        )
        if args.calibration:
            from axquant.probe import probe_tensor_sensitivity
            from axquant.schema import ProbeConfig

            analysis_report = probe_tensor_sensitivity(
                inventory,
                config=ProbeConfig(
                    model=inventory.model,
                    calibration_cache=args.calibration,
                    profile=args.profile,
                    candidate_bits=args.bits,
                    candidate_methods=args.methods,
                    target_tensors=tuple(args.target_tensor),
                    group_size=args.group_size,
                    token_budget_per_candidate=args.token_budget,
                    replay_batch_size=args.replay_batch_size,
                    metric_positions_per_sample=args.metric_positions,
                    long_context_min_tokens=args.long_context_min_tokens,
                    warmup_replays=args.warmup_replays,
                ),
                state_path=(args.state or str(Path(args.output).with_suffix(".progress.json"))),
                base_report=(
                    load_model(args.base_sensitivity, SensitivityReport)
                    if args.base_sensitivity
                    else None
                ),
            )
        else:
            analysis_report = architecture_prior_report(
                inventory,
                profile=args.profile,
                candidate_bits=args.bits,
                group_size=args.group_size,
            )
        write_data(args.output, analysis_report)
        log_method = log.info if analysis_report.evidence_kind.release_quality else log.warning
        log_method(
            "sensitivity_analysis_created",
            output=str(args.output),
            entries=len(analysis_report.entries),
            evidence=analysis_report.evidence_kind.value,
        )
        return 0

    if args.command == "analyze-kv":
        from axquant.kv_probe import measure_kv_sensitivity

        inventory = inspect_model(
            args.model,
            model_id=args.model_id,
            revision=args.revision,
        )
        kv_report = measure_kv_sensitivity(
            inventory,
            model_dir=args.model,
            calibration_cache=args.calibration,
            profile=args.profile,
            candidate_bits=args.bits,
            group_size=args.group_size,
            token_budget=args.token_budget,
            metric_positions=args.metric_positions,
        )
        write_data(args.output, kv_report)
        log.warning(
            "kv_sensitivity_created",
            output=str(args.output),
            layers=kv_report.text_layer_count,
            evidence=kv_report.evidence_kind.value,
        )
        return 0

    if args.command == "plan":
        analysis_report = load_model(args.analysis, SensitivityReport)
        request = PlanRequest(
            profile=analysis_report.profile,
            target_bpw=args.target_bpw,
            candidate_bits=args.bits,
            group_size=args.group_size,
            allow_unmeasured=args.allow_unmeasured,
            candidate_count=args.candidates,
            random_seed=args.seed,
            target_mode=args.mode,
            hardware=HardwareProfile(supported_methods=args.methods),
            max_model_size_ratio_to_uniform4=args.max_size_ratio,
            minimum_quality_retention=args.minimum_quality,
            minimum_mtp_acceptance_retention=args.minimum_mtp_retention,
            minimum_mtp_speedup=args.minimum_mtp_speedup,
            mtp=MtpPolicy(
                mode=args.mtp,
                candidate_bits=args.mtp_bits,
                min_bits=args.mtp_min_bits,
            ),
        )
        plan = plan_quantization(analysis_report, request)
        if args.kv_cache == "prior":
            layer_count = analysis_report.architecture_profile.text_layer_count
            if layer_count is None:
                raise PlanningError("KV-cache planning requires a known text layer count")
            plan.kv_cache = allocate_kv_cache(
                layer_count,
                default_bits=args.kv_default_bits,
                group_size=args.group_size,
            )
        elif args.kv_cache == "measured":
            if not args.kv_analysis:
                raise PlanningError("--kv-cache measured requires --kv-analysis")
            kv_report = load_model(args.kv_analysis, KvSensitivityReport)
            if kv_report.model.model_id != plan.source_model.model_id:
                raise PlanningError("KV sensitivity report model does not match the plan model")
            plan.kv_cache = allocate_kv_cache_measured(
                kv_report,
                max_output_kl=args.kv_max_kl,
            )
        output = _output_json(args.output, "plan-01.json")
        write_data(output, plan)
        log.info(
            "plan_created",
            output=str(output),
            effective_bpw=round(plan.effective_bpw, 6),
            evidence=plan.evidence_kind.value,
            kv_cache=(plan.kv_cache.allocation_basis if plan.kv_cache else "off"),
        )
        return 0

    if args.command == "plan-manual":
        inventory = load_model(args.inventory, Inventory)
        recipe = load_model(args.recipe, ManualPlanRecipe)
        plan = manual_quantization_plan(inventory, recipe)
        write_data(args.output, plan)
        if args.markdown_output:
            write_text(args.markdown_output, plan_markdown(plan))
        log.warning(
            "unmeasured_manual_plan_created",
            output=str(args.output),
            effective_bpw=round(plan.effective_bpw, 6),
        )
        return 0

    if args.command == "convert":
        plan = load_model(args.plan, QuantizationPlan)
        convert_model(
            model=args.model,
            plan=plan,
            output=args.output,
            revision=args.revision,
            mtp_sidecar=args.mtp_sidecar,
            mtp_layout=MtpSidecarLayout(args.mtp_layout),
            calibration_manifest=args.calibration_manifest,
            kv_sensitivity=args.kv_sensitivity,
            allow_unmeasured=args.allow_unmeasured,
            ax_engine_manifest=args.ax_engine_manifest,
            ax_engine_bench=args.ax_engine_bench,
        )
        return 0

    if args.command == "quantize":
        summary = quick_convert(
            model=args.model,
            output=args.output,
            model_id=args.model_id,
            revision=args.revision,
            profile=ProfileName(args.profile),
            target_bpw=args.target_bpw,
            kv_cache=args.kv_cache,
            recipe=args.recipe,
            calibration_manifest=args.calibration_manifest,
            mtp_sidecar=args.mtp_sidecar,
            runtime_smoke=args.runtime_smoke,
            ax_engine=args.ax_engine,
            mlx_lm=args.mlx_lm,
            ax_engine_manifest=args.ax_engine_manifest,
        )
        if args.json_output:
            write_data(args.json_output, summary)
        log.info(
            "quantize_completed",
            output=summary.output_path,
            family=summary.product_family,
            tier=summary.support_tier.value,
            evidence=summary.evidence_kind.value,
            plan_source=summary.plan_source,
            measured_bpw=round(summary.measured_total_bpw, 4),
            runtime_smoke=summary.runtime_smoke,
            runtime_smoke_passed=summary.runtime_smoke_passed,
        )
        if summary.development_evidence:
            log.warning("development_evidence", note=DEVELOPMENT_NOTE)
        return 0 if summary.runtime_smoke_passed is not False else 1

    if args.command == "head-to-head":
        from axquant.head_to_head import render_head_to_head

        page = render_head_to_head(args.benchmark_index, title=args.title)
        write_text(args.output, page)
        log.info("head_to_head_written", output=str(args.output))
        return 0

    if args.command == "support-matrix":
        families = support_matrix()
        for family_entry in families.entries:
            log.info(
                "support_matrix_entry",
                adapter=family_entry.adapter_id,
                family=family_entry.product_family,
                tier=family_entry.support_tier.value,
            )
        if args.output:
            write_data(args.output, families)
            log.info("support_matrix_written", output=str(args.output))
        return 0

    if args.command == "recipe-export":
        lineage: dict[str, str] = {}
        for item in args.lineage:
            name, separator, digest = item.partition("=")
            if not separator or not name or not digest:
                raise ValueError(f"lineage entries use NAME=SHA256 form, got {item!r}")
            lineage[name] = digest
        bundle_path = export_recipe_bundle(
            plan=args.plan,
            output_dir=args.output_dir,
            bundle_id=args.bundle_id,
            lineage=lineage,
            notes=args.note,
        )
        log.info(
            "recipe_bundle_exported",
            bundle=str(bundle_path),
            bundle_id=args.bundle_id,
        )
        return 0

    if args.command == "validate":
        reference = load_model(args.reference_evaluation, EvaluationBundle)
        candidate_direct = load_model(args.candidate_direct_evaluation, EvaluationBundle)
        candidate = load_model(args.candidate_evaluation, EvaluationBundle)
        calibration = (
            load_model(args.calibration_manifest, CalibrationManifest)
            if args.calibration_manifest
            else None
        )
        from axquant.schema import ArtifactSizeEvidence

        size_reference = (
            load_model(args.size_reference, ArtifactSizeEvidence) if args.size_reference else None
        )
        candidate_size = (
            load_model(args.candidate_size, ArtifactSizeEvidence) if args.candidate_size else None
        )
        validation_report = validate_evaluations(
            reference,
            candidate_direct,
            candidate,
            profile=args.profile,
            thresholds=thresholds_for(args.profile),
            calibration=calibration,
            size_reference=size_reference,
            candidate_size=candidate_size,
        )
        if args.release_exception:
            if not args.plan or size_reference is None or candidate_size is None:
                raise ValueError(
                    "--release-exception requires --plan, --size-reference, and --candidate-size"
                )
            from axquant.release_exceptions import apply_release_exception
            from axquant.schema import ReleaseException

            exception = load_model(args.release_exception, ReleaseException)
            exception_evidence = _named_paths(args.exception_evidence)
            reserved_evidence = {
                "plan": Path(args.plan).expanduser().resolve(),
                "candidate_size": Path(args.candidate_size).expanduser().resolve(),
                "size_reference": Path(args.size_reference).expanduser().resolve(),
            }
            duplicated = sorted(set(exception_evidence) & set(reserved_evidence))
            if duplicated:
                raise ValueError(f"reserved exception evidence names were supplied: {duplicated}")
            validation_report = apply_release_exception(
                validation_report,
                exception,
                plan=load_model(args.plan, QuantizationPlan),
                evidence_files={**reserved_evidence, **exception_evidence},
            )
        elif args.exception_evidence:
            raise ValueError("--exception-evidence requires --release-exception")
        write_data(args.output, validation_report)
        log.info(
            "validation_completed",
            output=str(args.output),
            passed=validation_report.passed,
        )
        return 0 if validation_report.passed else 1

    if args.command == "size-evidence":
        from axquant.schema import (
            ArtifactManifest,
            ArtifactSizeEvidence,
            FeasibilityReport,
        )

        sources = [
            value
            for value in (args.artifact_manifest, args.feasibility_report)
            if value is not None
        ]
        if len(sources) != 1:
            raise ValueError(
                "size-evidence requires exactly one of --artifact-manifest or --feasibility-report"
            )
        if args.artifact_manifest:
            if not args.model_id or not args.revision:
                raise ValueError(
                    "candidate size evidence requires --model-id and immutable --revision"
                )
            artifact = load_model(args.artifact_manifest, ArtifactManifest)
            evidence = ArtifactSizeEvidence(
                kind="candidate",
                model=ModelIdentity(
                    model_id=args.model_id,
                    revision=args.revision,
                    local_path=str(Path(args.artifact_manifest).expanduser().resolve().parent),
                ),
                logical_parameters=artifact.logical_parameters,
                weight_bytes=artifact.weight_file_size_bytes,
                measured_bpw=artifact.measured_total_bpw,
                source_sha256=file_sha256(args.artifact_manifest),
            )
        else:
            feasibility = load_model(args.feasibility_report, FeasibilityReport)
            baseline = next(
                (
                    audit
                    for audit in feasibility.baselines
                    if audit.kind == BaselineKind.UNIFORM_4BIT
                ),
                None,
            )
            if baseline is None or not baseline.complete:
                raise ValueError("feasibility report has no complete uniform-4-bit baseline")
            evidence = ArtifactSizeEvidence(
                kind="uniform-4bit",
                model=baseline.model,
                logical_parameters=baseline.logical_parameters,
                weight_bytes=baseline.weight_bytes,
                measured_bpw=baseline.effective_bpw,
                source_sha256=file_sha256(args.feasibility_report),
            )
        write_data(args.output, evidence)
        log.info(
            "artifact_size_evidence_written",
            output=str(args.output),
            kind=evidence.kind,
            weight_bytes=evidence.weight_bytes,
        )
        return 0

    if args.command == "release-exception":
        from axquant.schema import (
            ArtifactSizeEvidence,
            ReleaseException,
            ReleaseExceptionTarget,
        )

        plan_path = Path(args.plan).expanduser().resolve()
        candidate_size_path = Path(args.candidate_size).expanduser().resolve()
        size_reference_path = Path(args.size_reference).expanduser().resolve()
        tradeoff_path = Path(args.tradeoff_evidence).expanduser().resolve()
        plan = load_model(plan_path, QuantizationPlan)
        candidate_size = load_model(candidate_size_path, ArtifactSizeEvidence)
        size_reference = load_model(size_reference_path, ArtifactSizeEvidence)
        if candidate_size.kind != "candidate" or size_reference.kind != "uniform-4bit":
            raise ValueError("release exception requires candidate and uniform-4bit size evidence")
        if candidate_size.logical_parameters != size_reference.logical_parameters:
            raise ValueError("release exception size evidence parameter counts differ")
        additional_evidence = _named_paths(args.evidence)
        reserved_evidence = {
            "plan": plan_path,
            "candidate_size": candidate_size_path,
            "size_reference": size_reference_path,
            "tradeoff": tradeoff_path,
        }
        duplicated = sorted(set(additional_evidence) & set(reserved_evidence))
        if duplicated:
            raise ValueError(f"reserved release exception evidence names: {duplicated}")
        evidence_files = {**reserved_evidence, **additional_evidence}
        missing_evidence = sorted(
            name for name, path in evidence_files.items() if not path.is_file()
        )
        if missing_evidence:
            raise ValueError(f"release exception evidence files are missing: {missing_evidence}")
        weight_size_ratio = candidate_size.weight_bytes / size_reference.weight_bytes
        exception = ReleaseException(
            exception_id=args.exception_id,
            candidate_model=candidate_size.model,
            plan_sha256=stable_sha256(plan),
            targets=[
                ReleaseExceptionTarget(
                    metric="artifact.weight_size_ratio",
                    observed_value=weight_size_ratio,
                    required_maximum=args.max_weight_size_ratio,
                    requirement="candidate weight bytes must be at most the uniform-4bit limit",
                ),
                ReleaseExceptionTarget(
                    metric="artifact.candidate_measured_bpw",
                    observed_value=candidate_size.measured_bpw,
                    required_minimum=args.minimum_measured_bpw,
                    required_maximum=args.maximum_measured_bpw,
                    requirement="candidate measured BPW must remain within the target range",
                ),
            ],
            measured_tradeoff=args.measured_tradeoff,
            owner=args.owner,
            approved_by=args.approved_by,
            approval_reference=args.approval_reference,
            approved_at=args.approved_at,
            expires_at=args.expires_at,
            evidence_sha256={
                name: file_sha256(path) for name, path in sorted(evidence_files.items())
            },
        )
        write_data(args.output, exception)
        log.warning(
            "governed_release_exception_recorded",
            output=str(args.output),
            exception_id=exception.exception_id,
            expires_at=exception.expires_at.isoformat(),
        )
        return 0

    if args.command == "report":
        plan = load_model(args.plan, QuantizationPlan)
        sections = [plan_markdown(plan)]
        if args.validation:
            validation = load_model(args.validation, ValidationReport)
            sections.append(validation_markdown(validation))
        write_text(args.output, "\n\n".join(sections))
        log.info("report_created", output=str(args.output))
        return 0

    if args.command == "publish-prepare":
        prepared_files = prepare_publication(
            model_dir=args.model,
            repo_id=args.repo,
            validation_index_path=args.validation_index,
            hardware_registry_path=args.hardware_registry,
            pareto_report_path=args.pareto_report,
        )
        log.info(
            "publication_prepared",
            model=args.model,
            files=len(prepared_files),
        )
        return 0

    if args.command == "publish":
        published_files = publish_model(
            model_dir=args.model,
            repo_id=args.repo,
            validation_index_path=args.validation_index,
            hardware_registry_path=args.hardware_registry,
            pareto_report_path=args.pareto_report,
            release_audit_path=args.release_audit,
            release_audit_request_path=args.release_audit_request,
            execute=args.yes,
            private=args.private,
        )
        log.info(
            "publication_finished" if args.yes else "publication_previewed",
            repo=args.repo,
            files=len(published_files),
        )
        return 0

    if args.command == "verify-reproduction":
        verification = verify_reproduction(
            recipe_path=args.recipe,
            artifact_dir=args.artifact,
        )
        write_data(args.output, verification)
        log_method = log.info if verification.passed else log.error
        log_method(
            "reproduction_verification_completed",
            artifact=args.artifact,
            output=str(args.output),
            passed=verification.passed,
            verified_weight_files=len(verification.verified_weight_files),
        )
        return 0 if verification.passed else 1

    if args.command == "name":
        name = model_name(
            args.base,
            target_class=args.target_class,
            mtp=args.mtp_suffix,
            include_mlx=not args.no_mlx,
        )
        sys.stdout.write(f"{args.owner}/{name}\n")
        return 0

    if args.command == "runtime-check":
        runtime = RuntimeName(args.runtime)
        runtime_model_path = Path(args.model).expanduser()
        runtime_model = ModelIdentity(
            model_id=args.model_id or args.model,
            revision=args.revision,
            local_path=(str(runtime_model_path.resolve()) if runtime_model_path.is_dir() else None),
        )
        result = (
            check_ax_engine(
                args.model,
                executable=args.ax_engine,
                model_identity=runtime_model,
            )
            if runtime == RuntimeName.AX_ENGINE
            else (
                check_mlx_lm_static(args.model, model_identity=runtime_model)
                if args.static_only
                else check_mlx_lm_generation(
                    args.model,
                    executable=args.mlx_lm,
                    model_identity=runtime_model,
                )
            )
        )
        write_data(args.output, result)
        log.info(
            "runtime_check_completed",
            runtime=runtime.value,
            output=str(args.output),
            passed=result.passed,
        )
        return 0 if result.passed else 1

    if args.command == "benchmark":
        from axquant.serde import file_sha256 as _file_sha256

        dataset_path = Path(args.prompts).expanduser().resolve()
        dataset_sha = _file_sha256(dataset_path)
        benchmark_model_path = Path(args.model).expanduser()
        model_identity = ModelIdentity(
            model_id=args.model_id or args.model,
            revision=args.revision,
            local_path=(
                str(benchmark_model_path.resolve()) if benchmark_model_path.is_dir() else None
            ),
        )
        runtime_env = parse_runtime_env_items(args.runtime_env)
        config = BenchmarkConfig(
            model=model_identity,
            mtp_enabled=args.mtp,
            baseline_kind=args.baseline_kind,
            workload=args.workload,
            dataset_sha256=dataset_sha,
            prompt_count=args.trials,
            warmup_trials=args.warmup,
            measured_trials=args.trials,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            draft_depth=args.draft_depth,
            power_mode=args.power_mode,
            quantizer=args.quantizer,
            quantizer_version=args.quantizer_version,
            random_seed=args.seed,
            timeout_seconds=args.timeout,
            runtime_env=runtime_env,
        )
        quality_result = _load_matching_quality_evaluation(
            args.quality_evaluation,
            model_identity,
        )
        bench_result = run_benchmark(
            config,
            dataset_path=dataset_path,
            executable=args.ax_engine,
            output_dir=args.log_dir,
        )
        bundle = result_to_evaluation_bundle(bench_result)
        if quality_result is not None:
            bundle.quality = quality_result.metrics
            bundle.benchmark_metadata["quality_dataset_sha256"] = quality_result.dataset_sha256
        write_data(args.output, bundle)
        log.info(
            "benchmark_completed",
            output=str(args.output),
            mtp=args.mtp,
            measured=bench_result.measured_count,
            failed=bench_result.failed_count,
        )
        return 0

    if args.command == "benchmark-ab":
        from axquant.serde import file_sha256 as _file_sha256

        dataset_path = Path(args.prompts).expanduser().resolve()
        dataset_sha = _file_sha256(dataset_path)
        model_identity = ModelIdentity(
            model_id=args.model_id or args.model,
            revision=args.revision,
            local_path=(
                str(Path(args.model).expanduser().resolve())
                if Path(args.model).expanduser().is_dir()
                else None
            ),
        )
        runtime_env = parse_runtime_env_items(args.runtime_env)
        config_direct = BenchmarkConfig(
            model=model_identity,
            mtp_enabled=False,
            baseline_kind=args.direct_baseline_kind,
            workload=args.workload,
            dataset_sha256=dataset_sha,
            prompt_count=args.trials,
            warmup_trials=args.warmup,
            measured_trials=args.trials,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            draft_depth=args.draft_depth,
            power_mode=args.power_mode,
            quantizer=args.quantizer,
            quantizer_version=args.quantizer_version,
            random_seed=args.seed,
            timeout_seconds=args.timeout,
            runtime_env=runtime_env,
        )
        config_mtp = BenchmarkConfig(
            model=model_identity,
            mtp_enabled=True,
            baseline_kind=args.mtp_baseline_kind,
            workload=args.workload,
            dataset_sha256=dataset_sha,
            prompt_count=args.trials,
            warmup_trials=args.warmup,
            measured_trials=args.trials,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            draft_depth=args.draft_depth,
            power_mode=args.power_mode,
            quantizer=args.quantizer,
            quantizer_version=args.quantizer_version,
            random_seed=args.seed,
            timeout_seconds=args.timeout,
            runtime_env=runtime_env,
        )
        quality_result = _load_matching_quality_evaluation(
            args.quality_evaluation,
            model_identity,
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        direct_bundle, mtp_bundle = run_mtp_ab(
            config_direct,
            config_mtp,
            dataset_path=dataset_path,
            executable=args.ax_engine,
            output_dir=output_dir,
            enforce_speedup=not args.record_failed_speedup,
            minimum_speedup=args.minimum_speedup,
        )
        if quality_result is not None:
            direct_bundle.quality = quality_result.metrics
            mtp_bundle.quality = quality_result.metrics
            direct_bundle.benchmark_metadata["quality_dataset_sha256"] = (
                quality_result.dataset_sha256
            )
            mtp_bundle.benchmark_metadata["quality_dataset_sha256"] = quality_result.dataset_sha256
        write_data(output_dir / "evaluation_mtp_off.json", direct_bundle)
        write_data(output_dir / "evaluation_mtp_on.json", mtp_bundle)
        log.info(
            "benchmark_ab_completed",
            output_dir=str(output_dir),
            runtime_env=runtime_env,
        )
        if args.record_failed_speedup:
            from axquant.schema import MtpAbComparison

            mtp_ab_comparison = load_model(
                output_dir / "mtp_ab_comparison.json",
                MtpAbComparison,
            )
            if not mtp_ab_comparison.speedup_pass:
                log.warning(
                    "benchmark_ab_speedup_gate_failed",
                    output_dir=str(output_dir),
                    speedup=mtp_ab_comparison.speedup,
                    minimum_speedup=mtp_ab_comparison.minimum_speedup,
                )
                return 1
        return 0

    if args.command == "mtp-diagnose":
        from axquant.serde import file_sha256 as _file_sha256

        dataset_path = Path(args.prompts).expanduser().resolve()
        dataset_sha = _file_sha256(dataset_path)
        model_path = Path(args.model).expanduser()
        model_identity = ModelIdentity(
            model_id=args.model_id or args.model,
            revision=args.revision,
            local_path=str(model_path.resolve()) if model_path.is_dir() else None,
        )
        runtime_env = parse_runtime_env_items(args.runtime_env)
        base_config = BenchmarkConfig(
            model=model_identity,
            mtp_enabled=False,
            baseline_kind="axquant-mtp-off",
            workload=args.workload,
            dataset_sha256=dataset_sha,
            prompt_count=args.trials,
            warmup_trials=args.warmup,
            measured_trials=args.trials,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            draft_depth=args.draft_depth,
            power_mode=args.power_mode,
            quantizer=args.quantizer,
            quantizer_version=args.quantizer_version,
            random_seed=args.seed,
            timeout_seconds=args.timeout,
            runtime_env=runtime_env,
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        diagnostic = run_mtp_diagnostics(
            base_config,
            dataset_path=dataset_path,
            executable=args.ax_engine,
            output_dir=output_dir,
            profiles=args.profiles or None,
            minimum_speedup=args.minimum_mtp_speedup,
        )
        report_path = Path(args.output).expanduser()
        if not report_path.is_absolute():
            report_path = output_dir / report_path
        write_data(report_path, diagnostic)
        log.info(
            "mtp_diagnose_completed",
            output=str(report_path),
            any_exactness_pass=diagnostic.any_exactness_pass,
            any_release_ready=diagnostic.any_release_ready,
            profiles=len(diagnostic.profiles),
        )
        return 0 if diagnostic.any_release_ready else 1

    if args.command == "evaluate-quality":
        from axquant.quality import evaluate_quality

        quality_model_path = Path(args.model).expanduser()
        quality_result = evaluate_quality(
            model=ModelIdentity(
                model_id=args.model_id or args.model,
                revision=args.revision,
                local_path=(
                    str(quality_model_path.resolve()) if quality_model_path.is_dir() else None
                ),
            ),
            dataset_path=args.dataset,
            max_sequence_length=args.max_seq_length,
            max_generation_tokens=args.max_tokens,
            random_seed=args.seed,
            max_samples=args.max_samples,
        )
        write_data(args.output, quality_result)
        log.info(
            "quality_evaluation_written",
            output=str(args.output),
            samples=quality_result.samples,
            perplexity=quality_result.metrics.perplexity,
        )
        return 0

    if args.command == "compare-quality":
        from axquant.quality import compare_quality

        comparison = compare_quality(
            load_model(args.reference, QualityEvaluationResult),
            load_model(args.candidate, QualityEvaluationResult),
        )
        write_data(args.output, comparison)
        log.info(
            "quality_comparison_written",
            output=str(args.output),
            aggregate_retention=comparison.aggregate.retention,
            perplexity_ratio=comparison.perplexity_ratio,
        )
        return 0

    if args.command == "tokenize-calibration":
        from axquant.activation_cache import tokenize_calibration

        tokenization_model_path = Path(args.model).expanduser()
        token_manifest = tokenize_calibration(
            model=ModelIdentity(
                model_id=args.model_id or args.model,
                revision=args.revision,
                local_path=(
                    str(tokenization_model_path.resolve())
                    if tokenization_model_path.is_dir()
                    else None
                ),
            ),
            dataset_path=args.dataset,
            output_dir=args.output,
            profile=args.profile,
            sequence_length=args.max_seq_length,
            random_seed=args.seed,
            tokenizer_revision=args.tokenizer_revision,
            separation_attested=args.attest_calibration_eval_separation,
            domains=args.domains,
        )
        log.info(
            "tokenization_completed",
            output=str(args.output),
            shards=token_manifest.shard_count,
            samples=token_manifest.samples,
            complete=token_manifest.complete,
        )
        return 0

    if args.command == "refine":
        from axquant.refinement import refine_candidates
        from axquant.schema import RefinementConfig

        analysis_report = load_model(args.analysis, SensitivityReport)
        request = PlanRequest(
            profile=analysis_report.profile,
            target_bpw=args.target_bpw,
            candidate_bits=args.bits,
            group_size=args.group_size,
            allow_unmeasured=args.allow_unmeasured,
            candidate_count=args.top_n,
            random_seed=args.seed,
        )
        refine_config = RefinementConfig(
            top_n=args.top_n,
            max_iterations=args.max_iterations,
            evaluation_budget=args.eval_budget,
            wall_clock_seconds=args.wall_clock,
            convergence_threshold=args.convergence,
            swap_radius=args.swap_radius,
            random_seed=args.seed,
        )
        refine_result = refine_candidates(analysis_report, request, refine_config)
        write_data(args.output, refine_result)
        log.info(
            "refinement_completed",
            output=str(args.output),
            iterations=refine_result.iterations_used,
            evaluations=refine_result.evaluations_used,
            converged=refine_result.converged,
        )
        return 0

    if args.command == "refine-select":
        from axquant.refinement import select_complete_candidate
        from axquant.schema import RefinementMeasurementSet, RefinementResult

        refinement = load_model(args.refinement, RefinementResult)
        measurements = load_model(args.measurements, RefinementMeasurementSet)
        if measurements.refinement_sha256 != stable_sha256(refinement):
            raise ValueError("measurement set does not match the refinement result")
        selected = select_complete_candidate(refinement, measurements)
        write_data(args.output, selected)
        log.info(
            "complete_candidate_selected",
            output=str(args.output),
            candidate_id=selected.selected_candidate_id,
            plan_sha256=selected.selected_plan_sha256,
        )
        return 0

    if args.command == "refine-measure":
        from axquant import __version__
        from axquant.refinement import (
            COMPLETE_OBJECTIVE_VERSION,
            build_complete_candidate_measurement,
        )
        from axquant.schema import (
            ArtifactManifest,
            QualityComparisonReport,
            RefinementMeasurementSet,
            RefinementResult,
        )

        refinement = load_model(args.refinement, RefinementResult)
        refine_measure_plan = refinement.candidate_plans.get(args.candidate_id)
        if refine_measure_plan is None:
            raise ValueError(f"unknown refinement candidate {args.candidate_id!r}")
        artifact = load_model(args.artifact_manifest, ArtifactManifest)
        quality = load_model(args.quality_comparison, QualityComparisonReport)
        validation = load_model(args.validation, ValidationReport)
        measurement = build_complete_candidate_measurement(
            candidate_id=args.candidate_id,
            measurement_id=args.measurement_id,
            plan=refine_measure_plan,
            artifact=artifact,
            artifact_sha256=file_sha256(args.artifact_manifest),
            quality=quality,
            quality_sha256=file_sha256(args.quality_comparison),
            validation=validation,
            validation_sha256=file_sha256(args.validation),
        )
        refinement_sha256 = stable_sha256(refinement)
        evaluator_version = f"{__version__}:{COMPLETE_OBJECTIVE_VERSION}"
        refine_measure_records = [measurement]
        if args.existing:
            existing_measurement_set = load_model(args.existing, RefinementMeasurementSet)
            if existing_measurement_set.refinement_sha256 != refinement_sha256:
                raise ValueError("existing measurement set does not match the refinement result")
            if existing_measurement_set.evaluator_version != evaluator_version:
                raise ValueError("existing measurement set uses another evaluator version")
            refine_measure_records = [*existing_measurement_set.measurements, measurement]
        measurement_set_result = RefinementMeasurementSet(
            refinement_sha256=refinement_sha256,
            evaluator_version=evaluator_version,
            measurements=refine_measure_records,
        )
        write_data(args.output, measurement_set_result)
        log.info(
            "complete_candidate_measured",
            output=str(args.output),
            candidate_id=args.candidate_id,
            measurement_id=measurement.measurement_id,
            validation_passed=measurement.validation_passed,
            objective_loss=measurement.objective_loss,
        )
        return 0

    if args.command == "refine-export":
        from axquant.schema import RefinementResult

        refinement = load_model(args.refinement, RefinementResult)
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        targets = {
            output_dir / f"{candidate_id}.json": plan
            for candidate_id, plan in refinement.candidate_plans.items()
        }
        targets[output_dir / "selected-plan.json"] = refinement.selected_plan
        export_existing = [str(path) for path in targets if path.exists()]
        if export_existing:
            raise ValueError(f"refinement export targets already exist: {export_existing[:5]}")
        for path, plan in targets.items():
            write_data(path, plan)
        log.info(
            "refinement_plans_exported",
            output_dir=str(output_dir),
            candidates=len(refinement.candidate_plans),
            selection_basis=refinement.selection_basis,
        )
        return 0

    if args.command == "refine-run":
        from axquant.refinement_runner import execute_refinement

        execution = execute_refinement(
            request_path=args.request,
            output_dir=args.output_dir,
            execute=args.execute,
        )
        log_method = (
            log.info if not args.execute or execution.selected_result is not None else log.warning
        )
        log_method(
            "refinement_execution_updated",
            output_dir=str(args.output_dir),
            execute=args.execute,
            complete=execution.complete,
            measured=len(execution.measured_candidate_ids),
            failed=len(execution.failed_candidate_ids),
            selected=execution.selected_result,
        )
        if not args.execute:
            return 0
        return (
            0
            if execution.complete
            and execution.selected_result is not None
            and execution.pareto_report is not None
            else 1
        )

    if args.command == "pareto":
        from axquant.pareto import build_pareto_report
        from axquant.schema import RefinementMeasurementSet

        measurement_set = load_model(args.measurements, RefinementMeasurementSet)
        pareto_report = build_pareto_report(measurement_set)
        write_data(args.output, pareto_report)
        log.info(
            "pareto_report_written",
            output=str(args.output),
            candidates=len(pareto_report.points),
            frontier=len(pareto_report.frontier_candidate_ids),
        )
        return 0

    if args.command == "hardware-registry":
        from axquant.hardware_registry import build_hardware_profile_registry

        registry = build_hardware_profile_registry(args.request)
        write_data(args.output, registry)
        log_method = log.info if registry.release_ready else log.warning
        log_method(
            "hardware_profile_registry_written",
            output=str(args.output),
            candidates=len(registry.entries),
            named_hosts=registry.distinct_named_hosts,
            release_ready=registry.release_ready,
        )
        return 0 if registry.release_ready else 1

    if args.command == "compatibility-matrix":
        from axquant.compatibility import build_compatibility_matrix

        matrix = build_compatibility_matrix(args.request)
        write_data(args.output, matrix)
        log_method = log.info if matrix.release_ready else log.warning
        log_method(
            "compatibility_matrix_written",
            output=str(args.output),
            candidates=len(matrix.entries),
            distinct_dense_sources=matrix.distinct_dense_source_checkpoints,
            release_ready=matrix.release_ready,
        )
        return 0 if matrix.release_ready else 1

    if args.command == "benchmark-index":
        from axquant.benchmark_evidence import build_benchmark_evidence_index

        index = build_benchmark_evidence_index(args.request)
        write_data(args.output, index)
        log_method = log.info if index.release_ready else log.warning
        log_method(
            "benchmark_evidence_index_written",
            output=str(args.output),
            profile=index.profile.value,
            release_ready=index.release_ready,
            unavailable=sum(entry.status == "unavailable" for entry in index.entries),
        )
        return 0 if index.release_ready else 1

    if args.command == "validation-index":
        from axquant.release_validation import build_release_validation_index

        release_index = build_release_validation_index(args.request)
        write_data(args.output, release_index)
        log_method = log.info if release_index.release_ready else log.warning
        log_method(
            "release_validation_index_written",
            output=str(args.output),
            release_ready=release_index.release_ready,
            profiles=[entry.profile.value for entry in release_index.entries],
        )
        return 0 if release_index.release_ready else 1

    if args.command == "release-audit":
        from axquant.release_audit import build_release_audit

        audit = build_release_audit(args.request)
        write_data(args.output, audit)
        log_method = log.info if audit.release_ready else log.warning
        log_method(
            "release_audit_written",
            output=str(args.output),
            release_ready=audit.release_ready,
            passed=sum(check.passed for check in audit.checks),
            blockers=len(audit.blockers),
        )
        return 0 if audit.release_ready else 1

    if args.command == "prepare-suite":
        from axquant.suites import build_benchmark_suites

        suite_manifest = build_benchmark_suites(args.output_dir, random_seed=args.seed)
        log.info(
            "benchmark_suite_created",
            output=str(args.output_dir),
            samples=sum(suite_manifest.samples.values()),
        )
        return 0

    raise AssertionError(f"unhandled command {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        return _run(args)
    except (AxquantError, ValidationError, OSError, ValueError) as exc:
        structlog.get_logger().error(
            "command_failed",
            command=args.command,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return 2


def entrypoint() -> None:
    raise SystemExit(main())
