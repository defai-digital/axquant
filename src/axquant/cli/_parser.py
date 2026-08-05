from __future__ import annotations

import argparse
from datetime import datetime

from axquant.profiles import implemented_profiles
from axquant.schema import (
    AX_ENGINE_EXECUTABLE_BITS,
    AX_ENGINE_EXECUTABLE_GROUP_SIZES,
    ConvertLadderName,
    MtpSidecarLayout,
    ProfileName,
    QuantMethod,
    RuntimeName,
)


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


def _group_sizes(value: str) -> tuple[int, ...]:
    """Parse comma-separated group sizes for multi-group planning (AXQ-028)."""
    parsed: list[int] = []
    for item in value.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        try:
            size = int(normalized)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid group size {item!r}") from exc
        if size < 1:
            raise argparse.ArgumentTypeError(f"group size must be positive: {size}")
        parsed.append(size)
    result = tuple(sorted(set(parsed)))
    if not result:
        raise argparse.ArgumentTypeError("at least one group size is required")
    return result


def _kv_bits(value: str) -> tuple[int, ...]:
    result = _bits(value)
    unsupported = sorted(set(result) - AX_ENGINE_EXECUTABLE_BITS)
    if unsupported:
        choices = ",".join(str(bits) for bits in sorted(AX_ENGINE_EXECUTABLE_BITS))
        raise argparse.ArgumentTypeError(
            f"unsupported KV precision(s) {unsupported}; choose from {choices}"
        )
    return result


def _kv_bit(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid KV precision {value!r}") from exc
    if result not in AX_ENGINE_EXECUTABLE_BITS:
        choices = ",".join(str(bits) for bits in sorted(AX_ENGINE_EXECUTABLE_BITS))
        raise argparse.ArgumentTypeError(
            f"unsupported KV precision {result}; choose from {choices}"
        )
    return result


def _kv_default_bit(value: str) -> int:
    result = _kv_bit(value)
    if result < 4:
        # Probe grids may measure 2/3-bit KV, but every allocator enforces the
        # 4-bit policy floor, so sub-4 defaults would only fail at plan time.
        choices = ",".join(str(bits) for bits in sorted(AX_ENGINE_EXECUTABLE_BITS) if bits >= 4)
        raise argparse.ArgumentTypeError(
            f"unsupported KV default precision {result}; the policy floor is 4-bit "
            f"(choose from {choices})"
        )
    return result


def _kv_group_size(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid KV group size {value!r}") from exc
    if result not in AX_ENGINE_EXECUTABLE_GROUP_SIZES:
        choices = ",".join(str(size) for size in sorted(AX_ENGINE_EXECUTABLE_GROUP_SIZES))
        raise argparse.ArgumentTypeError(
            f"unsupported KV group size {result}; choose from {choices}"
        )
    return result


def _domains(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one calibration domain is required")
    return result


_VALID_CAPTURE_POINTS = ("output", "hidden")


def _capture_points(value: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    invalid = sorted(set(result) - set(_VALID_CAPTURE_POINTS))
    if not result or invalid:
        choices = ",".join(_VALID_CAPTURE_POINTS)
        raise argparse.ArgumentTypeError(f"capture points must be a subset of: {choices}")
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
    unsupported = set(result) - {
        QuantMethod.AFFINE,
        QuantMethod.DWQ,
        QuantMethod.AWQ,
        QuantMethod.GPTQ,
    }
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


def _ladder(value: str) -> ConvertLadderName:
    try:
        return ConvertLadderName(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in ConvertLadderName)
        raise argparse.ArgumentTypeError(f"ladder must be one of {choices}") from exc


def _iso_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed


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

    source_manifest_parser = subparsers.add_parser("source-checkpoint-manifest")
    source_manifest_parser.add_argument("--model", required=True)
    source_manifest_parser.add_argument("--inventory", required=True)
    source_manifest_parser.add_argument("--output", default="source_checkpoint_manifest.json")

    certification_policy_parser = subparsers.add_parser("certification-policy")
    certification_policy_parser.add_argument("--output", default="qwen3_next_direct_policy.json")

    direct_validation_parser = subparsers.add_parser(
        "direct-validation-index",
        help="Recompute the non-MTP direct-track BF16/candidate quality gates",
    )
    direct_validation_parser.add_argument("--request", required=True)
    direct_validation_parser.add_argument(
        "--output",
        default="direct-release-validation-index.json",
    )

    general_overlap_parser = subparsers.add_parser(
        "prepare-general-overlap",
        help="Compare the direct-track general holdout with release calibration",
    )
    general_overlap_parser.add_argument("--general-dataset", required=True)
    general_overlap_parser.add_argument("--calibration", required=True)
    general_overlap_parser.add_argument("--output", default="general-overlap-report.json")

    coding_suite_parser = subparsers.add_parser(
        "prepare-coding-suite",
        help="Build the frozen Qwen3-Next coding-suite v2 and calibration-overlap evidence",
    )
    coding_suite_parser.add_argument("--output-dir", required=True)
    coding_suite_parser.add_argument("--calibration", required=True)
    coding_suite_parser.add_argument("--seed", type=int, default=20260803)
    coding_suite_parser.add_argument(
        "--toolchain",
        action="append",
        default=[],
        metavar="NAME=EXECUTABLE",
        help="Override python/node/typescript/rust/go/sandbox executable; repeatable",
    )

    coding_evaluate_parser = subparsers.add_parser(
        "evaluate-coding-suite",
        help="Run a resumable coding-suite v2 evaluation with fail-closed sandbox scoring",
    )
    coding_evaluate_parser.add_argument("--model", required=True)
    coding_evaluate_parser.add_argument("--model-id")
    coding_evaluate_parser.add_argument("--revision", required=True)
    coding_evaluate_parser.add_argument("--manifest", required=True)
    coding_evaluate_parser.add_argument("--model-artifact-sha256", required=True)
    coding_evaluate_parser.add_argument("--tokenizer-sha256", required=True)
    coding_evaluate_parser.add_argument("--max-seq-length", type=int, default=4096)
    coding_evaluate_parser.add_argument("--seed", type=int, default=20260803)
    coding_evaluate_parser.add_argument("--output", default="coding-quality-evaluation.json")
    coding_evaluate_parser.add_argument("--state", default="coding-evaluation-state.json")
    coding_evaluate_parser.add_argument("--raw-log-dir", default="coding-raw-logs")
    coding_evaluate_parser.add_argument(
        "--work-root",
        default=".internal/tmp/coding-sandbox-work",
    )
    coding_evaluate_parser.add_argument(
        "--toolchain",
        action="append",
        default=[],
        metavar="NAME=EXECUTABLE",
        help="Override python/node/typescript/rust/go/sandbox executable; repeatable",
    )

    general_evaluate_parser = subparsers.add_parser(
        "evaluate-general-quality",
        help="Evaluate the non-MTP direct track's general holdout with raw-output evidence",
    )
    general_evaluate_parser.add_argument("--model", required=True)
    general_evaluate_parser.add_argument("--model-id")
    general_evaluate_parser.add_argument("--revision", required=True)
    general_evaluate_parser.add_argument("--dataset", required=True)
    general_evaluate_parser.add_argument("--model-artifact-sha256", required=True)
    general_evaluate_parser.add_argument("--tokenizer-sha256", required=True)
    general_evaluate_parser.add_argument("--max-seq-length", type=int, default=4096)
    general_evaluate_parser.add_argument("--max-generation-tokens", type=int, default=256)
    general_evaluate_parser.add_argument("--seed", type=int, default=20260803)
    general_evaluate_parser.add_argument("--output", default="general-quality-evaluation.json")
    general_evaluate_parser.add_argument("--state", default="general-quality-state.json")
    general_evaluate_parser.add_argument("--raw-log-dir", default="general-quality-raw-logs")

    coding_verify_parser = subparsers.add_parser(
        "verify-coding-suite",
        help="Prove every coding-suite oracle passes and an empty mutant is rejected",
    )
    coding_verify_parser.add_argument("--manifest", required=True)
    coding_verify_parser.add_argument("--output", default="coding-suite-self-test.json")
    coding_verify_parser.add_argument("--raw-log-dir", default="coding-self-test-logs")
    coding_verify_parser.add_argument(
        "--work-root",
        default=".internal/tmp/coding-self-test-work",
    )
    coding_verify_parser.add_argument(
        "--toolchain",
        action="append",
        default=[],
        metavar="NAME=EXECUTABLE",
        help="Override python/node/typescript/rust/go/sandbox executable; repeatable",
    )

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
        "--candidate-group-sizes",
        type=_group_sizes,
        default=(),
        help="optional multi-group probe/prior grid (e.g. 32,64,128); empty uses --group-size",
    )
    analyze_parser.add_argument(
        "--methods",
        type=_probe_methods,
        default=(QuantMethod.AFFINE,),
    )
    analyze_parser.add_argument("--group-size", type=int, default=64)
    analyze_parser.add_argument("--calibration")
    analyze_parser.add_argument(
        "--calibration-activations",
        help="path to a capture-activations artifact directory "
        "(required when --methods includes AWQ or GPTQ)",
    )
    analyze_parser.add_argument("--base-sensitivity")
    analyze_parser.add_argument("--target-tensor", action="append", default=[])
    analyze_parser.add_argument(
        "--capture-points",
        type=_capture_points,
        default=None,
        help="probe capture points (subset of output,hidden); plain dense backbones "
        "expose logits only and need '--capture-points output'",
    )
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
    analyze_kv_parser.add_argument("--bits", type=_kv_bits, default=(4, 6, 8))
    analyze_kv_parser.add_argument("--group-size", type=_kv_group_size, default=64)
    analyze_kv_parser.add_argument("--token-budget", type=int, default=2048)
    analyze_kv_parser.add_argument("--metric-positions", type=int, default=32)
    analyze_kv_parser.add_argument("--output", default="kv_sensitivity.json")

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--sensitivity", "--analysis", dest="analysis", required=True)
    plan_parser.add_argument(
        "--ladder",
        type=_ladder,
        help="apply a convert-ladder preset for bits/groups/methods/target BPW defaults (P1)",
    )
    plan_parser.add_argument("--target-bpw", type=float, default=None)
    plan_parser.add_argument("--bits", type=_bits, default=None)
    plan_parser.add_argument(
        "--candidate-group-sizes",
        type=_group_sizes,
        default=None,
        help="optional multi-group planner grid (e.g. 32,64,128); empty uses --group-size",
    )
    plan_parser.add_argument(
        "--methods",
        type=_methods,
        default=None,
    )
    plan_parser.add_argument("--group-size", type=int, default=None)
    plan_parser.add_argument(
        "--bind-kv-sensitivity",
        help="optional KV sensitivity JSON; writes unified sensitivity binding alongside the plan",
    )
    plan_parser.add_argument(
        "--unified-binding-output",
        default="unified-sensitivity.json",
        help="where to write the unified weight+KV sensitivity binding when binding",
    )
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
    plan_parser.add_argument(
        "--lm-head-floor",
        choices=["bf16", "8bit"],
        default="bf16",
        help="AXQ-026 governed size-gate path: 8bit lowers the LM-head weight "
        "floor for this plan; release certification still requires measured "
        "quality evidence",
    )
    plan_parser.add_argument("--allow-unmeasured", action="store_true")
    plan_parser.add_argument("--kv-cache", choices=["off", "prior", "measured"], default="off")
    plan_parser.add_argument("--kv-default-bits", type=_kv_default_bit, default=4)
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
        "--calibration-activations",
        help="path to a capture-activations artifact directory "
        "(required when the plan uses AWQ or GPTQ methods)",
    )
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
        help=(
            "Simple development convert (OptiQ-like): MODEL + optional --target-bpw. "
            "Always development evidence; release claims use the staged pipeline."
        ),
    )
    quantize_parser.add_argument(
        "model_positional",
        nargs="?",
        help="Local BF16 MLX directory or Hub id (org/name)",
    )
    quantize_parser.add_argument(
        "--model",
        dest="model_option",
        help="Same as positional MODEL (either form is accepted)",
    )
    quantize_parser.add_argument("--model-id")
    quantize_parser.add_argument("--revision")
    quantize_parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: ./AX-<model>-MLX-AXQ-<class>)",
    )
    quantize_parser.add_argument(
        "--profile",
        choices=[profile.value for profile in implemented_profiles()],
        default=None,
    )
    quantize_parser.add_argument(
        "--ladder",
        type=_ladder,
        default=ConvertLadderName.PRIOR,
        help="convert ladder (default: prior with multi-group 32,64 grid)",
    )
    quantize_parser.add_argument(
        "--target-bpw",
        type=float,
        default=None,
        help="target average BPW (prior default 4.8)",
    )
    quantize_parser.add_argument("--kv-cache", choices=["off", "prior"], default=None)
    quantize_parser.add_argument(
        "--recipe",
        help="Recipe bundle file or directory; replaces prior-based planning",
    )
    quantize_parser.add_argument("--calibration-manifest")
    quantize_parser.add_argument(
        "--kv-sensitivity",
        help="KV sensitivity report bound by a measured KV-cache plan (AXQ-025)",
    )
    quantize_parser.add_argument("--mtp-sidecar")
    quantize_parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face Hub download when MODEL is a Hub id (not a local path)",
    )
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

    simple_help_parser = subparsers.add_parser(
        "simple-convert-help",
        help="Print simple-convert best practices (two-door model)",
    )
    simple_help_parser.add_argument(
        "--output",
        default=None,
        help="Optional markdown file path (default: stdout via log)",
    )

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
        help="List every registered model family with tier, investment posture, and policy notes",
    )
    support_matrix_parser.add_argument("--certification-registry")
    support_matrix_parser.add_argument("--output", help="Optional JSON output path")

    support_policy_parser = subparsers.add_parser(
        "support-policy",
        help="Print family investment best practices (primary/secondary/thin/deferred)",
    )
    support_policy_parser.add_argument(
        "--output",
        default=None,
        help="Optional markdown path (default: stdout)",
    )

    head_to_head_parser = subparsers.add_parser(
        "head-to-head",
        help="Render the public comparison page from a bound benchmark evidence index",
    )
    head_to_head_parser.add_argument("--benchmark-index", required=True)
    head_to_head_parser.add_argument("--title")
    head_to_head_parser.add_argument("--output", default="head-to-head.md")

    scoreboard_parser = subparsers.add_parser(
        "scoreboard",
        help="Build a certification scoreboard from a plan and optional evidence (P0)",
    )
    scoreboard_parser.add_argument("--plan", required=True)
    scoreboard_parser.add_argument("--profile", type=_profile)
    scoreboard_parser.add_argument("--title")
    scoreboard_parser.add_argument("--candidate-size")
    scoreboard_parser.add_argument("--size-reference")
    scoreboard_parser.add_argument("--quality-comparison")
    scoreboard_parser.add_argument("--validation-report")
    scoreboard_parser.add_argument("--mtp-ab")
    scoreboard_parser.add_argument("--candidate-evaluation")
    scoreboard_parser.add_argument("--reference-evaluation")
    scoreboard_parser.add_argument("--minimum-quality", type=float, default=0.98)
    scoreboard_parser.add_argument("--max-size-ratio", type=float, default=1.10)
    scoreboard_parser.add_argument("--minimum-mtp-retention", type=float, default=0.95)
    scoreboard_parser.add_argument("--minimum-mtp-speedup", type=float, default=1.20)
    scoreboard_parser.add_argument("--require-complete", action="store_true")
    scoreboard_parser.add_argument("--output", default="scoreboard.json")
    scoreboard_parser.add_argument("--markdown-output", default="scoreboard.md")

    probe_capacity_parser = subparsers.add_parser(
        "probe-capacity",
        help="Recommend sensitivity probe mode under host memory limits (P0)",
    )
    probe_capacity_parser.add_argument("--inventory", help="Inventory JSON from axquant inspect")
    probe_capacity_parser.add_argument(
        "--parameter-count",
        type=int,
        help="Explicit parameter count when inventory is not provided",
    )
    probe_capacity_parser.add_argument("--model-id")
    probe_capacity_parser.add_argument(
        "--available-memory-bytes",
        type=int,
        help="Override detected unified memory (bytes)",
    )
    probe_capacity_parser.add_argument(
        "--headroom-fraction",
        type=float,
        default=0.70,
        help="Fraction of available memory usable for probes (default 0.70)",
    )
    probe_capacity_parser.add_argument("--output", default="probe-capacity.json")
    probe_capacity_parser.add_argument("--markdown-output", default="probe-capacity.md")

    ladder_parser = subparsers.add_parser(
        "ladders",
        help="List convert ladders (prior → measured-lite → measured-full → refine) (P1)",
    )
    ladder_parser.add_argument("--output", help="Optional JSON list output")
    ladder_parser.add_argument("--markdown-output", default="convert-ladders.md")

    deferred_parser = subparsers.add_parser(
        "deferred-features",
        help="List fail-closed deferred expansion features (P2)",
    )
    deferred_parser.add_argument("--output", help="Optional JSON output")

    recovery_rank_parser = subparsers.add_parser(
        "recovery-rank",
        help="Rank quantized tensors for opt-in recovery by sensitivity (P2)",
    )
    recovery_rank_parser.add_argument("--plan", required=True)
    recovery_rank_parser.add_argument("--sensitivity")
    recovery_rank_parser.add_argument("--limit", type=int)
    recovery_rank_parser.add_argument("--output", default="recovery-ranking.json")

    bind_parser = subparsers.add_parser(
        "bind-sensitivity",
        help="Bind weight (+ optional KV) sensitivity digests into one lineage artifact (P1)",
    )
    bind_parser.add_argument("--sensitivity", required=True, help="Weight sensitivity report")
    bind_parser.add_argument("--kv-sensitivity")
    bind_parser.add_argument("--plan")
    bind_parser.add_argument("--output", default="unified-sensitivity.json")

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
    prepare_parser.add_argument("--validation-index")
    prepare_parser.add_argument("--hardware-registry")
    prepare_parser.add_argument("--pareto-report")
    prepare_parser.add_argument("--release-audit-request")
    prepare_parser.add_argument("--release-audit")

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--model", required=True)
    publish_parser.add_argument("--repo", required=True)
    publish_parser.add_argument("--validation-index", required=True)
    publish_parser.add_argument("--hardware-registry", required=True)
    publish_parser.add_argument("--pareto-report", required=True)
    publish_parser.add_argument("--release-audit")
    publish_parser.add_argument("--release-audit-request")
    publish_parser.add_argument("--certification-registry")
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
        choices=[*(runtime.value for runtime in RuntimeName), "mlx-lm-kv"],
        default=RuntimeName.AX_ENGINE.value,
        help="mlx-lm-kv executes the artifact's planned per-layer KV-cache "
        "precision table through the public MLX-LM prompt-cache API",
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
            "gptq",
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
            "gptq",
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
            "gptq",
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
    benchmark_ab_parser.add_argument(
        "--qwen36-exact-profile",
        action="store_true",
        help="Apply the complete Qwen 3.6 exact-MTP measurement contract "
        "(the formal-suite env set); explicit --runtime-env values win",
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

    capture_parser = subparsers.add_parser(
        "capture-activations",
        help="Capture per-module calibration input activations into a checksum-bound artifact",
    )
    capture_parser.add_argument("--model", required=True)
    capture_parser.add_argument(
        "--revision",
        help="Immutable source revision; must match the tokenized calibration cache",
    )
    capture_parser.add_argument(
        "--calibration",
        required=True,
        help="Verified tokenized calibration cache directory",
    )
    capture_parser.add_argument("--output", default="activation-capture")
    capture_parser.add_argument("--max-rows", type=int, default=2048)
    capture_parser.add_argument("--token-budget", type=int)
    capture_parser.add_argument(
        "--segment-batches",
        type=int,
        default=8,
        help="Batches per resumable replay segment (checkpoint written after each)",
    )
    capture_parser.add_argument(
        "--modules-per-shard",
        type=int,
        default=1,
        help="Group this many modules per shared shard-NNNN.npz archive "
        "(default: one npz per module)",
    )
    capture_parser.add_argument(
        "--target-module",
        action="append",
        default=[],
        dest="target_modules",
        metavar="MODULE",
        help="Checkpoint-style module path to capture (repeatable); default: every eligible Linear",
    )
    capture_parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face Hub download when MODEL is a Hub id (not a local path)",
    )

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
    refine_parser.add_argument(
        "--lm-head-floor",
        choices=["bf16", "8bit"],
        default="bf16",
        help="AXQ-026 governed size-gate path: 8bit lowers the LM-head weight "
        "floor for this search; release certification still requires measured "
        "quality evidence",
    )
    refine_parser.add_argument(
        "--holdout-digest",
        dest="holdout_measurement_set_sha256",
        default=None,
        help="optional expected sha256 of a holdout RefinementMeasurementSet (QP1)",
    )
    refine_parser.add_argument("--output", default="refinement_result.json")

    recover_parser = subparsers.add_parser(
        "recover",
        help="optional post-PTQ recovery with provenance (never required by convert/quantize)",
    )
    recover_parser.add_argument("--artifact", required=True, help="source converted checkpoint")
    recover_parser.add_argument("--plan", required=True, help="quantization plan JSON")
    recover_parser.add_argument("--calibration-dataset-id", required=True)
    recover_parser.add_argument("--calibration-dataset-sha256", required=True)
    recover_parser.add_argument("--output", required=True)
    recover_parser.add_argument("--seed", type=int, default=0)
    recover_parser.add_argument("--steps", type=int, default=1)
    recover_parser.add_argument("--learning-rate", type=float, default=None)
    recover_parser.add_argument(
        "--scope",
        default="scales-and-biases",
        choices=["scales", "biases", "scales-and-biases", "lora-merged"],
    )
    recover_parser.add_argument("--quality-before-sha256", default=None)
    recover_parser.add_argument("--quality-after-sha256", default=None)
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
    campaign_overlap_parser = subparsers.add_parser(
        "campaign-overlap",
        help="Build a deterministic cross-dataset overlap report for a flagship campaign",
    )
    campaign_overlap_parser.add_argument("--dataset", required=True)
    campaign_overlap_parser.add_argument("--compare", action="append", required=True)
    campaign_overlap_parser.add_argument("--threshold", type=float, default=0.9)
    campaign_overlap_parser.add_argument("--id-field", default="id")
    campaign_overlap_parser.add_argument("--text-field", action="append")
    campaign_overlap_parser.add_argument("--max-comparison-pairs", type=int, default=10_000_000)
    campaign_overlap_parser.add_argument(
        "--output",
        default="campaign-overlap-report.json",
    )
    campaign_frontier_parser = subparsers.add_parser(
        "campaign-frontier",
        help="Build and verify a cheapest-failure-first flagship candidate frontier",
    )
    campaign_frontier_parser.add_argument("--request", required=True)
    campaign_frontier_parser.add_argument(
        "--output",
        default="flagship-frontier.json",
    )
    campaign_freeze_parser = subparsers.add_parser(
        "campaign-freeze",
        help="Freeze one strict qwen36-mtp-v2 flagship campaign",
    )
    campaign_freeze_parser.add_argument("--request", required=True)
    campaign_freeze_parser.add_argument("--output", default="flagship-campaign.json")
    campaign_preflight_parser = subparsers.add_parser(
        "campaign-preflight",
        help="Verify a frozen flagship campaign on its authorizing host",
    )
    campaign_preflight_parser.add_argument("--campaign", required=True)
    campaign_preflight_parser.add_argument(
        "--output",
        default="flagship-campaign-preflight.json",
    )
    campaign_start_parser = subparsers.add_parser(
        "campaign-start-formal",
        help="Transition a frozen, preflighted campaign to formal_running",
    )
    campaign_start_parser.add_argument("--campaign", required=True)
    campaign_start_parser.add_argument("--preflight", required=True)
    campaign_start_parser.add_argument("--output", required=True)
    campaign_complete_parser = subparsers.add_parser(
        "campaign-complete-formal",
        help="Consume the formal holdout and close the formal cycle",
    )
    campaign_complete_parser.add_argument("--campaign", required=True)
    campaign_complete_parser.add_argument("--completion", required=True)
    campaign_complete_parser.add_argument("--output", required=True)
    campaign_close_parser = subparsers.add_parser(
        "campaign-close-no-go",
        help="Close a pre-formal campaign without consuming its holdout",
    )
    campaign_close_parser.add_argument("--campaign", required=True)
    campaign_close_parser.add_argument("--no-go-record", required=True)
    campaign_close_parser.add_argument("--output", required=True)
    campaign_publication_parser = subparsers.add_parser(
        "campaign-record-publication",
        help="Close a release-ready campaign from downloaded publication verification",
    )
    campaign_publication_parser.add_argument("--campaign", required=True)
    campaign_publication_parser.add_argument("--verification", required=True)
    campaign_publication_parser.add_argument("--output", required=True)
    lifecycle_parser = subparsers.add_parser(
        "artifact-lifecycle",
        help="Append one validated lifecycle transition and write a new registry",
    )
    lifecycle_parser.add_argument("--registry", required=True)
    lifecycle_parser.add_argument("--candidate", required=True)
    lifecycle_parser.add_argument(
        "--to",
        required=True,
        choices=["development", "candidate", "frozen", "certified", "superseded", "revoked"],
    )
    lifecycle_parser.add_argument("--actor", required=True)
    lifecycle_parser.add_argument("--reviewer", required=True)
    lifecycle_parser.add_argument(
        "--reason",
        required=True,
        choices=[
            "certification_passed",
            "formal_cycle_failed",
            "new_certified_successor",
            "adapter_classification_changed",
            "packing_semantics_changed",
            "source_or_tokenizer_changed",
            "runtime_contract_invalidated",
            "provenance_error",
            "security_or_license_issue",
        ],
    )
    lifecycle_parser.add_argument("--narrative", required=True)
    lifecycle_parser.add_argument("--evidence", required=True)
    lifecycle_parser.add_argument("--replacement-candidate")
    lifecycle_parser.add_argument("--public-repository")
    lifecycle_parser.add_argument("--public-revision")
    lifecycle_parser.add_argument("--impact-scan")
    lifecycle_parser.add_argument("--output", required=True)
    claim_parser = subparsers.add_parser(
        "claim-render",
        help="Generate certified public claims and model card from bound evidence",
    )
    claim_parser.add_argument("--request", required=True)
    claim_parser.add_argument("--output", default="public-claim.json")
    claim_parser.add_argument("--model-card", default="README.md")

    validate_dataset_parser = subparsers.add_parser("validate-calibration-dataset")
    validate_dataset_parser.add_argument("--path", default=None)

    return parser
