from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Literal

from axquant.analyzer import architecture_prior_report
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.memory_budget import evaluate_budget
from axquant.planner import allocate_kv_cache, allocate_kv_cache_measured, plan_quantization
from axquant.profiles import objective_for_mode
from axquant.schema import (
    ArtifactManifest,
    DeploymentEvidenceKind,
    DeploymentPlan,
    EvidenceKind,
    Inventory,
    KernelLatencyTable,
    KvCachePlan,
    KvSensitivityReport,
    PlanRequest,
    ProfileName,
    QuantizationPlan,
    RuntimeName,
    SensitivityReport,
)
from axquant.serde import load_model, read_data, stable_sha256, write_data, write_text

OptimizationMode = Literal["balanced", "quality", "low-memory", "speed"]
KvCacheMode = Literal["off", "prior", "measured"]

_MEMORY_VALUE = re.compile(
    r"^(?P<amount>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)$",
    re.IGNORECASE,
)
_MEMORY_MULTIPLIERS = {
    "B": 1,
    "KB": 1_000,
    "MB": 1_000_000,
    "GB": 1_000_000_000,
    "TB": 1_000_000_000_000,
    "KIB": 1 << 10,
    "MIB": 1 << 20,
    "GIB": 1 << 30,
    "TIB": 1 << 40,
}


def parse_memory_bytes(value: str) -> int:
    """Parse an explicit decimal or IEC memory amount into bytes."""

    match = _MEMORY_VALUE.fullmatch(value.strip())
    if match is None:
        raise ValueError("memory must include a unit, for example 18GB or 16GiB")
    try:
        amount = Decimal(match.group("amount"))
    except InvalidOperation as exc:
        raise ValueError(f"invalid memory amount: {value}") from exc
    result = amount * _MEMORY_MULTIPLIERS[match.group("unit").upper()]
    if result != result.to_integral_value() or result < 0:
        raise ValueError("memory amount must resolve to a non-negative whole byte count")
    return int(result)


def _inventory_digest(inventory: Inventory) -> str:
    return stable_sha256(inventory.model_dump(mode="json", exclude={"created_at"}))


def _load_inventory(model: Path, inventory_path: str | Path | None) -> Inventory:
    if inventory_path is not None:
        return load_model(inventory_path, Inventory)
    return inspect_model(
        model,
        allow_quantized=(model / "axquant_manifest.json").is_file(),
    )


def _load_sensitivity(
    inventory: Inventory,
    sensitivity_path: str | Path | None,
    *,
    profile: ProfileName,
    allow_unmeasured: bool,
) -> SensitivityReport:
    if sensitivity_path is None:
        if not allow_unmeasured:
            raise PlanningError(
                "optimize without --sensitivity requires --allow-unmeasured because it uses "
                "architecture priors"
            )
        return architecture_prior_report(inventory, profile=profile)
    report = load_model(sensitivity_path, SensitivityReport)
    if report.inventory_sha256 != _inventory_digest(inventory):
        raise PlanningError("sensitivity report does not bind the selected inventory")
    return report


def _positive_config_int(config: dict[str, object], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if type(value) is int and value > 0:
            return value
    return None


def _kv_dimensions(model: Path) -> tuple[int, int]:
    payload = read_data(model / "config.json")
    if not isinstance(payload, dict):
        raise PlanningError("model config must be a JSON object for KV accounting")
    nested = payload.get("text_config")
    text = nested if isinstance(nested, dict) else payload
    kv_heads = _positive_config_int(text, "num_key_value_heads", "num_kv_heads")
    attention_heads = _positive_config_int(text, "num_attention_heads", "n_head")
    if kv_heads is None:
        kv_heads = attention_heads
    head_dim = _positive_config_int(text, "head_dim", "qk_head_dim")
    if head_dim is None:
        hidden_size = _positive_config_int(text, "hidden_size", "d_model")
        if hidden_size is not None and attention_heads is not None:
            if hidden_size % attention_heads != 0:
                raise PlanningError(
                    "hidden size is not divisible by attention heads for KV accounting"
                )
            head_dim = hidden_size // attention_heads
    if kv_heads is None or head_dim is None:
        raise PlanningError(
            "KV accounting requires num_key_value_heads (or num_attention_heads) and head_dim "
            "(or divisible hidden_size) in config.json"
        )
    return kv_heads, head_dim


def estimate_kv_bytes(
    kv_plan: KvCachePlan,
    *,
    model_dir: str | Path,
    context_length: int,
    batch_size: int,
) -> int:
    """Estimate packed K+V bytes, including affine scale overhead per group."""

    if type(context_length) is not int or context_length <= 0:
        raise PlanningError("context length must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise PlanningError("batch size must be a positive integer")
    model = Path(model_dir).expanduser().resolve()
    kv_heads, head_dim = _kv_dimensions(model)
    elements_per_layer = 2 * batch_size * context_length * kv_heads * head_dim
    total = 0
    for layer in kv_plan.layers:
        effective_bits = (
            Fraction(16, 1)
            if layer.bits == 16
            else Fraction(layer.bits, 1) + Fraction(32, layer.group_size)
        )
        total += math.ceil(Fraction(elements_per_layer, 8) * effective_bits)
    return total


def _estimate_weight_bytes(plan: QuantizationPlan) -> int:
    parameters = sum(allocation.parameters for allocation in plan.assignments)
    if parameters <= 0:
        raise PlanningError("quantization plan contains no logical parameters")
    return math.ceil(plan.effective_bpw * parameters / 8.0)


def _attach_kv_plan(
    plan: QuantizationPlan,
    *,
    mode: KvCacheMode,
    default_bits: int,
    kv_analysis: str | Path | None,
) -> None:
    if mode == "off":
        return
    if mode == "prior":
        layer_count = plan.architecture_profile.text_layer_count
        if layer_count is None:
            raise PlanningError("prior KV planning requires a known text layer count")
        plan.kv_cache = allocate_kv_cache(
            layer_count,
            default_bits=default_bits,
            group_size=plan.group_size,
        )
        return
    if kv_analysis is None:
        raise PlanningError("--kv-cache measured requires --kv-analysis")
    report = load_model(kv_analysis, KvSensitivityReport)
    if report.model.model_id != plan.source_model.model_id:
        raise PlanningError("KV sensitivity report model does not match the weight plan")
    plan.kv_cache = allocate_kv_cache_measured(report)


def _deployment_evidence(
    source: EvidenceKind,
    *,
    manifest_backed: bool,
) -> DeploymentEvidenceKind:
    if source is EvidenceKind.ARCHITECTURE_PRIOR:
        return "architecture_prior"
    if source is EvidenceKind.IMPORTED:
        return "imported-as-estimate"
    if source is EvidenceKind.MEASURED and manifest_backed:
        return "measured"
    return "estimate"


def deployment_plan_markdown(deployment: DeploymentPlan) -> str:
    bpw_label = (
        f"{deployment.measured_main_bpw:.6f} measured main BPW"
        if deployment.measured_main_bpw is not None
        else f"{deployment.estimated_main_bpw:.6f} estimated BPW"
    )
    notes = "\n".join(f"- {note}" for note in deployment.notes) or "- None."
    return f"""# AXQuant deployment plan

{deployment.explanation}

| Item | Bytes |
| --- | ---: |
| Weights | {deployment.weight_bytes} |
| KV cache | {deployment.kv_bytes} |
| Explicit runtime reserve | {deployment.reserve_bytes} |
| Requested limit | {deployment.limit_bytes} |
| Remainder | {deployment.remainder_bytes} |

- Feasible: `{deployment.feasible}`
- Evidence: `{deployment.evidence_kind}` (source: `{deployment.source_evidence_kind.value}`)
- Precision: {bpw_label}
- Product class: `{deployment.target_class}`
- Context / batch: `{deployment.context_length}` / `{deployment.batch_size}`
- Profile / mode: `{deployment.profile.value}` / `{deployment.mode}`
- Runtime: `{deployment.runtime.value}`

## Notes

{notes}
"""


def optimize_deployment(
    *,
    model_dir: str | Path,
    max_memory_bytes: int,
    context_length: int,
    profile: ProfileName,
    runtime: RuntimeName,
    minimum_quality_retention: float,
    mode: OptimizationMode,
    output_dir: str | Path,
    inventory_path: str | Path | None = None,
    sensitivity_path: str | Path | None = None,
    kv_analysis_path: str | Path | None = None,
    allow_unmeasured: bool = False,
    target_bpw: float = 4.8,
    kv_cache: KvCacheMode = "off",
    kv_default_bits: int = 4,
    reserve_bytes: int = 1_000_000_000,
    batch_size: int = 1,
    latency_table_path: str | Path | None = None,
) -> DeploymentPlan:
    """Orchestrate existing weight/KV planning under one explicit memory budget."""

    if runtime is not RuntimeName.AX_ENGINE:
        raise PlanningError("AX Engine is the only optimize runtime supported in v1.8")
    model = Path(model_dir).expanduser().resolve()
    if not model.is_dir():
        raise PlanningError(f"model directory does not exist: {model}")
    inventory = _load_inventory(model, inventory_path)
    report = _load_sensitivity(
        inventory,
        sensitivity_path,
        profile=profile,
        allow_unmeasured=allow_unmeasured,
    )
    objective = objective_for_mode(profile, mode)
    request = PlanRequest(
        profile=profile,
        target_bpw=target_bpw,
        allow_unmeasured=allow_unmeasured,
        target_mode=mode,
        primary_runtime=runtime,
        minimum_quality_retention=minimum_quality_retention,
    )
    latency_table = (
        load_model(latency_table_path, KernelLatencyTable)
        if latency_table_path is not None
        else None
    )
    plan = plan_quantization(
        report,
        request,
        kernel_latency=latency_table,
        objective_weights=objective,
    )
    _attach_kv_plan(
        plan,
        mode=kv_cache,
        default_bits=kv_default_bits,
        kv_analysis=kv_analysis_path,
    )
    kv_bytes = (
        estimate_kv_bytes(
            plan.kv_cache,
            model_dir=model,
            context_length=context_length,
            batch_size=batch_size,
        )
        if plan.kv_cache is not None
        else 0
    )

    manifest_path = model / "axquant_manifest.json"
    manifest = load_model(manifest_path, ArtifactManifest) if manifest_path.is_file() else None
    if manifest is not None:
        if manifest.target_class != plan.target_class:
            raise PlanningError(
                "artifact manifest target class does not match the optimized weight plan"
            )
        if manifest.profile != plan.profile:
            raise PlanningError("artifact manifest profile does not match the optimized plan")
        weight_bytes = manifest.weight_file_size_bytes
        weight_basis: Literal["plan-estimate", "artifact-manifest"] = "artifact-manifest"
        measured_main_bpw = manifest.measured_main_bpw
        estimated_main_bpw = None
    else:
        weight_bytes = _estimate_weight_bytes(plan)
        weight_basis = "plan-estimate"
        measured_main_bpw = None
        estimated_main_bpw = plan.effective_bpw

    breakdown = evaluate_budget(
        weight_bytes,
        kv_bytes,
        reserve_bytes,
        max_memory_bytes,
    )
    if not breakdown.feasible:
        required = weight_bytes + kv_bytes + reserve_bytes
        raise PlanningError(
            "deployment memory budget is infeasible: "
            f"weights={weight_bytes}, kv={kv_bytes}, reserve={reserve_bytes}, "
            f"required={required}, limit={max_memory_bytes}, "
            f"remainder={breakdown.remainder_bytes}"
        )

    evidence_kind = _deployment_evidence(
        plan.evidence_kind,
        manifest_backed=manifest is not None,
    )
    notes = [
        f"Runtime reserve is explicit: {reserve_bytes} bytes.",
        "Static accounting is a planning gate, not measured peak resident memory.",
    ]
    notes.extend(plan.warnings)
    deployment = DeploymentPlan(
        **breakdown.model_dump(),
        evidence_kind=evidence_kind,
        source_evidence_kind=plan.evidence_kind,
        context_length=context_length,
        batch_size=batch_size,
        profile=profile,
        target_class=plan.target_class,
        runtime=runtime,
        mode=mode,
        objective=plan.objective,
        minimum_quality_retention=minimum_quality_retention,
        weight_bytes_basis=weight_basis,
        measured_main_bpw=measured_main_bpw,
        estimated_main_bpw=estimated_main_bpw,
        plan_sha256=stable_sha256(plan),
        kv_plan_sha256=(stable_sha256(plan.kv_cache) if plan.kv_cache is not None else None),
        notes=notes,
        explanation=(
            "The selected existing weight and KV plans fit the requested limit under the "
            "normative weights + KV + explicit reserve constraint."
        ),
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_data(output / "axquant_plan.json", plan)
    write_data(output / "deployment-plan.json", deployment)
    write_text(output / "deployment-plan.md", deployment_plan_markdown(deployment))
    return deployment
