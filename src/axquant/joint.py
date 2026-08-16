"""Beta weight x KV interaction diagnostic.

This is the 1.9.0b1 try: measure whether isolated weight and KV losses add,
and whether the memory-feasible (weight BPW, KV bits) winner flips with
context length. It never converts weights and never emits a certificate.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from axquant.analyzer import architecture_prior_report
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.memory_budget import evaluate_budget
from axquant.optimizer import estimate_kv_bytes
from axquant.planner import allocate_kv_cache, plan_quantization
from axquant.schema import (
    EvidenceKind,
    Inventory,
    JointBudgetCandidate,
    JointContextWinner,
    JointCrossoverSummary,
    JointInteractionReport,
    JointMeasuredDeltas,
    JointProxyScores,
    KvCachePlan,
    KvSensitivityReport,
    PlanRequest,
    ProfileName,
    QualityEvaluationResult,
    QuantizationPlan,
    RuntimeName,
    SensitivityReport,
)
from axquant.serde import load_model, stable_sha256, write_data, write_text

_LAYER_INDEX = re.compile(r"\.layers\.(\d+)\.")
_BETA_NOTE = (
    "axquant diagnose-joint is a 1.9.0b1 development diagnostic. "
    "It is not a certification claim and cannot authorize a Hub pack."
)


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
                "diagnose-joint without --sensitivity requires --allow-unmeasured "
                "because it uses architecture priors"
            )
        return architecture_prior_report(inventory, profile=profile)
    report = load_model(sensitivity_path, SensitivityReport)
    if report.inventory_sha256 != _inventory_digest(inventory):
        raise PlanningError("sensitivity report does not bind the selected inventory")
    return report


def text_layer_count(inventory: Inventory, plan: QuantizationPlan) -> int:
    """Prefer the architecture profile, then infer from tensor names."""

    for count in (
        plan.architecture_profile.text_layer_count,
        inventory.architecture_profile.text_layer_count,
    ):
        if count is not None:
            return count
    indices: set[int] = set()
    for tensor in inventory.tensors:
        match = _LAYER_INDEX.search(tensor.name)
        if match is not None:
            indices.add(int(match.group(1)))
    if not indices:
        raise PlanningError(
            "diagnose-joint needs a text layer count; the inventory has no layers.* tensors"
        )
    return max(indices) + 1


def _weight_bytes(plan: QuantizationPlan) -> int:
    parameters = sum(allocation.parameters for allocation in plan.assignments)
    if parameters <= 0:
        raise PlanningError("quantization plan contains no logical parameters")
    return math.ceil(plan.effective_bpw * parameters / 8.0)


def _weight_proxy_kl(plan: QuantizationPlan) -> float:
    weighted = 0.0
    total = 0
    for allocation in plan.assignments:
        weighted += allocation.metrics.output_kl * allocation.parameters
        total += allocation.parameters
    if total <= 0:
        raise PlanningError("cannot form a weight proxy from an empty plan")
    return weighted / total


def _kv_proxy_kl(plan: KvCachePlan, report: KvSensitivityReport | None) -> float | None:
    if report is None:
        return None
    by_layer = {entry.layer_index: entry for entry in report.entries}
    scores: list[float] = []
    for layer in plan.layers:
        entry = by_layer.get(layer.layer_index)
        if entry is None:
            return None
        match = next(
            (
                candidate
                for candidate in entry.candidates
                if candidate.bits == layer.bits and candidate.supported
            ),
            None,
        )
        if match is None:
            return None
        scores.append(match.metrics.output_kl)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _proxy_scores(
    plan: QuantizationPlan,
    kv_plan: KvCachePlan,
    kv_report: KvSensitivityReport | None,
) -> JointProxyScores:
    weight = _weight_proxy_kl(plan)
    kv = _kv_proxy_kl(kv_plan, kv_report)
    if kv is None:
        return JointProxyScores(weight_output_kl=weight)
    return JointProxyScores(
        weight_output_kl=weight,
        kv_output_kl=kv,
        additive_output_kl=weight + kv,
    )


def _quality_delta(result: QualityEvaluationResult) -> float:
    if not result.metrics.task_scores:
        raise PlanningError(
            "measured joint quality files must include task_scores so a delta can be formed"
        )
    mean = sum(result.metrics.task_scores.values()) / len(result.metrics.task_scores)
    return max(0.0, 1.0 - mean)


def _same_model_id(left: str, right: str) -> bool:
    if left == right:
        return True
    left_path = Path(left)
    right_path = Path(right)
    if left_path.exists() and right_path.exists():
        return left_path.resolve() == right_path.resolve()
    return False


def _require_same_model(result: QualityEvaluationResult, inventory: Inventory, label: str) -> None:
    if not _same_model_id(result.model.model_id, inventory.model.model_id):
        raise PlanningError(
            f"{label} quality evaluation model_id does not match the inspected model"
        )


def _measured_deltas(
    *,
    inventory: Inventory,
    weight_only: QualityEvaluationResult,
    kv_only: QualityEvaluationResult,
    joint: QualityEvaluationResult,
    threshold: float,
) -> JointMeasuredDeltas:
    _require_same_model(weight_only, inventory, "weight-only")
    _require_same_model(kv_only, inventory, "kv-only")
    _require_same_model(joint, inventory, "joint")
    if threshold <= 0.0:
        raise PlanningError("interaction threshold must be positive")
    weight_delta = _quality_delta(weight_only)
    kv_delta = _quality_delta(kv_only)
    joint_delta = _quality_delta(joint)
    interaction = joint_delta - weight_delta - kv_delta
    return JointMeasuredDeltas(
        weight_only_delta=weight_delta,
        kv_only_delta=kv_delta,
        joint_delta=joint_delta,
        interaction=interaction,
        threshold=threshold,
        material=abs(interaction) >= threshold,
        weight_only_sha256=stable_sha256(weight_only),
        kv_only_sha256=stable_sha256(kv_only),
        joint_sha256=stable_sha256(joint),
    )


def _winner_key(candidate: JointBudgetCandidate) -> tuple[float, float, float, int]:
    """Prefer lower additive proxy, then fewer leftover bytes, then lower BPW."""

    proxy = (
        candidate.proxy.additive_output_kl
        if candidate.proxy.additive_output_kl is not None
        else candidate.proxy.weight_output_kl
    )
    if proxy is None:
        proxy = 0.0
    return (
        proxy,
        float(candidate.remainder_bytes),
        candidate.target_bpw,
        candidate.kv_default_bits,
    )


def _crossover(candidates: list[JointBudgetCandidate]) -> JointCrossoverSummary:
    by_context: dict[int, list[JointBudgetCandidate]] = {}
    for candidate in candidates:
        by_context.setdefault(candidate.context_length, []).append(candidate)
    winners: list[JointContextWinner] = []
    for context in sorted(by_context):
        cells = by_context[context]
        feasible = [cell for cell in cells if cell.feasible]
        if not feasible:
            winners.append(
                JointContextWinner(
                    context_length=context,
                    feasible_count=0,
                )
            )
            continue
        best = min(feasible, key=_winner_key)
        score = best.proxy.additive_output_kl
        if score is None:
            score = best.proxy.weight_output_kl
        winners.append(
            JointContextWinner(
                context_length=context,
                target_bpw=best.target_bpw,
                kv_default_bits=best.kv_default_bits,
                feasible_count=len(feasible),
                proxy_score=score,
            )
        )
    pairs = {
        (winner.target_bpw, winner.kv_default_bits)
        for winner in winners
        if winner.target_bpw is not None
    }
    return JointCrossoverSummary(winners=winners, detected=len(pairs) > 1)


def _verdict(interaction: JointMeasuredDeltas | None) -> str:
    if interaction is None:
        return "insufficient-measured-interaction"
    return "interaction-material" if interaction.material else "interaction-small"


def joint_interaction_markdown(report: JointInteractionReport) -> str:
    interaction_block = "Not measured. Supply the three quality evaluations to compute I(W, KV)."
    if report.interaction is not None:
        sign = "material" if report.interaction.material else "small"
        interaction_block = (
            f"| Side | Quality delta (1 - mean task score) |\n"
            f"| --- | ---: |\n"
            f"| Weight only | {report.interaction.weight_only_delta:.6f} |\n"
            f"| KV only | {report.interaction.kv_only_delta:.6f} |\n"
            f"| Joint | {report.interaction.joint_delta:.6f} |\n"
            f"| I(W, KV) | {report.interaction.interaction:.6f} ({sign}; "
            f"threshold {report.interaction.threshold:.6f}) |"
        )
    winner_lines = []
    for winner in report.crossover.winners:
        if winner.target_bpw is None:
            winner_lines.append(f"| {winner.context_length} | none (infeasible) | 0 |")
            continue
        winner_lines.append(
            f"| {winner.context_length} | {winner.target_bpw:.3f} bpw + "
            f"KV{winner.kv_default_bits} | {winner.feasible_count} |"
        )
    notes = "\n".join(f"- {note}" for note in report.notes) or "- None."
    return f"""# AXQuant joint interaction diagnostic (beta)

{_BETA_NOTE}

- Verdict: `{report.verdict}`
- Crossover detected: `{report.crossover.detected}`
- Evidence: `{report.evidence_kind.value}`
- Profile: `{report.profile.value}`
- Memory limit / reserve: `{report.limit_bytes}` / `{report.reserve_bytes}` bytes

## Interaction

{interaction_block}

## Context winners (feasible cells, lowest isolated proxy)

| Context | Winner | Feasible cells |
| --- | --- | ---: |
{chr(10).join(winner_lines)}

## Notes

{notes}
"""


def diagnose_joint_interaction(
    *,
    model_dir: str | Path,
    max_memory_bytes: int,
    contexts: tuple[int, ...],
    weight_bpws: tuple[float, ...],
    kv_bits: tuple[int, ...],
    profile: ProfileName,
    output_dir: str | Path,
    inventory_path: str | Path | None = None,
    sensitivity_path: str | Path | None = None,
    kv_analysis_path: str | Path | None = None,
    allow_unmeasured: bool = False,
    reserve_bytes: int = 1_000_000_000,
    batch_size: int = 1,
    interaction_threshold: float = 0.02,
    quality_weight_only_path: str | Path | None = None,
    quality_kv_only_path: str | Path | None = None,
    quality_joint_path: str | Path | None = None,
) -> JointInteractionReport:
    """Enumerate a small deployment grid and optionally compute I(W, KV)."""

    if max_memory_bytes <= 0:
        raise PlanningError("memory limit must be positive")
    if not contexts or any(context <= 0 for context in contexts):
        raise PlanningError("every context length must be a positive integer")
    if not weight_bpws or any(bpw <= 0.0 or bpw > 16.0 for bpw in weight_bpws):
        raise PlanningError("every weight BPW must be in (0, 16]")
    if not kv_bits:
        raise PlanningError("at least one KV bit-width is required")
    if batch_size <= 0:
        raise PlanningError("batch size must be a positive integer")

    quality_paths = (quality_weight_only_path, quality_kv_only_path, quality_joint_path)
    if any(path is None for path in quality_paths) and any(
        path is not None for path in quality_paths
    ):
        raise PlanningError(
            "measured interaction requires --quality-weight-only, --quality-kv-only, "
            "and --quality-joint together"
        )

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
    kv_report = (
        load_model(kv_analysis_path, KvSensitivityReport) if kv_analysis_path is not None else None
    )
    if kv_report is not None and kv_report.model.model_id != report.model.model_id:
        raise PlanningError("KV sensitivity report model does not match the weight sensitivity")

    interaction = None
    if (
        quality_weight_only_path is not None
        and quality_kv_only_path is not None
        and quality_joint_path is not None
    ):
        interaction = _measured_deltas(
            inventory=inventory,
            weight_only=load_model(quality_weight_only_path, QualityEvaluationResult),
            kv_only=load_model(quality_kv_only_path, QualityEvaluationResult),
            joint=load_model(quality_joint_path, QualityEvaluationResult),
            threshold=interaction_threshold,
        )

    plans: dict[float, QuantizationPlan] = {}
    for target_bpw in weight_bpws:
        request = PlanRequest(
            profile=profile,
            target_bpw=target_bpw,
            allow_unmeasured=allow_unmeasured,
            target_mode="balanced",
            primary_runtime=RuntimeName.AX_ENGINE,
            minimum_quality_retention=0.98,
        )
        plans[target_bpw] = plan_quantization(report, request)

    layer_count = text_layer_count(inventory, next(iter(plans.values())))
    group_size = next(iter(plans.values())).group_size
    kv_plans: dict[int, KvCachePlan] = {}
    for bits in kv_bits:
        kv_plans[bits] = allocate_kv_cache(
            layer_count,
            default_bits=bits,
            min_bits=4 if bits >= 4 else bits,
            group_size=group_size,
        )

    candidates: list[JointBudgetCandidate] = []
    for target_bpw, plan in plans.items():
        weight_bytes = _weight_bytes(plan)
        for bits, kv_plan in kv_plans.items():
            for context in contexts:
                kv_bytes = estimate_kv_bytes(
                    kv_plan,
                    model_dir=model,
                    context_length=context,
                    batch_size=batch_size,
                )
                breakdown = evaluate_budget(
                    weight_bytes,
                    kv_bytes,
                    reserve_bytes,
                    max_memory_bytes,
                )
                candidates.append(
                    JointBudgetCandidate(
                        target_bpw=target_bpw,
                        kv_default_bits=bits,
                        context_length=context,
                        weight_bytes=weight_bytes,
                        kv_bytes=kv_bytes,
                        reserve_bytes=reserve_bytes,
                        limit_bytes=max_memory_bytes,
                        remainder_bytes=breakdown.remainder_bytes,
                        feasible=breakdown.feasible,
                        estimated_main_bpw=plan.effective_bpw,
                        proxy=_proxy_scores(plan, kv_plan, kv_report),
                        plan_sha256=stable_sha256(plan),
                        kv_plan_sha256=stable_sha256(kv_plan),
                    )
                )

    if report.evidence_kind is EvidenceKind.ARCHITECTURE_PRIOR:
        evidence = EvidenceKind.ARCHITECTURE_PRIOR
    elif report.evidence_kind is EvidenceKind.IMPORTED:
        evidence = EvidenceKind.IMPORTED
    else:
        evidence = EvidenceKind.MEASURED_DEVELOPMENT

    notes = [
        _BETA_NOTE,
        "Independent planning is still used for each grid cell; this is not a joint solver.",
        "I(W, KV) is only defined when the three quality evaluations are supplied.",
    ]
    if kv_report is None:
        notes.append(
            "No --kv-analysis: KV proxy KL is omitted and crossover ranks on weight proxy "
            "and remainder."
        )
    if interaction is None:
        notes.append("No measured quality triple: verdict is insufficient-measured-interaction.")
    notes.extend(report.warnings)

    result = JointInteractionReport(
        evidence_kind=evidence,
        profile=profile,
        model=inventory.model,
        limit_bytes=max_memory_bytes,
        reserve_bytes=reserve_bytes,
        batch_size=batch_size,
        weight_bpws=tuple(weight_bpws),
        kv_default_bits=tuple(kv_bits),
        contexts=tuple(contexts),
        interaction=interaction,
        candidates=candidates,
        crossover=_crossover(candidates),
        verdict=_verdict(interaction),
        notes=notes,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_data(output / "joint-interaction.json", result)
    write_text(output / "joint-interaction.md", joint_interaction_markdown(result))
    return result
