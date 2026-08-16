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
from axquant.identity import same_model_identity
from axquant.inspector import inspect_model
from axquant.memory_budget import evaluate_budget
from axquant.optimizer import estimate_kv_bytes, optimize_deployment
from axquant.planner import allocate_kv_cache, plan_quantization
from axquant.schema import (
    CandidateMeasurement,
    EvidenceKind,
    Inventory,
    JointBudgetCandidate,
    JointContextWinner,
    JointCrossoverSummary,
    JointInteractionReport,
    JointMeasuredDeltas,
    JointProxyScores,
    JointScoredCell,
    JointSelectionReport,
    KvCachePlan,
    KvLayerAllocation,
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
            raise PlanningError(
                f"KV sensitivity is missing layer {layer.layer_index}; "
                "incomplete --kv-analysis cannot rank a joint grid"
            )
        match = next(
            (
                candidate
                for candidate in entry.candidates
                if _kv_candidate_matches(candidate, layer)
            ),
            None,
        )
        if match is None:
            raise PlanningError(
                f"KV sensitivity has no supported {layer.bits}-bit / group "
                f"{layer.group_size} candidate for layer {layer.layer_index}"
            )
        scores.append(match.metrics.output_kl)
    if not scores:
        raise PlanningError("KV sensitivity produced no layer scores")
    return sum(scores) / len(scores)


def _kv_candidate_matches(candidate: CandidateMeasurement, layer: KvLayerAllocation) -> bool:
    """16-bit KV probes store group_size=None; quantized cells must match the plan group."""

    if not candidate.supported or candidate.bits != layer.bits:
        return False
    if layer.bits == 16:
        return True
    return candidate.group_size == layer.group_size


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


def _mean_task_score(result: QualityEvaluationResult) -> float:
    if not result.metrics.task_scores:
        raise PlanningError(
            "measured joint quality files must include task_scores so a score can be formed"
        )
    return sum(result.metrics.task_scores.values()) / len(result.metrics.task_scores)


def _same_model_id(left: str, right: str) -> bool:
    if left == right:
        return True
    left_path = Path(left)
    right_path = Path(right)
    if left_path.exists() and right_path.exists():
        return left_path.resolve() == right_path.resolve()
    return False


def _require_same_model(result: QualityEvaluationResult, inventory: Inventory, label: str) -> None:
    if _same_model_id(result.model.model_id, inventory.model.model_id):
        return
    if same_model_identity(result.model, inventory.model):
        return
    raise PlanningError(
        f"{label} quality evaluation model identity does not match the inspected model"
    )


def _require_matched_quality(
    results: tuple[tuple[str, QualityEvaluationResult], ...],
    inventory: Inventory,
) -> None:
    first_label, first = results[0]
    _require_same_model(first, inventory, first_label)
    first_tasks = {task.task_id: task.category for task in first.task_results}
    for label, result in results[1:]:
        _require_same_model(result, inventory, label)
        if result.dataset_sha256 != first.dataset_sha256:
            raise PlanningError(f"{label} dataset_sha256 does not match {first_label}")
        if result.random_seed != first.random_seed:
            raise PlanningError(f"{label} random_seed does not match {first_label}")
        if result.generation != first.generation:
            raise PlanningError(f"{label} generation config does not match {first_label}")
        got_tasks = {task.task_id: task.category for task in result.task_results}
        if got_tasks != first_tasks:
            raise PlanningError(f"{label} task id/category mapping does not match {first_label}")


def _measured_deltas(
    *,
    inventory: Inventory,
    baseline: QualityEvaluationResult,
    weight_only: QualityEvaluationResult,
    kv_only: QualityEvaluationResult,
    joint: QualityEvaluationResult,
    threshold: float,
) -> JointMeasuredDeltas:
    _require_matched_quality(
        (
            ("baseline", baseline),
            ("weight-only", weight_only),
            ("kv-only", kv_only),
            ("joint", joint),
        ),
        inventory,
    )
    if threshold <= 0.0:
        raise PlanningError("interaction threshold must be positive")
    baseline_score = _mean_task_score(baseline)
    weight_score = _mean_task_score(weight_only)
    kv_score = _mean_task_score(kv_only)
    joint_score = _mean_task_score(joint)
    weight_delta = baseline_score - weight_score
    kv_delta = baseline_score - kv_score
    joint_delta = baseline_score - joint_score
    interaction = joint_delta - weight_delta - kv_delta
    return JointMeasuredDeltas(
        baseline_score=baseline_score,
        weight_only_score=weight_score,
        kv_only_score=kv_score,
        joint_score=joint_score,
        weight_only_delta=weight_delta,
        kv_only_delta=kv_delta,
        joint_delta=joint_delta,
        interaction=interaction,
        threshold=threshold,
        material=abs(interaction) >= threshold,
        baseline_sha256=stable_sha256(baseline),
        weight_only_sha256=stable_sha256(weight_only),
        kv_only_sha256=stable_sha256(kv_only),
        joint_sha256=stable_sha256(joint),
        dataset_sha256=baseline.dataset_sha256,
    )


def _winner_key(candidate: JointBudgetCandidate) -> tuple[float, int, float, int]:
    """Prefer lower additive proxy, then lower KV bits, then lower BPW, then more slack."""

    proxy = candidate.proxy.additive_output_kl
    if proxy is None:
        raise PlanningError("cannot rank a candidate without an additive proxy")
    return (
        proxy,
        candidate.kv_default_bits,
        candidate.target_bpw,
        -candidate.remainder_bytes,
    )


def _crossover(candidates: list[JointBudgetCandidate]) -> JointCrossoverSummary:
    by_context: dict[int, list[JointBudgetCandidate]] = {}
    for candidate in candidates:
        by_context.setdefault(candidate.context_length, []).append(candidate)
    winners: list[JointContextWinner] = []
    for context in sorted(by_context):
        cells = by_context[context]
        feasible = [cell for cell in cells if cell.feasible]
        rankable = [cell for cell in feasible if cell.ranking_available]
        if not rankable:
            winners.append(
                JointContextWinner(
                    context_length=context,
                    feasible_count=len(feasible),
                    rankable_count=0,
                )
            )
            continue
        best = min(rankable, key=_winner_key)
        winners.append(
            JointContextWinner(
                context_length=context,
                target_bpw=best.target_bpw,
                kv_default_bits=best.kv_default_bits,
                feasible_count=len(feasible),
                rankable_count=len(rankable),
                proxy_score=best.proxy.additive_output_kl,
            )
        )
    pairs = {
        (winner.target_bpw, winner.kv_default_bits)
        for winner in winners
        if winner.target_bpw is not None
    }
    return JointCrossoverSummary(
        winners=winners,
        detected=len(pairs) > 1,
        ranking_complete=all(winner.rankable_count == winner.feasible_count for winner in winners),
    )


def _verdict(interaction: JointMeasuredDeltas | None) -> str:
    if interaction is None:
        return "insufficient-measured-interaction"
    return "interaction-material" if interaction.material else "interaction-small"


def joint_interaction_markdown(report: JointInteractionReport) -> str:
    interaction_block = (
        "Not measured. Supply the BF16 baseline plus three treatment quality "
        "evaluations to compute I(W, KV)."
    )
    if report.interaction is not None:
        sign = "material" if report.interaction.material else "small"
        interaction_block = (
            f"| Side | Mean task score | Delta vs baseline |\n"
            f"| --- | ---: | ---: |\n"
            f"| Baseline | {report.interaction.baseline_score:.6f} | 0.000000 |\n"
            f"| Weight only | {report.interaction.weight_only_score:.6f} | "
            f"{report.interaction.weight_only_delta:.6f} |\n"
            f"| KV only | {report.interaction.kv_only_score:.6f} | "
            f"{report.interaction.kv_only_delta:.6f} |\n"
            f"| Joint | {report.interaction.joint_score:.6f} | "
            f"{report.interaction.joint_delta:.6f} |\n"
            f"| I(W, KV) |  | {report.interaction.interaction:.6f} ({sign}; "
            f"threshold {report.interaction.threshold:.6f}) |"
        )
    feasible_lines: list[str] = []
    by_context: dict[int, list[str]] = {}
    for candidate in report.candidates:
        if not candidate.feasible:
            continue
        by_context.setdefault(candidate.context_length, []).append(
            f"{candidate.target_bpw:.3f}+KV{candidate.kv_default_bits}"
        )
    for context in report.contexts:
        cells = ", ".join(by_context.get(context, [])) or "none"
        feasible_lines.append(f"| {context} | {cells} |")
    winner_lines = []
    for winner in report.crossover.winners:
        if winner.target_bpw is None:
            reason = "infeasible" if winner.feasible_count == 0 else "not rankable"
            winner_lines.append(
                f"| {winner.context_length} | none ({reason}) | "
                f"{winner.feasible_count}/{winner.rankable_count} |"
            )
            continue
        winner_lines.append(
            f"| {winner.context_length} | {winner.target_bpw:.3f} bpw + "
            f"KV{winner.kv_default_bits} | "
            f"{winner.feasible_count}/{winner.rankable_count} |"
        )
    notes = "\n".join(f"- {note}" for note in report.notes) or "- None."
    return f"""# AXQuant joint interaction diagnostic (beta)

{_BETA_NOTE}

- Verdict: `{report.verdict}`
- Crossover detected: `{report.crossover.detected}`
- Evidence: `{report.evidence_kind}`
- Profile: `{report.profile.value}`
- Memory limit / reserve: `{report.limit_bytes}` / `{report.reserve_bytes}` bytes

## Interaction

{interaction_block}

## Estimated feasible cells

| Context | Feasible target BPW + KV |
| --- | --- |
{chr(10).join(feasible_lines)}

## Context winners (feasible rankable cells, lowest additive proxy)

| Context | Winner | Feasible / rankable |
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
    quality_baseline_path: str | Path | None = None,
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

    quality_paths = (
        quality_baseline_path,
        quality_weight_only_path,
        quality_kv_only_path,
        quality_joint_path,
    )
    if any(path is None for path in quality_paths) and any(
        path is not None for path in quality_paths
    ):
        raise PlanningError(
            "measured interaction requires --quality-baseline, --quality-weight-only, "
            "--quality-kv-only, and --quality-joint together"
        )

    model = Path(model_dir).expanduser().resolve()
    if not model.is_dir():
        raise PlanningError(f"model directory does not exist: {model}")
    live_inventory = inspect_model(
        model,
        allow_quantized=(model / "axquant_manifest.json").is_file(),
    )
    if inventory_path is None:
        inventory = live_inventory
    else:
        inventory = load_model(inventory_path, Inventory)
        if _inventory_digest(inventory) != _inventory_digest(live_inventory):
            raise PlanningError("supplied --inventory does not match the inspected --model")
    report = _load_sensitivity(
        inventory,
        sensitivity_path,
        profile=profile,
        allow_unmeasured=allow_unmeasured,
    )
    kv_report = (
        load_model(kv_analysis_path, KvSensitivityReport) if kv_analysis_path is not None else None
    )
    if kv_report is not None:
        if not same_model_identity(kv_report.model, report.model) and not _same_model_id(
            kv_report.model.model_id, report.model.model_id
        ):
            raise PlanningError("KV sensitivity report model does not match the weight sensitivity")
        if kv_report.inventory_sha256 != _inventory_digest(inventory):
            raise PlanningError("KV sensitivity report does not bind the selected inventory")
        if kv_report.profile != profile:
            raise PlanningError("KV sensitivity profile does not match the requested profile")

    interaction = None
    if (
        quality_baseline_path is not None
        and quality_weight_only_path is not None
        and quality_kv_only_path is not None
        and quality_joint_path is not None
    ):
        interaction = _measured_deltas(
            inventory=inventory,
            baseline=load_model(quality_baseline_path, QualityEvaluationResult),
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
                proxy = _proxy_scores(plan, kv_plan, kv_report)
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
                        ranking_available=proxy.additive_output_kl is not None,
                        estimated_main_bpw=plan.effective_bpw,
                        proxy=proxy,
                        plan_sha256=stable_sha256(plan),
                        kv_plan_sha256=stable_sha256(kv_plan),
                    )
                )

    if kv_report is not None:
        expected_layers = next(iter(kv_plans.values())).layers
        if kv_report.text_layer_count != len(expected_layers):
            raise PlanningError(
                "KV sensitivity text layer count does not match the planned KV grid"
            )
        if any(plan.default_group_size != kv_report.group_size for plan in kv_plans.values()):
            raise PlanningError("KV sensitivity group size does not match the planned KV grid")

    evidence = (
        EvidenceKind.ARCHITECTURE_PRIOR
        if report.evidence_kind is EvidenceKind.ARCHITECTURE_PRIOR
        else EvidenceKind.MEASURED_DEVELOPMENT
    )

    notes = [
        _BETA_NOTE,
        "Independent planning is still used for each grid cell; this is not a joint solver.",
        "I(W, KV) is only defined when the BF16 baseline and three treatment evaluations "
        "are supplied.",
        "Memory feasibility is analytical (estimated weights + KV + reserve), not measured RSS.",
    ]
    if kv_report is None:
        notes.append(
            "No --kv-analysis: cells are not rankable, so no winner or crossover is claimed."
        )
    if interaction is None:
        notes.append("No measured quality quadruple: verdict is insufficient-measured-interaction.")
    notes.extend(report.warnings)

    result = JointInteractionReport(
        evidence_kind=evidence,
        profile=profile,
        model=inventory.model,
        inventory_sha256=_inventory_digest(inventory),
        weight_sensitivity_sha256=stable_sha256(report),
        kv_sensitivity_sha256=stable_sha256(kv_report) if kv_report is not None else None,
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
    for target_bpw, plan in plans.items():
        write_data(output / f"weight-plan-{target_bpw:.3f}.json", plan)
    for bits, kv_plan in kv_plans.items():
        write_data(output / f"kv-plan-{bits}.json", kv_plan)
    write_data(output / "joint-interaction.json", result)
    write_text(output / "joint-interaction.md", joint_interaction_markdown(result))
    return result


def compression_intensity(value: float, maximum: float) -> float:
    """Map an isolated loss onto [0, 1] relative to the grid maximum."""

    if maximum <= 0.0:
        return 0.0
    return min(1.0, max(0.0, value / maximum))


def coupled_joint_loss(
    candidate: JointBudgetCandidate,
    *,
    interaction: float,
    max_weight_kl: float,
    max_kv_kl: float,
) -> float:
    """Predict joint loss: additive proxies plus an I-weighted coupling term.

    When I > 0, compressing weights and KV together is worse than the sum.
    When I < 0, compressing both is better than the sum. A constant I does
    not change ranking; the product of compression intensities does.
    """

    weight = candidate.proxy.weight_output_kl
    kv = candidate.proxy.kv_output_kl
    additive = candidate.proxy.additive_output_kl
    if weight is None or kv is None or additive is None:
        raise PlanningError("coupled ranking requires a complete additive proxy")
    weight_intensity = compression_intensity(weight, max_weight_kl)
    kv_intensity = compression_intensity(kv, max_kv_kl)
    # A one-sided grid (every cell shares the same isolated loss on one
    # side) would otherwise force that intensity to 0 and make I a no-op.
    # Keep the product form, but treat a zero-span side as fully present
    # so I still changes ranking through the side that varies.
    if max_weight_kl <= 0.0:
        weight_intensity = 1.0
    if max_kv_kl <= 0.0:
        kv_intensity = 1.0
    return additive + interaction * weight_intensity * kv_intensity


def _kv_layer_signature(plan: QuantizationPlan) -> tuple[tuple[int, int, int], ...]:
    if plan.kv_cache is None:
        return ()
    return tuple(
        (layer.layer_index, layer.bits, layer.group_size) for layer in plan.kv_cache.layers
    )


def _same_effective_configuration(left: QuantizationPlan, right: QuantizationPlan) -> bool:
    """Compare deployable precision, not KV allocation_basis / digest wrappers."""

    return left.target_bpw == right.target_bpw and _kv_layer_signature(left) == _kv_layer_signature(
        right
    )


def select_coupled_candidate(
    candidates: list[JointBudgetCandidate],
    *,
    context_length: int,
    interaction: float,
) -> tuple[JointBudgetCandidate, list[JointScoredCell]]:
    """Pick the feasible rankable cell with the lowest coupled loss at one context."""

    pool = [
        candidate
        for candidate in candidates
        if candidate.context_length == context_length
        and candidate.feasible
        and candidate.ranking_available
    ]
    if not pool:
        raise PlanningError(
            f"no feasible rankable joint cell at context {context_length}; "
            "widen the grid or raise --max-memory"
        )
    max_weight = max(
        candidate.proxy.weight_output_kl
        for candidate in pool
        if candidate.proxy.weight_output_kl is not None
    )
    max_kv = max(
        candidate.proxy.kv_output_kl
        for candidate in pool
        if candidate.proxy.kv_output_kl is not None
    )
    scored: list[tuple[float, JointBudgetCandidate]] = [
        (
            coupled_joint_loss(
                candidate,
                interaction=interaction,
                max_weight_kl=max_weight,
                max_kv_kl=max_kv,
            ),
            candidate,
        )
        for candidate in pool
    ]
    scored.sort(
        key=lambda item: (
            item[0],
            item[1].kv_default_bits,
            item[1].target_bpw,
            -item[1].remainder_bytes,
        )
    )
    winner = scored[0][1]
    cells = [
        JointScoredCell(
            target_bpw=candidate.target_bpw,
            kv_default_bits=candidate.kv_default_bits,
            context_length=candidate.context_length,
            additive_loss=candidate.proxy.additive_output_kl or 0.0,
            coupled_loss=loss,
            selected=candidate is winner,
        )
        for loss, candidate in scored
    ]
    return winner, cells


def joint_selection_markdown(selection: JointSelectionReport) -> str:
    rows = []
    for cell in selection.scored_cells:
        mark = " yes" if cell.selected else ""
        rows.append(
            f"| {cell.target_bpw:.3f} | KV{cell.kv_default_bits} | "
            f"{cell.additive_loss:.6f} | {cell.coupled_loss:.6f} |{mark} |"
        )
    scored_table = "\n".join(rows) or "| (none) | | | | |"
    notes = "\n".join(f"- {note}" for note in selection.notes) or "- None."
    return f"""# AXQuant joint plan selection (beta)

I-gated WeightPlan x KVPlan search. This writes a development plan for
`axquant convert`. It is not a certificate.

- Selection basis: `{selection.selection_basis}`
- Differs from 1.8 independent optimize: `{selection.differs_from_independent}`
- Context: `{selection.context_length}`
- Independent target BPW: `{selection.independent_target_bpw:.3f}`
- Selected: `{selection.selected_target_bpw:.3f}` bpw + KV{selection.selected_kv_bits}
- Interaction: `{selection.interaction}` material=`{selection.interaction_material}`

## Scored feasible cells

| Target BPW | KV | Additive | Coupled | Selected |
| ---: | --- | ---: | ---: | --- |
{scored_table}

## Notes

{notes}
"""


def plan_joint_allocation(
    *,
    model_dir: str | Path,
    max_memory_bytes: int,
    context_length: int,
    profile: ProfileName,
    output_dir: str | Path,
    contexts: tuple[int, ...] | None = None,
    weight_bpws: tuple[float, ...] = (4.0, 4.8, 6.0),
    kv_bits: tuple[int, ...] = (4, 8, 16),
    inventory_path: str | Path | None = None,
    sensitivity_path: str | Path | None = None,
    kv_analysis_path: str | Path | None = None,
    allow_unmeasured: bool = False,
    reserve_bytes: int = 1_000_000_000,
    batch_size: int = 1,
    interaction_threshold: float = 0.02,
    independent_bpw: float = 4.8,
    independent_kv_bits: int = 4,
    quality_baseline_path: str | Path | None = None,
    quality_weight_only_path: str | Path | None = None,
    quality_kv_only_path: str | Path | None = None,
    quality_joint_path: str | Path | None = None,
) -> JointSelectionReport:
    """Emit a convert-ready plan: 1.8 independent if I is small, else coupled search."""

    if context_length <= 0:
        raise PlanningError("deployment context must be a positive integer")
    if kv_analysis_path is None:
        raise PlanningError("plan-joint requires --kv-analysis so cells can be ranked")
    extra_contexts = contexts or (4096, 32768)
    diagnostic_contexts = tuple(dict.fromkeys((context_length, *extra_contexts)))
    output = Path(output_dir).expanduser().resolve()
    diagnostic = diagnose_joint_interaction(
        model_dir=model_dir,
        max_memory_bytes=max_memory_bytes,
        contexts=diagnostic_contexts,
        weight_bpws=weight_bpws,
        kv_bits=kv_bits,
        profile=profile,
        output_dir=output,
        inventory_path=inventory_path,
        sensitivity_path=sensitivity_path,
        kv_analysis_path=kv_analysis_path,
        allow_unmeasured=allow_unmeasured,
        reserve_bytes=reserve_bytes,
        batch_size=batch_size,
        interaction_threshold=interaction_threshold,
        quality_baseline_path=quality_baseline_path,
        quality_weight_only_path=quality_weight_only_path,
        quality_kv_only_path=quality_kv_only_path,
        quality_joint_path=quality_joint_path,
    )
    if diagnostic.interaction is None:
        raise PlanningError(
            "plan-joint requires the BF16 baseline and three treatment quality "
            "evaluations so I(W, KV) can gate independent vs coupled search"
        )

    notes = [
        "plan-joint is a 1.9.0b1 development search. It is not a certificate.",
        "I is a gate: small keeps the 1.8 independent plan; material ranks "
        "feasible WeightPlan x KVPlan cells with a coupled proxy.",
    ]
    independent = None
    independent_plan = None
    try:
        independent = optimize_deployment(
            model_dir=model_dir,
            max_memory_bytes=max_memory_bytes,
            context_length=context_length,
            profile=profile,
            runtime=RuntimeName.AX_ENGINE,
            minimum_quality_retention=0.98,
            mode="balanced",
            output_dir=output / "independent",
            inventory_path=inventory_path,
            sensitivity_path=sensitivity_path,
            kv_analysis_path=kv_analysis_path,
            allow_unmeasured=allow_unmeasured,
            target_bpw=independent_bpw,
            kv_cache="measured",
            kv_default_bits=independent_kv_bits,
            reserve_bytes=reserve_bytes,
            batch_size=batch_size,
        )
        independent_plan = load_model(
            output / "independent" / "axquant_plan.json", QuantizationPlan
        )
    except PlanningError as exc:
        if not diagnostic.interaction.material:
            raise
        notes.append(f"1.8 independent optimize is infeasible: {exc}")
    if not diagnostic.interaction.material:
        if independent is None or independent_plan is None:
            raise PlanningError("independent optimize did not emit a plan")
        notes.append("Interaction is small; emitting the 1.8 independent optimize plan.")
        selection = JointSelectionReport(
            selection_basis="independent",
            context_length=context_length,
            independent_target_bpw=independent_bpw,
            selected_target_bpw=independent_plan.target_bpw,
            selected_kv_bits=(
                independent_plan.kv_cache.default_bits
                if independent_plan.kv_cache is not None
                else independent_kv_bits
            ),
            differs_from_independent=False,
            interaction=diagnostic.interaction.interaction,
            interaction_material=False,
            plan_sha256=independent.plan_sha256,
            kv_plan_sha256=independent.kv_plan_sha256,
            diagnostic_sha256=stable_sha256(diagnostic),
            notes=notes,
        )
        write_data(output / "axquant_plan.json", independent_plan)
        write_data(output / "joint-selection.json", selection)
        write_text(output / "joint-selection.md", joint_selection_markdown(selection))
        return selection

    winner, scored = select_coupled_candidate(
        diagnostic.candidates,
        context_length=context_length,
        interaction=diagnostic.interaction.interaction,
    )
    model = Path(model_dir).expanduser().resolve()
    inventory = inspect_model(
        model,
        allow_quantized=(model / "axquant_manifest.json").is_file(),
    )
    sensitivity = _load_sensitivity(
        inventory,
        sensitivity_path,
        profile=profile,
        allow_unmeasured=allow_unmeasured,
    )
    request = PlanRequest(
        profile=profile,
        target_bpw=winner.target_bpw,
        allow_unmeasured=allow_unmeasured,
        target_mode="balanced",
        primary_runtime=RuntimeName.AX_ENGINE,
        minimum_quality_retention=0.98,
    )
    plan = plan_quantization(sensitivity, request)
    layer_count = text_layer_count(inventory, plan)
    plan.kv_cache = allocate_kv_cache(
        layer_count,
        default_bits=winner.kv_default_bits,
        min_bits=4 if winner.kv_default_bits >= 4 else winner.kv_default_bits,
        group_size=plan.group_size,
    )
    notes.append(
        "Interaction is material; emitting the coupled WeightPlan x KVPlan cell "
        f"{winner.target_bpw:.3f} bpw + KV{winner.kv_default_bits}."
    )
    same_configuration = independent_plan is not None and _same_effective_configuration(
        plan, independent_plan
    )
    if same_configuration:
        if independent_plan is None:
            raise PlanningError("independent plan disappeared after a same-configuration match")
        plan = independent_plan
        notes.append(
            "Coupled search selected the same effective (weight BPW, per-layer KV) "
            "configuration as independent optimize; reusing that measured plan."
        )
    else:
        notes.append("Coupled search selected a configuration 1.8 independent optimize would not.")
    differs = not same_configuration
    selected_kv_sha = stable_sha256(plan.kv_cache) if plan.kv_cache is not None else None
    winner_score = next(cell for cell in scored if cell.selected)
    selection = JointSelectionReport(
        selection_basis="coupled-interaction",
        context_length=context_length,
        independent_target_bpw=independent_bpw,
        selected_target_bpw=winner.target_bpw,
        selected_kv_bits=winner.kv_default_bits,
        differs_from_independent=differs,
        interaction=diagnostic.interaction.interaction,
        interaction_material=True,
        coupled_loss=winner_score.coupled_loss,
        additive_loss=winner_score.additive_loss,
        scored_cells=scored,
        plan_sha256=stable_sha256(plan),
        kv_plan_sha256=selected_kv_sha,
        diagnostic_sha256=stable_sha256(diagnostic),
        notes=notes,
    )
    write_data(output / "axquant_plan.json", plan)
    write_data(output / "joint-selection.json", selection)
    write_text(output / "joint-selection.md", joint_selection_markdown(selection))
    return selection
