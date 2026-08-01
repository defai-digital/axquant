from __future__ import annotations

from dataclasses import dataclass

from axquant.architectures.registry import declared_tier_for
from axquant.errors import PlanningError
from axquant.module_paths import fused_expert_module
from axquant.profiles import objective_for
from axquant.schema import (
    Allocation,
    ArchitectureProfile,
    ArchitectureSupportLevel,
    CandidateMeasurement,
    EvidenceKind,
    KvCachePlan,
    KvLayerAllocation,
    KvSensitivityReport,
    MetricVector,
    PlanningConstraints,
    PlanRequest,
    PrecisionShare,
    QuantizationPlan,
    SensitivityReport,
    TensorRole,
    TensorSensitivity,
)
from axquant.serde import stable_sha256
from axquant.versioning import collect_versions

_PROTECTED_MIN_BITS = {
    TensorRole.EMBEDDING: 8,
    TensorRole.NORM: 16,
    TensorRole.LM_HEAD: 16,
    TensorRole.ROUTER: 8,
    TensorRole.VISION: 16,
}


@dataclass(frozen=True)
class _Option:
    measurement: CandidateMeasurement
    loss: float
    storage_bpw: float


@dataclass
class _Choice:
    entry: TensorSensitivity
    options: list[_Option]
    index: int = 0
    upgraded: bool = False
    policy_reason: str | None = None

    @property
    def selected(self) -> _Option:
        return self.options[self.index]


def _current_policy_profile(profile: ArchitectureProfile) -> ArchitectureProfile:
    """Stamp the registry's current tier onto a recorded profile (AXQ-017).

    Sensitivity reports are immutable evidence, but the tier is current policy:
    a plan built today from an older report carries today's declared tier for
    the same adapter, not the tier (or its absence) recorded before promotion.
    Unsupported profiles and unknown adapters stay fail-closed.
    """
    if profile.support_level != ArchitectureSupportLevel.SUPPORTED:
        return profile
    current = declared_tier_for(profile.adapter_id)
    if current is None or current == profile.support_tier:
        return profile
    return profile.model_copy(update={"support_tier": current})


def storage_bpw(bits: int, group_size: int | None) -> float:
    if bits == 16:
        return 16.0
    if group_size is None:
        raise PlanningError("quantized precision is missing a group size")
    return bits + 32.0 / group_size


def allocate_kv_cache(
    layer_count: int,
    *,
    default_bits: int = 4,
    min_bits: int = 4,
    group_size: int = 64,
) -> KvCachePlan:
    """Prior-based per-layer KV-cache precision allocation (AXQ-021 phase one).

    Boundary layers receive at least 8-bit KV because they carry
    disproportionate attention mass under the same architecture priors the
    weight path uses. This allocator exists to establish the plan and metadata
    contract; it makes no measured-quality claim.
    """
    if layer_count < 1:
        raise PlanningError("KV-cache planning requires a positive text layer count")
    if default_bits < min_bits:
        raise PlanningError("KV-cache default bits cannot fall below the policy floor")
    boundary = max(1, -(-layer_count // 10))
    layers: list[KvLayerAllocation] = []
    for index in range(layer_count):
        is_boundary = index < boundary or index >= layer_count - boundary
        bits = max(8, default_bits) if is_boundary else default_bits
        reason = (
            "architecture prior: boundary layers keep at least 8-bit KV"
            if is_boundary
            else "architecture prior: interior layer at the default KV precision"
        )
        layers.append(
            KvLayerAllocation(
                layer_index=index,
                bits=bits,
                group_size=group_size,
                reason=reason,
            )
        )
    return KvCachePlan(
        allocation_basis="architecture-prior",
        min_bits=min_bits,
        default_bits=default_bits,
        default_group_size=group_size,
        layers=layers,
    )


def allocate_kv_cache_measured(
    report: KvSensitivityReport,
    *,
    max_output_kl: float = 0.005,
    min_bits: int = 4,
) -> KvCachePlan:
    """Allocate per-layer KV precision from a measured sensitivity report (AXQ-024).

    Each layer receives the lowest candidate bit-width whose measured output KL
    stays within ``max_output_kl``; layers with no passing quantized candidate
    keep BF16 KV. The plan is bound to the producing report by semantic digest.
    """
    if report.evidence_kind == EvidenceKind.ARCHITECTURE_PRIOR:
        raise PlanningError("measured KV allocation requires a measured sensitivity report")
    if max_output_kl <= 0.0:
        raise PlanningError("the KV output-KL budget must be positive")
    layers: list[KvLayerAllocation] = []
    chosen_bits: list[int] = []
    for entry in sorted(report.entries, key=lambda item: item.layer_index):
        quantized = sorted(
            (
                candidate
                for candidate in entry.candidates
                if candidate.bits < 16 and candidate.bits >= min_bits and candidate.supported
            ),
            key=lambda candidate: candidate.bits,
        )
        selected: KvLayerAllocation | None = None
        for candidate in quantized:
            if candidate.metrics.output_kl <= max_output_kl:
                selected = KvLayerAllocation(
                    layer_index=entry.layer_index,
                    bits=candidate.bits,
                    group_size=candidate.group_size or report.group_size,
                    reason=(
                        f"measured output KL {candidate.metrics.output_kl:.6f} at "
                        f"{candidate.bits}-bit within budget {max_output_kl}"
                    ),
                )
                break
        if selected is None:
            any_supported = any(
                candidate.bits < 16 and candidate.supported for candidate in entry.candidates
            )
            selected = KvLayerAllocation(
                layer_index=entry.layer_index,
                bits=16,
                group_size=report.group_size,
                reason=(
                    f"no quantized KV candidate met the output-KL budget {max_output_kl}"
                    if any_supported
                    else "layer KV is not quantizable (non-KV recurrent cache)"
                ),
            )
        layers.append(selected)
        chosen_bits.append(selected.bits)
    default_bits = min(chosen_bits)
    return KvCachePlan(
        allocation_basis="measured",
        min_bits=min_bits,
        default_bits=default_bits,
        default_group_size=report.group_size,
        sensitivity_sha256=stable_sha256(report),
        max_output_kl=max_output_kl,
        layers=layers,
    )


def _loss(metrics: MetricVector, weights: dict[str, float]) -> float:
    values = metrics.model_dump()
    return sum(float(values[key]) * weight for key, weight in weights.items())


def _external_mtp_sidecar(entry: TensorSensitivity) -> bool:
    return entry.tensor.role.is_mtp


def _minimum_bits(entry: TensorSensitivity, request: PlanRequest) -> tuple[int, str | None]:
    role = entry.tensor.role
    if not entry.tensor.quantizable:
        return 16, "non-quantizable tensor preserved"
    if _external_mtp_sidecar(entry) and request.mtp.preserve_external_sidecar:
        return 16, "external MTP sidecar preserved byte-for-byte"
    if role.is_mtp and request.mtp.mode == "protected":
        return request.mtp.min_bits, "protected MTP policy"
    minimum = _PROTECTED_MIN_BITS.get(role, min(request.candidate_bits))
    reason = f"protected {role.value} policy" if role in _PROTECTED_MIN_BITS else None
    if role == TensorRole.LM_HEAD and request.lm_head_min_bits < minimum:
        minimum = request.lm_head_min_bits
        reason = (
            "protected lm_head floor lowered to 8-bit under AXQ-026; "
            "release certification requires measured quality evidence"
        )
    return minimum, reason


def _options_for(
    entry: TensorSensitivity,
    request: PlanRequest,
    weights: dict[str, float],
) -> tuple[list[_Option], str | None]:
    minimum_bits, reason = _minimum_bits(entry, request)
    allowed_bits = set(request.candidate_bits)
    if entry.tensor.role.is_mtp and request.mtp.mode != "disabled":
        allowed_bits &= set(request.mtp.candidate_bits)
        if minimum_bits == 16:
            allowed_bits.add(16)
    candidates = [
        candidate
        for candidate in entry.candidates
        if candidate.supported
        and candidate.bits in allowed_bits
        and candidate.bits >= minimum_bits
        and candidate.bits in request.hardware.supported_bits
        and candidate.method in request.hardware.supported_methods
        and (candidate.bits == 16 or candidate.group_size in request.hardware.supported_group_sizes)
    ]
    if not candidates:
        raise PlanningError(
            f"{entry.tensor.name} has no candidate satisfying the precision and hardware policy"
        )
    best_by_bits: dict[int, _Option] = {}
    for candidate in candidates:
        option = _Option(
            measurement=candidate,
            loss=_loss(candidate.metrics, weights),
            storage_bpw=storage_bpw(candidate.bits, candidate.group_size),
        )
        current = best_by_bits.get(candidate.bits)
        if current is None or option.loss < current.loss:
            best_by_bits[candidate.bits] = option
    return [best_by_bits[bits] for bits in sorted(best_by_bits)], reason


def _distribution(
    choices: list[_Choice],
    *,
    mtp_only: bool = False,
) -> dict[str, PrecisionShare]:
    selected = [choice for choice in choices if not mtp_only or choice.entry.tensor.role.is_mtp]
    total = sum(choice.entry.tensor.parameters for choice in selected)
    if total == 0:
        return {}
    parameters_by_precision: dict[str, int] = {}
    for choice in selected:
        bits = choice.selected.measurement.bits
        label = "bf16" if bits == 16 else f"{bits}bit"
        parameters_by_precision[label] = (
            parameters_by_precision.get(label, 0) + choice.entry.tensor.parameters
        )
    return {
        label: PrecisionShare(parameters=parameters, fraction=parameters / total)
        for label, parameters in sorted(parameters_by_precision.items())
    }


def plan_quantization(
    report: SensitivityReport,
    request: PlanRequest,
) -> QuantizationPlan:
    if request.candidate_count > 1:
        # Top-N generation is handled by the refinement module;
        # the planner itself always produces a single deterministic plan.
        # Accept candidate_count > 1 but only produce one plan here.
        pass
    if report.profile != request.profile:
        raise PlanningError(
            f"analysis profile {report.profile} does not match plan profile {request.profile}"
        )
    if request.hardware.runtime != request.primary_runtime:
        raise PlanningError("hardware profile runtime does not match the primary runtime")
    if not report.evidence_kind.release_quality and not request.allow_unmeasured:
        raise PlanningError(
            f"{report.evidence_kind.value} evidence is not release quality; "
            "pass --allow-unmeasured "
            "only for development dry runs"
        )
    if (
        report.architecture_profile.support_level != ArchitectureSupportLevel.SUPPORTED
        and not request.allow_unmeasured
    ):
        raise PlanningError(
            "AXQuant release planning supports validated Qwen 3.6 dense checkpoints only"
        )
    weights_model = objective_for(request.profile)
    weights = weights_model.normalized()
    choices: list[_Choice] = []
    for entry in report.entries:
        options, reason = _options_for(entry, request, weights)
        choices.append(_Choice(entry=entry, options=options, policy_reason=reason))
    total_parameters = sum(choice.entry.tensor.parameters for choice in choices)
    if total_parameters <= 0:
        raise PlanningError("analysis report contains no parameters")

    target_storage_bits = request.target_bpw * total_parameters

    def current_storage_bits() -> float:
        return sum(
            choice.selected.storage_bpw * choice.entry.tensor.parameters for choice in choices
        )

    minimum_storage_bits = current_storage_bits()
    if minimum_storage_bits > target_storage_bits + 1e-6:
        minimum_bpw = minimum_storage_bits / total_parameters
        raise PlanningError(
            f"target {request.target_bpw:.4f} BPW is infeasible; policy minimum is "
            f"{minimum_bpw:.4f} BPW"
        )

    while True:
        best: tuple[float, int, float] | None = None
        for choice_index, choice in enumerate(choices):
            if choice.index + 1 >= len(choice.options):
                continue
            current = choice.options[choice.index]
            upgraded = choice.options[choice.index + 1]
            delta_storage = (
                upgraded.storage_bpw - current.storage_bpw
            ) * choice.entry.tensor.parameters
            if delta_storage <= 0:
                continue
            if current_storage_bits() + delta_storage > target_storage_bits + 1e-6:
                continue
            benefit = current.loss - upgraded.loss
            if benefit <= 0:
                continue
            efficiency = benefit * choice.entry.tensor.parameters / delta_storage
            candidate = (efficiency, choice_index, delta_storage)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break
        choice = choices[best[1]]
        choice.index += 1
        choice.upgraded = True

    # Fused MoE experts quantize as one MLX-LM switch module, so every member
    # of a group must share one precision. Normalize each group to its
    # minimum selected storage option: strictly budget-safe (storage can only
    # shrink) and deterministic.
    expert_groups: dict[str, list[_Choice]] = {}
    for choice in choices:
        fused = fused_expert_module(choice.entry.tensor.module_path)
        if fused is not None:
            expert_groups.setdefault(fused, []).append(choice)
    for members in expert_groups.values():
        floor_index = min(member.index for member in members)
        for member in members:
            if member.index != floor_index:
                member.index = floor_index
                member.upgraded = floor_index > 0

    allocations: list[Allocation] = []
    for choice in choices:
        selected = choice.selected
        measurement = selected.measurement
        if choice.policy_reason:
            reason = choice.policy_reason
        elif choice.upgraded:
            reason = "selected by marginal quality gain per storage bit"
        else:
            reason = "minimum storage candidate"
        allocations.append(
            Allocation(
                tensor=choice.entry.tensor.name,
                module_path=choice.entry.tensor.module_path,
                role=choice.entry.tensor.role,
                parameters=choice.entry.tensor.parameters,
                bits=measurement.bits,
                method=measurement.method,
                group_size=measurement.group_size,
                predicted_loss=selected.loss,
                metrics=measurement.metrics,
                reason=reason,
            )
        )

    nominal_bpw = (
        sum(choice.selected.measurement.bits * choice.entry.tensor.parameters for choice in choices)
        / total_parameters
    )
    effective_bpw = current_storage_bits() / total_parameters
    quantized_bits = [bits for bits in request.candidate_bits if bits < 16]
    target_class = f"{min(quantized_bits)}bit" if quantized_bits else "bf16"
    warnings = list(report.warnings)
    if not report.evidence_kind.release_quality:
        warnings.append(
            f"Plan uses non-release {report.evidence_kind.value} evidence and requires "
            "complete-model validation."
        )
    return QuantizationPlan(
        source_model=report.model,
        architecture_profile=_current_policy_profile(report.architecture_profile),
        profile=request.profile,
        target_class=target_class,
        target_bpw=request.target_bpw,
        nominal_bpw=nominal_bpw,
        effective_bpw=effective_bpw,
        candidate_bits=request.candidate_bits,
        group_size=request.group_size,
        objective=weights_model,
        hardware=request.hardware,
        mtp=request.mtp,
        constraints=PlanningConstraints(
            effective_bpw_limit=request.target_bpw,
            max_model_size_ratio_to_uniform4=request.max_model_size_ratio_to_uniform4,
            minimum_quality_retention=request.minimum_quality_retention,
            minimum_mtp_acceptance_retention=request.minimum_mtp_acceptance_retention,
            minimum_mtp_speedup=request.minimum_mtp_speedup,
            lm_head_min_bits=request.lm_head_min_bits,
        ),
        target_mode=request.target_mode,
        primary_runtime=request.primary_runtime,
        random_seed=request.random_seed,
        software_versions=collect_versions(),
        analysis_sha256=stable_sha256(report),
        evidence_kind=report.evidence_kind,
        calibration=report.calibration,
        assignments=allocations,
        weight_distribution=_distribution(choices),
        mtp_distribution=_distribution(choices, mtp_only=True),
        warnings=warnings,
    )
