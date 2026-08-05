from __future__ import annotations

from dataclasses import dataclass

from axquant.architectures.registry import declared_tier_for
from axquant.errors import PlanningError
from axquant.experimental_bits import annotate_experimental_low_bit_plan
from axquant.module_paths import fused_expert_module, packed_expert_runtime_modules
from axquant.naming import target_class_for_bpw
from axquant.profiles import objective_for
from axquant.role_policy import prefer_method_on_tie, ranking_loss
from axquant.schema import (
    AX_ENGINE_EXECUTABLE_BITS,
    AX_ENGINE_EXECUTABLE_GROUP_SIZES,
    Allocation,
    ArchitectureProfile,
    ArchitectureSupportLevel,
    CandidateMeasurement,
    EvidenceKind,
    KvCachePlan,
    KvLayerAllocation,
    KvSensitivityReport,
    MetricVector,
    OutlierStrategy,
    PlanningConstraints,
    PlanRequest,
    PrecisionShare,
    QuantizationPlan,
    QuantMethod,
    ScaleStrategy,
    SensitivityReport,
    TensorRole,
    TensorSensitivity,
)
from axquant.schema import (
    PROTECTED_MIN_BITS as _PROTECTED_MIN_BITS,
)
from axquant.serde import stable_sha256
from axquant.versioning import collect_versions


@dataclass(frozen=True)
class _Option:
    measurement: CandidateMeasurement
    loss: float
    storage_bpw: float
    ranking_loss: float


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


_PrecisionSignature = tuple[int, QuantMethod, int | None]


def _harmonize_choice_group(
    members: list[_Choice],
    *,
    label: str,
    running_storage_bits: float,
    target_storage_bits: float,
    reason: str | None = None,
) -> float:
    """Select one common executable signature for tensors sharing physical storage."""

    option_maps = [
        {
            (
                option.measurement.bits,
                option.measurement.method,
                option.measurement.group_size,
            ): (index, option)
            for index, option in enumerate(member.options)
        }
        for member in members
    ]
    common_signatures = set(option_maps[0])
    for option_map in option_maps[1:]:
        common_signatures &= set(option_map)
    feasible: list[tuple[float, float, _PrecisionSignature]] = []
    current_group_storage = sum(
        member.selected.storage_bpw * member.entry.tensor.parameters for member in members
    )
    for signature in common_signatures:
        options = [option_map[signature][1] for option_map in option_maps]
        candidate_group_storage = sum(
            option.storage_bpw * member.entry.tensor.parameters
            for member, option in zip(members, options, strict=True)
        )
        if (
            running_storage_bits - current_group_storage + candidate_group_storage
            > target_storage_bits + 1e-6
        ):
            continue
        aggregate_loss = sum(
            option.ranking_loss * member.entry.tensor.parameters
            for member, option in zip(members, options, strict=True)
        )
        feasible.append((aggregate_loss, -candidate_group_storage, signature))
    if not feasible:
        raise PlanningError(f"{label} has no common precision within the budget")
    _, _, selected_signature = min(
        feasible,
        key=lambda candidate: (
            candidate[0],
            candidate[1],
            candidate[2][0],
            candidate[2][1].value,
            candidate[2][2] or 0,
        ),
    )
    selected_group_storage = 0.0
    for member, option_map in zip(members, option_maps, strict=True):
        selected_index, selected_option = option_map[selected_signature]
        member.index = selected_index
        member.upgraded = selected_index > 0
        if reason is not None:
            member.policy_reason = reason
        selected_group_storage += selected_option.storage_bpw * member.entry.tensor.parameters
    return running_storage_bits + selected_group_storage - current_group_storage


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


def _require_executable_kv_bits(*values: int) -> None:
    unsupported = sorted(set(values) - AX_ENGINE_EXECUTABLE_BITS)
    if unsupported:
        raise PlanningError(f"AX Engine KV cache does not support bit widths {unsupported}")


def _require_executable_kv_group_sizes(*values: int) -> None:
    unsupported = sorted(set(values) - AX_ENGINE_EXECUTABLE_GROUP_SIZES)
    if unsupported:
        raise PlanningError(f"AX Engine KV cache does not support group sizes {unsupported}")


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
    _require_executable_kv_bits(default_bits, min_bits)
    _require_executable_kv_group_sizes(group_size)
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
    _require_executable_kv_bits(min_bits)
    _require_executable_kv_group_sizes(report.group_size)
    for entry in report.entries:
        _require_executable_kv_bits(*(candidate.bits for candidate in entry.candidates))
        _require_executable_kv_group_sizes(
            *(
                candidate.group_size
                for candidate in entry.candidates
                if candidate.group_size is not None
            )
        )
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


def strategy_for_measurement(
    measurement: CandidateMeasurement,
) -> tuple[ScaleStrategy, OutlierStrategy]:
    """Map packing method to recorded scale/outlier strategy (AXQ-028)."""
    if measurement.bits == 16 or measurement.method == QuantMethod.BF16:
        return ScaleStrategy.NONE, OutlierStrategy.NONE
    if measurement.method == QuantMethod.AWQ:
        return ScaleStrategy.CHANNEL_AWQ, OutlierStrategy.NONE
    if measurement.method == QuantMethod.GPTQ:
        return ScaleStrategy.GPTQ_HESSIAN, OutlierStrategy.NONE
    if measurement.method == QuantMethod.DWQ:
        return ScaleStrategy.GROUP_AFFINE, OutlierStrategy.PERCENTILE_CLIP_DWQ
    return ScaleStrategy.GROUP_AFFINE, OutlierStrategy.NONE


def _pareto_options(options: list[_Option]) -> list[_Option]:
    """Keep storage-ordered non-dominated options (lower storage, lower loss)."""
    ordered = sorted(
        options,
        key=lambda option: (
            option.storage_bpw,
            option.loss,
            option.measurement.bits,
            option.measurement.group_size or 0,
            option.measurement.method.value,
        ),
    )
    frontier: list[_Option] = []
    best_loss = float("inf")
    for option in ordered:
        # Ascending storage order (one option per storage key): a later option
        # can only join the frontier by strictly improving on the best loss.
        if option.loss < best_loss - 1e-15:
            frontier.append(option)
            best_loss = option.loss
    return frontier


def _options_for(
    entry: TensorSensitivity,
    request: PlanRequest,
    weights: dict[str, float],
    *,
    evidence_kind: EvidenceKind,
) -> tuple[list[_Option], str | None]:
    """Build a bits x group x method option ladder under hard floors (AXQ-028/QP1)."""
    minimum_bits, reason = _minimum_bits(entry, request)
    allowed_bits = set(request.candidate_bits)
    allowed_groups = set(request.effective_group_sizes())
    method_filter = set(request.candidate_methods)
    role = entry.tensor.role
    # MLX-LM exposes both per-expert checkpoint tensors and packed expert
    # tensors through fused SwitchLinear modules.  Predicate/conversion can
    # only execute affine packing for those modules, so do not let refinement
    # candidates produce a plan that the conversion preflight must reject.
    fused_expert = fused_expert_module(entry.tensor.module_path) is not None
    packed_expert = bool(packed_expert_runtime_modules(entry.tensor.module_path))
    if role.is_mtp and request.mtp.mode != "disabled":
        allowed_bits &= set(request.mtp.candidate_bits)
    if minimum_bits == 16:
        # Policy-preserved tensors (norms, LM head, vision, MTP floors) inject
        # BF16 even when the requested quantized search grid omits 16-bit,
        # matching manual.py and the schema loader contract.
        allowed_bits.add(16)
    candidates = [
        candidate
        for candidate in entry.candidates
        if candidate.supported
        and candidate.bits in allowed_bits
        and candidate.bits >= minimum_bits
        and candidate.bits in request.hardware.supported_bits
        and candidate.method in request.hardware.supported_methods
        and (
            not (fused_expert or packed_expert)
            or candidate.bits == 16
            or candidate.method == QuantMethod.AFFINE
        )
        and (
            candidate.bits == 16
            or (
                candidate.group_size in request.hardware.supported_group_sizes
                and candidate.group_size in allowed_groups
            )
        )
        and (
            not method_filter
            or candidate.method in method_filter
            or candidate.method == QuantMethod.BF16
        )
    ]
    if not candidates:
        raise PlanningError(
            f"{entry.tensor.name} has no candidate satisfying the precision and hardware policy"
        )
    # Collapse same storage key (bits, group_size) with role-aware method preference.
    options_by_key: dict[tuple[int, int | None], list[_Option]] = {}
    for candidate in candidates:
        raw_loss = _loss(candidate.metrics, weights)
        option = _Option(
            measurement=candidate,
            loss=raw_loss,
            storage_bpw=storage_bpw(candidate.bits, candidate.group_size),
            ranking_loss=ranking_loss(
                loss=raw_loss,
                role=role,
                method=candidate.method,
                group_size=candidate.group_size,
                evidence_kind=evidence_kind,
            ),
        )
        key = (candidate.bits, candidate.group_size)
        options_by_key.setdefault(key, []).append(option)

    best_by_storage_key: dict[tuple[int, int | None], _Option] = {}
    for key, options in options_by_key.items():
        best_loss = min(option.loss for option in options)
        selected = options[0]
        for option in options[1:]:
            if prefer_method_on_tie(
                role,
                current_method=selected.measurement.method,
                current_loss=selected.loss,
                candidate_method=option.measurement.method,
                candidate_loss=option.loss,
                evidence_kind=evidence_kind,
                best_loss_at_key=best_loss,
            ):
                selected = option
        best_by_storage_key[key] = selected
    return _pareto_options(list(best_by_storage_key.values())), reason


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
        raise PlanningError("AXQuant release planning requires a supported architecture adapter")
    weights_model = objective_for(request.profile)
    weights = weights_model.normalized()
    choices: list[_Choice] = []
    for entry in report.entries:
        options, reason = _options_for(entry, request, weights, evidence_kind=report.evidence_kind)
        choices.append(_Choice(entry=entry, options=options, policy_reason=reason))
    tensor_names = [choice.entry.tensor.name for choice in choices]
    if len(tensor_names) != len(set(tensor_names)):
        raise PlanningError("analysis report contains duplicate tensor entries")
    total_parameters = sum(choice.entry.tensor.parameters for choice in choices)
    if total_parameters <= 0:
        raise PlanningError("analysis report contains no parameters")

    target_storage_bits = request.target_bpw * total_parameters

    def current_storage_bits() -> float:
        return sum(
            choice.selected.storage_bpw * choice.entry.tensor.parameters for choice in choices
        )

    # Keep a running total so the upgrade search stays O(options) per pass, not
    # O(tensors) inside the inner loop (critical for hybrid MoE with 6k+ tensors).
    running_storage_bits = current_storage_bits()
    minimum_storage_bits = running_storage_bits
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
            if running_storage_bits + delta_storage > target_storage_bits + 1e-6:
                continue
            # Use ranking_loss so measured role preferences influence upgrade order (QP1).
            benefit = current.ranking_loss - upgraded.ranking_loss
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
        running_storage_bits += best[2]

    # Fused MoE experts quantize as one MLX-LM switch module, so every member
    # of a group must share one executable (bits, method, group-size)
    # signature. Option ladders are tensor-specific; equal list indexes do not
    # imply equal signatures.
    expert_groups: dict[str, list[_Choice]] = {}
    for choice in choices:
        fused = fused_expert_module(choice.entry.tensor.module_path)
        if fused is not None:
            expert_groups.setdefault(fused, []).append(choice)
    for fused_module, members in expert_groups.items():
        running_storage_bits = _harmonize_choice_group(
            members,
            label=f"fused expert module {fused_module}",
            running_storage_bits=running_storage_bits,
            target_storage_bits=target_storage_bits,
        )

    # Tied tensors share one physical weight. A plan that assigns different
    # packing signatures cannot be executed consistently and can also cause
    # the fail-closed conversion predicate to see only one side of the tie.
    choices_by_tensor = {choice.entry.tensor.name: choice for choice in choices}
    tied_graph: dict[str, set[str]] = {}
    for choice in choices:
        name = choice.entry.tensor.name
        tied_to = choice.entry.tensor.tied_to
        if tied_to is None:
            continue
        if tied_to == name:
            raise PlanningError(f"tensor {name} cannot be tied to itself")
        if tied_to not in choices_by_tensor:
            raise PlanningError(f"tensor {name} is tied to missing tensor {tied_to}")
        tied_graph.setdefault(name, set()).add(tied_to)
        tied_graph.setdefault(tied_to, set()).add(name)
    visited_tied: set[str] = set()
    for start in sorted(tied_graph):
        if start in visited_tied:
            continue
        stack = [start]
        component: list[str] = []
        while stack:
            name = stack.pop()
            if name in visited_tied:
                continue
            visited_tied.add(name)
            component.append(name)
            stack.extend(sorted(tied_graph.get(name, ()), reverse=True))
        if len(component) < 2:
            continue
        group_name = ", ".join(sorted(component))
        running_storage_bits = _harmonize_choice_group(
            [choices_by_tensor[name] for name in sorted(component)],
            label=f"tied-weight group [{group_name}]",
            running_storage_bits=running_storage_bits,
            target_storage_bits=target_storage_bits,
            reason="tied-weight group harmonized at one executable precision",
        )

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
        scale_strategy, outlier_strategy = strategy_for_measurement(measurement)
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
                scale_strategy=scale_strategy,
                outlier_strategy=outlier_strategy,
                strategy_metadata={
                    "storage_bpw": selected.storage_bpw,
                    "selected_from_candidates": len(choice.options),
                },
            )
        )

    nominal_bpw = (
        sum(choice.selected.measurement.bits * choice.entry.tensor.parameters for choice in choices)
        / total_parameters
    )
    effective_bpw = current_storage_bits() / total_parameters
    quantized_bits = [bits for bits in request.candidate_bits if bits < 16]
    target_class = target_class_for_bpw(request.target_bpw) if quantized_bits else "bf16"
    warnings = list(report.warnings)
    if not report.evidence_kind.release_quality:
        warnings.append(
            f"Plan uses non-release {report.evidence_kind.value} evidence and requires "
            "complete-model validation."
        )
    plan = QuantizationPlan(
        source_model=report.model,
        architecture_profile=_current_policy_profile(report.architecture_profile),
        profile=request.profile,
        target_class=target_class,
        target_bpw=request.target_bpw,
        nominal_bpw=nominal_bpw,
        effective_bpw=effective_bpw,
        candidate_bits=request.candidate_bits,
        group_size=request.group_size,
        candidate_group_sizes=request.effective_group_sizes(),
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
    return annotate_experimental_low_bit_plan(plan)
