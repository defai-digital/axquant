from __future__ import annotations

from fnmatch import fnmatchcase

from axquant.errors import PlanningError
from axquant.experimental_bits import annotate_experimental_low_bit_plan
from axquant.module_paths import fused_expert_module, packed_expert_runtime_modules
from axquant.naming import target_class_for_bpw
from axquant.package_data import message_template
from axquant.planner import storage_bpw, strategy_for_measurement
from axquant.profiles import objective_for
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    PROTECTED_MIN_BITS as _PROTECTED_MIN_BITS,
)
from axquant.schema import (
    Allocation,
    ArchitectureSupportLevel,
    CandidateMeasurement,
    EvidenceKind,
    Inventory,
    ManualPlanRecipe,
    ManualPrecisionRule,
    MetricVector,
    PlanningConstraints,
    PrecisionShare,
    QuantizationPlan,
    QuantMethod,
    TensorRole,
    TensorSpec,
)
from axquant.serde import stable_sha256
from axquant.versioning import collect_versions


def _matches(rule: ManualPrecisionRule, tensor: TensorSpec) -> bool:
    return (
        (rule.bits == 16 or tensor.quantizable)
        and (not rule.roles or tensor.role in rule.roles)
        and (rule.tensor_glob is None or fnmatchcase(tensor.name, rule.tensor_glob))
        and (rule.module_glob is None or fnmatchcase(tensor.module_path, rule.module_glob))
    )


def _external_mtp(tensor: TensorSpec) -> bool:
    return tensor.role.is_mtp


def _minimum_bits(tensor: TensorSpec, recipe: ManualPlanRecipe) -> tuple[int, str | None]:
    if not tensor.quantizable:
        return 16, message_template("floor_reasons", "non_quantizable")
    if _external_mtp(tensor) and recipe.mtp.preserve_external_sidecar:
        return 16, message_template("floor_reasons", "external_mtp_sidecar")
    if tensor.role.is_mtp and recipe.mtp.mode == "protected":
        return recipe.mtp.min_bits, message_template("floor_reasons", "protected_mtp")
    # Unlike planner.py's automatic search, manual recipes have no
    # `candidate_bits` search space to respect -- each rule states its bits
    # explicitly, already schema-validated to >=2 (ManualPrecisionRule.bits),
    # so `2` here is a harmless floor for unprotected roles, not a policy
    # choice that needs to mirror planner.py's `min(candidate_bits)`.
    minimum = _PROTECTED_MIN_BITS.get(tensor.role, 2)
    reason = (
        message_template("floor_reasons", "protected_role").format(role=tensor.role.value)
        if tensor.role in _PROTECTED_MIN_BITS
        else None
    )
    if tensor.role == TensorRole.LM_HEAD and recipe.lm_head_min_bits < minimum:
        minimum = recipe.lm_head_min_bits
        reason = message_template("floor_reasons", "lm_head_lowered")
    return minimum, reason


def _precision(
    tensor: TensorSpec,
    recipe: ManualPlanRecipe,
    rule: ManualPrecisionRule | None,
) -> tuple[int, QuantMethod, int | None, str]:
    minimum, policy_reason = _minimum_bits(tensor, recipe)
    bits = rule.bits if rule is not None else recipe.default_bits
    method = rule.method if rule is not None else recipe.default_method
    group_size = rule.group_size if rule is not None else None
    if rule is not None and bits < minimum:
        raise PlanningError(
            f"manual rule {rule.rule_id!r} assigns {bits}-bit to {tensor.name}, "
            f"below the protected minimum of {minimum}-bit"
        )
    if rule is None and bits < minimum:
        bits = minimum
        method = QuantMethod.BF16 if bits == 16 else QuantMethod.AFFINE
    # BF16 is always admissible for MTP tensors regardless of the policy's
    # quantized candidate grid: protection floors (external sidecar, non-
    # quantizable shapes) legitimately force 16-bit, matching planner.py's
    # re-admittance of 16 for floored tensors.
    if tensor.role.is_mtp and recipe.mtp.mode != "disabled" and bits < 16:
        allowed = set(recipe.mtp.candidate_bits)
        if bits not in allowed:
            if rule is not None:
                raise PlanningError(
                    f"manual rule {rule.rule_id!r} assigns unsupported MTP precision "
                    f"{bits}-bit to {tensor.name}"
                )
            candidates = sorted(candidate for candidate in allowed if candidate >= bits)
            if not candidates:
                raise PlanningError(f"no MTP precision satisfies the policy for {tensor.name}")
            bits = candidates[0]
            method = QuantMethod.BF16 if bits == 16 else QuantMethod.AFFINE
    if bits == 16:
        method = QuantMethod.BF16
        group_size = None
    else:
        group_size = group_size or recipe.group_size
    if bits not in recipe.hardware.supported_bits:
        raise PlanningError(f"hardware profile does not support {bits}-bit for {tensor.name}")
    if method not in recipe.hardware.supported_methods:
        raise PlanningError(f"hardware profile does not support {method.value} for {tensor.name}")
    if bits < 16 and group_size not in recipe.hardware.supported_group_sizes:
        raise PlanningError(
            f"hardware profile does not support group size {group_size} for {tensor.name}"
        )
    # Per-expert checkpoint tensors and packed expert tensors both fuse into
    # MLX-LM switch modules that conversion can only pack affinely (see
    # planner.py and predicate.py), so reject refinement methods at plan time
    # instead of producing a plan the conversion preflight must reject.
    fused_module = fused_expert_module(tensor.module_path)
    packed_modules = packed_expert_runtime_modules(tensor.module_path)
    if bits < 16 and method != QuantMethod.AFFINE and (fused_module or packed_modules):
        raise PlanningError(
            f"manual recipe assigns {method.value} to expert tensor {tensor.name}; "
            "conversion can only execute affine packing for fused/packed expert modules"
        )
    if rule is not None:
        reason = f"manual rule {rule.rule_id}: {rule.reason}"
    elif policy_reason is not None:
        reason = policy_reason
    else:
        reason = "manual recipe default"
    return bits, method, group_size, reason


def _harmonize_tied_weights(
    allocations: list[Allocation],
    tied_weight_groups: list[list[str]],
    recipe: ManualPlanRecipe,
) -> None:
    by_tensor = {allocation.tensor: allocation for allocation in allocations}
    for group in tied_weight_groups:
        tied = [by_tensor[name] for name in group if name in by_tensor]
        if len(tied) < 2:
            continue
        bits = max(allocation.bits for allocation in tied)
        if bits == 16:
            method = QuantMethod.BF16
            group_size = None
        else:
            selected_signatures = {
                (allocation.method, allocation.group_size)
                for allocation in tied
                if allocation.bits == bits
            }
            if len(selected_signatures) != 1:
                names = sorted(allocation.tensor for allocation in tied)
                raise PlanningError(
                    f"tied-weight group {names} has conflicting manual methods or group sizes"
                )
            method, group_size = next(iter(selected_signatures))
        scale_strategy, outlier_strategy = strategy_for_measurement(
            CandidateMeasurement(
                bits=bits,
                method=method,
                group_size=group_size,
                metrics=MetricVector(),
            )
        )
        for allocation in tied:
            allocation.bits = bits
            allocation.method = method
            allocation.group_size = group_size
            allocation.reason = "tied-weight group harmonized at maximum selected precision"
            allocation.scale_strategy = scale_strategy
            allocation.outlier_strategy = outlier_strategy
            allocation.strategy_metadata["storage_bpw"] = storage_bpw(bits, group_size)


def _enforce_fused_expert_signatures(allocations: list[Allocation]) -> None:
    """Reject mixed signatures inside one fused MLX-LM switch module group.

    Planner harmonizes fused expert groups to one executable (bits, method,
    group-size) signature because MLX-LM quantizes a switch module as a single
    unit. Manual recipes state each precision explicitly, so a mixed group is a
    recipe error and fails fast here rather than at the conversion predicate.
    """
    groups: dict[str, list[Allocation]] = {}
    for allocation in allocations:
        fused = fused_expert_module(allocation.module_path)
        if fused is not None:
            groups.setdefault(fused, []).append(allocation)
    for fused, members in sorted(groups.items()):
        signatures = {(member.bits, member.method.value, member.group_size) for member in members}
        if len(signatures) > 1:
            raise PlanningError(
                f"fused expert module {fused} mixes precisions {sorted(signatures)}; "
                "every expert in a switch group must share one assignment"
            )


def _distribution(
    allocations: list[Allocation],
    *,
    mtp_only: bool = False,
) -> dict[str, PrecisionShare]:
    selected = [
        allocation
        for allocation in allocations
        if allocation.parameters > 0 and (not mtp_only or allocation.role.is_mtp)
    ]
    total = sum(allocation.parameters for allocation in selected)
    if total <= 0:
        return {}
    by_precision: dict[str, int] = {}
    for allocation in selected:
        label = "bf16" if allocation.bits == 16 else f"{allocation.bits}bit"
        by_precision[label] = by_precision.get(label, 0) + allocation.parameters
    return {
        label: PrecisionShare(parameters=parameters, fraction=parameters / total)
        for label, parameters in sorted(by_precision.items())
    }


def manual_quantization_plan(
    inventory: Inventory,
    recipe: ManualPlanRecipe,
) -> QuantizationPlan:
    if inventory.quantized_source:
        # Mixed-precision exports (e.g. DeepSeek V4 Flash FP4+FP8) may still be
        # re-packed when inventory was produced with --allow-quantized and the
        # expert/MLP/attention weights are marked quantizable for dequant+affine.
        if not any(tensor.quantizable for tensor in inventory.tensors):
            raise PlanningError(
                "manual planning requires an unquantized source inventory "
                "(or --allow-quantized inventory with quantizable re-pack weights)"
            )
    if not is_immutable_revision(inventory.model.revision):
        raise PlanningError("manual planning requires a revision-pinned source inventory")
    profile = inventory.architecture_profile
    if profile.support_level != ArchitectureSupportLevel.SUPPORTED:
        raise PlanningError("manual planning requires a supported architecture adapter")
    matched_rules: set[str] = set()
    allocations: list[Allocation] = []
    for tensor in inventory.tensors:
        matching = next((rule for rule in recipe.rules if _matches(rule, tensor)), None)
        if matching is not None:
            matched_rules.add(matching.rule_id)
        bits, method, group_size, reason = _precision(tensor, recipe, matching)
        measurement = CandidateMeasurement(
            bits=bits,
            method=method,
            group_size=group_size,
            metrics=MetricVector(),
        )
        scale_strategy, outlier_strategy = strategy_for_measurement(measurement)
        allocations.append(
            Allocation(
                tensor=tensor.name,
                module_path=tensor.module_path,
                role=tensor.role,
                parameters=tensor.parameters,
                bits=bits,
                method=method,
                group_size=group_size,
                predicted_loss=0.0,
                metrics=MetricVector(),
                reason=reason,
                scale_strategy=scale_strategy,
                outlier_strategy=outlier_strategy,
                strategy_metadata={
                    "storage_bpw": storage_bpw(bits, group_size),
                    "selected_from_candidates": 1,
                },
            )
        )
    unmatched = [rule.rule_id for rule in recipe.rules if rule.rule_id not in matched_rules]
    if unmatched and not recipe.allow_unmatched_rules:
        raise PlanningError(f"manual precision rules matched no tensors: {unmatched}")
    _harmonize_tied_weights(allocations, inventory.tied_weight_groups, recipe)
    _enforce_fused_expert_signatures(allocations)
    total_parameters = sum(allocation.parameters for allocation in allocations)
    if total_parameters <= 0:
        raise PlanningError("source inventory contains no logical parameters")
    nominal_bpw = (
        sum(allocation.bits * allocation.parameters for allocation in allocations)
        / total_parameters
    )
    effective_bpw = (
        sum(
            storage_bpw(allocation.bits, allocation.group_size) * allocation.parameters
            for allocation in allocations
        )
        / total_parameters
    )
    if effective_bpw > recipe.target_bpw + 1e-6:
        raise PlanningError(
            f"manual recipe produces {effective_bpw:.4f} BPW, above its "
            f"{recipe.target_bpw:.4f} BPW limit"
        )
    candidate_bits = tuple(sorted({allocation.bits for allocation in allocations}))
    quantized_bits = [bits for bits in candidate_bits if bits < 16]
    target_class = target_class_for_bpw(recipe.target_bpw) if quantized_bits else "bf16"
    objective = objective_for(recipe.profile)
    fingerprint = {
        "inventory": inventory.model_dump(mode="json", exclude={"created_at"}),
        "recipe": recipe.model_dump(mode="json"),
    }
    warnings = [
        "Manual assignments are unmeasured development evidence.",
        "Conversion requires --allow-unmeasured and cannot pass publication gates.",
    ]
    if unmatched:
        warnings.append(f"Unmatched manual rules were allowed: {unmatched}")
    plan = QuantizationPlan(
        source_model=inventory.model,
        architecture_profile=inventory.architecture_profile,
        profile=recipe.profile,
        target_class=target_class,
        target_bpw=recipe.target_bpw,
        nominal_bpw=nominal_bpw,
        effective_bpw=effective_bpw,
        candidate_bits=candidate_bits,
        group_size=recipe.group_size,
        objective=objective,
        hardware=recipe.hardware,
        mtp=recipe.mtp,
        constraints=PlanningConstraints(
            effective_bpw_limit=recipe.target_bpw,
            max_model_size_ratio_to_uniform4=recipe.max_model_size_ratio_to_uniform4,
            minimum_quality_retention=recipe.minimum_quality_retention,
            minimum_mtp_acceptance_retention=recipe.minimum_mtp_acceptance_retention,
            minimum_mtp_speedup=recipe.minimum_mtp_speedup,
            lm_head_min_bits=recipe.lm_head_min_bits,
        ),
        target_mode=recipe.target_mode,
        primary_runtime=recipe.primary_runtime,
        random_seed=recipe.random_seed,
        software_versions=collect_versions(),
        analysis_sha256=stable_sha256(fingerprint),
        evidence_kind=EvidenceKind.ARCHITECTURE_PRIOR,
        assignments=allocations,
        weight_distribution=_distribution(allocations),
        mtp_distribution=_distribution(allocations, mtp_only=True),
        warnings=warnings,
    )
    return annotate_experimental_low_bit_plan(plan)
