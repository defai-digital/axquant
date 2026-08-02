from __future__ import annotations

from fnmatch import fnmatchcase

from axquant.errors import PlanningError
from axquant.experimental_bits import annotate_experimental_low_bit_plan
from axquant.planner import storage_bpw, strategy_for_measurement
from axquant.profiles import objective_for
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
        return 16, "non-quantizable tensor preserved"
    if _external_mtp(tensor) and recipe.mtp.preserve_external_sidecar:
        return 16, "external MTP sidecar preserved byte-for-byte"
    if tensor.role.is_mtp and recipe.mtp.mode == "protected":
        return recipe.mtp.min_bits, "protected MTP policy"
    # Unlike planner.py's automatic search, manual recipes have no
    # `candidate_bits` search space to respect -- each rule states its bits
    # explicitly, already schema-validated to >=2 (ManualPrecisionRule.bits),
    # so `2` here is a harmless floor for unprotected roles, not a policy
    # choice that needs to mirror planner.py's `min(candidate_bits)`.
    minimum = _PROTECTED_MIN_BITS.get(tensor.role, 2)
    reason = f"protected {tensor.role.value} policy" if tensor.role in _PROTECTED_MIN_BITS else None
    if tensor.role == TensorRole.LM_HEAD and recipe.lm_head_min_bits < minimum:
        minimum = recipe.lm_head_min_bits
        reason = (
            "protected lm_head floor lowered to 8-bit under AXQ-026; "
            "release certification requires measured quality evidence"
        )
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
    if tensor.role.is_mtp and recipe.mtp.mode != "disabled":
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
        method = QuantMethod.BF16 if bits == 16 else QuantMethod.AFFINE
        group_size = None if bits == 16 else recipe.group_size
        for allocation in tied:
            allocation.bits = bits
            allocation.method = method
            allocation.group_size = group_size
            allocation.reason = "tied-weight group harmonized at maximum selected precision"


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
        raise PlanningError("manual planning requires an unquantized source inventory")
    if inventory.model.revision is None:
        raise PlanningError("manual planning requires a revision-pinned source inventory")
    profile = inventory.architecture_profile
    if profile.support_level != ArchitectureSupportLevel.SUPPORTED:
        raise PlanningError("manual planning is restricted to the supported Qwen 3.6 adapter")
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
    target_class = f"{min(quantized_bits)}bit" if quantized_bits else "bf16"
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
