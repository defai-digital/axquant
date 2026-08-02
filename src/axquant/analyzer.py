from __future__ import annotations

from axquant.schema import (
    CandidateMeasurement,
    EvidenceKind,
    Inventory,
    MetricVector,
    ProfileName,
    QuantMethod,
    SensitivityReport,
    TensorRole,
    TensorSensitivity,
)
from axquant.serde import stable_sha256

_ROLE_SENSITIVITY = {
    TensorRole.EMBEDDING: 0.90,
    TensorRole.ATTENTION: 0.72,
    TensorRole.MLP: 0.45,
    TensorRole.NORM: 1.20,
    TensorRole.LM_HEAD: 1.25,
    TensorRole.ROUTER: 1.10,
    TensorRole.EXPERT: 0.38,
    TensorRole.MTP_PROJECTION: 1.20,
    TensorRole.MTP_BLOCK: 1.10,
    TensorRole.MTP_OUTPUT: 1.35,
    TensorRole.VISION: 0.90,
    TensorRole.OTHER: 0.60,
}


def _prior_metrics(
    role: TensorRole,
    bits: int,
    *,
    group_size: int | None = 64,
) -> MetricVector:
    if bits == 16:
        return MetricVector()
    # Smaller groups weakly reduce prior noise (AXQ-028 development heuristic only).
    group_factor = 1.0 if group_size is None else (float(group_size) / 64.0) ** 0.5
    noise = (2.0 ** (4 - bits)) * group_factor
    sensitivity = _ROLE_SENSITIVITY[role]
    mtp_factor = 1.4 if role.is_mtp else 0.08
    long_context_factor = 1.25 if role == TensorRole.ATTENTION else 0.30
    return MetricVector(
        output_kl=sensitivity * noise,
        hidden_state_error=sensitivity * 0.80 * noise,
        cosine_distance=sensitivity * 0.45 * noise,
        token_disagreement=sensitivity * 0.55 * noise,
        task_loss_delta=sensitivity * 0.65 * noise,
        mtp_acceptance_loss=sensitivity * mtp_factor * noise,
        long_context_loss=sensitivity * long_context_factor * noise,
    )


def architecture_prior_report(
    inventory: Inventory,
    *,
    profile: ProfileName,
    candidate_bits: tuple[int, ...] = (4, 6, 8, 16),
    group_size: int = 64,
    candidate_group_sizes: tuple[int, ...] = (),
) -> SensitivityReport:
    """Build architecture-prior sensitivity with optional multi-group candidates (AXQ-028)."""
    effective_groups = candidate_group_sizes or (group_size,)
    entries: list[TensorSensitivity] = []
    for tensor in inventory.tensors:
        bits_for_tensor = candidate_bits if tensor.quantizable else (16,)
        candidates: list[CandidateMeasurement] = []
        for bits in bits_for_tensor:
            if bits == 16:
                candidates.append(
                    CandidateMeasurement(
                        bits=16,
                        method=QuantMethod.BF16,
                        group_size=None,
                        metrics=_prior_metrics(tensor.role, 16),
                        note="architecture prior; not a measured quality result",
                    )
                )
                continue
            for size in effective_groups:
                candidates.append(
                    CandidateMeasurement(
                        bits=bits,
                        method=QuantMethod.AFFINE,
                        group_size=size,
                        metrics=_prior_metrics(tensor.role, bits, group_size=size),
                        note="architecture prior; not a measured quality result",
                    )
                )
        entries.append(TensorSensitivity(tensor=tensor, candidates=candidates))
    warnings = [
        "This report contains architecture priors, not calibration measurements.",
        "Conversion planning requires --allow-unmeasured for this report.",
    ]
    if len(effective_groups) > 1:
        warnings.append(
            "Multi-group architecture priors weakly prefer smaller group sizes; "
            "this is development evidence only (AXQ-028)."
        )
    return SensitivityReport(
        model=inventory.model,
        architecture_profile=inventory.architecture_profile,
        profile=profile,
        evidence_kind=EvidenceKind.ARCHITECTURE_PRIOR,
        inventory_sha256=stable_sha256(inventory.model_dump(mode="json", exclude={"created_at"})),
        entries=entries,
        warnings=warnings,
    )
