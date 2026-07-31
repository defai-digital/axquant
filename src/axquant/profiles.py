from __future__ import annotations

from axquant.schema import ObjectiveWeights, ProfileName, ValidationThresholds

_OBJECTIVES = {
    ProfileName.GENERAL: ObjectiveWeights(
        output_kl=0.25,
        hidden_state_error=0.12,
        cosine_distance=0.05,
        token_disagreement=0.08,
        task_loss_delta=0.15,
        mtp_acceptance_loss=0.15,
        long_context_loss=0.08,
        peak_memory_cost=0.04,
        prefill_latency_cost=0.03,
        decode_latency_cost=0.05,
    ),
    ProfileName.AGENT_CODING: ObjectiveWeights(
        output_kl=0.15,
        hidden_state_error=0.10,
        cosine_distance=0.03,
        token_disagreement=0.12,
        task_loss_delta=0.20,
        mtp_acceptance_loss=0.22,
        long_context_loss=0.05,
        peak_memory_cost=0.03,
        prefill_latency_cost=0.03,
        decode_latency_cost=0.07,
    ),
    ProfileName.AGENT: ObjectiveWeights(
        output_kl=0.20,
        hidden_state_error=0.15,
        cosine_distance=0.03,
        token_disagreement=0.07,
        task_loss_delta=0.25,
        mtp_acceptance_loss=0.20,
        long_context_loss=0.05,
        peak_memory_cost=0.02,
        prefill_latency_cost=0.01,
        decode_latency_cost=0.02,
    ),
    ProfileName.CODING: ObjectiveWeights(
        output_kl=0.20,
        hidden_state_error=0.12,
        cosine_distance=0.03,
        token_disagreement=0.15,
        task_loss_delta=0.30,
        mtp_acceptance_loss=0.12,
        long_context_loss=0.05,
        peak_memory_cost=0.01,
        prefill_latency_cost=0.01,
        decode_latency_cost=0.01,
    ),
    ProfileName.TRANSLATION: ObjectiveWeights(
        output_kl=0.25,
        hidden_state_error=0.12,
        cosine_distance=0.08,
        token_disagreement=0.10,
        task_loss_delta=0.25,
        mtp_acceptance_loss=0.10,
        long_context_loss=0.05,
        peak_memory_cost=0.01,
        prefill_latency_cost=0.01,
        decode_latency_cost=0.03,
    ),
    ProfileName.REASONING: ObjectiveWeights(
        output_kl=0.20,
        hidden_state_error=0.15,
        cosine_distance=0.05,
        token_disagreement=0.08,
        task_loss_delta=0.30,
        mtp_acceptance_loss=0.15,
        long_context_loss=0.05,
        peak_memory_cost=0.005,
        prefill_latency_cost=0.005,
        decode_latency_cost=0.01,
    ),
    ProfileName.LONG_CONTEXT: ObjectiveWeights(
        output_kl=0.15,
        hidden_state_error=0.15,
        cosine_distance=0.05,
        token_disagreement=0.05,
        task_loss_delta=0.15,
        mtp_acceptance_loss=0.10,
        long_context_loss=0.25,
        peak_memory_cost=0.04,
        prefill_latency_cost=0.03,
        decode_latency_cost=0.03,
    ),
}

_THRESHOLDS = {
    ProfileName.GENERAL: ValidationThresholds(min_effective_speedup=1.20),
    ProfileName.AGENT_CODING: ValidationThresholds(
        max_task_score_drop=0.01,
        max_mtp_acceptance_drop=0.02,
        max_mtp_token_accuracy_drop=0.03,
        max_repetition_increase=0.005,
        max_divergence_increase=0.005,
        min_effective_speedup=1.20,
        required_task_scores=(
            "coding",
            "tool",
            "json",
            "multilingual",
            "long_context",
        ),
    ),
    ProfileName.AGENT: ValidationThresholds(
        max_task_score_drop=0.015,
        max_mtp_acceptance_drop=0.02,
        max_mtp_token_accuracy_drop=0.03,
        max_repetition_increase=0.005,
        max_divergence_increase=0.005,
        min_effective_speedup=1.02,
    ),
    ProfileName.CODING: ValidationThresholds(
        max_task_score_drop=0.01,
        max_mtp_acceptance_drop=0.025,
        max_mtp_token_accuracy_drop=0.035,
        max_repetition_increase=0.005,
        max_divergence_increase=0.005,
    ),
    ProfileName.TRANSLATION: ValidationThresholds(max_task_score_drop=0.01),
    ProfileName.REASONING: ValidationThresholds(max_task_score_drop=0.01),
    ProfileName.LONG_CONTEXT: ValidationThresholds(
        max_task_score_drop=0.015,
        max_mtp_acceptance_drop=0.03,
    ),
}


def objective_for(profile: ProfileName) -> ObjectiveWeights:
    if profile not in _OBJECTIVES:
        raise ValueError(f"profile {profile.value!r} is reserved but not implemented")
    return _OBJECTIVES[profile].model_copy(deep=True)


def thresholds_for(profile: ProfileName) -> ValidationThresholds:
    if profile not in _THRESHOLDS:
        raise ValueError(f"profile {profile.value!r} is reserved but not implemented")
    return _THRESHOLDS[profile].model_copy(deep=True)


def implemented_profiles() -> tuple[ProfileName, ...]:
    return tuple(_OBJECTIVES)
