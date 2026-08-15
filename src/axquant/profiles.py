from __future__ import annotations

from typing import Any, Literal

from axquant.package_data import load_package_yaml, message_template
from axquant.schema import ObjectiveWeights, ProfileName, ValidationThresholds


def _require_mapping(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return payload


def _build_tables() -> tuple[
    dict[ProfileName, ObjectiveWeights],
    dict[ProfileName, ValidationThresholds],
]:
    raw = _require_mapping(load_package_yaml("profiles.yaml"), "profiles.yaml")
    objectives_raw = _require_mapping(raw.get("objectives"), "profiles.yaml objectives")
    thresholds_raw = _require_mapping(raw.get("thresholds"), "profiles.yaml thresholds")
    objectives: dict[ProfileName, ObjectiveWeights] = {}
    thresholds: dict[ProfileName, ValidationThresholds] = {}
    for key, value in objectives_raw.items():
        profile = ProfileName(str(key))
        objectives[profile] = ObjectiveWeights.model_validate(value)
    for key, value in thresholds_raw.items():
        profile = ProfileName(str(key))
        thresholds[profile] = ValidationThresholds.model_validate(value)
    if set(objectives) != set(thresholds):
        raise ValueError("profiles.yaml objectives and thresholds must cover the same profiles")
    return objectives, thresholds


_OBJECTIVES, _THRESHOLDS = _build_tables()


def objective_for(profile: ProfileName) -> ObjectiveWeights:
    if profile not in _OBJECTIVES:
        raise ValueError(
            message_template("profiles", "not_implemented").format(profile=profile.value)
        )
    return _OBJECTIVES[profile].model_copy(deep=True)


def objective_for_mode(
    profile: ProfileName,
    mode: Literal["balanced", "quality", "low-memory", "speed"],
) -> ObjectiveWeights:
    """Return the profile objective with the documented v1.8 mode overlay."""

    objective = objective_for(profile)
    if mode == "balanced":
        return objective
    values = objective.model_dump()
    if mode == "quality":
        for key in ("task_loss_delta", "output_kl", "token_disagreement"):
            values[key] *= 1.5
        for key in ("peak_memory_cost", "decode_latency_cost"):
            values[key] *= 0.5
    elif mode == "low-memory":
        values["peak_memory_cost"] *= 2.0
    elif mode == "speed":
        values["decode_latency_cost"] *= 2.0
        values["prefill_latency_cost"] *= 2.0
    else:
        raise ValueError(f"unsupported optimization mode: {mode}")
    return ObjectiveWeights.model_validate(values)


def thresholds_for(profile: ProfileName) -> ValidationThresholds:
    if profile not in _THRESHOLDS:
        raise ValueError(
            message_template("profiles", "not_implemented").format(profile=profile.value)
        )
    return _THRESHOLDS[profile].model_copy(deep=True)


def implemented_profiles() -> tuple[ProfileName, ...]:
    return tuple(_OBJECTIVES)
