from __future__ import annotations

from typing import Any, Literal

from axquant.package_data import load_package_yaml, message_template
from axquant.schema import ObjectiveWeights, ProfileName, ValidationThresholds

# Package-level MTP speed floors used when profiles.yaml carries no
# mtp_speed_floors table for a profile. Mirrors the benchmark.py defaults;
# numeric changes are deferred by ADR AXQ-045 (mechanism only).
DEFAULT_MTP_EFFECTIVE_SPEEDUP = 1.20
DEFAULT_MTP_PROMPT_MEDIAN_SPEEDUP = 1.10

# Architecture class used when a caller has no architecture information.
DEFAULT_SPEED_CLASS = "default"


def _require_mapping(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return payload


def _non_negative_float(payload: Any, label: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        raise ValueError(f"{label} must be a number")
    value = float(payload)
    if value < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _build_tables() -> tuple[
    dict[ProfileName, ObjectiveWeights],
    dict[ProfileName, ValidationThresholds],
    Any,
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
    return objectives, thresholds, raw


def _parse_mtp_speed_floors(
    payload: Any,
    *,
    implemented: set[ProfileName],
) -> dict[ProfileName, dict[str, tuple[float, float]]]:
    """Validate the optional mtp_speed_floors table: profile -> class -> floors."""

    if payload is None:
        return {}
    table_raw = _require_mapping(payload, "profiles.yaml mtp_speed_floors")
    table: dict[ProfileName, dict[str, tuple[float, float]]] = {}
    for profile_key, classes_raw in table_raw.items():
        profile = ProfileName(str(profile_key))
        if profile not in implemented:
            raise ValueError(
                f"profiles.yaml mtp_speed_floors defines unknown profile: {profile_key}"
            )
        classes = _require_mapping(classes_raw, f"profiles.yaml mtp_speed_floors.{profile_key}")
        entries: dict[str, tuple[float, float]] = {}
        for class_key, entry_raw in classes.items():
            label = f"profiles.yaml mtp_speed_floors.{profile_key}.{class_key}"
            entry = _require_mapping(entry_raw, label)
            min_effective = entry.get("min_effective_speedup")
            min_prompt_median = entry.get("min_prompt_median_speedup")
            if min_effective is None or min_prompt_median is None:
                raise ValueError(
                    f"{label} requires min_effective_speedup and min_prompt_median_speedup"
                )
            entries[str(class_key)] = (
                _non_negative_float(min_effective, f"{label}.min_effective_speedup"),
                _non_negative_float(min_prompt_median, f"{label}.min_prompt_median_speedup"),
            )
        table[profile] = entries
    return table


_OBJECTIVES, _THRESHOLDS, _RAW_PROFILES = _build_tables()
_MTP_SPEED_FLOORS = _parse_mtp_speed_floors(
    _RAW_PROFILES.get("mtp_speed_floors"),
    implemented=set(_THRESHOLDS),
)


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


def has_mtp_speed_floors(profile: ProfileName) -> bool:
    """Whether the profile defines an explicit mtp_speed_floors table."""

    return profile in _MTP_SPEED_FLOORS


def mtp_speed_floors_for(profile: ProfileName, arch_class: str) -> tuple[float, float]:
    """Resolve (min_effective_speedup, min_prompt_median_speedup) for a profile.

    Unknown architecture classes fall back to the profile's ``default`` class
    entry; profiles without a table fall back to the package constants
    (1.20 / 1.10), mirroring the benchmark defaults.
    """

    entries = _MTP_SPEED_FLOORS.get(profile)
    if entries:
        floors = entries.get(arch_class) or entries.get(DEFAULT_SPEED_CLASS)
        if floors is not None:
            return floors
    return (DEFAULT_MTP_EFFECTIVE_SPEEDUP, DEFAULT_MTP_PROMPT_MEDIAN_SPEEDUP)


def architecture_speed_class(dense: bool | None) -> str:
    """Map an ArchitectureProfile.dense flag to an mtp_speed_floors class key.

    Sparse-expert (MoE) models classify as ``moe``; dense and unknown layouts
    use the ``default`` class entry.
    """

    return "moe" if dense is False else DEFAULT_SPEED_CLASS


def implemented_profiles() -> tuple[ProfileName, ...]:
    return tuple(_OBJECTIVES)
