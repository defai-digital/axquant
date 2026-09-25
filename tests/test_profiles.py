from __future__ import annotations

import pytest

from axquant.profiles import (
    _parse_mtp_speed_floors,
    architecture_speed_class,
    has_mtp_speed_floors,
    implemented_profiles,
    mtp_speed_floors_for,
    objective_for,
    thresholds_for,
)
from axquant.schema import ProfileName


def test_reserved_profiles_fail_closed() -> None:
    assert ProfileName.AGENT_CODING in implemented_profiles()
    assert ProfileName.VLM not in implemented_profiles()
    with pytest.raises(ValueError, match="reserved but not implemented"):
        objective_for(ProfileName.VLM)
    with pytest.raises(ValueError, match="reserved but not implemented"):
        thresholds_for(ProfileName.OCR)


def test_v1_release_profiles_enforce_product_mtp_speed_floor() -> None:
    assert thresholds_for(ProfileName.AGENT_CODING).min_effective_speedup == 1.20
    assert thresholds_for(ProfileName.GENERAL).min_effective_speedup == 1.20


def test_mtp_speed_floors_table_loads_current_values() -> None:
    # AXQ-045 MH2: the table ships with each profile's current floors
    # (mechanism only; numeric changes are deferred by the ADR).
    assert mtp_speed_floors_for(ProfileName.GENERAL, "default") == (1.20, 1.10)
    assert mtp_speed_floors_for(ProfileName.GENERAL, "moe") == (1.20, 1.10)
    assert mtp_speed_floors_for(ProfileName.AGENT_CODING, "default") == (1.20, 1.10)
    assert mtp_speed_floors_for(ProfileName.AGENT_CODING, "moe") == (1.20, 1.10)
    assert mtp_speed_floors_for(ProfileName.AGENT, "default") == (1.02, 1.10)
    assert mtp_speed_floors_for(ProfileName.AGENT, "moe") == (1.02, 1.10)


def test_mtp_speed_floors_unknown_class_falls_back_to_default_entry() -> None:
    assert mtp_speed_floors_for(ProfileName.GENERAL, "vision") == mtp_speed_floors_for(
        ProfileName.GENERAL, "default"
    )
    # No class information at all resolves identically.
    assert mtp_speed_floors_for(ProfileName.GENERAL, "default") == (1.20, 1.10)


def test_mtp_speed_floors_profile_without_table_falls_back_to_constants() -> None:
    assert has_mtp_speed_floors(ProfileName.CODING) is False
    assert mtp_speed_floors_for(ProfileName.CODING, "moe") == (1.20, 1.10)


def test_mtp_speed_floors_absent_table_falls_back_to_constants() -> None:
    assert _parse_mtp_speed_floors(None, implemented=set(ProfileName)) == {}


def test_mtp_speed_floors_parser_rejects_bad_shapes() -> None:
    implemented = {ProfileName.GENERAL}
    with pytest.raises(ValueError, match="must be a mapping"):
        _parse_mtp_speed_floors(["not-a-mapping"], implemented=implemented)
    with pytest.raises(ValueError, match="unknown profile"):
        _parse_mtp_speed_floors(
            {"vlm": {"default": {"min_effective_speedup": 1.2, "min_prompt_median_speedup": 1.1}}},
            implemented=implemented,
        )
    with pytest.raises(ValueError, match="requires min_effective_speedup"):
        _parse_mtp_speed_floors(
            {"general": {"default": {"min_effective_speedup": 1.2}}},
            implemented=implemented,
        )
    with pytest.raises(ValueError, match="must be a number"):
        _parse_mtp_speed_floors(
            {
                "general": {
                    "default": {"min_effective_speedup": "1.2", "min_prompt_median_speedup": 1.1}
                }
            },
            implemented=implemented,
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        _parse_mtp_speed_floors(
            {
                "general": {
                    "default": {"min_effective_speedup": -1.0, "min_prompt_median_speedup": 1.1}
                }
            },
            implemented=implemented,
        )


def test_architecture_speed_class_mapping() -> None:
    assert architecture_speed_class(False) == "moe"
    assert architecture_speed_class(True) == "default"
    assert architecture_speed_class(None) == "default"
