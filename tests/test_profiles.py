from __future__ import annotations

import pytest

from axquant.profiles import implemented_profiles, objective_for, thresholds_for
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
