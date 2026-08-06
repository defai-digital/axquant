"""Package YAML loaders for profiles, ladders, role prefs, and messages."""

from __future__ import annotations

import pytest

from axquant.errors import ArtifactError, PlanningError
from axquant.ladders import get_ladder, list_ladders
from axquant.numeric import as_finite_float32_matrix
from axquant.package_data import load_package_yaml, message_template
from axquant.profiles import implemented_profiles, objective_for, thresholds_for
from axquant.role_policy import ROLE_PREFERENCES
from axquant.schema import ConvertLadderName, ProfileName, QuantMethod, TensorRole


def test_load_package_yaml_profiles_matches_objective_api() -> None:
    raw = load_package_yaml("profiles.yaml")
    assert set(raw["objectives"]) == {profile.value for profile in implemented_profiles()}
    general = objective_for(ProfileName.GENERAL)
    assert general.output_kl == raw["objectives"]["general"]["output_kl"]
    assert thresholds_for(ProfileName.AGENT_CODING).min_effective_speedup == 1.20


def test_load_package_yaml_rejects_invalid_names() -> None:
    with pytest.raises(ArtifactError, match="invalid package data name"):
        load_package_yaml("../secrets.yaml")
    with pytest.raises(ArtifactError, match="missing package data"):
        load_package_yaml("does-not-exist.yaml")


def test_message_templates_preserve_floor_wording() -> None:
    assert message_template("floor_reasons", "non_quantizable") == (
        "non-quantizable tensor preserved"
    )
    assert message_template("floor_reasons", "protected_role").format(role="norm") == (
        "protected norm policy"
    )
    assert "AXQ-026" in message_template("floor_reasons", "lm_head_lowered")


def test_role_preferences_loaded_from_yaml() -> None:
    assert TensorRole.ATTENTION in ROLE_PREFERENCES
    assert ROLE_PREFERENCES[TensorRole.ATTENTION].preferred_methods[0] is QuantMethod.AWQ
    assert ROLE_PREFERENCES[TensorRole.ATTENTION].preferred_max_group_size == 32
    assert ROLE_PREFERENCES[TensorRole.MLP].preferred_max_group_size == 64


def test_convert_ladders_loaded_from_yaml() -> None:
    ladders = list_ladders()
    assert [ladder.name for ladder in ladders] == [
        ConvertLadderName.PRIOR,
        ConvertLadderName.MEASURED_LITE,
        ConvertLadderName.MEASURED_FULL,
        ConvertLadderName.REFINE_AWQ_DWQ,
    ]
    prior = get_ladder("prior")
    assert prior.allow_unmeasured is True
    assert prior.candidate_group_sizes == (32, 64)
    assert prior.default_target_bpw == 4.8
    refine = get_ladder(ConvertLadderName.REFINE_AWQ_DWQ)
    assert QuantMethod.GPTQ in refine.candidate_methods
    assert refine.requires_refinement is True


def test_as_finite_float32_matrix_uses_component_label() -> None:
    import numpy as np

    matrix = as_finite_float32_matrix([[1.0, 2.0], [3.0, 4.0]], component="AWQ")
    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 2)
    with pytest.raises(PlanningError, match="GPTQ weight matrix must not be empty"):
        as_finite_float32_matrix([], component="GPTQ")
    with pytest.raises(PlanningError, match="AWQ weight matrix must contain only finite values"):
        as_finite_float32_matrix([[1.0, float("nan")]], component="AWQ")
