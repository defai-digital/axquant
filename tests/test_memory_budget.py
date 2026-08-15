from __future__ import annotations

import pytest
from pydantic import ValidationError

from axquant.errors import PlanningError
from axquant.memory_budget import evaluate_budget
from axquant.profiles import objective_for
from axquant.schema import DeploymentPlan, EvidenceKind, ProfileName, RuntimeName


def test_budget_is_feasible_exactly_at_the_limit() -> None:
    result = evaluate_budget(700, 200, 100, 1_000)
    assert result.feasible is True
    assert result.remainder_bytes == 0


def test_budget_reports_an_infeasible_remainder() -> None:
    result = evaluate_budget(800, 200, 100, 1_000)
    assert result.feasible is False
    assert result.remainder_bytes == -100


def test_budget_rejects_negative_reserve() -> None:
    with pytest.raises(PlanningError, match="reserve_bytes"):
        evaluate_budget(1, 1, -1, 10)


def test_architecture_prior_deployment_cannot_be_labeled_measured() -> None:
    with pytest.raises(ValidationError, match="architecture-prior"):
        DeploymentPlan(
            weight_bytes=100,
            kv_bytes=0,
            reserve_bytes=10,
            limit_bytes=1_000,
            remainder_bytes=890,
            feasible=True,
            evidence_kind="measured",
            source_evidence_kind=EvidenceKind.ARCHITECTURE_PRIOR,
            context_length=128,
            batch_size=1,
            profile=ProfileName.GENERAL,
            target_class="4bit",
            runtime=RuntimeName.AX_ENGINE,
            mode="balanced",
            objective=objective_for(ProfileName.GENERAL),
            minimum_quality_retention=0.98,
            weight_bytes_basis="artifact-manifest",
            measured_main_bpw=4.5,
            plan_sha256="a" * 64,
            explanation="fixture",
        )
