from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.schema import EvidenceKind, ProfileName, QuantMethod


def test_architecture_prior_is_explicitly_unmeasured(tiny_model_dir: Path) -> None:
    inventory = inspect_model(tiny_model_dir)
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
        candidate_bits=(4, 6, 8, 16),
    )
    assert report.evidence_kind == EvidenceKind.ARCHITECTURE_PRIOR
    assert report.calibration is None
    assert any("not calibration" in warning for warning in report.warnings)
    q_proj = next(entry for entry in report.entries if entry.tensor.name.endswith("q_proj.weight"))
    # AXQ-047: the 4-bit rung carries the enforced MXFP4 candidate (group 32)
    # alongside the affine prior candidate.
    assert [candidate.bits for candidate in q_proj.candidates] == [4, 4, 6, 8, 16]
    mxfp4 = [candidate for candidate in q_proj.candidates if candidate.method is QuantMethod.MXFP4]
    assert len(mxfp4) == 1 and mxfp4[0].group_size == 32
    assert q_proj.candidates[0].metrics.output_kl > q_proj.candidates[-1].metrics.output_kl
    inventory.created_at += timedelta(seconds=1)
    repeated = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
        candidate_bits=(4, 6, 8, 16),
    )
    assert repeated.inventory_sha256 == report.inventory_sha256


def test_architecture_prior_rejects_empty_inventory_and_candidate_grid(
    tiny_model_dir: Path,
) -> None:
    inventory = inspect_model(tiny_model_dir)
    empty = inventory.model_copy(
        update={
            "tensors": [],
            "total_parameters": 0,
            "quantizable_parameters": 0,
        }
    )
    with pytest.raises(PlanningError, match="non-empty tensor inventory"):
        architecture_prior_report(empty, profile=ProfileName.GENERAL)
    with pytest.raises(PlanningError, match="at least one candidate"):
        architecture_prior_report(
            inventory,
            profile=ProfileName.GENERAL,
            candidate_bits=(),
        )


def test_architecture_prior_canonicalizes_and_validates_candidate_grid(
    tiny_model_dir: Path,
) -> None:
    inventory = inspect_model(tiny_model_dir)
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.GENERAL,
        candidate_bits=(16, 4, 4),
        candidate_group_sizes=(64, 32, 64),
    )
    quantizable = next(entry for entry in report.entries if entry.tensor.quantizable)
    # AXQ-047: the 4-bit rung appends the enforced MXFP4 (group 32) candidate.
    assert [(candidate.bits, candidate.group_size) for candidate in quantizable.candidates] == [
        (4, 32),
        (4, 64),
        (4, 32),
        (16, None),
    ]
    assert quantizable.candidates[2].method is QuantMethod.MXFP4

    with pytest.raises(PlanningError, match=r"bit-widths.*5"):
        architecture_prior_report(
            inventory,
            profile=ProfileName.GENERAL,
            candidate_bits=(5, 16),
        )
    with pytest.raises(PlanningError, match=r"group sizes.*7"):
        architecture_prior_report(
            inventory,
            profile=ProfileName.GENERAL,
            candidate_group_sizes=(7,),
        )
