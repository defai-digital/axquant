from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from axquant.analyzer import architecture_prior_report
from axquant.inspector import inspect_model
from axquant.schema import EvidenceKind, ProfileName


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
    assert [candidate.bits for candidate in q_proj.candidates] == [4, 6, 8, 16]
    assert q_proj.candidates[0].metrics.output_kl > q_proj.candidates[-1].metrics.output_kl
    inventory.created_at += timedelta(seconds=1)
    repeated = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
        candidate_bits=(4, 6, 8, 16),
    )
    assert repeated.inventory_sha256 == report.inventory_sha256
