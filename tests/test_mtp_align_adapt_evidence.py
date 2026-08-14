"""Committed stage-1 adapt campaign evidence is coherent and non-theatrical."""

from __future__ import annotations

import json
from pathlib import Path

from axquant.mtp_align.evaluate import load_report_metrics
from axquant.mtp_align.gates import AlignRecommendation, evaluate_ladder
from axquant.mtp_align.provenance import ADAPTED_GRAFT_KIND

_EV = (
    Path(__file__).resolve().parents[1]
    / "docs/certifications/evidence/holo3-35b-axq6-mtp-adapt-stage1"
)


def test_stage1_evidence_shows_offline_top1_gain() -> None:
    before = json.loads((_EV / "teacher_force_before.json").read_text(encoding="utf-8"))
    after = json.loads((_EV / "teacher_force_after.json").read_text(encoding="utf-8"))
    assert before["top1"] == 0.0
    assert after["positions"] == before["positions"] == 32
    assert after["top1"] > before["top1"]
    assert after["correct"] == 8
    assert after["top1"] == after["correct"] / after["positions"]


def test_stage1_graft_provenance_is_adapted_not_cotrained() -> None:
    graft = json.loads((_EV / "axquant_mtp_graft.json").read_text(encoding="utf-8"))
    assert graft["graft_kind"] == ADAPTED_GRAFT_KIND
    notes = " ".join(graft["notes"]).lower()
    assert "not full co-training" in notes
    assert "tier 2" in notes


def test_stage1_online_probe_still_not_tier2_ready() -> None:
    metrics = load_report_metrics(_EV / "online_probe_after.json")
    assert metrics.online_accept_rate is not None
    assert metrics.online_accept_rate > 0.0  # left absolute zero
    assert metrics.online_accept_rate < 0.5
    assert metrics.token_weighted_decode_speedup is not None
    assert metrics.token_weighted_decode_speedup < 1.0
    decision = evaluate_ladder(metrics)
    # still below viable accept → keep adapting, not Tier 2
    assert decision.recommendation == AlignRecommendation.ADAPT_FC_NORMS


def test_stage1_comparison_record_matches_files() -> None:
    cmp = json.loads((_EV / "before_after_comparison.json").read_text(encoding="utf-8"))
    assert cmp["offline_top1_before"] == 0.0
    assert cmp["offline_top1_after"] == 0.25
    assert cmp["tier2_status"] == "not_certified"
    assert cmp["online_accept_after"] is not None
    assert cmp["online_accept_after"] > 0.0
