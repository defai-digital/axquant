"""V3 scaled stage-1 adapt evidence: more labels, modest online accept gain."""

from __future__ import annotations

import json
from pathlib import Path

from axquant.mtp_align.provenance import ADAPTED_GRAFT_KIND

_EV = (
    Path(__file__).resolve().parents[1]
    / "docs/certifications/evidence/holo3-35b-axq6-mtp-adapt-v3"
)
_S1 = (
    Path(__file__).resolve().parents[1]
    / "docs/certifications/evidence/holo3-35b-axq6-mtp-adapt-stage1"
)


def test_v3_improves_online_accept_over_stage1() -> None:
    s1 = json.loads((_S1 / "before_after_comparison.json").read_text(encoding="utf-8"))
    v3 = json.loads((_EV / "comparison.json").read_text(encoding="utf-8"))
    assert v3["v3_online_accept"] > s1["online_accept_after"]
    assert v3["v3_online_accept"] < 0.5
    assert v3["tier2_status"] == "not_certified"
    assert v3["main_digests_unchanged"] is True


def test_v3_offline_top1_nontrivial() -> None:
    before = json.loads((_EV / "teacher_force_before.json").read_text(encoding="utf-8"))
    after = json.loads((_EV / "teacher_force_after.json").read_text(encoding="utf-8"))
    assert after["positions"] == 48
    assert after["top1"] >= before["top1"]
    assert after["top1"] >= 0.25
    assert after["correct"] == 12


def test_v3_graft_still_adapted_not_cotrained() -> None:
    graft = json.loads((_EV / "axquant_mtp_graft.json").read_text(encoding="utf-8"))
    assert graft["graft_kind"] == ADAPTED_GRAFT_KIND
    assert graft["train"]["stage"] == "fc_norms"
    assert int(graft["train"]["steps"]) == 1200
    assert "not full co-training" in graft["notes"][0].lower()


def test_v3_campaign_summary_sample_budget() -> None:
    camp = json.loads((_EV / "campaign_summary.json").read_text(encoding="utf-8"))
    assert camp["prepare"]["samples"] == 1024
    assert camp["adapt"]["steps"] == 1200
    assert camp["improved"] is True
