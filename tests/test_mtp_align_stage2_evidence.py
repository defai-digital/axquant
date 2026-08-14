"""Stage-2 full-layer adapt evidence: honest comparison to stage-1."""

from __future__ import annotations

import json
from pathlib import Path

from axquant.mtp_align.provenance import ADAPTED_FULL_GRAFT_KIND

_S1 = (
    Path(__file__).resolve().parents[1]
    / "docs/certifications/evidence/holo3-35b-axq6-mtp-adapt-stage1"
)
_S2 = (
    Path(__file__).resolve().parents[1]
    / "docs/certifications/evidence/holo3-35b-axq6-mtp-adapt-stage2"
)


def test_stage2_summary_beats_grafted_baseline() -> None:
    s2 = json.loads((_S2 / "stage2_summary.json").read_text(encoding="utf-8"))
    s1_before = json.loads((_S1 / "teacher_force_before.json").read_text(encoding="utf-8"))
    assert s2["baseline_top1"] == s1_before["top1"] == 0.0
    assert s2["stage2_top1"] > s2["baseline_top1"]
    assert s2["tier2_status"] == "not_certified"
    assert s2["main_digests_unchanged"] is True


def test_stage2_did_not_beat_stage1_on_this_budget() -> None:
    s2 = json.loads((_S2 / "stage2_summary.json").read_text(encoding="utf-8"))
    assert s2["stage1_top1"] == 0.25
    assert s2["stage2_top1"] <= s2["stage1_top1"]
    assert s2["stage2_improved_vs_stage1"] is False


def test_stage2_graft_kind_is_full_layer() -> None:
    graft = json.loads((_S2 / "axquant_mtp_graft.json").read_text(encoding="utf-8"))
    assert graft["graft_kind"] == ADAPTED_FULL_GRAFT_KIND
    assert graft["train"]["stage"] == "full_layer"
    assert int(graft["train"]["steps"]) == 300
    assert "not full co-training" in graft["notes"][0].lower()


def test_stage2_online_accept_still_low() -> None:
    s2 = json.loads((_S2 / "stage2_summary.json").read_text(encoding="utf-8"))
    online = s2["online"]
    assert online["accept_rate"] is not None
    assert 0.0 < float(online["accept_rate"]) < 0.1
    assert float(online["speedup"]) < 1.0
