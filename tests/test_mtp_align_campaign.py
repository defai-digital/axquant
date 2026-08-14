"""Campaign script next-command guidance uses real decision recommendations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from axquant.mtp_align.evaluate import load_report_metrics
from axquant.mtp_align.gates import AlignRecommendation, evaluate_ladder


def _load_campaign_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_holo3_mtp_align_campaign.py"
    spec = importlib.util.spec_from_file_location("holo3_mtp_align_campaign", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_campaign_next_commands_for_adapt() -> None:
    mod = _load_campaign_module()
    cmds = mod._next_commands("adapt_fc_norms", Path("/pack"), Path("/out"))
    assert any("mtp-align-prepare-data" in c for c in cmds)
    assert any("mtp-align-adapt-fc" in c for c in cmds)
    assert any("compose-grafted-mtp" in c for c in cmds)


def test_holo3_evidence_drives_adapt_recommendation() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/certifications/evidence/holo3-35b-axq6-mtp-tier2/probe_decision.json"
    )
    metrics = load_report_metrics(path)
    decision = evaluate_ladder(metrics)
    assert decision.recommendation == AlignRecommendation.ADAPT_FC_NORMS
    assert metrics.online_accept_rate == 0.0
