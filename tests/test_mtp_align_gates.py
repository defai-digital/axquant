"""MTP align ladder gates."""

from __future__ import annotations

from pathlib import Path

from axquant.mtp_align.evaluate import extract_align_metrics, load_report_metrics
from axquant.mtp_align.gates import AlignMetrics, AlignRecommendation, evaluate_ladder


def test_zero_accept_recommends_adapt_fc() -> None:
    decision = evaluate_ladder(
        AlignMetrics(
            online_accept_rate=0.0,
            token_weighted_decode_speedup=0.50,
            exactness_pass=True,
        )
    )
    assert decision.recommendation == AlignRecommendation.ADAPT_FC_NORMS
    assert decision.stage_reached == "L0"


def test_viable_accept_recommends_speedup_sweep() -> None:
    decision = evaluate_ladder(
        AlignMetrics(online_accept_rate=0.55, token_weighted_decode_speedup=1.05)
    )
    assert decision.recommendation == AlignRecommendation.ONLINE_SPEEDUP_SWEEP


def test_formal_gates_certified_path() -> None:
    decision = evaluate_ladder(
        AlignMetrics(
            online_accept_rate=0.8,
            token_weighted_decode_speedup=1.25,
            prompt_median_speedup=1.15,
            exactness_pass=True,
        )
    )
    assert decision.recommendation == AlignRecommendation.CERTIFIED_PATH


def test_extract_metrics_from_holo3_evidence() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/certifications/evidence/holo3-35b-axq6-mtp-tier2/probe_decision.json"
    )
    if not path.is_file():
        return
    metrics = load_report_metrics(path)
    assert metrics.online_accept_rate == 0.0
    assert metrics.token_weighted_decode_speedup is not None
    assert metrics.token_weighted_decode_speedup < 1.0
    decision = evaluate_ladder(metrics)
    assert decision.recommendation == AlignRecommendation.ADAPT_FC_NORMS


def test_extract_phase_accept_nested() -> None:
    metrics = extract_align_metrics(
        {
            "exactness_pass": True,
            "token_weighted_decode_speedup": 0.5,
            "phase_accept": {"accepted_tokens": 0, "proposed_tokens": 128, "accept_rate": 0.0},
        }
    )
    assert metrics.online_accept_rate == 0.0
