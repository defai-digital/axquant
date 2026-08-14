"""Gate ladder for Holo3-aligned MTP adaptation campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AlignRecommendation(str, Enum):
    """Next action after measuring MTP align metrics."""

    STOP_ENV_TUNING = "stop_env_tuning"
    """accept≈0 under formal env — do not spend more on flags/host only."""

    ADAPT_FC_NORMS = "adapt_fc_norms"
    """Stage-1: freeze MTP transformer; train fc + pre_fc norms + mtp.norm."""

    ADAPT_FULL_LAYER = "adapt_full_layer"
    """Stage-2: unfreeze mtp.layers.0 after stage-1 stalls."""

    CO_TRAIN_OR_MORE_DATA = "co_train_or_more_data"
    """Stage-1/2 insufficient — need more data or trunk-joint training."""

    ONLINE_SPEEDUP_SWEEP = "online_speedup_sweep"
    """Accept viable; optimize for ≥1.20× under MoE exact profile."""

    READY_FOR_FORMAL_TIER2 = "ready_for_formal_tier2"
    """Medium probe looks strong enough for authorizing scoreboard."""

    CERTIFIED_PATH = "certified_path"
    """Meets formal Tier 2 numeric gates (still requires formal host binding)."""


@dataclass(frozen=True, slots=True)
class AlignMetrics:
    """Measured signals for the align ladder (any field may be None if unknown)."""

    online_accept_rate: float | None = None
    offline_top1: float | None = None
    token_weighted_decode_speedup: float | None = None
    prompt_median_speedup: float | None = None
    exactness_pass: bool | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class AlignDecision:
    recommendation: AlignRecommendation
    stage_reached: str
    reasons: tuple[str, ...]
    metrics: AlignMetrics
    thresholds: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "axquant.mtp-align-decision.v1",
            "recommendation": self.recommendation.value,
            "stage_reached": self.stage_reached,
            "reasons": list(self.reasons),
            "metrics": {
                "online_accept_rate": self.metrics.online_accept_rate,
                "offline_top1": self.metrics.offline_top1,
                "token_weighted_decode_speedup": self.metrics.token_weighted_decode_speedup,
                "prompt_median_speedup": self.metrics.prompt_median_speedup,
                "exactness_pass": self.metrics.exactness_pass,
                "source": self.metrics.source,
            },
            "thresholds": dict(self.thresholds),
        }


# Product heuristics (not formal cert constants — those stay 1.20 / 1.10).
ONLINE_ACCEPT_SMOKE = 0.0  # must be strictly greater to leave smoke
ONLINE_ACCEPT_VIABLE = 0.50
OFFLINE_TOP1_TRAIN_OK = 0.70
OFFLINE_TOP1_STRONG = 0.80
SPEEDUP_BEATS_DIRECT = 1.0
SPEEDUP_MEDIUM_PROBE = 1.15
SPEEDUP_TIER2_WEIGHTED = 1.20
SPEEDUP_TIER2_MEDIAN = 1.10


def evaluate_ladder(metrics: AlignMetrics) -> AlignDecision:
    """Map measured metrics to the next best-practice action."""
    thresholds = {
        "online_accept_smoke_gt": ONLINE_ACCEPT_SMOKE,
        "online_accept_viable": ONLINE_ACCEPT_VIABLE,
        "offline_top1_train_ok": OFFLINE_TOP1_TRAIN_OK,
        "offline_top1_strong": OFFLINE_TOP1_STRONG,
        "speedup_beats_direct": SPEEDUP_BEATS_DIRECT,
        "speedup_medium_probe": SPEEDUP_MEDIUM_PROBE,
        "speedup_tier2_weighted": SPEEDUP_TIER2_WEIGHTED,
        "speedup_tier2_median": SPEEDUP_TIER2_MEDIAN,
    }
    reasons: list[str] = []
    accept = metrics.online_accept_rate
    offline = metrics.offline_top1
    weighted = metrics.token_weighted_decode_speedup
    median = metrics.prompt_median_speedup

    # Formal Tier 2 numeric shape (still needs host/evidence packaging).
    if (
        metrics.exactness_pass is True
        and weighted is not None
        and median is not None
        and weighted >= SPEEDUP_TIER2_WEIGHTED
        and median >= SPEEDUP_TIER2_MEDIAN
        and accept is not None
        and accept > ONLINE_ACCEPT_SMOKE
    ):
        return AlignDecision(
            recommendation=AlignRecommendation.CERTIFIED_PATH,
            stage_reached="L4",
            reasons=(
                "exactness pass and speedup gates meet formal Tier 2 thresholds",
            ),
            metrics=metrics,
            thresholds=thresholds,
        )

    if (
        weighted is not None
        and weighted >= SPEEDUP_MEDIUM_PROBE
        and accept is not None
        and accept >= ONLINE_ACCEPT_VIABLE
    ):
        return AlignDecision(
            recommendation=AlignRecommendation.READY_FOR_FORMAL_TIER2,
            stage_reached="L3",
            reasons=(
                f"accept_rate={accept:.3f} viable and weighted speedup "
                f"{weighted:.3f}≥{SPEEDUP_MEDIUM_PROBE}",
            ),
            metrics=metrics,
            thresholds=thresholds,
        )

    if accept is not None and accept >= ONLINE_ACCEPT_VIABLE:
        if weighted is not None and weighted > SPEEDUP_BEATS_DIRECT:
            reasons.append(
                f"accept viable ({accept:.3f}); speedup {weighted:.3f} beats direct "
                "but below formal gates — sweep MoE-exact profiles"
            )
        else:
            reasons.append(
                f"accept viable ({accept:.3f}) but speedup not yet above direct"
            )
        return AlignDecision(
            recommendation=AlignRecommendation.ONLINE_SPEEDUP_SWEEP,
            stage_reached="L2",
            reasons=tuple(reasons),
            metrics=metrics,
            thresholds=thresholds,
        )

    # Zero / near-zero online accept: stop env tuning; drive adaptation.
    if accept is not None and accept <= ONLINE_ACCEPT_SMOKE + 1e-12:
        reasons.append(
            f"online accept_rate={accept:.4f} — env/host tuning will not create accepts"
        )
        if offline is not None and offline < 0.10:
            reasons.append(
                f"offline top-1={offline:.3f} confirms distributional mismatch "
                "(or broken head wiring)"
            )
            return AlignDecision(
                recommendation=AlignRecommendation.ADAPT_FC_NORMS,
                stage_reached="L0",
                reasons=tuple(reasons),
                metrics=metrics,
                thresholds=thresholds,
            )
        if offline is not None and offline >= OFFLINE_TOP1_TRAIN_OK:
            reasons.append(
                f"offline top-1={offline:.3f} high but online accept≈0 — "
                "suspect offline/engine wiring divergence"
            )
            return AlignDecision(
                recommendation=AlignRecommendation.STOP_ENV_TUNING,
                stage_reached="L0-wiring-check",
                reasons=tuple(reasons),
                metrics=metrics,
                thresholds=thresholds,
            )
        return AlignDecision(
            recommendation=AlignRecommendation.ADAPT_FC_NORMS,
            stage_reached="L0",
            reasons=tuple(reasons)
            + ("run offline teacher-force baseline then stage-1 fc/norm adapt",),
            metrics=metrics,
            thresholds=thresholds,
        )

    # Partial accept: escalate training depth.
    if accept is not None and 0.0 < accept < ONLINE_ACCEPT_VIABLE:
        reasons.append(f"partial accept_rate={accept:.3f} below viable {ONLINE_ACCEPT_VIABLE}")
        if offline is not None and offline >= OFFLINE_TOP1_TRAIN_OK:
            return AlignDecision(
                recommendation=AlignRecommendation.ADAPT_FULL_LAYER,
                stage_reached="L1-partial",
                reasons=tuple(reasons)
                + ("offline strong enough — unfreeze mtp.layers.0",),
                metrics=metrics,
                thresholds=thresholds,
            )
        return AlignDecision(
            recommendation=AlignRecommendation.ADAPT_FC_NORMS,
            stage_reached="L1-partial",
            reasons=tuple(reasons) + ("continue stage-1 adapt with more data",),
            metrics=metrics,
            thresholds=thresholds,
        )

    # Offline-only path (no online metrics yet).
    if offline is not None:
        if offline >= OFFLINE_TOP1_STRONG:
            return AlignDecision(
                recommendation=AlignRecommendation.ONLINE_SPEEDUP_SWEEP,
                stage_reached="L0-offline-strong",
                reasons=(f"offline top-1={offline:.3f} strong — measure online A/B",),
                metrics=metrics,
                thresholds=thresholds,
            )
        if offline >= OFFLINE_TOP1_TRAIN_OK:
            return AlignDecision(
                recommendation=AlignRecommendation.ADAPT_FULL_LAYER,
                stage_reached="L0-offline-ok",
                reasons=(f"offline top-1={offline:.3f} train-ok — deepen adapt if online weak",),
                metrics=metrics,
                thresholds=thresholds,
            )
        if offline < 0.10:
            return AlignDecision(
                recommendation=AlignRecommendation.ADAPT_FC_NORMS,
                stage_reached="L0-offline-low",
                reasons=(f"offline top-1={offline:.3f} — start stage-1 fc/norm adapt",),
                metrics=metrics,
                thresholds=thresholds,
            )
        return AlignDecision(
            recommendation=AlignRecommendation.CO_TRAIN_OR_MORE_DATA,
            stage_reached="L0-offline-mid",
            reasons=(f"offline top-1={offline:.3f} mid — more data or full-layer adapt",),
            metrics=metrics,
            thresholds=thresholds,
        )

    return AlignDecision(
        recommendation=AlignRecommendation.STOP_ENV_TUNING,
        stage_reached="unknown",
        reasons=("insufficient metrics; run online probe and/or teacher-force",),
        metrics=metrics,
        thresholds=thresholds,
    )
