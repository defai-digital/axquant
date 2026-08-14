"""Holo3-aligned MTP adaptation: measure → decide → adapt head (freeze trunk).

Best-practice KPI order: accept_rate → speedup → formal Tier 2.
Online ground truth is AX Engine A/B; offline teacher-forced top-1 is the
cheap training metric (mlx_lm strips mtp.* at load).
"""

from __future__ import annotations

from axquant.mtp_align.evaluate import extract_align_metrics, load_report_metrics
from axquant.mtp_align.gates import (
    AlignDecision,
    AlignMetrics,
    AlignRecommendation,
    evaluate_ladder,
)

__all__ = [
    "AlignDecision",
    "AlignMetrics",
    "AlignRecommendation",
    "evaluate_ladder",
    "extract_align_metrics",
    "load_report_metrics",
]
