"""Extract AlignMetrics from engine A/B / probe JSON reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axquant.mtp_align.gates import AlignMetrics


def load_report_metrics(path: str | Path) -> AlignMetrics:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return extract_align_metrics(payload, source=str(path))


def extract_align_metrics(payload: dict[str, Any] | list[Any], *, source: str | None = None) -> AlignMetrics:
    """Best-effort parse of probe_decision, profile_summary, or mtp_ab_comparison."""
    if isinstance(payload, list):
        # list of profile summaries
        return _from_profiles(payload, source=source)

    if not isinstance(payload, dict):
        return AlignMetrics(source=source)

    # probe_decision.json
    profiles = payload.get("profiles")
    if isinstance(profiles, list) and profiles:
        return _from_profiles(profiles, source=source, exactness_hint=payload.get("all_exactness_pass"))

    # single profile_summary / mtp_ab_comparison
    return _from_profile_dict(payload, source=source)


def _from_profiles(
    profiles: list[Any],
    *,
    source: str | None,
    exactness_hint: Any = None,
) -> AlignMetrics:
    dicts = [p for p in profiles if isinstance(p, dict)]
    if not dicts:
        return AlignMetrics(source=source)
    # Prefer first profile; aggregate accept if present on all.
    primary = _from_profile_dict(dicts[0], source=source)
    if exactness_hint is not None and primary.exactness_pass is None:
        primary = AlignMetrics(
            online_accept_rate=primary.online_accept_rate,
            offline_top1=primary.offline_top1,
            token_weighted_decode_speedup=primary.token_weighted_decode_speedup,
            prompt_median_speedup=primary.prompt_median_speedup,
            exactness_pass=bool(exactness_hint),
            source=primary.source,
        )
    return primary


def _from_profile_dict(data: dict[str, Any], *, source: str | None) -> AlignMetrics:
    accept = _accept_rate(data)
    weighted = _float(data.get("token_weighted_decode_speedup"))
    median = _float(data.get("prompt_median_speedup"))
    # comparison often nests speedup under same keys
    if weighted is None:
        weighted = _float(data.get("speedup")) if data.get("speedup_metric") else None
    exact = data.get("exactness_pass")
    if exact is not None:
        exact = bool(exact)
    offline = _float(data.get("offline_top1"))
    if offline is None:
        offline = _float((data.get("teacher_force") or {}).get("top1"))
    return AlignMetrics(
        online_accept_rate=accept,
        offline_top1=offline,
        token_weighted_decode_speedup=weighted,
        prompt_median_speedup=median,
        exactness_pass=exact,
        source=source,
    )


def _accept_rate(data: dict[str, Any]) -> float | None:
    phase = data.get("phase_accept")
    if isinstance(phase, dict):
        rate = _float(phase.get("accept_rate"))
        if rate is not None:
            return rate
        accepted = phase.get("accepted_tokens")
        proposed = phase.get("proposed_tokens")
        if isinstance(accepted, int) and isinstance(proposed, int) and proposed > 0:
            return accepted / proposed
    # trial-level aggregation
    trials = data.get("trial_comparisons")
    if isinstance(trials, list) and trials:
        acc = 0
        prop = 0
        for trial in trials:
            if not isinstance(trial, dict):
                continue
            a = trial.get("mtp_accepted_tokens")
            p = trial.get("mtp_proposed_tokens")
            if isinstance(a, int) and isinstance(p, int):
                acc += a
                prop += p
        if prop > 0:
            return acc / prop
    # nested phase_timing on comparison objects
    phase_timing = data.get("phase_timing")
    if isinstance(phase_timing, dict):
        accepted = phase_timing.get("accepted_tokens")
        proposed = phase_timing.get("proposed_tokens")
        if isinstance(accepted, int) and isinstance(proposed, int) and proposed > 0:
            return accepted / proposed
        rate = _float(phase_timing.get("accept_rate"))
        if rate is not None:
            return rate
    return _float(data.get("online_accept_rate"))


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
