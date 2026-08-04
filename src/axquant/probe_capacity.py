"""Probe capacity / RAM-mode assessment (P0).

Recommends which sensitivity measurement mode can run on a host without OOM,
and records the strongest EvidenceKind each mode may produce.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from axquant.errors import PlanningError
from axquant.schema import (
    EvidenceKind,
    Inventory,
    ModelIdentity,
    ProbeCapacityModeAssessment,
    ProbeCapacityReport,
    ProbeMode,
)
from axquant.serde import load_model

# Storage heuristics (bytes per parameter) for capacity planning only — not BPW claims.
_BF16_BYTES_PER_PARAM = 2.0
_UNIFORM4_BYTES_PER_PARAM = 0.55  # ~4-bit + group scales
_PROBE_OVERHEAD = 1.35  # activations / workspace headroom multiplier
_DEFAULT_HEADROOM = 0.70  # use at most this fraction of available unified memory


def _read_available_memory_bytes(explicit: int | None) -> int | None:
    if explicit is not None:
        if type(explicit) is not int or explicit <= 0 or explicit > sys.maxsize:
            raise PlanningError("available memory must be a positive platform-sized integer")
        return explicit
    # Best-effort macOS / Linux detection; absence falls back to prior-only advice.
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if (
            type(page_size) is int
            and type(phys_pages) is int
            and page_size > 0
            and phys_pages > 0
            and page_size <= sys.maxsize // phys_pages
        ):
            return page_size * phys_pages
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _mode_assessment(
    mode: ProbeMode,
    *,
    required_bytes: int,
    available_bytes: int | None,
    headroom_fraction: float,
    evidence_kind: EvidenceKind,
    release_quality: bool,
    notes: list[str],
) -> ProbeCapacityModeAssessment:
    budget = None if available_bytes is None else int(available_bytes * headroom_fraction)
    if budget is None:
        feasible = mode is ProbeMode.PRIOR_ONLY
        reason = (
            "prior-only is always feasible without a memory measurement"
            if feasible
            else "host memory unknown; cannot confirm this measured mode"
        )
    else:
        feasible = required_bytes <= budget
        reason = (
            f"requires ~{required_bytes / 1024**3:.2f} GiB within {budget / 1024**3:.2f} GiB budget"
            if feasible
            else f"needs ~{required_bytes / 1024**3:.2f} GiB but budget is "
            f"{budget / 1024**3:.2f} GiB"
        )
    return ProbeCapacityModeAssessment(
        mode=mode,
        feasible=feasible,
        estimated_bytes=required_bytes,
        evidence_kind=evidence_kind,
        release_quality_eligible=release_quality and feasible,
        reason=reason,
        notes=notes,
    )


def assess_probe_capacity(
    *,
    parameter_count: int,
    model: ModelIdentity | None = None,
    available_memory_bytes: int | None = None,
    headroom_fraction: float = _DEFAULT_HEADROOM,
) -> ProbeCapacityReport:
    """Assess which probe modes fit the host memory budget."""
    if type(parameter_count) is not int or parameter_count <= 0 or parameter_count > sys.maxsize:
        raise PlanningError("parameter_count must be a positive platform-sized integer")
    if (
        isinstance(headroom_fraction, bool)
        or not isinstance(headroom_fraction, (int, float))
        or not math.isfinite(float(headroom_fraction))
        or not 0.0 < float(headroom_fraction) <= 1.0
    ):
        raise PlanningError("headroom_fraction must be in (0, 1]")
    resolved_headroom = float(headroom_fraction)
    available = _read_available_memory_bytes(available_memory_bytes)
    bf16_estimate = parameter_count * _BF16_BYTES_PER_PARAM * _PROBE_OVERHEAD
    u4_estimate = parameter_count * _UNIFORM4_BYTES_PER_PARAM * _PROBE_OVERHEAD
    if not math.isfinite(bf16_estimate) or not math.isfinite(u4_estimate):
        raise PlanningError("probe capacity estimate exceeds the supported numeric range")
    bf16 = max(1, math.ceil(bf16_estimate))
    u4 = max(1, math.ceil(u4_estimate))
    lite = max(1, math.ceil(u4 * 1.15))  # uniform-4 resident + modest probe workspace
    modes = [
        _mode_assessment(
            ProbeMode.BF16_FULL,
            required_bytes=bf16,
            available_bytes=available,
            headroom_fraction=resolved_headroom,
            evidence_kind=EvidenceKind.MEASURED,
            release_quality=True,
            notes=[
                "Gold-standard: bf16 weights resident; per-tensor probes against full precision.",
                "Required for certification-grade measured sensitivity when the host allows it.",
            ],
        ),
        _mode_assessment(
            ProbeMode.MEASURED_LITE,
            required_bytes=lite,
            available_bytes=available,
            headroom_fraction=resolved_headroom,
            evidence_kind=EvidenceKind.MEASURED_DEVELOPMENT,
            release_quality=False,
            notes=[
                "Fewer bit/group candidates; labeled measured_development.",
                "Use convert ladder measured-lite.",
            ],
        ),
        _mode_assessment(
            ProbeMode.STREAMING_PARTIAL,
            required_bytes=u4,
            available_bytes=available,
            headroom_fraction=resolved_headroom,
            evidence_kind=EvidenceKind.MEASURED_DEVELOPMENT,
            release_quality=False,
            notes=[
                "Uniform-4bit-scale resident footprint with partial/streaming probes.",
                "Weaker signal than bf16-full; never auto-promoted to release measured.",
            ],
        ),
        _mode_assessment(
            ProbeMode.PRIOR_ONLY,
            required_bytes=0,
            available_bytes=available,
            headroom_fraction=resolved_headroom,
            evidence_kind=EvidenceKind.ARCHITECTURE_PRIOR,
            release_quality=False,
            notes=[
                "Architecture priors + multi-group planner grid; development evidence only.",
                "Always available; pass --allow-unmeasured for plan/convert.",
            ],
        ),
    ]
    recommended = ProbeMode.PRIOR_ONLY
    for preferred in (
        ProbeMode.BF16_FULL,
        ProbeMode.MEASURED_LITE,
        ProbeMode.STREAMING_PARTIAL,
        ProbeMode.PRIOR_ONLY,
    ):
        match = next(item for item in modes if item.mode is preferred)
        if match.feasible:
            recommended = preferred
            break
    warnings: list[str] = []
    if available is None:
        warnings.append(
            "Host unified memory could not be detected; only prior-only is confirmed feasible."
        )
    if recommended is not ProbeMode.BF16_FULL:
        warnings.append(
            f"Recommended mode is {recommended.value}; release-quality measured "
            "probes may be unavailable on this host."
        )
    return ProbeCapacityReport(
        model=model,
        parameter_count=parameter_count,
        available_memory_bytes=available,
        headroom_fraction=resolved_headroom,
        recommended_mode=recommended,
        modes=modes,
        warnings=warnings,
    )


def assess_probe_capacity_from_inventory(
    inventory: Inventory | str | Path,
    *,
    available_memory_bytes: int | None = None,
    headroom_fraction: float = _DEFAULT_HEADROOM,
) -> ProbeCapacityReport:
    """Assess capacity from an Inventory object or JSON path."""
    report = inventory if isinstance(inventory, Inventory) else load_model(inventory, Inventory)
    try:
        report = Inventory.model_validate(report.model_dump(mode="python"))
    except ValidationError as exc:
        raise PlanningError(f"invalid inventory for probe capacity: {exc}") from exc
    if not report.tensors:
        raise PlanningError("probe capacity requires a non-empty tensor inventory")
    tensor_parameters = sum(tensor.parameters for tensor in report.tensors)
    if report.total_parameters != tensor_parameters:
        raise PlanningError(
            "inventory total_parameters does not match its tensor parameter records"
        )
    return assess_probe_capacity(
        parameter_count=report.total_parameters,
        model=report.model,
        available_memory_bytes=available_memory_bytes,
        headroom_fraction=headroom_fraction,
    )


def probe_capacity_markdown(report: ProbeCapacityReport) -> str:
    """Render a short operator-facing capacity summary."""
    avail = (
        "unknown"
        if report.available_memory_bytes is None
        else f"{report.available_memory_bytes / 1024**3:.2f} GiB"
    )
    model = report.model.model_id if report.model is not None else "unspecified"
    lines = [
        "# Probe capacity",
        "",
        f"- Model: `{model}`",
        f"- Parameters: {report.parameter_count:,}",
        f"- Available memory: {avail}",
        f"- Headroom fraction: {report.headroom_fraction:.2f}",
        f"- **Recommended mode:** `{report.recommended_mode.value}`",
        "",
        "| Mode | Feasible | Est. bytes | Evidence | Release-eligible | Reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode in report.modes:
        lines.append(
            f"| `{mode.mode.value}` | {'yes' if mode.feasible else 'no'} | "
            f"{mode.estimated_bytes / 1024**3:.2f} GiB | `{mode.evidence_kind.value}` | "
            f"{'yes' if mode.release_quality_eligible else 'no'} | {mode.reason} |"
        )
    if report.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)
