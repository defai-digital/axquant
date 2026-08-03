"""Named convert ladders: prior → measured-lite → measured-full → refine (P1).

Ladders package target BPW, candidate bits/groups/methods, evidence posture, and
relative cost so CLI defaults stay simple without hiding the evidence contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from axquant.errors import PlanningError
from axquant.schema import (
    ConvertLadderName,
    EvidenceKind,
    PlanRequest,
    ProfileName,
    QuantMethod,
)


@dataclass(frozen=True, slots=True)
class ConvertLadder:
    """Resolved convert-ladder specification."""

    name: ConvertLadderName
    evidence_kind: EvidenceKind
    allow_unmeasured: bool
    candidate_bits: tuple[int, ...]
    candidate_group_sizes: tuple[int, ...]
    candidate_methods: tuple[QuantMethod, ...]
    requires_calibration: bool
    requires_measured_sensitivity: bool
    requires_refinement: bool
    default_target_bpw: float
    estimated_relative_cost: float
    estimated_ram_multiplier_vs_4bit: float
    description: str
    notes: tuple[str, ...]


_LADDERS: dict[ConvertLadderName, ConvertLadder] = {
    ConvertLadderName.PRIOR: ConvertLadder(
        name=ConvertLadderName.PRIOR,
        evidence_kind=EvidenceKind.ARCHITECTURE_PRIOR,
        allow_unmeasured=True,
        candidate_bits=(4, 6, 8, 16),
        candidate_group_sizes=(32, 64),
        candidate_methods=(QuantMethod.AFFINE, QuantMethod.BF16),
        requires_calibration=False,
        requires_measured_sensitivity=False,
        requires_refinement=False,
        default_target_bpw=4.8,
        estimated_relative_cost=0.01,
        estimated_ram_multiplier_vs_4bit=0.0,
        description=(
            "Architecture-prior multi-group plan. Development evidence only; "
            "release claims require measured sensitivity or a measured recipe bundle."
        ),
        notes=(
            "Always available (no forward probes).",
            "Planner grid includes group sizes 32 and 64 (AXQ-028 / P0).",
        ),
    ),
    ConvertLadderName.MEASURED_LITE: ConvertLadder(
        name=ConvertLadderName.MEASURED_LITE,
        evidence_kind=EvidenceKind.MEASURED_DEVELOPMENT,
        allow_unmeasured=False,
        candidate_bits=(4, 8, 16),
        candidate_group_sizes=(64,),
        candidate_methods=(QuantMethod.AFFINE, QuantMethod.BF16),
        requires_calibration=True,
        requires_measured_sensitivity=True,
        requires_refinement=False,
        default_target_bpw=5.0,
        estimated_relative_cost=0.25,
        estimated_ram_multiplier_vs_4bit=1.0,
        description=(
            "Lightweight measured probes: fewer bit widths and a single group size. "
            "Produces measured_development evidence suitable for iteration, not certification."
        ),
        notes=(
            "Prefer when probe capacity is measured-lite or streaming-partial.",
            "Does not run AWQ/DWQ refine.",
        ),
    ),
    ConvertLadderName.MEASURED_FULL: ConvertLadder(
        name=ConvertLadderName.MEASURED_FULL,
        evidence_kind=EvidenceKind.MEASURED,
        allow_unmeasured=False,
        candidate_bits=(4, 6, 8, 16),
        candidate_group_sizes=(32, 64, 128),
        candidate_methods=(QuantMethod.AFFINE, QuantMethod.DWQ, QuantMethod.BF16),
        requires_calibration=True,
        requires_measured_sensitivity=True,
        requires_refinement=False,
        default_target_bpw=4.8,
        estimated_relative_cost=1.0,
        estimated_ram_multiplier_vs_4bit=4.0,
        description=(
            "Full measured grid over bits x groups x affine/DWQ methods. "
            "Release-quality when the probe backend records measured evidence."
        ),
        notes=(
            "Requires probe capacity bf16-full (or an explicit measured protocol).",
            "Use refine-awq-dwq after this when channel scales are desired.",
        ),
    ),
    ConvertLadderName.REFINE_AWQ_DWQ: ConvertLadder(
        name=ConvertLadderName.REFINE_AWQ_DWQ,
        evidence_kind=EvidenceKind.MEASURED,
        allow_unmeasured=False,
        candidate_bits=(4, 6, 8, 16),
        candidate_group_sizes=(32, 64, 128),
        candidate_methods=(
            QuantMethod.AFFINE,
            QuantMethod.AWQ,
            QuantMethod.DWQ,
            QuantMethod.GPTQ,
            QuantMethod.BF16,
        ),
        requires_calibration=True,
        requires_measured_sensitivity=True,
        requires_refinement=True,
        default_target_bpw=4.8,
        estimated_relative_cost=1.6,
        estimated_ram_multiplier_vs_4bit=4.0,
        description=(
            "Measured full grid plus AWQ/DWQ refinement candidates. "
            "Highest convert cost; best scale/outlier strategy coverage."
        ),
        notes=(
            "Run after measured-full or bind via refine-select / refine-run.",
            "Still subject to EvidenceKind and release gates.",
        ),
    ),
}


def list_ladders() -> list[ConvertLadder]:
    """Return convert ladders in recommended progression order."""
    order = (
        ConvertLadderName.PRIOR,
        ConvertLadderName.MEASURED_LITE,
        ConvertLadderName.MEASURED_FULL,
        ConvertLadderName.REFINE_AWQ_DWQ,
    )
    return [_LADDERS[name] for name in order]


def get_ladder(name: ConvertLadderName | str) -> ConvertLadder:
    """Resolve a ladder by enum or CLI string."""
    if isinstance(name, ConvertLadderName):
        key = name
    else:
        try:
            key = ConvertLadderName(name)
        except ValueError as exc:
            choices = ", ".join(item.value for item in ConvertLadderName)
            raise PlanningError(
                f"unknown convert ladder {name!r}; choose one of {choices}"
            ) from exc
    return _LADDERS[key]


def plan_request_for_ladder(
    ladder: ConvertLadder | ConvertLadderName | str,
    *,
    profile: ProfileName,
    target_bpw: float | None = None,
    allow_unmeasured: bool | None = None,
) -> PlanRequest:
    """Build a PlanRequest from a ladder with optional overrides."""
    resolved = ladder if isinstance(ladder, ConvertLadder) else get_ladder(ladder)
    return PlanRequest(
        profile=profile,
        target_bpw=resolved.default_target_bpw if target_bpw is None else target_bpw,
        candidate_bits=resolved.candidate_bits,
        group_size=resolved.candidate_group_sizes[0],
        candidate_group_sizes=resolved.candidate_group_sizes,
        candidate_methods=resolved.candidate_methods,
        allow_unmeasured=(
            resolved.allow_unmeasured if allow_unmeasured is None else allow_unmeasured
        ),
        target_mode="low-memory" if resolved.name is ConvertLadderName.PRIOR else "balanced",
    )


def ladder_markdown(ladders: list[ConvertLadder] | None = None) -> str:
    """Human-readable ladder table for CLI and docs."""
    rows = ladders if ladders is not None else list_ladders()
    lines = [
        "# AXQuant convert ladders",
        "",
        "Progress from fast architecture-prior development converts to measured, "
        "refined release candidates. Evidence labels never upgrade automatically.",
        "",
        "| Ladder | Evidence | Target BPW | Bits | Groups | Methods | Rel. cost | Needs cal |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ladder in rows:
        methods = ",".join(method.value for method in ladder.candidate_methods)
        groups = ",".join(str(size) for size in ladder.candidate_group_sizes)
        bits = ",".join(str(bit) for bit in ladder.candidate_bits)
        lines.append(
            f"| `{ladder.name.value}` | `{ladder.evidence_kind.value}` | "
            f"{ladder.default_target_bpw:.1f} | {bits} | {groups} | {methods} | "
            f"{ladder.estimated_relative_cost:.2f} | "
            f"{'yes' if ladder.requires_calibration else 'no'} |"
        )
    lines.append("")
    for ladder in rows:
        lines.append(f"## `{ladder.name.value}`")
        lines.append("")
        lines.append(ladder.description)
        lines.append("")
        for note in ladder.notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)
