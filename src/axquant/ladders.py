"""Named convert ladders: prior → measured-lite → measured-full → refine (P1).

Ladders package target BPW, candidate bits/groups/methods, evidence posture, and
relative cost so CLI defaults stay simple without hiding the evidence contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from axquant.errors import PlanningError
from axquant.package_data import load_package_yaml
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


def _require_mapping(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return payload


def _as_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    return tuple(int(item) for item in value)


def _as_str_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return tuple(str(item) for item in value)


def _load_ladders() -> dict[ConvertLadderName, ConvertLadder]:
    raw = _require_mapping(load_package_yaml("convert_ladders.yaml"), "convert_ladders.yaml")
    ladders_raw = _require_mapping(raw.get("ladders"), "convert_ladders.yaml ladders")
    resolved: dict[ConvertLadderName, ConvertLadder] = {}
    for key, entry in ladders_raw.items():
        name = ConvertLadderName(str(key))
        item = _require_mapping(entry, f"convert_ladders.yaml ladders.{key}")
        methods = tuple(
            QuantMethod(str(method)) for method in _as_str_tuple(item.get("candidate_methods"), key)
        )
        notes = _as_str_tuple(item.get("notes"), f"{key}.notes")
        description = str(item["description"]).strip()
        resolved[name] = ConvertLadder(
            name=name,
            evidence_kind=EvidenceKind(str(item["evidence_kind"])),
            allow_unmeasured=bool(item["allow_unmeasured"]),
            candidate_bits=_as_int_tuple(item.get("candidate_bits"), f"{key}.candidate_bits"),
            candidate_group_sizes=_as_int_tuple(
                item.get("candidate_group_sizes"), f"{key}.candidate_group_sizes"
            ),
            candidate_methods=methods,
            requires_calibration=bool(item["requires_calibration"]),
            requires_measured_sensitivity=bool(item["requires_measured_sensitivity"]),
            requires_refinement=bool(item["requires_refinement"]),
            default_target_bpw=float(item["default_target_bpw"]),
            estimated_relative_cost=float(item["estimated_relative_cost"]),
            estimated_ram_multiplier_vs_4bit=float(item["estimated_ram_multiplier_vs_4bit"]),
            description=description,
            notes=notes,
        )
    return resolved


def _ladder_order() -> tuple[ConvertLadderName, ...]:
    raw = _require_mapping(load_package_yaml("convert_ladders.yaml"), "convert_ladders.yaml")
    order_raw = raw.get("order")
    if not isinstance(order_raw, list) or not order_raw:
        raise ValueError("convert_ladders.yaml order must be a non-empty list")
    return tuple(ConvertLadderName(str(name)) for name in order_raw)


_LADDERS: dict[ConvertLadderName, ConvertLadder] = _load_ladders()
_LADDER_ORDER: tuple[ConvertLadderName, ...] = _ladder_order()


def list_ladders() -> list[ConvertLadder]:
    """Return convert ladders in recommended progression order."""
    return [_LADDERS[name] for name in _LADDER_ORDER]


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
