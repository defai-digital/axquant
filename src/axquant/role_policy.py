"""Role-aware quantization preferences for measured planning (AXQ-028 QP1).

Preferences only reorder or soft-bias legal candidates. They never lower
protection floors from AXQ-007 / AXQ-026.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from axquant.package_data import load_package_yaml
from axquant.schema import EvidenceKind, QuantMethod, TensorRole


@dataclass(frozen=True)
class RolePreference:
    """Preferred packing choices for a tensor role under measured evidence."""

    preferred_methods: tuple[QuantMethod, ...]
    preferred_max_group_size: int
    # Relative loss margin: prefer preferred method if loss <= best * (1 + margin).
    method_loss_margin: float = 0.05
    # Soft discount applied only to knapsack ranking under measured evidence.
    ranking_discount: float = 0.97


def _require_mapping(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return payload


def _load_role_preferences() -> dict[TensorRole, RolePreference]:
    raw = _require_mapping(load_package_yaml("role_preferences.yaml"), "role_preferences.yaml")
    defaults = _require_mapping(raw.get("defaults"), "role_preferences.yaml defaults")
    preferences = _require_mapping(raw.get("preferences"), "role_preferences.yaml preferences")
    default_margin = float(defaults.get("method_loss_margin", 0.05))
    default_discount = float(defaults.get("ranking_discount", 0.97))
    resolved: dict[TensorRole, RolePreference] = {}
    for role_key, entry in preferences.items():
        role = TensorRole(str(role_key))
        item = _require_mapping(entry, f"role_preferences.yaml preferences.{role_key}")
        methods_raw = item.get("preferred_methods")
        if not isinstance(methods_raw, list) or not methods_raw:
            raise ValueError(f"role_preferences.yaml preferences.{role_key}.preferred_methods")
        methods = tuple(QuantMethod(str(method)) for method in methods_raw)
        resolved[role] = RolePreference(
            preferred_methods=methods,
            preferred_max_group_size=int(item["preferred_max_group_size"]),
            method_loss_margin=float(item.get("method_loss_margin", default_margin)),
            ranking_discount=float(item.get("ranking_discount", default_discount)),
        )
    return resolved


# Sensitive roles favor finer groups and AWQ when measured candidates exist.
ROLE_PREFERENCES: dict[TensorRole, RolePreference] = _load_role_preferences()


def role_preferences_active(evidence_kind: EvidenceKind) -> bool:
    """Role soft-preferences apply only to measured (or measured-development) evidence."""
    return evidence_kind in {
        EvidenceKind.MEASURED,
        EvidenceKind.MEASURED_DEVELOPMENT,
        EvidenceKind.IMPORTED,
    }


def method_preference_rank(role: TensorRole, method: QuantMethod) -> int:
    """Lower is better. Unknown methods rank last."""
    pref = ROLE_PREFERENCES.get(role)
    if pref is None:
        return 100
    try:
        return pref.preferred_methods.index(method)
    except ValueError:
        return 50 + ord(method.value[0])


def group_preference_rank(role: TensorRole, group_size: int | None) -> int:
    """Lower is better. Groups at or below the preferred max rank ahead of coarser ones."""
    if group_size is None:
        return 0
    pref = ROLE_PREFERENCES.get(role)
    if pref is None:
        return group_size
    if group_size <= pref.preferred_max_group_size:
        return group_size  # among preferred, smaller still slightly preferred
    return 1000 + group_size


def prefer_method_on_tie(
    role: TensorRole,
    *,
    current_method: QuantMethod,
    current_loss: float,
    candidate_method: QuantMethod,
    candidate_loss: float,
    evidence_kind: EvidenceKind,
    best_loss_at_key: float | None = None,
) -> bool:
    """Return True when candidate should replace current at the same storage key.

    Under measured evidence, the role's top preferred method wins when its loss is
    within ``method_loss_margin`` of the best loss at that storage key — even if
    another method is slightly better. Floors and legality are enforced upstream.
    """
    if role_preferences_active(evidence_kind):
        pref = ROLE_PREFERENCES.get(role)
        if pref is not None and pref.preferred_methods:
            preferred = pref.preferred_methods[0]
            ceiling = (
                best_loss_at_key
                if best_loss_at_key is not None
                else min(current_loss, candidate_loss)
            ) * (1.0 + pref.method_loss_margin)
            current_ok = current_method == preferred and current_loss <= ceiling + 1e-15
            candidate_ok = candidate_method == preferred and candidate_loss <= ceiling + 1e-15
            if candidate_ok and not current_ok:
                return True
            if current_ok and not candidate_ok:
                return False
            if current_ok and candidate_ok:
                return candidate_loss < current_loss - 1e-15 or (
                    abs(candidate_loss - current_loss) <= 1e-15
                    and candidate_method.value < current_method.value
                )
    if candidate_loss < current_loss - 1e-15:
        return True
    if candidate_loss > current_loss + 1e-15:
        return False
    if method_preference_rank(role, candidate_method) < method_preference_rank(
        role, current_method
    ):
        return True
    if method_preference_rank(role, candidate_method) > method_preference_rank(
        role, current_method
    ):
        return False
    return candidate_method.value < current_method.value


def ranking_loss(
    *,
    loss: float,
    role: TensorRole,
    method: QuantMethod,
    group_size: int | None,
    evidence_kind: EvidenceKind,
) -> float:
    """Soft-discount preferred options for knapsack ranking (measured only)."""
    if not role_preferences_active(evidence_kind):
        return loss
    pref = ROLE_PREFERENCES.get(role)
    if pref is None:
        return loss
    adjusted = loss
    if pref.preferred_methods and method == pref.preferred_methods[0]:
        adjusted *= pref.ranking_discount
    if group_size is not None and group_size <= pref.preferred_max_group_size:
        adjusted *= pref.ranking_discount
    return adjusted
