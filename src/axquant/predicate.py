from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from axquant.awq import apply_mlx_awq_scale as _apply_awq_scale
from axquant.dwq import apply_mlx_dwq_clip as _apply_dwq_clip
from axquant.errors import PlanningError
from axquant.module_paths import fused_expert_module, mlx_module_aliases
from axquant.schema import Allocation, QuantizationPlan

_EXECUTABLE_METHODS = frozenset({"affine", "dwq", "awq"})


def _without_weight_suffix(path: str) -> str:
    return path[: -len(".weight")] if path.endswith(".weight") else path


class PlanPredicate:
    def __init__(
        self,
        plan: QuantizationPlan,
        *,
        execute_refinement: bool = True,
        awq_activations: Mapping[str, Any] | None = None,
    ) -> None:
        self._assignments = {
            _without_weight_suffix(allocation.module_path): allocation
            for allocation in plan.assignments
        }
        if len(self._assignments) != len(plan.assignments):
            raise PlanningError("plan contains duplicate module paths")
        # Per-expert allocations fuse into one MLX-LM switch module that is
        # quantized as a single unit, so every member of a fused group must
        # agree on precision and every member counts as visited when the
        # fused module is quantized.
        self._fused_members: dict[str, list[Allocation]] = {}
        for module_path, allocation in self._assignments.items():
            fused = fused_expert_module(module_path)
            if fused is not None:
                self._fused_members.setdefault(fused, []).append(allocation)
        for fused, members in self._fused_members.items():
            signatures = {
                (member.bits, member.method.value, member.group_size) for member in members
            }
            if len(signatures) > 1:
                raise PlanningError(
                    f"fused expert module {fused} mixes precisions {sorted(signatures)}; "
                    "every expert in a switch group must share one assignment"
                )
            if members[0].bits < 16 and members[0].method.value != "affine":
                raise PlanningError(
                    f"fused expert module {fused} requires the affine method; "
                    f"got {members[0].method.value}"
                )
        self._aliases: dict[str, Allocation] = {}
        for module_path, allocation in self._assignments.items():
            aliases = set(mlx_module_aliases(module_path))
            fused = fused_expert_module(module_path)
            if fused is not None:
                for fused_alias in mlx_module_aliases(fused):
                    aliases.add(fused_alias)
            for alias in aliases:
                existing = self._aliases.get(alias)
                if existing is not None and existing.module_path != allocation.module_path:
                    if fused is not None and fused_expert_module(existing.module_path) == fused:
                        # Members of one fused group intentionally share the
                        # fused alias; keep the first representative.
                        continue
                    raise PlanningError(f"plan module alias is ambiguous: {alias}")
                self._aliases[alias] = allocation
        self.matched: set[str] = set()
        self._execute_refinement = execute_refinement
        self._awq_activations = dict(awq_activations or {})
        self.dwq_metadata: dict[str, dict[str, float | int]] = {}
        self.awq_metadata: dict[str, dict[str, float | int | list[float]]] = {}

    def lookup(self, path: str) -> Allocation | None:
        normalized = _without_weight_suffix(path)
        if allocation := self._aliases.get(normalized):
            return allocation
        suffix_matches = [
            allocation
            for module_path, allocation in self._aliases.items()
            if module_path.endswith(f".{normalized}") or normalized.endswith(f".{module_path}")
        ]
        suffix_matches = list(
            {allocation.module_path: allocation for allocation in suffix_matches}.values()
        )
        if len(suffix_matches) > 1:
            raise PlanningError(f"ambiguous module path {path}")
        return suffix_matches[0] if suffix_matches else None

    def _resolve_awq_calibration(self, allocation: Allocation) -> Any:
        candidates = [
            allocation.module_path,
            _without_weight_suffix(allocation.module_path),
            allocation.tensor,
            _without_weight_suffix(allocation.tensor),
        ]
        for key in candidates:
            if key in self._awq_activations:
                return self._awq_activations[key]
        for alias in mlx_module_aliases(allocation.module_path):
            if alias in self._awq_activations:
                return self._awq_activations[alias]
            if _without_weight_suffix(alias) in self._awq_activations:
                return self._awq_activations[_without_weight_suffix(alias)]
        raise PlanningError(
            f"AWQ conversion requires calibration activations for module {allocation.module_path}"
        )

    def __call__(self, path: str, module: Any, *args: Any) -> bool | dict[str, Any]:
        del args
        allocation = self.lookup(path)
        if allocation is None:
            return False
        self.matched.add(allocation.module_path)
        fused = fused_expert_module(allocation.module_path)
        if fused is not None:
            # Quantizing the fused switch module covers every member expert.
            for member in self._fused_members.get(fused, ()):
                self.matched.add(member.module_path)
        if allocation.bits == 16:
            return False
        if allocation.method.value == "dwq" and self._execute_refinement:
            self.dwq_metadata[allocation.module_path] = _apply_dwq_clip(module)
        if allocation.method.value == "awq" and self._execute_refinement:
            group_size = allocation.group_size
            if group_size is None:
                raise PlanningError(
                    f"AWQ allocation is missing group_size: {allocation.module_path}"
                )
            calibration = self._resolve_awq_calibration(allocation)
            self.awq_metadata[allocation.module_path] = _apply_awq_scale(
                module,
                activations=calibration,
                bits=allocation.bits,
                group_size=group_size,
            )
        return {
            "group_size": allocation.group_size,
            "bits": allocation.bits,
            "mode": "affine",
        }

    def unmatched_quantized_modules(self) -> set[str]:
        expected = {
            allocation.module_path
            for allocation in self._assignments.values()
            if allocation.bits < 16
        }
        return expected - self.matched


def build_quant_predicate(
    plan: QuantizationPlan,
    *,
    execute_refinement: bool = True,
    awq_activations: Mapping[str, Any] | None = None,
) -> PlanPredicate:
    undeclared = {
        allocation.method.value
        for allocation in plan.assignments
        if allocation.bits < 16 and allocation.method not in plan.hardware.supported_methods
    }
    if undeclared:
        raise PlanningError(
            f"plan uses methods absent from its hardware profile: {sorted(undeclared)}"
        )
    unsupported = {
        allocation.method.value
        for allocation in plan.assignments
        if allocation.bits < 16 and allocation.method.value not in _EXECUTABLE_METHODS
    }
    if unsupported:
        raise PlanningError(
            f"the MLX-LM predicate backend cannot execute methods {sorted(unsupported)}"
        )
    awq_assignments = [
        allocation
        for allocation in plan.assignments
        if allocation.bits < 16 and allocation.method.value == "awq"
    ]
    if awq_assignments and execute_refinement and not awq_activations:
        missing = sorted(allocation.module_path for allocation in awq_assignments)
        raise PlanningError(
            f"AWQ conversion requires calibration activations for modules: {missing[:10]}"
        )
    return PlanPredicate(
        plan,
        execute_refinement=execute_refinement,
        awq_activations=awq_activations,
    )


QuantPredicate = Callable[[str, Any], bool | dict[str, Any]]
