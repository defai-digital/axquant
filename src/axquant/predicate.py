from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from axquant.awq import apply_mlx_awq_scale as _apply_awq_scale
from axquant.dwq import apply_mlx_dwq_clip as _apply_dwq_clip
from axquant.errors import PlanningError
from axquant.gptq import apply_mlx_gptq_refine as _apply_gptq_refine
from axquant.module_paths import (
    fused_expert_module,
    mlx_module_aliases,
    packed_expert_runtime_modules,
)
from axquant.schema import Allocation, QuantizationPlan

_EXECUTABLE_METHODS = frozenset({"affine", "dwq", "awq", "gptq", "gptq-act"})


def _without_weight_suffix(path: str) -> str:
    return path[: -len(".weight")] if path.endswith(".weight") else path


class PlanPredicate:
    def __init__(
        self,
        plan: QuantizationPlan,
        *,
        execute_refinement: bool = True,
        calibration_activations: Mapping[str, Any] | None = None,
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
        self._packed_requirements: dict[str, tuple[frozenset[str], ...]] = {}
        self._packed_seen: dict[str, set[int]] = {}
        for module_path, allocation in self._assignments.items():
            fused = fused_expert_module(module_path)
            if fused is not None:
                self._fused_members.setdefault(fused, []).append(allocation)
            packed_modules = packed_expert_runtime_modules(module_path)
            if packed_modules:
                if allocation.bits < 16 and allocation.method.value != "affine":
                    raise PlanningError(
                        f"packed expert tensor {module_path} requires the affine method; "
                        f"got {allocation.method.value}"
                    )
                self._packed_requirements[module_path] = tuple(
                    frozenset(mlx_module_aliases(runtime_module))
                    for runtime_module in packed_modules
                )
                self._packed_seen[module_path] = set()
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
                    existing_path = _without_weight_suffix(existing.module_path)
                    if fused is not None and fused_expert_module(existing_path) == fused:
                        # Members of one fused group intentionally share the
                        # fused alias; keep the first representative.
                        continue
                    raise PlanningError(f"plan module alias is ambiguous: {alias}")
                self._aliases[alias] = allocation
        self.matched: set[str] = set()
        self._execute_refinement = execute_refinement
        self._calibration_activations = dict(calibration_activations or {})
        self.dwq_metadata: dict[str, dict[str, float | int]] = {}
        self.method_metadata: dict[str, dict[str, Any]] = {}

    def lookup(self, path: str) -> Allocation | None:
        normalized = _without_weight_suffix(path)
        if allocation := self._aliases.get(normalized):
            return allocation
        # Prefer longest alias that is a path suffix of the runtime module.
        # Bare names like ``norm`` would otherwise match every ``*.norm`` path
        # (DeepSeek V4 has both root norms and per-layer compressor norms).
        ending_matches = [
            (module_path, allocation)
            for module_path, allocation in self._aliases.items()
            if normalized == module_path or normalized.endswith(f".{module_path}")
        ]
        if ending_matches:
            ending_matches.sort(key=lambda item: len(item[0]), reverse=True)
            best_len = len(ending_matches[0][0])
            best = [
                allocation
                for module_path, allocation in ending_matches
                if len(module_path) == best_len
            ]
            unique = list({allocation.module_path: allocation for allocation in best}.values())
            if len(unique) == 1:
                return unique[0]
            if len(unique) > 1:
                raise PlanningError(f"ambiguous module path {path}")
        prefix_matches = [
            allocation
            for module_path, allocation in self._aliases.items()
            if module_path.endswith(f".{normalized}")
        ]
        prefix_matches = list(
            {allocation.module_path: allocation for allocation in prefix_matches}.values()
        )
        if len(prefix_matches) > 1:
            raise PlanningError(f"ambiguous module path {path}")
        return prefix_matches[0] if prefix_matches else None

    def _resolve_calibration(self, allocation: Allocation) -> Any:
        candidates = [
            allocation.module_path,
            _without_weight_suffix(allocation.module_path),
            allocation.tensor,
            _without_weight_suffix(allocation.tensor),
        ]
        for key in candidates:
            if key in self._calibration_activations:
                return self._calibration_activations[key]
        for alias in mlx_module_aliases(allocation.module_path):
            if alias in self._calibration_activations:
                return self._calibration_activations[alias]
            if _without_weight_suffix(alias) in self._calibration_activations:
                return self._calibration_activations[_without_weight_suffix(alias)]
        raise PlanningError(
            "AWQ/GPTQ conversion requires calibration activations for module "
            f"{allocation.module_path}; run capture-activations to produce them"
        )

    def __call__(self, path: str, module: Any, *args: Any) -> bool | dict[str, Any]:
        del args
        normalized_path = _without_weight_suffix(path)
        allocation = self.lookup(normalized_path)
        if allocation is None:
            return False
        allocation_path = _without_weight_suffix(allocation.module_path)
        packed_requirements = self._packed_requirements.get(allocation_path)
        if packed_requirements is None:
            self.matched.add(allocation.module_path)
        else:
            seen = self._packed_seen[allocation_path]
            for index, aliases in enumerate(packed_requirements):
                if normalized_path in aliases or any(
                    alias.endswith(f".{normalized_path}") or normalized_path.endswith(f".{alias}")
                    for alias in aliases
                ):
                    seen.add(index)
            if len(seen) == len(packed_requirements):
                self.matched.add(allocation.module_path)
        fused = fused_expert_module(allocation_path)
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
            calibration = self._resolve_calibration(allocation)
            self.method_metadata[allocation.module_path] = _apply_awq_scale(
                module,
                activations=calibration,
                bits=allocation.bits,
                group_size=group_size,
            )
        if allocation.method.value in ("gptq", "gptq-act") and self._execute_refinement:
            group_size = allocation.group_size
            if group_size is None:
                raise PlanningError(
                    f"GPTQ allocation is missing group_size: {allocation.module_path}"
                )
            calibration = self._resolve_calibration(allocation)
            self.method_metadata[allocation.module_path] = _apply_gptq_refine(
                module,
                activations=calibration,
                bits=allocation.bits,
                group_size=group_size,
                act_order=allocation.method.value == "gptq-act",
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
    calibration_activations: Mapping[str, Any] | None = None,
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
    calibration_assignments = [
        allocation
        for allocation in plan.assignments
        if allocation.bits < 16 and allocation.method.value in ("awq", "gptq", "gptq-act")
    ]
    if calibration_assignments and execute_refinement and not calibration_activations:
        missing = sorted(allocation.module_path for allocation in calibration_assignments)
        raise PlanningError(
            "AWQ/GPTQ conversion requires calibration activations for modules: "
            f"{missing[:10]}; run capture-activations to produce them"
        )
    return PlanPredicate(
        plan,
        execute_refinement=execute_refinement,
        calibration_activations=calibration_activations,
    )


QuantPredicate = Callable[[str, Any], bool | dict[str, Any]]
