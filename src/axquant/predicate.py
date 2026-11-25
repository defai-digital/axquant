from __future__ import annotations

from collections.abc import Callable
from typing import Any

from axquant.errors import PlanningError
from axquant.model_support import require_model_supported
from axquant.module_paths import (
    fused_expert_module,
    mlx_module_aliases,
    packed_expert_runtime_modules,
)
from axquant.product_formats import (
    MLX_GROUP_SIZE,
    mlx_storage_mode,
    require_mlx_product_storage,
)
from axquant.schema import Allocation, QuantizationPlan

# AXQ-070: convert executes unrefined MXFP4 / MXFP8 packing only. Fused and
# packed SwitchLinear stacks take the same packing (validated end to end:
# quantize_model -> save -> load -> forward).
FUSED_STACK_METHODS = frozenset({"affine", "mxfp4"})
MXFP4_GROUP_SIZE = MLX_GROUP_SIZE
# ``--q-mode`` no longer selects storage; it is accepted for compatibility.
_Q_MODES = frozenset({"mxfp4", "mxfp8"})


def fused_stack_method_allowed(method: str) -> bool:
    """Whether convert can execute *method* on a fused or packed expert stack."""

    return method in FUSED_STACK_METHODS


def normalize_q_mode(q_mode: str | None) -> str:
    """Validate a legacy ``--q-mode`` value; storage follows allocation bits."""

    requested = str(q_mode or "mxfp4").lower()
    if requested not in _Q_MODES:
        raise PlanningError(
            f"unsupported convert q-mode {q_mode!r}: AXQ-070 removed affine storage; "
            "MLX storage follows allocation bits (4-bit MXFP4, 8-bit MXFP8)"
        )
    return requested


def allocation_physical_mode(allocation: Allocation, q_mode: str | None = None) -> str:
    """Return the MLX ``to_quantized`` mode for one quantized allocation.

    AXQ-070: 4-bit allocations are stored as MXFP4 and 8-bit allocations as
    MXFP8, both at group 32. Every other width or method is refused. MXFP8 is
    a physical packing of the 8-bit slot; affine sensitivity measured for that
    slot does not substantiate an MXFP8 quality claim.
    """

    normalize_q_mode(q_mode)
    return mlx_storage_mode(allocation)


def allocation_quant_params(allocation: Allocation, q_mode: str | None = None) -> dict[str, Any]:
    mode = allocation_physical_mode(allocation, q_mode)
    return {"group_size": allocation.group_size, "bits": allocation.bits, "mode": mode}


def _without_weight_suffix(path: str) -> str:
    return path[: -len(".weight")] if path.endswith(".weight") else path


class PlanPredicate:
    def __init__(
        self,
        plan: QuantizationPlan,
        *,
        q_mode: str | None = None,
    ) -> None:
        require_model_supported(plan.source_model.model_id, plan.source_model.local_path)
        self._q_mode = normalize_q_mode(q_mode)
        require_mlx_product_storage(plan.assignments)
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
                if allocation.bits < 16 and not fused_stack_method_allowed(allocation.method.value):
                    raise PlanningError(
                        f"packed expert tensor {module_path} requires MXFP4 or MXFP8 "
                        f"packing; got {allocation.method.value}"
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
            if members[0].bits < 16 and not fused_stack_method_allowed(members[0].method.value):
                raise PlanningError(
                    f"fused expert module {fused} requires MXFP4 or MXFP8 "
                    f"packing; got {members[0].method.value}"
                )
        self._aliases: dict[str, Allocation] = {}
        for module_path, allocation in self._assignments.items():
            aliases = set(
                mlx_module_aliases(module_path, adapter_id=plan.architecture_profile.adapter_id)
            )
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
        # Refinement (AWQ / GPTQ / DWQ) is outside AXQ-070; the execution
        # manifest still reads these per-allocation maps, which stay empty.
        self.dwq_metadata: dict[str, dict[str, Any]] = {}
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
        return allocation_quant_params(allocation, self._q_mode)

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
    q_mode: str | None = None,
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
    return PlanPredicate(plan, q_mode=q_mode)


QuantPredicate = Callable[[str, Any], bool | dict[str, Any]]
