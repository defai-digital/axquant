"""Frozen pre-MXFP4 schema envelopes (AXQ-042 freeze / AXQ-046 MH5).

Adding ``QuantMethod.MXFP4`` widens the shared ``$defs/QuantMethod`` rendered
into 12 frozen contracts. Under the freeze rule each bumped version keeps its
original envelope: the legacy classes below pin the pre-MXFP4 ``QuantMethod``
enum and the method-bearing nested models, so the old snapshots keep rendering
byte-identically and old artifacts keep loading with their original meaning
(a v1 artifact can never smuggle an ``mxfp4`` value past its own envelope).

Every legacy class subclasses its current-version counterpart (defined in the
home schema modules), so an old-version instance remains a valid instance of
the current type for read-only consumption. Load sites dispatch explicitly on
``schema_version`` via the helpers in ``axquant.schema.loading`` — never by
mutating a frozen literal.

The nested legacy models reuse the current class names (``Allocation``,
``TensorSpec``, ...) because the frozen JSON Schema snapshots key ``$defs`` by
class ``__name__``; they live here, not in their home modules, to keep the
import graph acyclic (home modules must not import this module). The
per-field ``type: ignore[assignment]`` comments pin the narrower frozen
envelope on purpose — do not "fix" them into the live enum types.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from axquant.schema import artifacts, inventory, kernel_latency, planning, sensitivity


class QuantMethod(StrEnum):
    """Pre-MXFP4 method envelope frozen into the v1/v2/v3 snapshots."""

    AFFINE = "affine"
    AWQ = "awq"
    DWQ = "dwq"
    GPTQ = "gptq"
    GPTQ_ACT = "gptq-act"
    BF16 = "bf16"


class TensorSpec(inventory.TensorSpec):
    current_method: QuantMethod | None = None  # type: ignore[assignment]


class CandidateMeasurement(sensitivity.CandidateMeasurement):
    method: QuantMethod  # type: ignore[assignment]


class TensorSensitivity(sensitivity.TensorSensitivity):
    tensor: TensorSpec
    candidates: list[CandidateMeasurement]  # type: ignore[assignment]


class KvLayerSensitivity(planning.KvLayerSensitivity):
    candidates: list[CandidateMeasurement]  # type: ignore[assignment]


class KernelLatencyEntry(kernel_latency.KernelLatencyEntry):
    method: QuantMethod  # type: ignore[assignment]


class HardwareProfile(planning.HardwareProfile):
    supported_methods: tuple[QuantMethod, ...] = (  # type: ignore[assignment]
        QuantMethod.AFFINE,
        QuantMethod.AWQ,
        QuantMethod.DWQ,
        QuantMethod.GPTQ,
        QuantMethod.GPTQ_ACT,
        QuantMethod.BF16,
    )


class ManualPrecisionRule(planning.ManualPrecisionRule):
    method: QuantMethod  # type: ignore[assignment]


class Allocation(planning.Allocation):
    method: QuantMethod  # type: ignore[assignment]


class MethodNearTie(planning.MethodNearTie):
    selected_method: QuantMethod  # type: ignore[assignment]
    runner_up_method: QuantMethod  # type: ignore[assignment]


class QuantizationPlan(planning.QuantizationPlan):
    """Legacy ``axquant.plan.v1`` envelope embedded by ``refinement.v2``."""

    schema_version: Literal["axquant.plan.v1"] = "axquant.plan.v1"  # type: ignore[assignment]
    hardware: HardwareProfile
    assignments: list[Allocation] = Field(min_length=1)  # type: ignore[assignment]
    method_near_ties: list[MethodNearTie] = Field(default_factory=list)  # type: ignore[assignment]


QuantizationPlanV1 = QuantizationPlan


class SensitivityReportV1(sensitivity.SensitivityReport):
    schema_version: Literal["axquant.sensitivity.v1"] = "axquant.sensitivity.v1"  # type: ignore[assignment]
    entries: list[TensorSensitivity]  # type: ignore[assignment]


class ProbeConfigV1(sensitivity.ProbeConfig):
    schema_version: Literal["axquant.probe-config.v1"] = "axquant.probe-config.v1"  # type: ignore[assignment]
    candidate_methods: tuple[QuantMethod, ...] = (QuantMethod.AFFINE,)  # type: ignore[assignment]


class ProbeProgressV1(sensitivity.ProbeProgress):
    schema_version: Literal["axquant.probe-progress.v1"] = "axquant.probe-progress.v1"  # type: ignore[assignment]
    completed_tensors: dict[str, list[CandidateMeasurement]] = Field(  # type: ignore[assignment]
        default_factory=dict
    )


class InventoryV1(inventory.Inventory):
    schema_version: Literal["axquant.inventory.v1"] = "axquant.inventory.v1"  # type: ignore[assignment]
    tensors: list[TensorSpec]  # type: ignore[assignment]


class KernelLatencyTableV1(kernel_latency.KernelLatencyTable):
    schema_version: Literal["axquant.kernel-latency.v1"] = "axquant.kernel-latency.v1"  # type: ignore[assignment]
    entries: list[KernelLatencyEntry] = Field(min_length=1)  # type: ignore[assignment]


class KvSensitivityReportV1(planning.KvSensitivityReport):
    schema_version: Literal["axquant.kv-sensitivity.v1"] = "axquant.kv-sensitivity.v1"  # type: ignore[assignment]
    entries: list[KvLayerSensitivity]  # type: ignore[assignment]


class ManualPlanRecipeV1(planning.ManualPlanRecipe):
    schema_version: Literal["axquant.manual-recipe.v1"] = "axquant.manual-recipe.v1"  # type: ignore[assignment]
    default_method: QuantMethod = QuantMethod.AFFINE  # type: ignore[assignment]
    rules: list[ManualPrecisionRule] = Field(default_factory=list)  # type: ignore[assignment]
    hardware: HardwareProfile = Field(default_factory=HardwareProfile)


class PlanRequestV1(planning.PlanRequest):
    schema_version: Literal["axquant.plan-request.v1"] = "axquant.plan-request.v1"  # type: ignore[assignment]
    candidate_methods: tuple[QuantMethod, ...] = ()  # type: ignore[assignment]
    hardware: HardwareProfile = Field(default_factory=HardwareProfile)


class QuantizerExecutionRecord(artifacts.QuantizerExecutionRecord):
    method: QuantMethod  # type: ignore[assignment]


class QuantizerExecutionManifestV1(artifacts.QuantizerExecutionManifest):
    schema_version: Literal["axquant.quantizer-execution.v1"] = "axquant.quantizer-execution.v1"  # type: ignore[assignment]
    records: list[QuantizerExecutionRecord]  # type: ignore[assignment]


class HardwareKernelCoverage(artifacts.HardwareKernelCoverage):
    method: QuantMethod  # type: ignore[assignment]


class HardwareRegistryEntry(artifacts.HardwareRegistryEntry):
    coverage: list[HardwareKernelCoverage] = Field(min_length=1)  # type: ignore[assignment]


class HardwareProfileRegistryV3(artifacts.HardwareProfileRegistry):
    schema_version: Literal["axquant.hardware-registry.v3"] = "axquant.hardware-registry.v3"  # type: ignore[assignment]
    entries: list[HardwareRegistryEntry] = Field(min_length=1)  # type: ignore[assignment]


class RefinementResultV2(artifacts.RefinementResult):
    schema_version: Literal["axquant.refinement.v2"] = "axquant.refinement.v2"  # type: ignore[assignment]
    candidate_plans: dict[str, QuantizationPlan]  # type: ignore[assignment]
    selected_plan: QuantizationPlan
