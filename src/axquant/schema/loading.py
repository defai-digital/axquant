"""Explicit ``schema_version`` dispatch for bumped frozen contracts (AXQ-046 MH5).

Twelve contracts gained a new version literal when ``QuantMethod.MXFP4``
landed. Every load site for those artifacts must accept the old and the new
version: this module dispatches on the payload's ``schema_version`` (AXQ-042 —
never by silently coercing one version into another) and returns the instance
as persisted. Legacy classes subclass their current-version counterparts, so a
returned old-version instance is still a valid instance of the current type
for read-only consumption, and its ``stable_sha256`` digest still matches the
digests recorded when the artifact was produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin

from pydantic import BaseModel

from axquant.errors import ArtifactError
from axquant.schema import artifacts, frozen_v1, inventory, kernel_latency, planning, sensitivity
from axquant.serde import read_data

ModelT = TypeVar("ModelT", bound=BaseModel)


def _declared_versions(model_type: type[BaseModel]) -> tuple[str, ...]:
    field = model_type.model_fields.get("schema_version")
    if field is None:
        return ()
    annotation = field.annotation
    if get_origin(annotation) is Literal:
        return tuple(arg for arg in get_args(annotation) if isinstance(arg, str))
    default = field.default
    return (default,) if isinstance(default, str) else ()


def load_versioned(path: str | Path, *model_types: type[ModelT]) -> ModelT:
    """Load a persisted artifact whose ``schema_version`` may be any of ``model_types``.

    The payload's version selects the model class; a payload with any other
    version is rejected fail-closed. ``model_types[0]`` must be the current
    version so the return type is the current model.
    """

    if not model_types:
        raise ArtifactError("load_versioned requires at least one model type")
    payload: Any = read_data(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("schema_version"), str):
        raise ArtifactError(f"{path}: missing required 'schema_version' field")
    version = payload["schema_version"]
    for model_type in model_types:
        if version in _declared_versions(model_type):
            return model_type.model_validate(payload)
    accepted = sorted(
        {item for model_type in model_types for item in _declared_versions(model_type)}
    )
    raise ArtifactError(
        f"{path}: unsupported schema_version {version!r} (expected one of: {', '.join(accepted)})"
    )


def load_quantization_plan(path: str | Path) -> planning.QuantizationPlan:
    return load_versioned(path, planning.QuantizationPlan, frozen_v1.QuantizationPlanV1)


def load_sensitivity_report(path: str | Path) -> sensitivity.SensitivityReport:
    return load_versioned(path, sensitivity.SensitivityReport, frozen_v1.SensitivityReportV1)


def load_probe_config(path: str | Path) -> sensitivity.ProbeConfig:
    return load_versioned(path, sensitivity.ProbeConfig, frozen_v1.ProbeConfigV1)


def load_probe_progress(path: str | Path) -> sensitivity.ProbeProgress:
    return load_versioned(path, sensitivity.ProbeProgress, frozen_v1.ProbeProgressV1)


def load_inventory(path: str | Path) -> inventory.Inventory:
    return load_versioned(path, inventory.Inventory, frozen_v1.InventoryV1)


def load_kernel_latency_table(path: str | Path) -> kernel_latency.KernelLatencyTable:
    return load_versioned(path, kernel_latency.KernelLatencyTable, frozen_v1.KernelLatencyTableV1)


def load_kv_sensitivity_report(path: str | Path) -> planning.KvSensitivityReport:
    return load_versioned(path, planning.KvSensitivityReport, frozen_v1.KvSensitivityReportV1)


def load_manual_plan_recipe(path: str | Path) -> planning.ManualPlanRecipe:
    return load_versioned(path, planning.ManualPlanRecipe, frozen_v1.ManualPlanRecipeV1)


def load_plan_request(path: str | Path) -> planning.PlanRequest:
    return load_versioned(path, planning.PlanRequest, frozen_v1.PlanRequestV1)


def load_quantizer_execution_manifest(path: str | Path) -> artifacts.QuantizerExecutionManifest:
    return load_versioned(
        path,
        artifacts.QuantizerExecutionManifest,
        frozen_v1.QuantizerExecutionManifestV1,
    )


def load_refinement_result(path: str | Path) -> artifacts.RefinementResult:
    return load_versioned(path, artifacts.RefinementResult, frozen_v1.RefinementResultV2)


def load_hardware_profile_registry(path: str | Path) -> artifacts.HardwareProfileRegistry:
    return load_versioned(
        path,
        artifacts.HardwareProfileRegistry,
        frozen_v1.HardwareProfileRegistryV3,
    )
