"""One-command development conversion (AXQ-019).

``quick_convert`` chains the existing inspect → plan → convert stages with
working defaults so a convertible checkpoint becomes a runnable development
artifact from a single command. It is a front end over the staged pipeline,
not a parallel implementation: every stage keeps its own fail-closed checks,
and the output is always labeled development evidence until measured planning
is bound through the staged release pipeline or a measured recipe bundle
(AXQ-020).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from axquant.analyzer import architecture_prior_report
from axquant.converter import convert_model
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.planner import allocate_kv_cache, plan_quantization
from axquant.recipes import resolve_recipe_plan
from axquant.runtime import check_ax_engine, check_mlx_lm_generation
from axquant.schema import (
    ModelIdentity,
    PlanRequest,
    ProfileName,
    QuickConversionSummary,
    RuntimeCheck,
    SupportTier,
)

DEVELOPMENT_NOTE = "This artifact is development evidence; it is not a certified AXQuant release."

RuntimeSmoke = Literal["none", "mlx-lm", "ax-engine"]


def _runtime_smoke_check(
    smoke: RuntimeSmoke,
    output: Path,
    *,
    model_id: str,
    ax_engine: str,
    mlx_lm: str,
) -> RuntimeCheck | None:
    if smoke == "none":
        return None
    identity = ModelIdentity(model_id=model_id, local_path=str(output))
    if smoke == "ax-engine":
        return check_ax_engine(str(output), executable=ax_engine, model_identity=identity)
    return check_mlx_lm_generation(str(output), executable=mlx_lm, model_identity=identity)


def quick_convert(
    *,
    model: str,
    output: str | Path,
    model_id: str | None = None,
    revision: str | None = None,
    profile: ProfileName = ProfileName.GENERAL,
    target_bpw: float = 4.8,
    kv_cache: Literal["off", "prior"] = "off",
    recipe: str | Path | None = None,
    calibration_manifest: str | Path | None = None,
    mtp_sidecar: str | Path | None = None,
    runtime_smoke: RuntimeSmoke = "none",
    ax_engine: str = "ax-engine",
    mlx_lm: str = "mlx_lm.generate",
    ax_engine_manifest: Literal["required", "if-available", "skip"] = "if-available",
) -> QuickConversionSummary:
    inventory = inspect_model(model, model_id=model_id, revision=revision)
    architecture = inventory.architecture_profile
    if architecture.support_tier is SupportTier.INSPECT_ONLY:
        raise PlanningError(
            f"the {architecture.product_family} family is inspect-only; quantize requires the "
            "convertible or certified tier and its promotion evidence (AXQ-017)"
        )
    bundle_id: str | None = None
    if recipe is not None:
        bundle, plan = resolve_recipe_plan(recipe, inventory=inventory)
        bundle_id = bundle.bundle_id
        plan_source: Literal["architecture-prior", "recipe-bundle"] = "recipe-bundle"
    else:
        report = architecture_prior_report(inventory, profile=profile)
        plan = plan_quantization(
            report,
            PlanRequest(
                profile=profile,
                target_bpw=target_bpw,
                allow_unmeasured=True,
            ),
        )
        plan_source = "architecture-prior"
    if kv_cache == "prior" and plan.kv_cache is None:
        layer_count = architecture.text_layer_count
        if layer_count is None:
            raise PlanningError("KV-cache planning requires a known text layer count")
        plan.kv_cache = allocate_kv_cache(layer_count, group_size=plan.group_size)
    sidecar = Path(mtp_sidecar).expanduser() if mtp_sidecar is not None else None
    if (
        sidecar is None
        and plan.mtp.preserve_external_sidecar
        and any(allocation.role.is_mtp for allocation in plan.assignments)
    ):
        model_dir = Path(model).expanduser()
        if (model_dir / "mtp.safetensors").is_file():
            sidecar = model_dir
    manifest = convert_model(
        model=model,
        plan=plan,
        output=output,
        revision=revision,
        mtp_sidecar=sidecar,
        calibration_manifest=calibration_manifest,
        allow_unmeasured=True,
        ax_engine_manifest=ax_engine_manifest,
    )
    output_dir = Path(output).expanduser().resolve()
    smoke_result = _runtime_smoke_check(
        runtime_smoke,
        output_dir,
        model_id=manifest.source_model.model_id,
        ax_engine=ax_engine,
        mlx_lm=mlx_lm,
    )
    development = not plan.evidence_kind.release_quality
    notes = [f"Plan derived from {plan.evidence_kind.value} evidence."]
    if bundle_id is not None:
        notes.append(f"Plan bound from recipe bundle {bundle_id}.")
    if development:
        notes.append(DEVELOPMENT_NOTE)
    return QuickConversionSummary(
        source_model=plan.source_model,
        product_family=architecture.product_family,
        support_tier=architecture.support_tier,
        evidence_kind=plan.evidence_kind,
        plan_source=plan_source,
        recipe_bundle_id=bundle_id,
        profile=plan.profile,
        target_bpw=plan.target_bpw,
        measured_total_bpw=manifest.measured_total_bpw,
        output_path=str(output_dir),
        runtime_smoke=runtime_smoke,
        runtime_smoke_passed=(smoke_result.passed if smoke_result is not None else None),
        development_evidence=development,
        notes=notes,
    )
