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

import math
import re
from pathlib import Path
from typing import Literal

from axquant.analyzer import architecture_prior_report
from axquant.converter import convert_model
from axquant.errors import PlanningError
from axquant.inspector import inspect_model
from axquant.ladders import get_ladder, plan_request_for_ladder
from axquant.mtp_sidecar import EXTERNAL_MTP_SIDECAR_FILENAMES
from axquant.planner import allocate_kv_cache, plan_quantization
from axquant.recipes import resolve_recipe_plan
from axquant.runtime import (
    check_ax_engine,
    check_mlx_audio_transcription,
    check_mlx_lm_generation,
    check_mlx_vlm_generation,
)
from axquant.schema import (
    ConvertLadderName,
    ModelIdentity,
    ProfileName,
    QuickConversionSummary,
    RuntimeCheck,
    SupportTier,
)

_POLICY_MIN_BPW = re.compile(
    r"target (?P<requested>[0-9.]+) BPW is infeasible; policy minimum is (?P<minimum>[0-9.]+) BPW"
)

DEVELOPMENT_NOTE = "This artifact is development evidence; it is not a certified AXQuant release."

RuntimeSmoke = Literal["none", "mlx-lm", "mlx-audio", "mlx-vlm", "ax-engine"]


def _validate_runtime_smoke(
    smoke: RuntimeSmoke,
    *,
    adapter_id: str,
    audio_input: str | Path | None,
    image_input: str | Path | None,
) -> None:
    required = {
        "qwen3-asr-v1": "mlx-audio",
        "qwen3-vl-v1": "mlx-vlm",
    }.get(adapter_id)
    if required is not None and smoke not in {"none", required}:
        raise PlanningError(
            f"{adapter_id} uses {required}; --runtime-smoke {smoke} would test the wrong runtime"
        )
    if required is None and smoke in {"mlx-audio", "mlx-vlm"}:
        raise PlanningError(
            f"--runtime-smoke {smoke} is only valid for its promoted multimodal adapter"
        )
    media = audio_input if smoke == "mlx-audio" else image_input if smoke == "mlx-vlm" else None
    if smoke in {"mlx-audio", "mlx-vlm"}:
        option = "--audio-input" if smoke == "mlx-audio" else "--image-input"
        if media is None:
            raise PlanningError(f"--runtime-smoke {smoke} requires {option}")
        if not Path(media).expanduser().is_file():
            raise PlanningError(f"{option} does not identify a file: {media}")


def _runtime_smoke_check(
    smoke: RuntimeSmoke,
    output: Path,
    *,
    model_id: str,
    ax_engine: str,
    mlx_lm: str,
    python: str,
    audio_input: str | Path | None,
    image_input: str | Path | None,
) -> RuntimeCheck | None:
    if smoke == "none":
        return None
    identity = ModelIdentity(model_id=model_id, local_path=str(output))
    if smoke == "ax-engine":
        return check_ax_engine(str(output), executable=ax_engine, model_identity=identity)
    if smoke == "mlx-audio":
        if audio_input is None:
            raise PlanningError("--runtime-smoke mlx-audio requires --audio-input")
        return check_mlx_audio_transcription(
            str(output),
            audio=audio_input,
            executable=python,
            model_identity=identity,
        )
    if smoke == "mlx-vlm":
        if image_input is None:
            raise PlanningError("--runtime-smoke mlx-vlm requires --image-input")
        return check_mlx_vlm_generation(
            str(output),
            image=image_input,
            executable=python,
            model_identity=identity,
        )
    return check_mlx_lm_generation(str(output), executable=mlx_lm, model_identity=identity)


def quick_convert(
    *,
    model: str,
    output: str | Path,
    model_id: str | None = None,
    revision: str | None = None,
    profile: ProfileName = ProfileName.GENERAL,
    target_bpw: float | None = None,
    ladder: ConvertLadderName | str = ConvertLadderName.PRIOR,
    kv_cache: Literal["off", "prior"] = "off",
    recipe: str | Path | None = None,
    calibration_manifest: str | Path | None = None,
    kv_sensitivity: str | Path | None = None,
    mtp_sidecar: str | Path | None = None,
    runtime_smoke: RuntimeSmoke = "none",
    ax_engine: str = "ax-engine",
    mlx_lm: str = "mlx_lm.generate",
    python: str = "python3",
    audio_input: str | Path | None = None,
    image_input: str | Path | None = None,
    ax_engine_manifest: Literal["required", "if-available", "skip"] = "if-available",
    allow_download: bool = False,
) -> QuickConversionSummary:
    inventory = inspect_model(
        model,
        model_id=model_id,
        revision=revision,
        allow_download=allow_download,
    )
    architecture = inventory.architecture_profile
    if architecture.support_tier is SupportTier.INSPECT_ONLY:
        raise PlanningError(
            f"the {architecture.product_family} family is inspect-only; quantize requires the "
            "convertible or certified tier and its promotion evidence (AXQ-017)"
        )
    _validate_runtime_smoke(
        runtime_smoke,
        adapter_id=architecture.adapter_id,
        audio_input=audio_input,
        image_input=image_input,
    )
    resolved_ladder = get_ladder(ladder)
    if recipe is not None and ladder not in {
        ConvertLadderName.PRIOR,
        ConvertLadderName.PRIOR.value,
        "prior",
    }:
        raise PlanningError(
            "quantize --recipe cannot be combined with a non-prior --ladder; "
            "recipe bundles already encode planning evidence"
        )
    if resolved_ladder.requires_measured_sensitivity and recipe is None:
        raise PlanningError(
            f"ladder {resolved_ladder.name.value} requires measured sensitivity; "
            "use the staged analyze → plan → convert pipeline (or a measured recipe bundle)"
        )
    bundle_id: str | None = None
    ladder_name = resolved_ladder.name.value
    if recipe is not None:
        bundle, plan = resolve_recipe_plan(recipe, inventory=inventory)
        bundle_id = bundle.bundle_id
        plan_source: Literal["architecture-prior", "recipe-bundle"] = "recipe-bundle"
        effective_target = target_bpw if target_bpw is not None else plan.target_bpw
    else:
        request = plan_request_for_ladder(
            resolved_ladder,
            profile=profile,
            target_bpw=target_bpw,
            allow_unmeasured=True,
        )
        report = architecture_prior_report(
            inventory,
            profile=profile,
            candidate_bits=request.candidate_bits,
            group_size=request.group_size,
            candidate_group_sizes=request.candidate_group_sizes,
        )
        try:
            plan = plan_quantization(report, request)
        except PlanningError as exc:
            # Simple-convert UX: protected floors can push the policy minimum
            # above the user's target (seen on Gemma-4 ~4.89 vs default 4.8).
            # Raise once to the reported minimum (ceiled to 0.01 BPW).
            match = _POLICY_MIN_BPW.search(str(exc))
            if match is None:
                raise
            minimum = float(match.group("minimum"))
            raised = math.ceil(minimum * 100.0 - 1e-12) / 100.0
            if raised <= request.target_bpw + 1e-9:
                raise
            request = request.model_copy(update={"target_bpw": raised})
            plan = plan_quantization(report, request)
            plan.warnings.append(
                f"target BPW raised from {float(match.group('requested')):.4f} to "
                f"{raised:.4f} to satisfy protection floors (policy minimum "
                f"{minimum:.4f})"
            )
        plan_source = "architecture-prior"
        effective_target = request.target_bpw
        plan.warnings.append(f"convert ladder: {ladder_name}")
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
        # Prefer inventory local_path (resolved Hub cache) over the raw model string.
        candidates = [
            Path(model).expanduser(),
            Path(inventory.model.local_path) if inventory.model.local_path else None,
        ]
        for candidate in candidates:
            if candidate is not None and any(
                (candidate / name).is_file() for name in EXTERNAL_MTP_SIDECAR_FILENAMES
            ):
                sidecar = candidate
                break
    # Convert from the resolved local directory when inventory recorded one.
    convert_source = inventory.model.local_path or model
    manifest = convert_model(
        model=convert_source,
        plan=plan,
        output=output,
        revision=revision,
        mtp_sidecar=sidecar,
        calibration_manifest=calibration_manifest,
        kv_sensitivity=kv_sensitivity,
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
        python=python,
        audio_input=audio_input,
        image_input=image_input,
    )
    development = not plan.evidence_kind.release_quality
    notes = [
        f"Plan derived from {plan.evidence_kind.value} evidence.",
        f"Convert ladder: {ladder_name}.",
        f"Requested target BPW: {effective_target}.",
    ]
    notes.extend(
        warning
        for warning in plan.warnings
        if "raised from" in warning or "protection floors" in warning
    )
    if bundle_id is not None:
        notes.append(f"Plan bound from recipe bundle {bundle_id}.")
    if plan.candidate_group_sizes:
        notes.append(
            "Candidate group sizes: " + ",".join(str(size) for size in plan.candidate_group_sizes)
        )
    if development:
        notes.append(DEVELOPMENT_NOTE)
    return QuickConversionSummary(
        source_model=plan.source_model,
        product_family=architecture.product_family,
        support_tier=architecture.support_tier,
        evidence_kind=plan.evidence_kind,
        plan_source=plan_source,
        recipe_bundle_id=bundle_id,
        convert_ladder=ladder_name,
        profile=plan.profile,
        target_bpw=plan.target_bpw,
        measured_total_bpw=manifest.measured_total_bpw,
        output_path=str(output_dir),
        runtime_smoke=runtime_smoke,
        runtime_smoke_passed=(smoke_result.passed if smoke_result is not None else None),
        development_evidence=development,
        notes=notes,
    )
