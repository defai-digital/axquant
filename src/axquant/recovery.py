"""Optional post-PTQ quantization recovery (AXQ-029 QP2).

Recovery restores retention after quantization using calibration-only updates.
It never performs domain SFT/DPO and is never implied by convert/quantize.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from axquant.errors import PlanningError
from axquant.schema import QuantizationPlan, RecoveryTargetRanking, SensitivityReport
from axquant.schema._base import StrictModel, utc_now
from axquant.serde import load_model, stable_sha256, write_data


class ParameterUpdateScope(StrEnum):
    SCALES = "scales"
    BIASES = "biases"
    SCALES_AND_BIASES = "scales-and-biases"
    LORA_MERGED = "lora-merged"


class RecoveryManifest(StrictModel):
    """Provenance for an opt-in recovery stage (`axquant.recovery.v2`)."""

    schema_version: Literal["axquant.recovery.v2"] = "axquant.recovery.v2"
    source_artifact_sha256: str = Field(min_length=64, max_length=64)
    plan_sha256: str = Field(min_length=64, max_length=64)
    algorithm_id: str = Field(default="axquant-scale-bias-recovery-v1", min_length=1)
    calibration_dataset_id: str = Field(min_length=1)
    calibration_dataset_sha256: str = Field(min_length=64, max_length=64)
    random_seed: int = Field(ge=0)
    steps: int = Field(ge=1)
    learning_rate: float | None = Field(default=None, gt=0.0)
    parameter_update_scope: ParameterUpdateScope
    quality_before_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    quality_after_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    claim: Literal["retention-restore-only"] = "retention-restore-only"
    development_evidence: bool = True
    weight_mutation_applied: bool
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def fail_closed_provenance(self) -> RecoveryManifest:
        # Empty strings already blocked by min_length; reinforce algorithm claim.
        if self.claim != "retention-restore-only":
            raise ValueError("recovery may only claim retention-restore-only")
        if self.steps < 1:
            raise ValueError("recovery requires at least one step")
        return self


class RecoveryRequest(StrictModel):
    """Validated inputs for an opt-in recovery run."""

    schema_version: Literal["axquant.recovery-request.v1"] = "axquant.recovery-request.v1"
    source_artifact: str = Field(min_length=1)
    plan_path: str = Field(min_length=1)
    calibration_dataset_id: str = Field(min_length=1)
    calibration_dataset_sha256: str = Field(min_length=64, max_length=64)
    output: str = Field(min_length=1)
    random_seed: int = Field(default=0, ge=0)
    steps: int = Field(default=1, ge=1)
    learning_rate: float | None = Field(default=None, gt=0.0)
    parameter_update_scope: ParameterUpdateScope = ParameterUpdateScope.SCALES_AND_BIASES
    quality_before_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    quality_after_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    algorithm_id: str = Field(default="axquant-scale-bias-recovery-v1", min_length=1)


def _sha256_hex_of_path(path: Path) -> str:
    """Content digest for a file, or of sorted relative file digests for a directory."""
    import hashlib

    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise PlanningError(f"recovery source does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            rel = child.relative_to(path).as_posix().encode()
            digest.update(rel)
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def validate_recovery_request(request: RecoveryRequest) -> None:
    """Fail closed when required recovery provenance inputs are incomplete."""
    if not request.calibration_dataset_id.strip():
        raise PlanningError("recovery requires a calibration dataset id")
    if len(request.calibration_dataset_sha256) != 64:
        raise PlanningError("recovery requires a 64-hex calibration dataset sha256")
    if not all(ch in "0123456789abcdef" for ch in request.calibration_dataset_sha256.lower()):
        raise PlanningError("recovery calibration_dataset_sha256 must be hexadecimal")
    source = Path(request.source_artifact).expanduser()
    if not source.exists():
        raise PlanningError(f"recovery source artifact not found: {source}")
    plan_path = Path(request.plan_path).expanduser()
    if not plan_path.is_file():
        raise PlanningError(f"recovery plan not found: {plan_path}")
    # Plan must parse as QuantizationPlan.
    load_model(plan_path, QuantizationPlan)
    if request.steps < 1:
        raise PlanningError("recovery requires steps >= 1")


def build_recovery_manifest(
    request: RecoveryRequest,
    *,
    source_artifact_sha256: str,
    plan_sha256: str,
    weight_mutation_applied: bool,
) -> RecoveryManifest:
    """Build a recovery provenance manifest after validating the request."""
    validate_recovery_request(request)
    if len(source_artifact_sha256) != 64 or len(plan_sha256) != 64:
        raise PlanningError("recovery digests must be 64-character hex sha256 values")
    notes = [
        "Optional quantization recovery; retention-restore only.",
        "Not domain SFT/DPO; convert/quantize never require this stage.",
    ]
    if not weight_mutation_applied:
        notes.append("v1 recovery is an identity copy: no weight bytes were modified.")
    return RecoveryManifest(
        source_artifact_sha256=source_artifact_sha256,
        plan_sha256=plan_sha256,
        algorithm_id=request.algorithm_id,
        calibration_dataset_id=request.calibration_dataset_id,
        calibration_dataset_sha256=request.calibration_dataset_sha256,
        random_seed=request.random_seed,
        steps=request.steps,
        learning_rate=request.learning_rate,
        parameter_update_scope=request.parameter_update_scope,
        quality_before_sha256=request.quality_before_sha256,
        quality_after_sha256=request.quality_after_sha256,
        weight_mutation_applied=weight_mutation_applied,
        notes=notes,
    )


def rank_recovery_targets(
    plan: QuantizationPlan | str | Path,
    *,
    sensitivity: SensitivityReport | str | Path | None = None,
    limit: int | None = None,
) -> RecoveryTargetRanking:
    """Order quantized tensors by sensitivity / predicted loss for opt-in recovery (P2).

    Higher predicted_loss (or sensitivity output_kl when a report is bound) ranks first.
    Domain LoRA/SFT is not offered — only scale/bias recovery targets.
    """
    plan_model = plan if isinstance(plan, QuantizationPlan) else load_model(plan, QuantizationPlan)
    score_by_tensor: dict[str, float] = {}
    for allocation in plan_model.assignments:
        if allocation.bits >= 16:
            continue
        score_by_tensor[allocation.tensor] = float(allocation.predicted_loss)

    sens_digest: str | None = None
    if sensitivity is not None:
        report = (
            sensitivity
            if isinstance(sensitivity, SensitivityReport)
            else load_model(sensitivity, SensitivityReport)
        )
        sens_digest = stable_sha256(report)
        if report.model.model_id != plan_model.source_model.model_id:
            raise PlanningError("recovery ranking sensitivity model does not match the plan")
        by_name = {entry.tensor.name: entry for entry in report.entries}
        for name in list(score_by_tensor):
            entry = by_name.get(name)
            if entry is None:
                continue
            # Prefer the max measured KL among quantized candidates under the assigned bits.
            assigned = next(
                (a for a in plan_model.assignments if a.tensor == name),
                None,
            )
            if assigned is None:
                continue
            matching = [
                candidate.metrics.output_kl
                for candidate in entry.candidates
                if candidate.bits == assigned.bits and candidate.supported
            ]
            if matching:
                score_by_tensor[name] = max(float(value) for value in matching)

    ordered = sorted(score_by_tensor.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        if limit < 1:
            raise PlanningError("recovery ranking limit must be >= 1")
        ordered = ordered[:limit]
    return RecoveryTargetRanking(
        plan_sha256=stable_sha256(plan_model),
        sensitivity_sha256=sens_digest,
        targets=[name for name, _ in ordered],
        scores={name: score for name, score in ordered},
        notes=[
            "Opt-in ranking only; convert/quantize never imply recovery.",
            "Domain LoRA/SFT is deferred (DeferredFeature.LORA_DOMAIN_SFT).",
        ],
    )


def recover_checkpoint(request: RecoveryRequest) -> RecoveryManifest:
    """Opt-in recovery: copy artifact, attach recovery provenance, return manifest.

    Weight mutation is intentionally a no-op identity copy in this phase so the
    provenance contract and fail-closed validation are shippable without a
    training stack. Future algorithm_id versions may update scales/biases under
    the same manifest schema.
    """
    validate_recovery_request(request)
    source = Path(request.source_artifact).expanduser().resolve()
    plan_path = Path(request.plan_path).expanduser().resolve()
    output = Path(request.output).expanduser().resolve()
    plan = load_model(plan_path, QuantizationPlan)
    plan_sha = stable_sha256(plan)
    source_sha = _sha256_hex_of_path(source)

    if output.exists() and output.resolve() == source:
        raise PlanningError("recovery output must differ from the source artifact path")
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    if source.is_dir():
        shutil.copytree(source, output)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)

    # Record a recovery sidecar marker (no weight mutation for v1 identity recovery).
    marker = {
        "algorithm_id": request.algorithm_id,
        "parameter_update_scope": request.parameter_update_scope.value,
        "steps": request.steps,
        "random_seed": request.random_seed,
        "identity_copy": True,
        "note": "v1 recovery records provenance; scale/bias updates are algorithm-specific",
    }
    marker_path = output / "axquant_recovery_marker.json" if output.is_dir() else None
    if marker_path is not None:
        marker_path.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    manifest = build_recovery_manifest(
        request,
        source_artifact_sha256=source_sha,
        plan_sha256=plan_sha,
        weight_mutation_applied=False,
    )
    manifest_path = (
        output / "axquant_recovery.json"
        if output.is_dir()
        else output.with_suffix(output.suffix + ".recovery.json")
    )
    write_data(manifest_path, manifest)
    return manifest
