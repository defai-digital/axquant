"""QP2: opt-in recovery provenance and fail-closed validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.errors import PlanningError
from axquant.planner import plan_quantization
from axquant.recovery import (
    ParameterUpdateScope,
    RecoveryRequest,
    build_recovery_manifest,
    recover_checkpoint,
    validate_recovery_request,
)
from axquant.schema import (
    HardwareProfile,
    Inventory,
    ModelIdentity,
    PlanRequest,
    ProfileName,
    QuantizationPlan,
    TensorRole,
    TensorSpec,
)
from axquant.serde import stable_sha256, write_data


def _minimal_plan(tmp_path: Path) -> tuple[Path, QuantizationPlan]:

    tensors = [
        TensorSpec(
            name="model.layers.0.mlp.down_proj.weight",
            module_path="model.layers.0.mlp.down_proj",
            shape=(64, 64),
            dtype="BF16",
            parameters=4096,
            role=TensorRole.MLP,
            quantizable=True,
            file="model.safetensors",
            current_precision="bf16",
        ),
        TensorSpec(
            name="model.norm.weight",
            module_path="model.norm",
            shape=(64,),
            dtype="BF16",
            parameters=64,
            role=TensorRole.NORM,
            quantizable=False,
            file="model.safetensors",
            current_precision="bf16",
        ),
    ]
    inventory = Inventory(
        model=ModelIdentity(model_id="org/model", revision="abc"),
        tensors=tensors,
        total_parameters=sum(t.parameters for t in tensors),
        quantizable_parameters=4096,
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
    )
    report = architecture_prior_report(inventory, profile=ProfileName.GENERAL)
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.GENERAL,
            target_bpw=8.0,
            allow_unmeasured=True,
            hardware=HardwareProfile(),
        ),
    )
    plan_path = tmp_path / "plan.json"
    write_data(plan_path, plan)
    return plan_path, plan


def test_validate_recovery_fails_without_calibration_digest(tmp_path: Path) -> None:
    plan_path, _ = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"abc")
    with pytest.raises(PlanningError, match="hexadecimal"):
        validate_recovery_request(
            RecoveryRequest(
                source_artifact=str(artifact),
                plan_path=str(plan_path),
                calibration_dataset_id="ds",
                calibration_dataset_sha256="g" * 64,  # not hexadecimal
                output=str(tmp_path / "out"),
            )
        )


def test_validate_recovery_fails_missing_plan(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    with pytest.raises(PlanningError, match="plan not found"):
        validate_recovery_request(
            RecoveryRequest(
                source_artifact=str(artifact),
                plan_path=str(tmp_path / "missing.json"),
                calibration_dataset_id="ds",
                calibration_dataset_sha256="b" * 64,
                output=str(tmp_path / "out"),
            )
        )


def test_recover_writes_manifest_and_is_opt_in(tmp_path: Path) -> None:
    plan_path, plan = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}", encoding="utf-8")
    (artifact / "weights.bin").write_bytes(b"quant-weights")
    output = tmp_path / "recovered"
    request = RecoveryRequest(
        source_artifact=str(artifact),
        plan_path=str(plan_path),
        calibration_dataset_id="reference-calibration",
        calibration_dataset_sha256="c" * 64,
        output=str(output),
        steps=3,
        parameter_update_scope=ParameterUpdateScope.SCALES_AND_BIASES,
        random_seed=7,
    )
    manifest = recover_checkpoint(request)
    assert manifest.schema_version == "axquant.recovery.v1"
    assert manifest.claim == "retention-restore-only"
    assert manifest.development_evidence is True
    assert manifest.steps == 3
    assert manifest.plan_sha256 == stable_sha256(plan)
    assert (output / "axquant_recovery.json").is_file()
    assert (output / "weights.bin").read_bytes() == b"quant-weights"
    assert (output / "axquant_recovery_marker.json").is_file()


def test_build_manifest_requires_valid_request(tmp_path: Path) -> None:
    plan_path, plan = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    request = RecoveryRequest(
        source_artifact=str(artifact),
        plan_path=str(plan_path),
        calibration_dataset_id="ds",
        calibration_dataset_sha256="d" * 64,
        output=str(tmp_path / "out"),
    )
    manifest = build_recovery_manifest(
        request,
        source_artifact_sha256="e" * 64,
        plan_sha256=stable_sha256(plan),
    )
    assert manifest.algorithm_id.startswith("axquant-")
