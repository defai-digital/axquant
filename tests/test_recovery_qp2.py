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
    rank_recovery_targets,
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
    SensitivityReport,
    TensorRole,
    TensorSpec,
)
from axquant.serde import stable_sha256, write_data


def _minimal_plan_with_report(
    tmp_path: Path,
) -> tuple[Path, QuantizationPlan, SensitivityReport]:
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
    return plan_path, plan, report


def _minimal_plan(tmp_path: Path) -> tuple[Path, QuantizationPlan]:
    plan_path, plan, _ = _minimal_plan_with_report(tmp_path)
    return plan_path, plan


def _bind_artifact_plan(artifact: Path, plan: QuantizationPlan) -> None:
    write_data(artifact / "axquant_plan.json", plan)


def test_validate_recovery_fails_without_calibration_digest(tmp_path: Path) -> None:
    plan_path, plan = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"abc")
    _bind_artifact_plan(artifact, plan)
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
    _bind_artifact_plan(artifact, plan)
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
    assert manifest.schema_version == "axquant.recovery.v2"
    assert manifest.claim == "retention-restore-only"
    assert manifest.development_evidence is True
    assert manifest.weight_mutation_applied is False
    assert "identity copy" in " ".join(manifest.notes)
    assert manifest.steps == 3
    assert manifest.plan_sha256 == stable_sha256(plan)
    assert (output / "axquant_recovery.json").is_file()
    assert (output / "weights.bin").read_bytes() == b"quant-weights"
    assert (output / "axquant_recovery_marker.json").is_file()


def test_build_manifest_requires_valid_request(tmp_path: Path) -> None:
    plan_path, plan = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _bind_artifact_plan(artifact, plan)
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
        weight_mutation_applied=False,
    )
    assert manifest.algorithm_id.startswith("axquant-")


@pytest.mark.parametrize("output_location", ["inside", "parent"])
def test_recovery_rejects_source_output_overlap(
    tmp_path: Path,
    output_location: str,
) -> None:
    plan_path, plan = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    source_file = artifact / "weights.bin"
    source_file.write_bytes(b"quant-weights")
    _bind_artifact_plan(artifact, plan)
    output = artifact / "recovered" if output_location == "inside" else tmp_path
    request = RecoveryRequest(
        source_artifact=str(artifact),
        plan_path=str(plan_path),
        calibration_dataset_id="ds",
        calibration_dataset_sha256="d" * 64,
        output=str(output),
    )
    with pytest.raises(PlanningError, match="must not overlap"):
        recover_checkpoint(request)
    assert source_file.read_bytes() == b"quant-weights"


def test_recovery_rejects_symlink_output(tmp_path: Path) -> None:
    plan_path, plan = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"quant-weights")
    _bind_artifact_plan(artifact, plan)
    target = tmp_path / "existing-target"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    output = tmp_path / "output-link"
    output.symlink_to(target, target_is_directory=True)
    request = RecoveryRequest(
        source_artifact=str(artifact),
        plan_path=str(plan_path),
        calibration_dataset_id="ds",
        calibration_dataset_sha256="d" * 64,
        output=str(output),
    )
    with pytest.raises(PlanningError, match="symbolic link"):
        recover_checkpoint(request)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_recovery_rejects_output_containing_input_plan(tmp_path: Path) -> None:
    plan_path, plan = _minimal_plan(tmp_path)
    plan_directory = tmp_path / "plan-input"
    plan_directory.mkdir()
    relocated_plan = plan_directory / plan_path.name
    plan_path.replace(relocated_plan)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"quant-weights")
    _bind_artifact_plan(artifact, plan)
    request = RecoveryRequest(
        source_artifact=str(artifact),
        plan_path=str(relocated_plan),
        calibration_dataset_id="ds",
        calibration_dataset_sha256="d" * 64,
        output=str(plan_directory),
    )
    with pytest.raises(PlanningError, match="contain the input plan"):
        recover_checkpoint(request)
    assert relocated_plan.is_file()


def test_recovery_rejects_plan_not_bound_to_artifact(tmp_path: Path) -> None:
    plan_path, plan = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"quant-weights")
    wrong_plan = plan.model_copy(
        update={"source_model": plan.source_model.model_copy(update={"model_id": "org/other"})}
    )
    write_data(artifact / "axquant_plan.json", wrong_plan)
    request = RecoveryRequest(
        source_artifact=str(artifact),
        plan_path=str(plan_path),
        calibration_dataset_id="ds",
        calibration_dataset_sha256="d" * 64,
        output=str(tmp_path / "output"),
    )
    with pytest.raises(PlanningError, match="source artifact plan"):
        validate_recovery_request(request)


def test_recovery_ranking_requires_exact_bound_sensitivity(tmp_path: Path) -> None:
    _, plan, report = _minimal_plan_with_report(tmp_path)
    ranking = rank_recovery_targets(plan, sensitivity=report)
    assert ranking.sensitivity_sha256 == plan.analysis_sha256
    assert ranking.targets

    other_revision = report.model_copy(
        update={"model": report.model.model_copy(update={"revision": "other"})}
    )
    with pytest.raises(PlanningError, match="model does not match"):
        rank_recovery_targets(plan, sensitivity=other_revision)

    changed_report = report.model_copy(update={"warnings": [*report.warnings, "changed"]})
    with pytest.raises(PlanningError, match="not bound to the plan"):
        rank_recovery_targets(plan, sensitivity=changed_report)


def test_recovery_refuses_to_delete_existing_output(tmp_path: Path) -> None:
    plan_path, plan = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"quant-weights")
    _bind_artifact_plan(artifact, plan)
    output = tmp_path / "existing-output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    request = RecoveryRequest(
        source_artifact=str(artifact),
        plan_path=str(plan_path),
        calibration_dataset_id="ds",
        calibration_dataset_sha256="d" * 64,
        output=str(output),
    )

    with pytest.raises(PlanningError, match="refusing to overwrite"):
        recover_checkpoint(request)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_recovery_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    plan_path, plan = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"quant-weights")
    _bind_artifact_plan(artifact, plan)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    request = RecoveryRequest(
        source_artifact=str(artifact),
        plan_path=str(plan_path),
        calibration_dataset_id="ds",
        calibration_dataset_sha256="d" * 64,
        output=str(linked_parent / "recovered"),
    )

    with pytest.raises(PlanningError, match="traverses a symbolic link"):
        recover_checkpoint(request)

    assert not (outside / "recovered").exists()


def test_recovery_requires_a_plan_bound_source_artifact(tmp_path: Path) -> None:
    plan_path, _ = _minimal_plan(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"quant-weights")
    request = RecoveryRequest(
        source_artifact=str(artifact),
        plan_path=str(plan_path),
        calibration_dataset_id="ds",
        calibration_dataset_sha256="d" * 64,
        output=str(tmp_path / "recovered"),
    )

    with pytest.raises(PlanningError, match=r"must contain axquant_plan\.json"):
        validate_recovery_request(request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("random_seed", True),
        ("steps", True),
        ("learning_rate", True),
    ],
)
def test_recovery_request_rejects_boolean_numeric_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=r"must not be a boolean|must not be booleans"):
        RecoveryRequest(
            source_artifact="/artifact",
            plan_path="/plan",
            calibration_dataset_id="ds",
            calibration_dataset_sha256="d" * 64,
            output="/output",
            **{field: value},
        )
