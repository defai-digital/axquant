from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.cli import main
from axquant.compatibility import build_compatibility_matrix
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.profiles import thresholds_for
from axquant.runtime import build_runtime_metadata
from axquant.schema import (
    ArtifactFile,
    ArtifactManifest,
    CompatibilityCandidateInput,
    CompatibilityMatrix,
    CompatibilityMatrixRequest,
    ModelIdentity,
    OfficialDenseCheckpointRequirement,
    PlanRequest,
    ProfileName,
    QuantizationPlan,
    ReleaseException,
    ReleaseExceptionTarget,
    RuntimeCheck,
    RuntimeName,
    ValidationIssue,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data


def _artifact_files(directory: Path) -> list[ArtifactFile]:
    return [
        ArtifactFile(
            path=path.relative_to(directory).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=file_sha256(path),
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "axquant_manifest.json"
    ]


def _candidate(
    *,
    source: Path,
    directory: Path,
    source_id: str,
    candidate_id: str,
) -> list[CompatibilityCandidateInput]:
    shutil.copytree(source, directory)
    inventory = inspect_model(
        directory,
        model_id=source_id,
        revision="a" * 40,
    )
    sensitivity = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    plan = plan_quantization(
        sensitivity,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
        ),
    )
    write_data(directory / "axquant_plan.json", plan)
    (directory / "model-manifest.json").write_text("{}", encoding="utf-8")
    runtime = build_runtime_metadata(plan, directory)
    write_data(directory / "axquant_runtime.json", runtime)
    mtp_parameters = sum(tensor.parameters for tensor in inventory.tensors if tensor.role.is_mtp)
    main_parameters = inventory.total_parameters - mtp_parameters
    main_bytes = inventory.weight_bytes - inventory.mtp_weight_bytes
    manifest = ArtifactManifest(
        axquant_version=plan.software_versions.axquant,
        source_model=plan.source_model,
        plan_sha256=stable_sha256(plan),
        calibration=plan.calibration,
        profile=plan.profile,
        target_class=plan.target_class,
        effective_bpw=plan.effective_bpw,
        logical_parameters=inventory.total_parameters,
        main_logical_parameters=main_parameters,
        weight_file_size_bytes=inventory.weight_bytes,
        main_weight_file_size_bytes=main_bytes,
        mtp_weight_file_size_bytes=inventory.mtp_weight_bytes,
        protected_weight_file_size_bytes=0,
        measured_total_bpw=8.0 * inventory.weight_bytes / inventory.total_parameters,
        measured_main_bpw=8.0 * main_bytes / main_parameters,
        weight_distribution=plan.weight_distribution,
        mtp_distribution=plan.mtp_distribution,
        mtp_present=inventory.mtp_present,
        mtp_policy=plan.mtp,
        runtime=runtime,
        software_versions=plan.software_versions,
        files=_artifact_files(directory),
    )
    write_data(directory / "axquant_manifest.json", manifest)

    evidence = directory.parent / f"{directory.name}-evidence"
    evidence.mkdir()
    candidate_model = ModelIdentity(
        model_id=candidate_id,
        revision="c" * 40,
        local_path=str(directory.resolve()),
    )
    ax_check = evidence / "ax-engine.json"
    write_data(
        ax_check,
        RuntimeCheck(
            model=candidate_model,
            runtime=RuntimeName.AX_ENGINE,
            check_kind="doctor",
            available=True,
            passed=True,
            command=[
                "ax-engine",
                "doctor",
                "--json",
                "--mlx-model-artifacts-dir",
                str(directory.resolve()),
            ],
            exit_code=0,
            report={"result": "ready"},
        ),
    )
    mlx_check = evidence / "mlx-lm.json"
    write_data(
        mlx_check,
        RuntimeCheck(
            model=candidate_model,
            runtime=RuntimeName.MLX_LM,
            check_kind="generation-smoke",
            available=True,
            passed=True,
            command=["mlx_lm.generate", "--model", str(directory.resolve())],
            exit_code=0,
            report={"standard_inference": True},
        ),
    )
    result: list[CompatibilityCandidateInput] = []
    for profile in (ProfileName.AGENT_CODING, ProfileName.GENERAL):
        validation_path = evidence / f"validation-{profile.value}.json"
        write_data(
            validation_path,
            ValidationReport(
                reference_model=ModelIdentity(
                    model_id="Qwen/Qwen3.6-27B-MLX-6bit",
                    revision="b" * 40,
                ),
                candidate_model=candidate_model,
                profile=profile,
                passed=True,
                thresholds=thresholds_for(profile),
                issues=[],
                comparisons={
                    "artifact.candidate_weight_bytes": manifest.weight_file_size_bytes,
                    "artifact.weight_size_ratio": 1.05,
                    "quality.aggregate_retention": 0.99,
                    "mtp.acceptance_retention": 0.97,
                    "hardware.effective_speedup": 1.25,
                    "hardware.peak_memory_ratio": 0.80,
                    "hardware.device_name": "Test Mac",
                    "hardware.chip": "Apple M3 Max",
                    "hardware.unified_memory_bytes": 128 * 1024**3,
                    "hardware.os_version": "macOS test",
                    "hardware.power_mode": "AC power",
                    "hardware.kernel_fallbacks": 0,
                    "software.mlx_lm": "0.31",
                },
            ),
        )
        result.append(
            CompatibilityCandidateInput(
                artifact_directory=str(directory),
                ax_engine_check=str(ax_check),
                mlx_lm_check=str(mlx_check),
                validation_report=str(validation_path),
            )
        )
    return result


def _request(
    *,
    candidates: list[CompatibilityCandidateInput],
    requirements: list[tuple[str, str]],
) -> CompatibilityMatrixRequest:
    return CompatibilityMatrixRequest(
        catalog_verified_at=datetime.now(UTC),
        required_dense_models=[
            OfficialDenseCheckpointRequirement(
                model_id=model_id,
                parameter_size=parameter_size,
            )
            for model_id, parameter_size in requirements
        ],
        candidates=candidates,
    )


def test_compatibility_matrix_accepts_every_current_official_dense_size(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidates = _candidate(
        source=qwen36_model_dir,
        directory=tmp_path / "candidate-a",
        source_id="Qwen/Qwen3.6-27B",
        candidate_id="AutomatosX/candidate-a",
    )
    request = tmp_path / "request.json"
    write_data(
        request,
        _request(
            candidates=candidates,
            requirements=[("Qwen/Qwen3.6-27B", "27B")],
        ),
    )

    matrix = build_compatibility_matrix(request)

    assert all(entry.compatible for entry in matrix.entries)
    assert matrix.distinct_dense_source_checkpoints == 1
    assert matrix.required_dense_checkpoints == 1
    assert matrix.release_ready
    assert not matrix.issues


def test_compatibility_matrix_requires_both_release_profiles(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidates = _candidate(
        source=qwen36_model_dir,
        directory=tmp_path / "candidate-a",
        source_id="Qwen/Qwen3.6-27B",
        candidate_id="AutomatosX/candidate-a",
    )
    request = tmp_path / "request.json"
    write_data(
        request,
        _request(
            candidates=[candidates[0]],
            requirements=[("Qwen/Qwen3.6-27B", "27B")],
        ),
    )

    matrix = build_compatibility_matrix(request)

    assert not matrix.release_ready
    assert any("missing compatible profiles: ['general']" in issue for issue in matrix.issues)


@pytest.mark.parametrize(
    "requirements",
    [
        [
            ("Qwen/Qwen3.6-27B", "27B"),
            ("Qwen/Qwen3.6-27B", "32B"),
        ],
        [
            ("Qwen/Qwen3.6-27B", "27B"),
            ("Qwen/Qwen3.6-32B", "27B"),
        ],
    ],
)
def test_compatibility_request_rejects_duplicate_official_scope(
    requirements: list[tuple[str, str]],
) -> None:
    candidate = CompatibilityCandidateInput(
        artifact_directory="artifact",
        ax_engine_check="ax-engine.json",
        mlx_lm_check="mlx-lm.json",
        validation_report="validation.json",
    )
    with pytest.raises(ValueError, match="must be unique"):
        _request(candidates=[candidate], requirements=requirements)


def test_compatibility_matrix_release_ready_and_cli(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        source=qwen36_model_dir,
        directory=tmp_path / "candidate-a",
        source_id="Qwen/Qwen3.6-27B",
        candidate_id="AutomatosX/candidate-a",
    )
    request = tmp_path / "request.json"
    output = tmp_path / "matrix.json"
    write_data(
        request,
        _request(
            candidates=candidate,
            requirements=[("Qwen/Qwen3.6-27B", "27B")],
        ),
    )

    assert (
        main(
            [
                "compatibility-matrix",
                "--request",
                str(request),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    matrix = load_model(output, CompatibilityMatrix)
    assert matrix.release_ready
    assert matrix.distinct_dense_source_checkpoints == 1
    assert all(entry.compatible for entry in matrix.entries)


def test_compatibility_matrix_rejects_runtime_evidence_for_another_artifact(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidates = _candidate(
        source=qwen36_model_dir,
        directory=tmp_path / "candidate-a",
        source_id="Qwen/Qwen3.6-27B",
        candidate_id="AutomatosX/candidate-a",
    )
    check_path = Path(candidates[0].ax_engine_check)
    check = load_model(check_path, RuntimeCheck)
    check.command[-1] = str(tmp_path / "another-artifact")
    write_data(check_path, check)
    request = tmp_path / "request.json"
    write_data(
        request,
        _request(
            candidates=candidates,
            requirements=[("Qwen/Qwen3.6-27B", "27B")],
        ),
    )

    matrix = build_compatibility_matrix(request)

    assert not matrix.entries[0].compatible
    assert "does not target the candidate artifact" in matrix.entries[0].issues[0]


def test_compatibility_matrix_rejects_ambiguous_runtime_target(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidates = _candidate(
        source=qwen36_model_dir,
        directory=tmp_path / "candidate-a",
        source_id="Qwen/Qwen3.6-27B",
        candidate_id="AutomatosX/candidate-a",
    )
    check_path = Path(candidates[0].ax_engine_check)
    check = load_model(check_path, RuntimeCheck)
    check.command.extend(["--mlx-model-artifacts-dir", str(tmp_path / "another-artifact")])
    write_data(check_path, check)
    request = tmp_path / "request.json"
    write_data(
        request,
        _request(
            candidates=candidates,
            requirements=[("Qwen/Qwen3.6-27B", "27B")],
        ),
    )

    matrix = build_compatibility_matrix(request)

    assert not matrix.entries[0].compatible
    assert any(
        "does not target the candidate artifact" in issue for issue in matrix.entries[0].issues
    )


def test_compatibility_matrix_rechecks_validation_thresholds(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidates = _candidate(
        source=qwen36_model_dir,
        directory=tmp_path / "candidate-a",
        source_id="Qwen/Qwen3.6-27B",
        candidate_id="AutomatosX/candidate-a",
    )
    validation_path = Path(candidates[0].validation_report)
    validation = load_model(validation_path, ValidationReport)
    validation.comparisons["hardware.effective_speedup"] = 0.5
    write_data(validation_path, validation)
    request = tmp_path / "request.json"
    write_data(
        request,
        _request(
            candidates=candidates,
            requirements=[("Qwen/Qwen3.6-27B", "27B")],
        ),
    )

    matrix = build_compatibility_matrix(request)

    assert not matrix.entries[0].compatible
    assert (
        "validation comparison violates its release threshold: hardware.effective_speedup"
        in matrix.entries[0].issues
    )


def test_compatibility_matrix_requires_complete_weight_manifest_membership(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidates = _candidate(
        source=qwen36_model_dir,
        directory=tmp_path / "candidate-a",
        source_id="Qwen/Qwen3.6-27B",
        candidate_id="AutomatosX/candidate-a",
    )
    manifest_path = Path(candidates[0].artifact_directory) / "axquant_manifest.json"
    manifest = load_model(manifest_path, ArtifactManifest)
    manifest.files = [
        record for record in manifest.files if not record.path.endswith(".safetensors")
    ]
    write_data(manifest_path, manifest)
    request = tmp_path / "request.json"
    write_data(
        request,
        _request(
            candidates=candidates,
            requirements=[("Qwen/Qwen3.6-27B", "27B")],
        ),
    )

    matrix = build_compatibility_matrix(request)

    assert not matrix.entries[0].compatible
    assert any("Safetensors membership differs" in issue for issue in matrix.entries[0].issues)


def test_compatibility_matrix_rejects_wrong_comparison_types(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidates = _candidate(
        source=qwen36_model_dir,
        directory=tmp_path / "candidate-a",
        source_id="Qwen/Qwen3.6-27B",
        candidate_id="AutomatosX/candidate-a",
    )
    validation_path = Path(candidates[0].validation_report)
    validation = load_model(validation_path, ValidationReport)
    validation.comparisons["hardware.effective_speedup"] = "fast"
    validation.comparisons["hardware.chip"] = 123
    write_data(validation_path, validation)
    request = tmp_path / "request.json"
    write_data(
        request,
        _request(
            candidates=candidates,
            requirements=[("Qwen/Qwen3.6-27B", "27B")],
        ),
    )

    matrix = build_compatibility_matrix(request)

    assert not matrix.entries[0].compatible
    assert (
        "validation comparison must be numeric: hardware.effective_speedup"
        in matrix.entries[0].issues
    )
    assert (
        "validation comparison must be a non-empty string: hardware.chip"
        in matrix.entries[0].issues
    )


def test_compatibility_matrix_accepts_a_governed_size_exception(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    candidates = _candidate(
        source=qwen36_model_dir,
        directory=tmp_path / "candidate-a",
        source_id="Qwen/Qwen3.6-27B",
        candidate_id="AutomatosX/candidate-a",
    )
    validation_path = Path(candidates[0].validation_report)
    validation = load_model(validation_path, ValidationReport)
    validation.thresholds = thresholds_for(validation.profile)
    plan = load_model(
        Path(candidates[0].artifact_directory) / "axquant_plan.json",
        QuantizationPlan,
    )
    approved_at = datetime.now(UTC) - timedelta(days=1)
    exception = ReleaseException(
        exception_id="AXQ-COMPAT-SIZE",
        candidate_model=validation.candidate_model,
        plan_sha256=stable_sha256(plan),
        targets=[
            ReleaseExceptionTarget(
                metric="artifact.weight_size_ratio",
                observed_value=1.2,
                required_maximum=1.1,
                requirement="candidate size must remain within the release limit",
            ),
            ReleaseExceptionTarget(
                metric="artifact.candidate_measured_bpw",
                observed_value=5.76,
                required_minimum=4.3,
                required_maximum=4.8,
                requirement="candidate BPW must remain within the target range",
            ),
        ],
        measured_tradeoff="Measured quality and memory tradeoff.",
        owner="test owner",
        approved_by="test authority",
        approval_reference="test-decision",
        approved_at=approved_at,
        expires_at=approved_at + timedelta(days=30),
        evidence_sha256={
            "plan": "1" * 64,
            "candidate_size": "2" * 64,
            "size_reference": "3" * 64,
            "tradeoff": "4" * 64,
        },
    )
    validation.comparisons.update(
        {
            "artifact.weight_size_ratio": 1.2,
            "artifact.candidate_measured_bpw": 5.76,
        }
    )
    validation.issues = [
        ValidationIssue(
            severity="warning",
            metric="artifact.weight_size_ratio",
            message=("ratio 1.2000 exceeds 1.1000; governed exception AXQ-COMPAT-SIZE"),
        )
    ]
    validation.release_exceptions = [exception]
    write_data(validation_path, validation)
    request = tmp_path / "request.json"
    write_data(
        request,
        _request(
            candidates=candidates,
            requirements=[("Qwen/Qwen3.6-27B", "27B")],
        ),
    )

    matrix = build_compatibility_matrix(request)

    assert matrix.entries[0].compatible, matrix.entries[0].issues
