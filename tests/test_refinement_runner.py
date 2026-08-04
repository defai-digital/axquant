from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.calibration import calibration_manifest_sha256
from axquant.cli import main
from axquant.errors import RefinementError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.refinement_runner import execute_refinement
from axquant.schema import (
    CalibrationEvidence,
    CalibrationManifest,
    CandidateEntry,
    CompleteCandidateHardware,
    CompleteCandidateMeasurement,
    EvidenceKind,
    ModelIdentity,
    PlanRequest,
    ProfileName,
    RefinementConfig,
    RefinementExecutionManifest,
    RefinementExecutionRequest,
    RefinementMeasurementSet,
    RefinementResult,
)
from axquant.serde import load_model, stable_sha256, write_data


def _inputs(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> tuple[Path, RefinementResult]:
    calibration_path = tmp_path / "calibration.json"
    inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )
    calibration = CalibrationManifest(
        model=inventory.model,
        profile=ProfileName.AGENT_CODING,
        dataset_id="internal/calibration",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding"],
        sequence_length=2048,
        random_seed=7,
        calibration_evaluation_separation_attested=True,
    )
    write_data(calibration_path, calibration)
    sensitivity = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
    )
    plan = plan_quantization(
        sensitivity,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
        ),
    )
    plan.evidence_kind = EvidenceKind.MEASURED
    plan.calibration = CalibrationEvidence(
        dataset_id=calibration.dataset_id,
        dataset_sha256=calibration.dataset_sha256,
        samples=calibration.samples,
        domains=calibration.domains,
        sequence_length=calibration.sequence_length,
        backend="mlx",
        reference=calibration_path.name,
        metadata={"calibration_manifest_sha256": calibration_manifest_sha256(calibration)},
    )
    candidate_id = "cand-0007-000"
    plan_sha256 = stable_sha256(plan)
    refinement = RefinementResult(
        config=RefinementConfig(random_seed=7, top_n=1),
        history=[
            CandidateEntry(
                candidate_id=candidate_id,
                plan_sha256=plan_sha256,
                change_description="initial measured candidate",
                reason="test",
                predicted_bpw=plan.effective_bpw,
                predicted_loss=0.1,
                budget_impact=0.0,
                state="selected",
            )
        ],
        candidate_plans={candidate_id: plan},
        selected_candidate_id=candidate_id,
        selected_plan=plan,
        selected_plan_sha256=plan_sha256,
        selection_basis="proxy",
        iterations_used=0,
        evaluations_used=1,
        converged=True,
    )
    refinement_path = tmp_path / "refinement.json"
    write_data(refinement_path, refinement)
    supporting: dict[str, Path] = {}
    for name in (
        "quality-dataset",
        "reference-quality",
        "reference-evaluation",
        "benchmark-prompts",
        "size-reference",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        supporting[name] = path
    request_path = tmp_path / "request.json"
    write_data(
        request_path,
        RefinementExecutionRequest(
            refinement_file=str(refinement_path),
            source_model=str(qwen36_model_dir),
            mtp_sidecar=str(qwen36_model_dir / "mtp.safetensors"),
            calibration_manifest=str(calibration_path),
            quality_dataset=str(supporting["quality-dataset"]),
            reference_quality=str(supporting["reference-quality"]),
            reference_evaluation=str(supporting["reference-evaluation"]),
            benchmark_prompts=str(supporting["benchmark-prompts"]),
            size_reference=str(supporting["size-reference"]),
            candidate_repository_prefix="AutomatosX/AX-Qwen3.6-test",
            benchmark_power_mode="AC power",
            random_seed=7,
        ),
    )
    return request_path, refinement


def test_refinement_execution_dry_run_is_exact_and_resumable(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    request, _ = _inputs(qwen36_model_dir, tmp_path)
    output = tmp_path / "run"

    first = execute_refinement(request_path=request, output_dir=output)
    second = execute_refinement(request_path=request, output_dir=output)

    assert first == second
    assert len(first.steps) == 9
    assert all(step.state == "pending" for step in first.steps)
    convert = first.steps[0]
    assert convert.step_id == "convert"
    assert "--allow-unmeasured" not in convert.command
    assert "--calibration-manifest" in convert.command
    assert convert.command[convert.command.index("--mtp-layout") + 1] == "byte-preserved"
    assert (output / "candidates" / "cand-0007-000" / "plan.json").is_file()


def test_refinement_execution_cli_defaults_to_dry_run(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    request, _ = _inputs(qwen36_model_dir, tmp_path)
    output = tmp_path / "cli-run"

    assert (
        main(
            [
                "refine-run",
                "--request",
                str(request),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    manifest = load_model(output / "execution-manifest.json", RefinementExecutionManifest)
    assert not manifest.complete
    assert all(step.state == "pending" for step in manifest.steps)


def test_refinement_execution_resume_rejects_changed_input(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    request_path, _ = _inputs(qwen36_model_dir, tmp_path)
    output = tmp_path / "run"
    execute_refinement(request_path=request_path, output_dir=output)
    request = load_model(request_path, RefinementExecutionRequest)
    Path(request.quality_dataset).write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(RefinementError, match="inputs changed on resume"):
        execute_refinement(request_path=request_path, output_dir=output)


def test_refinement_execution_skips_candidate_after_failed_conversion(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    request, _ = _inputs(qwen36_model_dir, tmp_path)

    def failed_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(command), 2, stdout="", stderr="failed")

    result = execute_refinement(
        request_path=request,
        output_dir=tmp_path / "run",
        execute=True,
        runner=failed_runner,
    )

    assert result.complete
    assert result.failed_candidate_ids == ["cand-0007-000"]
    assert result.steps[0].state == "failed"
    assert all(step.state == "skipped" for step in result.steps[1:])
    assert result.selected_result is None


def test_refinement_execution_merges_measurement_and_selects(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    request, refinement = _inputs(qwen36_model_dir, tmp_path)
    candidate_id = "cand-0007-000"
    plan = refinement.candidate_plans[candidate_id]

    def successful_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(command)
        operation = args[1]
        if operation == "convert":
            destination = Path(args[args.index("--output") + 1]) / "axquant_manifest.json"
            destination.parent.mkdir(parents=True)
            destination.write_text("{}\n", encoding="utf-8")
        elif operation in {"evaluate-quality", "compare-quality", "size-evidence", "validate"}:
            destination = Path(args[args.index("--output") + 1])
            destination.write_text("{}\n", encoding="utf-8")
        elif operation == "benchmark-ab":
            destination = Path(args[args.index("--output-dir") + 1])
            destination.mkdir(parents=True)
            (destination / "evaluation_mtp_off.json").write_text("{}\n", encoding="utf-8")
            (destination / "evaluation_mtp_on.json").write_text("{}\n", encoding="utf-8")
            for mode in ("mtp-off", "mtp-on"):
                raw_log = destination / mode / "benchmark_raw_log.json"
                raw_log.parent.mkdir(parents=True)
                raw_log.write_text("{}\n", encoding="utf-8")
        elif operation == "refine-measure":
            destination = Path(args[args.index("--output") + 1])
            write_data(
                destination,
                RefinementMeasurementSet(
                    refinement_sha256=stable_sha256(refinement),
                    evaluator_version="test-v1",
                    measurements=[
                        CompleteCandidateMeasurement(
                            candidate_id=candidate_id,
                            candidate_model=ModelIdentity(
                                model_id="AutomatosX/AX-Qwen3.6-test-cand-0007-000",
                                revision=stable_sha256(plan)[:40],
                            ),
                            profile=plan.profile,
                            plan_sha256=stable_sha256(plan),
                            artifact_manifest_sha256="a" * 64,
                            quality_comparison_sha256="b" * 64,
                            validation_sha256="c" * 64,
                            measured_bpw=4.7,
                            objective_loss=0.1,
                            quality_retention=0.99,
                            mtp_acceptance_retention=0.97,
                            mtp_speedup=1.25,
                            peak_memory_ratio=0.8,
                            hardware=CompleteCandidateHardware(
                                device_name="Test Mac",
                                chip="M3 Max",
                                unified_memory_bytes=128 * 1024**3,
                                os_version="macOS",
                                ax_engine_version="6.11.1",
                                mlx_version="0.32",
                                mlx_lm_version="0.31",
                                power_mode="AC power",
                                kernel_fallbacks=0,
                            ),
                            validation_passed=True,
                        )
                    ],
                ),
            )
        elif operation in {"refine-select", "pareto"}:
            destination = Path(args[args.index("--output") + 1])
            destination.write_text("{}\n", encoding="utf-8")
        else:
            raise AssertionError(f"unexpected command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    result = execute_refinement(
        request_path=request,
        output_dir=tmp_path / "run",
        execute=True,
        runner=successful_runner,
    )

    assert result.complete
    assert result.measured_candidate_ids == [candidate_id]
    assert result.selected_result is not None
    assert result.pareto_report is not None
    assert (tmp_path / "run" / "measurements.json").is_file()
