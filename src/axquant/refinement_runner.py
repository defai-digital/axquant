from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from axquant.calibration import calibration_manifest_matches
from axquant.errors import ArtifactError, RefinementError
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    CalibrationManifest,
    QuantizationPlan,
    RefinementExecutionManifest,
    RefinementExecutionRequest,
    RefinementExecutionStep,
    RefinementMeasurementSet,
    RefinementResult,
    ValidationReport,
    utc_now,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )


def _resolved(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _required_file(base: Path, value: str, label: str) -> Path:
    path = _resolved(base, value)
    if not path.is_file():
        raise ArtifactError(f"{label} does not exist: {path}")
    return path


def _required_directory(base: Path, value: str, label: str) -> Path:
    path = _resolved(base, value)
    if not path.is_dir():
        raise ArtifactError(f"{label} does not exist: {path}")
    return path


def _candidate_revision(plan_sha256: str) -> str:
    return plan_sha256[:40]


def _candidate_model_id(prefix: str, candidate_id: str) -> str:
    return f"{prefix}-{candidate_id}"


def _command_steps(
    *,
    request: RefinementExecutionRequest,
    refinement: RefinementResult,
    output: Path,
    source_model: Path,
    mtp_sidecar: Path,
    calibration: Path,
    quality_dataset: Path,
    reference_quality: Path,
    reference_evaluation: Path,
    benchmark_prompts: Path,
    size_reference: Path,
) -> list[RefinementExecutionStep]:
    executable = request.axquant_executable
    steps: list[RefinementExecutionStep] = []
    for candidate_id, plan in sorted(refinement.candidate_plans.items()):
        candidate_root = output / "candidates" / candidate_id
        candidate_root.mkdir(parents=True, exist_ok=True)
        plan_path = candidate_root / "plan.json"
        if plan_path.exists():
            if stable_sha256(load_model(plan_path, QuantizationPlan)) != stable_sha256(plan):
                raise RefinementError(f"candidate plan changed on resume: {candidate_id}")
        else:
            write_data(plan_path, plan)
        artifact = candidate_root / "artifact"
        quality = candidate_root / "quality.json"
        quality_comparison = candidate_root / "quality-comparison.json"
        benchmark = candidate_root / "benchmark-ab"
        candidate_size = candidate_root / "size-evidence.json"
        validation = candidate_root / "validation.json"
        measurement = candidate_root / "measurement.json"
        plan_sha256 = stable_sha256(plan)
        revision = _candidate_revision(plan_sha256)
        model_id = _candidate_model_id(request.candidate_repository_prefix, candidate_id)

        convert = [
            executable,
            "convert",
            "--model",
            str(source_model),
            "--revision",
            plan.source_model.revision or "",
            "--plan",
            str(plan_path),
            "--mtp-sidecar",
            str(mtp_sidecar),
            "--mtp-layout",
            request.mtp_layout.value,
            "--calibration-manifest",
            str(calibration),
            "--ax-engine-manifest",
            "required",
            "--ax-engine-bench",
            request.ax_engine_executable,
            "--output",
            str(artifact),
        ]
        quality_command = [
            executable,
            "evaluate-quality",
            "--model",
            str(artifact),
            "--model-id",
            model_id,
            "--revision",
            revision,
            "--dataset",
            str(quality_dataset),
            "--max-seq-length",
            str(request.quality_max_sequence_length),
            "--max-tokens",
            str(request.quality_max_tokens),
            "--seed",
            str(request.random_seed),
            "--output",
            str(quality),
        ]
        if request.quality_max_samples is not None:
            quality_command.extend(["--max-samples", str(request.quality_max_samples)])
        benchmark_command = [
            executable,
            "benchmark-ab",
            "--model",
            str(artifact),
            "--model-id",
            model_id,
            "--revision",
            revision,
            "--prompts",
            str(benchmark_prompts),
            "--workload",
            request.profile.value,
            "--trials",
            str(request.benchmark_trials),
            "--warmup",
            str(request.benchmark_warmup),
            "--max-tokens",
            str(request.benchmark_max_tokens),
            "--power-mode",
            request.benchmark_power_mode,
            "--quantizer",
            plan.quantizer,
            "--quantizer-version",
            plan.software_versions.axquant,
            "--seed",
            str(request.random_seed),
            "--timeout",
            str(request.benchmark_timeout_seconds),
            "--ax-engine",
            request.ax_engine_executable,
            "--output-dir",
            str(benchmark),
            "--quality-evaluation",
            str(quality),
        ]
        if request.benchmark_draft_depth is not None:
            benchmark_command.extend(["--draft-depth", str(request.benchmark_draft_depth)])
        candidate_steps = [
            RefinementExecutionStep(
                candidate_id=candidate_id,
                step_id="convert",
                command=convert,
                expected_outputs=[str(artifact / "axquant_manifest.json")],
            ),
            RefinementExecutionStep(
                candidate_id=candidate_id,
                step_id="quality",
                command=quality_command,
                expected_outputs=[str(quality)],
            ),
            RefinementExecutionStep(
                candidate_id=candidate_id,
                step_id="quality-comparison",
                command=[
                    executable,
                    "compare-quality",
                    "--reference",
                    str(reference_quality),
                    "--candidate",
                    str(quality),
                    "--output",
                    str(quality_comparison),
                ],
                expected_outputs=[str(quality_comparison)],
            ),
            RefinementExecutionStep(
                candidate_id=candidate_id,
                step_id="benchmark-ab",
                command=benchmark_command,
                expected_outputs=[
                    str(benchmark / "evaluation_mtp_off.json"),
                    str(benchmark / "evaluation_mtp_on.json"),
                    str(benchmark / "mtp-off" / "benchmark_raw_log.json"),
                    str(benchmark / "mtp-on" / "benchmark_raw_log.json"),
                ],
            ),
            RefinementExecutionStep(
                candidate_id=candidate_id,
                step_id="size-evidence",
                command=[
                    executable,
                    "size-evidence",
                    "--artifact-manifest",
                    str(artifact / "axquant_manifest.json"),
                    "--model-id",
                    model_id,
                    "--revision",
                    revision,
                    "--output",
                    str(candidate_size),
                ],
                expected_outputs=[str(candidate_size)],
            ),
            RefinementExecutionStep(
                candidate_id=candidate_id,
                step_id="validate",
                command=[
                    executable,
                    "validate",
                    "--reference-evaluation",
                    str(reference_evaluation),
                    "--candidate-direct-evaluation",
                    str(benchmark / "evaluation_mtp_off.json"),
                    "--candidate-evaluation",
                    str(benchmark / "evaluation_mtp_on.json"),
                    "--calibration-manifest",
                    str(calibration),
                    "--size-reference",
                    str(size_reference),
                    "--candidate-size",
                    str(candidate_size),
                    "--profile",
                    request.profile.value,
                    "--output",
                    str(validation),
                ],
                expected_outputs=[str(validation)],
                acceptable_exit_codes=[0, 1],
            ),
            RefinementExecutionStep(
                candidate_id=candidate_id,
                step_id="measure",
                command=[
                    executable,
                    "refine-measure",
                    "--refinement",
                    str(_resolved(output, request.refinement_file)),
                    "--candidate-id",
                    candidate_id,
                    "--artifact-manifest",
                    str(artifact / "axquant_manifest.json"),
                    "--quality-comparison",
                    str(quality_comparison),
                    "--validation",
                    str(validation),
                    "--output",
                    str(measurement),
                ],
                expected_outputs=[str(measurement)],
            ),
        ]
        steps.extend(candidate_steps)

    measurements = output / "measurements.json"
    steps.extend(
        [
            RefinementExecutionStep(
                candidate_id=None,
                step_id="select",
                command=[
                    request.axquant_executable,
                    "refine-select",
                    "--refinement",
                    str(_resolved(output, request.refinement_file)),
                    "--measurements",
                    str(measurements),
                    "--output",
                    str(output / "selected-refinement.json"),
                ],
                expected_outputs=[str(output / "selected-refinement.json")],
            ),
            RefinementExecutionStep(
                candidate_id=None,
                step_id="pareto",
                command=[
                    request.axquant_executable,
                    "pareto",
                    "--measurements",
                    str(measurements),
                    "--output",
                    str(output / "pareto-report.json"),
                ],
                expected_outputs=[str(output / "pareto-report.json")],
            ),
        ]
    )
    return steps


def prepare_refinement_execution(
    *,
    request_path: str | Path,
    output_dir: str | Path,
) -> tuple[RefinementExecutionRequest, RefinementExecutionManifest]:
    request_source = Path(request_path).expanduser().resolve()
    request = load_model(request_source, RefinementExecutionRequest)
    base = request_source.parent
    refinement_path = _required_file(base, request.refinement_file, "refinement result")
    source_model = _required_directory(base, request.source_model, "source model")
    mtp_sidecar = _required_file(base, request.mtp_sidecar, "MTP sidecar")
    calibration = _required_file(base, request.calibration_manifest, "calibration manifest")
    quality_dataset = _required_file(base, request.quality_dataset, "quality dataset")
    reference_quality = _required_file(base, request.reference_quality, "reference quality")
    reference_evaluation = _required_file(
        base,
        request.reference_evaluation,
        "reference evaluation",
    )
    benchmark_prompts = _required_file(base, request.benchmark_prompts, "benchmark prompts")
    size_reference = _required_file(base, request.size_reference, "size reference")
    refinement = load_model(refinement_path, RefinementResult)
    calibration_manifest = load_model(calibration, CalibrationManifest)
    if refinement.selected_plan.profile != request.profile:
        raise RefinementError("refinement profile does not match execution request")
    for candidate_id, plan in refinement.candidate_plans.items():
        if plan.profile != request.profile:
            raise RefinementError(f"candidate profile differs: {candidate_id}")
        if not plan.evidence_kind.release_quality or plan.calibration is None:
            raise RefinementError(f"candidate lacks release-quality evidence: {candidate_id}")
        if (
            plan.source_model.model_id != calibration_manifest.model.model_id
            or plan.source_model.revision != calibration_manifest.model.revision
        ):
            raise RefinementError(f"candidate source differs from calibration: {candidate_id}")
        if not is_immutable_revision(plan.source_model.revision):
            raise RefinementError(f"candidate source revision is not immutable: {candidate_id}")
        expected_calibration_sha256 = plan.calibration.metadata.get("calibration_manifest_sha256")
        if not isinstance(expected_calibration_sha256, str) or not calibration_manifest_matches(
            calibration,
            calibration_manifest,
            expected_calibration_sha256,
        ):
            raise RefinementError(f"candidate calibration checksum differs: {candidate_id}")

    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "execution-manifest.json"
    if output.exists() and not manifest_path.is_file() and any(output.iterdir()):
        raise ArtifactError(f"refinement output exists without a manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    local_refinement = output / "refinement.json"
    if local_refinement.exists():
        if file_sha256(local_refinement) != file_sha256(refinement_path):
            raise RefinementError("refinement input changed on resume")
    else:
        shutil.copy2(refinement_path, local_refinement)
        if file_sha256(local_refinement) != file_sha256(refinement_path):
            raise RefinementError("refinement input checksum changed during copy")
    request = request.model_copy(update={"refinement_file": str(local_refinement)})
    input_sha256 = {
        "refinement": file_sha256(refinement_path),
        "source_config": file_sha256(source_model / "config.json"),
        "mtp_sidecar": file_sha256(mtp_sidecar),
        "calibration_manifest": file_sha256(calibration),
        "quality_dataset": file_sha256(quality_dataset),
        "reference_quality": file_sha256(reference_quality),
        "reference_evaluation": file_sha256(reference_evaluation),
        "benchmark_prompts": file_sha256(benchmark_prompts),
        "size_reference": file_sha256(size_reference),
    }
    steps = _command_steps(
        request=request,
        refinement=refinement,
        output=output,
        source_model=source_model,
        mtp_sidecar=mtp_sidecar,
        calibration=calibration,
        quality_dataset=quality_dataset,
        reference_quality=reference_quality,
        reference_evaluation=reference_evaluation,
        benchmark_prompts=benchmark_prompts,
        size_reference=size_reference,
    )
    request_sha256 = stable_sha256(request)
    refinement_sha256 = stable_sha256(refinement)
    if manifest_path.is_file():
        manifest = load_model(manifest_path, RefinementExecutionManifest)
        if (
            manifest.request_sha256 != request_sha256
            or manifest.refinement_sha256 != refinement_sha256
            or manifest.input_sha256 != input_sha256
        ):
            raise RefinementError("refinement execution inputs changed on resume")
        expected_commands = [
            (step.candidate_id, step.step_id, step.command, step.expected_outputs) for step in steps
        ]
        actual_commands = [
            (step.candidate_id, step.step_id, step.command, step.expected_outputs)
            for step in manifest.steps
        ]
        if actual_commands != expected_commands:
            raise RefinementError("refinement execution commands changed on resume")
        return request, manifest

    manifest = RefinementExecutionManifest(
        request_sha256=request_sha256,
        refinement_sha256=refinement_sha256,
        profile=request.profile,
        input_sha256=input_sha256,
        steps=steps,
    )
    write_data(manifest_path, manifest)
    return request, manifest


def _verified_outputs(step: RefinementExecutionStep) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for value in step.expected_outputs:
        path = Path(value)
        if not path.is_file():
            raise RefinementError(f"execution output is missing: {path}")
        outputs[value] = file_sha256(path)
    return outputs


def _merge_measurements(
    manifest: RefinementExecutionManifest,
    output: Path,
) -> list[str]:
    measurement_steps = [
        step
        for step in manifest.steps
        if step.candidate_id is not None and step.step_id == "measure" and step.state == "completed"
    ]
    measurements = []
    evaluator_version: str | None = None
    refinement_sha256: str | None = None
    for step in measurement_steps:
        measurement_set = load_model(step.expected_outputs[0], RefinementMeasurementSet)
        if evaluator_version is None:
            evaluator_version = measurement_set.evaluator_version
            refinement_sha256 = measurement_set.refinement_sha256
        elif (
            measurement_set.evaluator_version != evaluator_version
            or measurement_set.refinement_sha256 != refinement_sha256
        ):
            raise RefinementError("candidate measurement sets are incompatible")
        measurements.extend(measurement_set.measurements)
    if not measurements or evaluator_version is None or refinement_sha256 is None:
        return []
    measurement_ids = [measurement.measurement_id for measurement in measurements]
    if len(measurement_ids) != len(set(measurement_ids)):
        raise RefinementError("complete-candidate measurement IDs are duplicated")
    write_data(
        output / "measurements.json",
        RefinementMeasurementSet(
            refinement_sha256=refinement_sha256,
            evaluator_version=evaluator_version,
            measurements=measurements,
        ),
    )
    return sorted({measurement.candidate_id for measurement in measurements})


def execute_refinement(
    *,
    request_path: str | Path,
    output_dir: str | Path,
    execute: bool = False,
    runner: CommandRunner = _run,
) -> RefinementExecutionManifest:
    _, manifest = prepare_refinement_execution(
        request_path=request_path,
        output_dir=output_dir,
    )
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "execution-manifest.json"
    if not execute:
        return manifest

    failed_candidates = set(manifest.failed_candidate_ids)
    for step in manifest.steps:
        if step.candidate_id is None:
            continue
        if step.state == "completed":
            if _verified_outputs(step) != step.output_sha256:
                raise RefinementError(
                    f"completed execution output changed: {step.candidate_id}/{step.step_id}"
                )
            continue
        if step.state in {"failed", "skipped"}:
            if step.state == "failed" and step.candidate_id is not None:
                failed_candidates.add(step.candidate_id)
            continue
        if step.candidate_id in failed_candidates:
            step.state = "skipped"
            manifest.updated_at = utc_now()
            write_data(manifest_path, manifest)
            continue
        existing = [Path(value).is_file() for value in step.expected_outputs]
        if existing and all(existing):
            step.state = "completed"
            step.output_sha256 = _verified_outputs(step)
            if step.step_id == "validate":
                step.gate_passed = load_model(
                    step.expected_outputs[0],
                    ValidationReport,
                ).passed
            manifest.updated_at = utc_now()
            write_data(manifest_path, manifest)
            continue
        if any(existing):
            raise RefinementError(
                f"partial execution outputs exist: {step.candidate_id}/{step.step_id}"
            )
        try:
            completed = runner(step.command)
        except OSError as exc:
            step.state = "failed"
            step.stderr = str(exc)[:4096]
            failed_candidates.add(step.candidate_id)
        else:
            step.exit_code = completed.returncode
            step.stderr = completed.stderr[-4096:]
            if completed.returncode not in step.acceptable_exit_codes:
                step.state = "failed"
                failed_candidates.add(step.candidate_id)
            else:
                try:
                    step.output_sha256 = _verified_outputs(step)
                except RefinementError as exc:
                    step.state = "failed"
                    step.stderr = str(exc)[:4096]
                    failed_candidates.add(step.candidate_id)
                else:
                    step.state = "completed"
                    if step.step_id == "validate":
                        step.gate_passed = completed.returncode == 0
        manifest.failed_candidate_ids = sorted(failed_candidates)
        manifest.updated_at = utc_now()
        write_data(manifest_path, manifest)

    manifest.measured_candidate_ids = _merge_measurements(manifest, output)
    measurements_path = output / "measurements.json"
    for step in manifest.steps:
        if step.candidate_id is not None:
            continue
        if step.state == "completed":
            if _verified_outputs(step) != step.output_sha256:
                raise RefinementError(f"completed execution output changed: {step.step_id}")
            if step.step_id == "select":
                manifest.selected_result = step.expected_outputs[0]
            elif step.step_id == "pareto":
                manifest.pareto_report = step.expected_outputs[0]
            continue
        if step.state != "pending":
            continue
        if not measurements_path.is_file():
            step.state = "skipped"
            continue
        try:
            completed = runner(step.command)
        except OSError as exc:
            step.state = "failed"
            step.stderr = str(exc)[:4096]
            continue
        step.exit_code = completed.returncode
        step.stderr = completed.stderr[-4096:]
        if completed.returncode != 0:
            step.state = "failed"
            continue
        try:
            step.output_sha256 = _verified_outputs(step)
        except RefinementError as exc:
            step.state = "failed"
            step.stderr = str(exc)[:4096]
            continue
        step.state = "completed"
        if step.step_id == "select":
            manifest.selected_result = step.expected_outputs[0]
        elif step.step_id == "pareto":
            manifest.pareto_report = step.expected_outputs[0]
        manifest.updated_at = utc_now()
        write_data(manifest_path, manifest)

    manifest.complete = all(step.state != "pending" for step in manifest.steps)
    manifest.updated_at = utc_now()
    write_data(manifest_path, manifest)
    return manifest
