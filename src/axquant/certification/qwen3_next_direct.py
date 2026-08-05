from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path

from axquant.calibration import calibration_manifest_matches
from axquant.capture_binding import CAPTURE_METADATA_KEYS, activation_capture_evidence_issues
from axquant.certification.common import (
    architecture_fingerprint,
    bound_file,
    finite_median,
    required_directory,
    required_file,
    source_manifest_issues,
)
from axquant.certification.policy import direct_policy, direct_policy_sha256
from axquant.coding_suite import (
    NEAR_DUPLICATE_THRESHOLD,
    SANDBOX_PROFILE_SHA256,
    coding_general_overlap_issues,
    load_coding_payloads,
)
from axquant.errors import ArtifactError
from axquant.quality import load_quality_tasks
from axquant.release_audit import _artifact_issues, _wheel_identity
from axquant.reproduction import verify_reproduction
from axquant.schema import (
    ActivationCaptureManifest,
    ArtifactManifest,
    BaselineKind,
    CalibrationManifest,
    CodingOverlapReport,
    CodingScorer,
    CodingSuiteManifest,
    CodingSuiteSelfTestReport,
    DirectBaselineKind,
    DirectBenchmarkArm,
    DirectBenchmarkEvidenceIndex,
    DirectBenchmarkTrial,
    DirectHardwareProfileRegistry,
    DirectParetoPoint,
    DirectParetoReport,
    DirectQualityEvaluation,
    DirectQualityTaskOutcome,
    DirectRefinementMeasurementSet,
    DirectReleaseValidationIndex,
    DirectReleaseValidationRequest,
    DirectValidationEntry,
    EvidenceArchiveIndex,
    EvidenceKind,
    FeasibilityReport,
    Inventory,
    NonMtpGateId,
    ProfileName,
    QuantizationPlan,
    Qwen3NextCompatibilityMatrix,
    Qwen3NextCompatibilityRequest,
    Qwen3NextReleaseAudit,
    Qwen3NextReleaseAuditCheck,
    Qwen3NextReleaseAuditRequest,
    Qwen3NextTargetClass,
    RefinementResult,
    ReproductionRecipe,
    ReproductionVerification,
    RuntimeCheck,
    RuntimeName,
    SensitivityReport,
    SourceCheckpointManifest,
    SourceConversionProvenance,
    TensorRole,
)
from axquant.serde import file_sha256, load_model, stable_sha256

_REQUIRED_CODING_CATEGORIES = {
    "python": 24,
    "javascript-typescript": 20,
    "rust": 16,
    "go": 16,
    "repository-context": 16,
    "json-tool": 16,
    "algorithm-reasoning": 12,
    "long-context": 8,
}
_EXECUTABLE_SCORERS = {CodingScorer.UNIT_TEST, CodingScorer.COMPILE}
_TOOLCHAIN_BY_LANGUAGE = {
    "python": "python",
    "javascript": "node",
    "typescript": "typescript",
    "rust": "rust",
    "go": "go",
}
_FORBIDDEN_MODEL_CARD_CLAIMS = (
    "family certified",
    "qwen3-next is certified",
    "mtp speedup",
    "speculative decoding",
    "vlm certified",
    "kv-cache certified",
    "batching certified",
    "serving concurrency certified",
)


def _same_source(left: object, right: object) -> bool:
    return bool(
        getattr(left, "model_id", None) == getattr(right, "model_id", None)
        and getattr(left, "revision", None) == getattr(right, "revision", None)
    )


def _add_check(
    checks: list[Qwen3NextReleaseAuditCheck],
    gate_id: NonMtpGateId,
    name: str,
    issues: list[str],
    evidence: dict[str, Path],
) -> None:
    checks.append(
        Qwen3NextReleaseAuditCheck(
            gate_id=gate_id,
            name=name,
            passed=not issues,
            evidence_sha256={key: file_sha256(path) for key, path in evidence.items()},
            issues=issues,
        )
    )


def _candidate_target_bpw(target_class: Qwen3NextTargetClass) -> float:
    return 4.8 if target_class is Qwen3NextTargetClass.FOUR_BIT else 6.0


def _artifact_weight_ratio_limit(target_class: Qwen3NextTargetClass) -> float:
    policy = direct_policy()
    if target_class is Qwen3NextTargetClass.FOUR_BIT:
        return policy.four_bit_artifact_weight_ratio_max
    return policy.six_bit_artifact_weight_ratio_max


def _eligibility_issues(
    *,
    source_dir: Path,
    inventory: Inventory,
    plan: QuantizationPlan,
    manifest: ArtifactManifest,
) -> list[str]:
    issues: list[str] = []
    try:
        config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot load Qwen3-Next source config: {exc}") from exc
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_next":
        issues.append("source model_type is not qwen3_next")
    architectures = config.get("architectures") if isinstance(config, dict) else None
    if not isinstance(architectures, list) or "Qwen3NextForCausalLM" not in architectures:
        issues.append("source does not declare Qwen3NextForCausalLM")
    profile = inventory.architecture_profile
    if profile.adapter_id != "qwen3-next-v1":
        issues.append("source inventory is not classified by qwen3-next-v1")
    if profile.mtp_declared or inventory.mtp_present or inventory.mtp_weight_bytes:
        issues.append("source inventory declares or contains MTP")
    if any(tensor.role.is_mtp for tensor in inventory.tensors):
        issues.append("source inventory contains an MTP tensor role")
    if (source_dir / "mtp.safetensors").exists():
        issues.append("source checkpoint contains a root MTP sidecar")
    if plan.mtp.mode != "disabled":
        issues.append("candidate plan does not disable MTP")
    if manifest.mtp_present or manifest.mtp_weight_file_size_bytes:
        issues.append("candidate artifact declares MTP")
    if manifest.runtime.mtp.detected:
        issues.append("candidate runtime metadata declares MTP")
    if manifest.runtime.primary_runtime.mtp_support != "none" or any(
        runtime.mtp_support != "none" for runtime in manifest.runtime.compatible_runtimes
    ):
        issues.append("candidate runtime profiles declare MTP capability")
    return issues


def _feasibility_issues(
    report: FeasibilityReport,
    *,
    plan: QuantizationPlan,
    inventory: Inventory,
) -> list[str]:
    issues: list[str] = []
    if report.status != "ready-for-conversion" or report.blockers:
        issues.append("feasibility report is not ready for conversion")
    source = report.source
    if source is None:
        issues.append("feasibility report has no BF16 source")
    else:
        if source.kind is not BaselineKind.BF16_SOURCE or source.quantized is not False:
            issues.append("feasibility source is not the unquantized BF16 checkpoint")
        if not source.complete or source.issues or not source.inspected:
            issues.append("feasibility BF16 source is incomplete")
        if not _same_source(source.model, plan.source_model):
            issues.append("feasibility BF16 source differs from the candidate plan")
        if source.mtp_logical_parameters != 0 or source.mtp_weight_bytes != 0:
            issues.append("feasibility BF16 source unexpectedly contains MTP")
        if source.logical_parameters != inventory.total_parameters:
            issues.append("feasibility and inventory logical parameter counts differ")

    required = {
        BaselineKind.UNIFORM_4BIT,
        BaselineKind.UNIFORM_6BIT,
        BaselineKind.MIXED_PRECISION,
    }
    kinds = [baseline.kind for baseline in report.baselines]
    if len(kinds) != len(set(kinds)) or set(kinds) != required:
        issues.append("feasibility report lacks one complete 4/6/mixed baseline")
    for baseline in report.baselines:
        if not baseline.complete or baseline.issues or not baseline.inspected:
            issues.append(f"feasibility {baseline.kind.value} baseline is incomplete")
        if baseline.mtp_logical_parameters or baseline.mtp_weight_bytes:
            issues.append(f"feasibility {baseline.kind.value} baseline unexpectedly contains MTP")
        if baseline.logical_parameters != inventory.total_parameters:
            issues.append(f"feasibility {baseline.kind.value} parameter count differs")
    return issues


def _runtime_check_issues(
    check: RuntimeCheck,
    *,
    runtime: RuntimeName,
    kinds: set[str],
    artifact: Path,
    candidate_model_id: str,
    candidate_revision: str | None,
) -> list[str]:
    issues: list[str] = []
    if check.runtime is not runtime or check.check_kind not in kinds:
        issues.append(f"{runtime.value} check has the wrong runtime or check kind")
    if not check.available or not check.passed or check.exit_code not in {None, 0}:
        issues.append(f"{runtime.value} check did not pass")
    if check.model.model_id != candidate_model_id or check.model.revision != candidate_revision:
        issues.append(f"{runtime.value} check identifies another candidate")
    if check.model.local_path is None or Path(check.model.local_path).resolve() != artifact:
        issues.append(f"{runtime.value} check does not bind the candidate artifact")
    report_fallbacks = check.report.get("kernel_fallbacks", 0)
    if not isinstance(report_fallbacks, int) or isinstance(report_fallbacks, bool):
        issues.append(f"{runtime.value} check has malformed kernel fallback evidence")
    elif report_fallbacks != 0:
        issues.append(f"{runtime.value} check reports kernel fallback")
    return issues


def _matched_arm_fields(arm: DirectBenchmarkArm) -> tuple[object, ...]:
    return (
        arm.tokenizer_sha256,
        arm.prompt_sha256,
        arm.ordered_prompt_ids_sha256,
        arm.random_seed,
        arm.temperature,
        arm.top_p,
        arm.top_k,
        arm.max_tokens,
        arm.runtime,
        arm.runtime_version,
        arm.runtime_executable_sha256,
        tuple(sorted(arm.runtime_environment.items())),
        arm.hardware_scope_id,
        arm.os_version,
        arm.power_mode,
        arm.background_policy,
        sum(trial.warmup for trial in arm.trials),
        sum(not trial.warmup for trial in arm.trials),
    )


def _successful_trials(arm: DirectBenchmarkArm, *, warmup: bool) -> list[DirectBenchmarkTrial]:
    return [trial for trial in arm.trials if trial.warmup is warmup and trial.success]


def _benchmark_issues(
    index: DirectBenchmarkEvidenceIndex,
    *,
    target_class: Qwen3NextTargetClass,
    artifact_manifest_sha256: str,
    hardware_scope_ids: set[str],
) -> list[str]:
    policy = direct_policy()
    issues: list[str] = []
    expected_uniform = (
        DirectBaselineKind.UNIFORM_4BIT
        if target_class is Qwen3NextTargetClass.FOUR_BIT
        else DirectBaselineKind.UNIFORM_6BIT
    )
    for profile in index.profiles:
        arms = {arm.kind: arm for arm in profile.arms}
        controls = {_matched_arm_fields(arm) for arm in arms.values()}
        if len(controls) != 1:
            issues.append(f"{profile.profile.value} benchmark arms are not matched")
        for kind, arm in arms.items():
            warmups = _successful_trials(arm, warmup=True)
            measured = _successful_trials(arm, warmup=False)
            if len(warmups) < policy.benchmark_warmups_min:
                issues.append(f"{profile.profile.value}/{kind.value} has too few warmups")
            if len(measured) < policy.benchmark_successful_trials_min:
                issues.append(f"{profile.profile.value}/{kind.value} has too few measured trials")
            if any(trial.kernel_fallbacks for trial in arm.trials):
                issues.append(f"{profile.profile.value}/{kind.value} reports kernel fallback")
            agreement = arm.mlx_lm_matching_tokens / arm.mlx_lm_parity_tokens
            if agreement < policy.runtime_token_agreement_min:
                issues.append(f"{profile.profile.value}/{kind.value} fails MLX-LM token parity")
            if arm.hardware_scope_id not in hardware_scope_ids:
                issues.append(f"{profile.profile.value}/{kind.value} uses an unscoped host")

        candidate = arms[DirectBaselineKind.CANDIDATE]
        if candidate.artifact_manifest_sha256 != artifact_manifest_sha256:
            issues.append(f"{profile.profile.value} candidate benchmark binds another artifact")
        bf16 = arms[DirectBaselineKind.BF16]
        uniform = arms[expected_uniform]
        candidate_trials = _successful_trials(candidate, warmup=False)
        bf16_trials = _successful_trials(bf16, warmup=False)
        uniform_trials = _successful_trials(uniform, warmup=False)
        if candidate_trials and bf16_trials and uniform_trials:
            candidate_decode = finite_median(
                [float(trial.decode_tokens_per_second or 0.0) for trial in candidate_trials]
            )
            bf16_decode = finite_median(
                [float(trial.decode_tokens_per_second or 0.0) for trial in bf16_trials]
            )
            uniform_decode = finite_median(
                [float(trial.decode_tokens_per_second or 0.0) for trial in uniform_trials]
            )
            candidate_ttft = finite_median(
                [float(trial.ttft_seconds or 0.0) for trial in candidate_trials]
            )
            uniform_ttft = finite_median(
                [float(trial.ttft_seconds or 0.0) for trial in uniform_trials]
            )
            if candidate_decode / bf16_decode < policy.decode_speedup_vs_bf16_min:
                issues.append(f"{profile.profile.value} candidate decode speedup is too low")
            if candidate_decode / uniform_decode < policy.throughput_retention_vs_uniform_min:
                issues.append(f"{profile.profile.value} candidate uniform throughput is too low")
            if candidate_ttft / uniform_ttft > policy.ttft_ratio_vs_uniform_max:
                issues.append(f"{profile.profile.value} candidate TTFT regression is too high")
    return issues


def _sensitivity_issues(
    report: SensitivityReport,
    *,
    inventory: Inventory,
    plan: QuantizationPlan,
) -> list[str]:
    policy = direct_policy()
    issues: list[str] = []
    if report.evidence_kind is not EvidenceKind.MEASURED:
        issues.append("sensitivity evidence is not formal measured evidence")
    if not _same_source(report.model, plan.source_model):
        issues.append("sensitivity source differs from the candidate plan")
    if report.inventory_sha256 != stable_sha256(inventory):
        issues.append("sensitivity does not bind the source inventory")
    if plan.analysis_sha256 != stable_sha256(report):
        issues.append("candidate plan does not bind the selected sensitivity report")
    if plan.evidence_kind is not EvidenceKind.MEASURED or plan.calibration is None:
        issues.append("candidate plan is not based on formal measured evidence")
    if report.calibration is None:
        issues.append("sensitivity report lacks calibration provenance")
        return issues
    if report.calibration.samples < policy.calibration_samples_min:
        issues.append("calibration sample count is below certification minimum")
    if plan.calibration != report.calibration:
        issues.append("plan and sensitivity calibration evidence differ")
    tensor_entries = {entry.tensor.name: entry for entry in report.entries}
    inventory_names = {tensor.name for tensor in inventory.tensors}
    if set(tensor_entries) != inventory_names:
        issues.append("sensitivity does not cover every inventory tensor exactly once")
    for assignment in plan.assignments:
        entry = tensor_entries.get(assignment.tensor)
        if entry is None:
            continue
        candidates = [
            candidate
            for candidate in entry.candidates
            if candidate.bits == assignment.bits
            and candidate.method == assignment.method
            and candidate.group_size == assignment.group_size
        ]
        if not candidates:
            issues.append(f"selected sensitivity candidate is missing: {assignment.tensor}")
            continue
        candidate = candidates[0]
        metric_values = candidate.metrics.model_dump().values()
        if not candidate.supported or any(not isfinite(float(value)) for value in metric_values):
            issues.append(f"selected sensitivity candidate is invalid: {assignment.tensor}")
        if assignment.bits < 16 and candidate.measured_tokens < policy.sensitivity_tokens_min:
            issues.append(f"selected sensitivity candidate has too few tokens: {assignment.tensor}")
    return issues


def _coding_manifest_issues(manifest_path: Path, manifest: CodingSuiteManifest) -> list[str]:
    policy = direct_policy()
    issues: list[str] = []
    if len(manifest.tasks) < policy.coding_tasks_min:
        issues.append("coding suite has fewer than 128 tasks")
    categories = Counter(task.category for task in manifest.tasks)
    for category, minimum in _REQUIRED_CODING_CATEGORIES.items():
        if categories[category] < minimum:
            issues.append(f"coding suite category {category!r} has fewer than {minimum} tasks")
    executable = sum(task.scorer in _EXECUTABLE_SCORERS for task in manifest.tasks)
    if executable * 2 < len(manifest.tasks):
        issues.append("fewer than half of coding-suite tasks use executable scorers")
    if any(not task.license_id.strip() or not task.provenance.strip() for task in manifest.tasks):
        issues.append("coding suite contains a task without provenance or license")
    if sum(task.target_tokens for task in manifest.tasks) < policy.coding_scored_tokens_min:
        issues.append("coding suite target-token budget is below certification minimum")
    if manifest.sandbox_profile_sha256 != SANDBOX_PROFILE_SHA256:
        issues.append("coding suite uses an unknown sandbox policy")
    if manifest.near_duplicate_threshold != NEAR_DUPLICATE_THRESHOLD:
        issues.append("coding suite near-duplicate threshold differs from frozen policy")
    required_toolchains = {"python", "node", "typescript", "rust", "go", "sandbox"}
    if not required_toolchains <= set(manifest.toolchains) or any(
        manifest.toolchains.get(name) == "unavailable" for name in required_toolchains
    ):
        issues.append("coding suite does not bind every required toolchain identity")
    for relative_name, expected_sha256 in manifest.task_shards.items():
        try:
            bound_file(
                manifest_path.parent,
                relative_name,
                expected_sha256,
                f"coding suite shard {relative_name}",
            )
        except ArtifactError as exc:
            issues.append(str(exc))
    if manifest.dataset_sha256 != stable_sha256(manifest.task_shards):
        issues.append("coding suite dataset digest differs from its task shards")
    try:
        load_coding_payloads(manifest_path, manifest)
    except ArtifactError as exc:
        issues.append(str(exc))
    return issues


def _coding_self_test_issues(
    report: CodingSuiteSelfTestReport,
    *,
    report_path: Path,
    manifest: CodingSuiteManifest,
    manifest_path: Path,
) -> tuple[list[str], list[Path]]:
    issues = list(report.issues)
    if report.suite_manifest_sha256 != file_sha256(manifest_path):
        issues.append("coding suite self-test binds another manifest")
    if report.toolchains != manifest.toolchains:
        issues.append("coding suite self-test uses another toolchain set")
    if report.sandbox_profile_sha256 != manifest.sandbox_profile_sha256:
        issues.append("coding suite self-test uses another sandbox policy")
    tasks = {task.task_id: task for task in manifest.tasks}
    expected_ids = [task.task_id for task in manifest.tasks]
    if [outcome.task_id for outcome in report.oracle_outcomes] != expected_ids:
        issues.append("coding suite oracle self-test membership differs")
    if [outcome.task_id for outcome in report.empty_mutant_outcomes] != expected_ids:
        issues.append("coding suite mutant self-test membership differs")
    for phase, outcomes, expected_score in (
        ("oracle", report.oracle_outcomes, 1.0),
        ("empty mutant", report.empty_mutant_outcomes, 0.0),
    ):
        for outcome in outcomes:
            task = tasks.get(outcome.task_id)
            if task is None:
                continue
            if outcome.scorer is not task.scorer:
                issues.append(f"coding suite {phase} scorer differs: {outcome.task_id}")
            if outcome.score != expected_score or outcome.model_error:
                issues.append(f"coding suite {phase} score failed: {outcome.task_id}")
            if outcome.infrastructure_error:
                issues.append(f"coding suite {phase} infrastructure failed: {outcome.task_id}")
            if outcome.output_file is None:
                issues.append(f"coding suite {phase} output is not archived: {outcome.task_id}")
            if task.scorer in _EXECUTABLE_SCORERS:
                if not outcome.sandboxed or not outcome.network_disabled:
                    issues.append(f"coding suite {phase} lacks sandbox proof: {outcome.task_id}")
                if outcome.timed_out is not False:
                    issues.append(f"coding suite {phase} timed out: {outcome.task_id}")
                toolchain_key = _TOOLCHAIN_BY_LANGUAGE.get(task.language)
                if toolchain_key is None or outcome.toolchain != manifest.toolchains.get(
                    toolchain_key
                ):
                    issues.append(f"coding suite {phase} toolchain differs: {outcome.task_id}")
                if outcome.sandbox_profile_sha256 != manifest.sandbox_profile_sha256:
                    issues.append(f"coding suite {phase} sandbox differs: {outcome.task_id}")
                if phase == "oracle" and outcome.syntax_valid is not True:
                    issues.append(f"coding suite oracle does not compile: {outcome.task_id}")
                if (
                    phase == "oracle"
                    and task.scorer is CodingScorer.UNIT_TEST
                    and outcome.unit_tests_passed is not True
                ):
                    issues.append(f"coding suite oracle tests failed: {outcome.task_id}")
    if report.passed != (not issues):
        issues.append("coding suite self-test declared status differs from raw outcomes")
    dependency_paths = _verify_quality_outcome_files(
        report_path.parent,
        report.oracle_outcomes,
        label="coding suite oracle self-test",
    )
    dependency_paths.extend(
        _verify_quality_outcome_files(
            report_path.parent,
            report.empty_mutant_outcomes,
            label="coding suite mutant self-test",
        )
    )
    return issues, dependency_paths


def _quality_metrics(evaluation: DirectQualityEvaluation) -> dict[str, float | int]:
    outcomes = evaluation.outcomes
    syntax = [outcome.syntax_valid for outcome in outcomes if outcome.syntax_valid is not None]
    tools = [outcome.tool_valid for outcome in outcomes if outcome.tool_valid is not None]
    unit_tests = [
        outcome.unit_tests_passed for outcome in outcomes if outcome.unit_tests_passed is not None
    ]
    return {
        "aggregate": sum(outcome.score for outcome in outcomes) / len(outcomes),
        "scored_tokens": sum(outcome.scored_tokens for outcome in outcomes),
        "syntax": sum(bool(value) for value in syntax) / len(syntax) if syntax else 0.0,
        "tools": sum(bool(value) for value in tools) / len(tools) if tools else 0.0,
        "unit_tests": (
            sum(bool(value) for value in unit_tests) / len(unit_tests) if unit_tests else 0.0
        ),
        "model_errors": sum(outcome.model_error for outcome in outcomes),
        "infra_errors": sum(outcome.infrastructure_error for outcome in outcomes),
    }


def _verify_quality_outcome_files(
    base: Path,
    outcomes: list[DirectQualityTaskOutcome],
    *,
    label: str,
) -> list[Path]:
    files: list[Path] = []
    for outcome in outcomes:
        if outcome.output_file is not None:
            files.append(
                bound_file(
                    base,
                    outcome.output_file,
                    outcome.output_sha256,
                    f"{label} model output for {outcome.task_id}",
                )
            )
        raw_fields = (
            outcome.stdout_file,
            outcome.stderr_file,
            outcome.stdout_sha256,
            outcome.stderr_sha256,
        )
        if any(value is not None for value in raw_fields) and not all(raw_fields):
            raise ArtifactError(f"{label} has an incomplete raw-log binding: {outcome.task_id}")
        if outcome.stdout_file is not None and outcome.stdout_sha256 is not None:
            files.append(
                bound_file(
                    base,
                    outcome.stdout_file,
                    outcome.stdout_sha256,
                    f"{label} stdout for {outcome.task_id}",
                )
            )
        if outcome.stderr_file is not None and outcome.stderr_sha256 is not None:
            files.append(
                bound_file(
                    base,
                    outcome.stderr_file,
                    outcome.stderr_sha256,
                    f"{label} stderr for {outcome.task_id}",
                )
            )
    return files


def _validation_evidence(
    index_path: Path,
    index: DirectReleaseValidationIndex,
) -> tuple[
    dict[ProfileName, tuple[DirectQualityEvaluation, DirectQualityEvaluation]],
    list[Path],
    dict[ProfileName, Path],
]:
    dependency_paths: list[Path] = [
        bound_file(
            index_path.parent,
            index.general_calibration_overlap_report_file,
            index.general_calibration_overlap_report_sha256,
            "general calibration-overlap report",
        )
    ]
    manifest_paths: dict[ProfileName, Path] = {}

    def load_evaluation(path: Path, label: str) -> DirectQualityEvaluation:
        evaluation = load_model(path, DirectQualityEvaluation)
        dependency_paths.extend(
            _verify_quality_outcome_files(path.parent, evaluation.outcomes, label=label)
        )
        return evaluation

    evidence: dict[ProfileName, tuple[DirectQualityEvaluation, DirectQualityEvaluation]] = {}
    for entry in index.entries:
        manifest_path = bound_file(
            index_path.parent,
            entry.evaluation_manifest_file,
            entry.evaluation_manifest_sha256,
            f"{entry.profile.value} evaluation manifest",
        )
        dependency_paths.append(manifest_path)
        manifest_paths[entry.profile] = manifest_path
        reference_path = bound_file(
            index_path.parent,
            entry.reference_evaluation_file,
            entry.reference_evaluation_sha256,
            f"{entry.profile.value} reference quality evaluation",
        )
        candidate_path = bound_file(
            index_path.parent,
            entry.candidate_evaluation_file,
            entry.candidate_evaluation_sha256,
            f"{entry.profile.value} candidate quality evaluation",
        )
        dependency_paths.extend((reference_path, candidate_path))
        evidence[entry.profile] = (
            load_evaluation(reference_path, f"{entry.profile.value} BF16 quality evaluation"),
            load_evaluation(candidate_path, f"{entry.profile.value} candidate quality evaluation"),
        )
        if any(
            evaluation.evaluation_manifest_sha256 != entry.evaluation_manifest_sha256
            for evaluation in evidence[entry.profile]
        ):
            raise ArtifactError(
                f"{entry.profile.value} quality evaluation binds another evaluation manifest"
            )
    return evidence, dependency_paths, manifest_paths


def _quality_issues(
    index: DirectReleaseValidationIndex,
    evidence: dict[ProfileName, tuple[DirectQualityEvaluation, DirectQualityEvaluation]],
    *,
    suite: CodingSuiteManifest,
    calibration_dataset_sha256: str,
    reference_model: object,
    candidate_model: object,
    tokenizer_sha256: str,
    reference_model_artifact_sha256: str,
    candidate_model_artifact_sha256: str,
    suite_manifest_sha256: str,
    toolkit_version: str,
    general_task_categories: dict[str, str],
    check_declared_status: bool = True,
) -> list[str]:
    policy = direct_policy()
    issues = list(index.issues)
    dataset_digests: set[str] = set()
    entries = {entry.profile: entry for entry in index.entries}
    suite_task_ids = {task.task_id for task in suite.tasks}
    task_categories = {task.task_id: task.category for task in suite.tasks}
    coding_tasks = {task.task_id: task for task in suite.tasks}
    for profile, (reference, candidate) in evidence.items():
        issue_count_before = len(issues)
        entry = entries[profile]
        if reference.model != reference_model:
            issues.append(f"{profile.value} reference quality uses the wrong BF16 model")
        if candidate.model != candidate_model:
            issues.append(f"{profile.value} candidate quality uses the wrong candidate model")
        if reference.model_artifact_sha256 != reference_model_artifact_sha256:
            issues.append(f"{profile.value} reference quality binds the wrong BF16 artifact")
        if candidate.model_artifact_sha256 != candidate_model_artifact_sha256:
            issues.append(f"{profile.value} candidate quality binds the wrong candidate artifact")
        if profile is ProfileName.AGENT_CODING and (
            entry.evaluation_manifest_sha256 != suite_manifest_sha256
        ):
            issues.append("agent-coding quality binds another coding-suite manifest")
        if profile is ProfileName.GENERAL and (
            reference.dataset_sha256 != entry.evaluation_manifest_sha256
            or candidate.dataset_sha256 != entry.evaluation_manifest_sha256
        ):
            issues.append("general quality dataset differs from its bound manifest")
        if (
            reference.tokenizer_sha256 != tokenizer_sha256
            or candidate.tokenizer_sha256 != tokenizer_sha256
        ):
            issues.append(f"{profile.value} quality uses the wrong tokenizer")
        if reference.generation != candidate.generation:
            issues.append(f"{profile.value} quality generation settings differ")
        if reference.random_seed != candidate.random_seed:
            issues.append(f"{profile.value} quality random seeds differ")
        if reference.evaluated_tokens != candidate.evaluated_tokens:
            issues.append(f"{profile.value} quality token coverage differs")
        if reference.software_versions != candidate.software_versions:
            issues.append(f"{profile.value} quality software environments differ")
        if (
            reference.software_versions.axquant != toolkit_version
            or candidate.software_versions.axquant != toolkit_version
        ):
            issues.append(f"{profile.value} quality uses another AXQuant version")
        if reference.profile is not profile or candidate.profile is not profile:
            issues.append(f"{profile.value} quality evidence has the wrong profile")
        if reference.dataset_sha256 != candidate.dataset_sha256:
            issues.append(f"{profile.value} reference and candidate datasets differ")
        dataset_digests.add(candidate.dataset_sha256)
        if candidate.dataset_sha256 == calibration_dataset_sha256:
            issues.append(f"{profile.value} evaluation overlaps calibration by digest")
        reference_tasks = {outcome.task_id for outcome in reference.outcomes}
        candidate_tasks = {outcome.task_id for outcome in candidate.outcomes}
        if reference_tasks != candidate_tasks:
            issues.append(f"{profile.value} quality task membership differs")
        if profile is ProfileName.AGENT_CODING and candidate_tasks != suite_task_ids:
            issues.append("agent-coding quality membership differs from coding-suite manifest")
        if profile is ProfileName.GENERAL and candidate_tasks != set(general_task_categories):
            issues.append("general quality membership differs from its evaluation manifest")
        if profile is ProfileName.GENERAL:
            for evaluation_name, evaluation in (
                ("BF16", reference),
                ("candidate", candidate),
            ):
                for outcome in evaluation.outcomes:
                    if outcome.output_file is None:
                        issues.append(
                            f"{evaluation_name} general output is not archived: {outcome.task_id}"
                        )

        reference_metrics = _quality_metrics(reference)
        candidate_metrics = _quality_metrics(candidate)
        if candidate_metrics["model_errors"] > policy.model_runtime_errors_max:
            issues.append(f"{profile.value} candidate has model errors")
        if candidate_metrics["infra_errors"] != 0 or reference_metrics["infra_errors"] != 0:
            issues.append(f"{profile.value} quality evidence has infrastructure errors")
        if candidate_metrics["scored_tokens"] < 1 or reference_metrics["scored_tokens"] < 1:
            issues.append(f"{profile.value} has no generated tokens to score")
        perplexity_limit = (
            policy.agent_coding_perplexity_ratio_max
            if profile is ProfileName.AGENT_CODING
            else policy.general_perplexity_ratio_max
        )
        if candidate.perplexity / reference.perplexity > perplexity_limit:
            issues.append(f"{profile.value} perplexity retention failed")
        reference_aggregate = float(reference_metrics["aggregate"])
        if reference_aggregate <= 0.0:
            issues.append(f"{profile.value} BF16 aggregate score is not meaningful")
        elif (
            float(candidate_metrics["aggregate"]) / reference_aggregate
            < policy.aggregate_retention_min
        ):
            issues.append(f"{profile.value} aggregate task retention failed")

        if profile is ProfileName.AGENT_CODING:
            reference_by_id = {outcome.task_id: outcome for outcome in reference.outcomes}
            candidate_by_id = {outcome.task_id: outcome for outcome in candidate.outcomes}
            for evaluation_name, outcomes_by_id in (
                ("BF16", reference_by_id),
                ("candidate", candidate_by_id),
            ):
                for task_id in suite_task_ids & outcomes_by_id.keys():
                    task = coding_tasks[task_id]
                    outcome = outcomes_by_id[task_id]
                    if outcome.scorer is not task.scorer:
                        issues.append(f"{evaluation_name} coding scorer differs for {task_id}")
                    if outcome.output_file is None:
                        issues.append(f"{evaluation_name} coding output is not archived: {task_id}")
                    if outcome.scored_tokens > task.target_tokens:
                        issues.append(
                            f"{evaluation_name} scored-token count exceeds target for {task_id}"
                        )
                    if task.scorer in _EXECUTABLE_SCORERS:
                        if not outcome.sandboxed or not outcome.network_disabled:
                            issues.append(
                                f"{evaluation_name} executable task lacks sandbox proof: {task_id}"
                            )
                        if outcome.timed_out is not False:
                            issues.append(f"{evaluation_name} executable task timed out: {task_id}")
                        if not all(
                            (
                                outcome.stdout_file,
                                outcome.stderr_file,
                                outcome.stdout_sha256,
                                outcome.stderr_sha256,
                            )
                        ):
                            issues.append(
                                f"{evaluation_name} executable task lacks raw logs: {task_id}"
                            )
                        if outcome.sandbox_profile_sha256 != suite.sandbox_profile_sha256:
                            issues.append(f"{evaluation_name} sandbox policy differs for {task_id}")
                        toolchain_key = _TOOLCHAIN_BY_LANGUAGE.get(task.language)
                        if toolchain_key is None or outcome.toolchain != suite.toolchains.get(
                            toolchain_key
                        ):
                            issues.append(
                                f"{evaluation_name} toolchain identity differs for {task_id}"
                            )
                        if outcome.syntax_valid is None:
                            issues.append(
                                f"{evaluation_name} executable task lacks compile result: {task_id}"
                            )
                        if (
                            task.scorer is CodingScorer.UNIT_TEST
                            and outcome.unit_tests_passed is None
                        ):
                            issues.append(
                                f"{evaluation_name} unit-test task lacks test result: {task_id}"
                            )
            category_scores: dict[str, tuple[list[float], list[float]]] = defaultdict(
                lambda: ([], [])
            )
            for task_id in suite_task_ids & reference_tasks & candidate_tasks:
                reference_scores, candidate_scores = category_scores[task_categories[task_id]]
                reference_scores.append(reference_by_id[task_id].score)
                candidate_scores.append(candidate_by_id[task_id].score)
            for category, (reference_scores, candidate_scores) in category_scores.items():
                reference_score = sum(reference_scores) / len(reference_scores)
                candidate_score = sum(candidate_scores) / len(candidate_scores)
                if reference_score <= 0.0 or (
                    candidate_score / reference_score < policy.aggregate_retention_min
                ):
                    issues.append(f"coding category retention failed: {category}")

            reference_syntax = float(reference_metrics["syntax"])
            candidate_syntax = float(candidate_metrics["syntax"])
            if reference_syntax < policy.syntax_validity_min:
                issues.append("BF16 syntax/compile validity is below the absolute floor")
            if candidate_syntax < policy.syntax_validity_min or (
                candidate_syntax - reference_syntax < policy.syntax_validity_delta_min
            ):
                issues.append("candidate syntax/compile validity failed")
            reference_tools = float(reference_metrics["tools"])
            candidate_tools = float(candidate_metrics["tools"])
            if reference_tools < policy.tool_validity_min:
                issues.append("BF16 JSON/tool validity is below the absolute floor")
            if candidate_tools < policy.tool_validity_min or (
                candidate_tools - reference_tools < policy.tool_validity_delta_min
            ):
                issues.append("candidate JSON/tool validity failed")
        else:
            reference_by_id = {outcome.task_id: outcome for outcome in reference.outcomes}
            candidate_by_id = {outcome.task_id: outcome for outcome in candidate.outcomes}
            general_category_scores: dict[str, tuple[list[float], list[float]]] = defaultdict(
                lambda: ([], [])
            )
            for task_id in set(general_task_categories) & reference_tasks & candidate_tasks:
                reference_scores, candidate_scores = general_category_scores[
                    general_task_categories[task_id]
                ]
                reference_scores.append(reference_by_id[task_id].score)
                candidate_scores.append(candidate_by_id[task_id].score)
            for category, (
                reference_scores,
                candidate_scores,
            ) in general_category_scores.items():
                reference_score = sum(reference_scores) / len(reference_scores)
                candidate_score = sum(candidate_scores) / len(candidate_scores)
                if reference_score <= 0.0 or (
                    candidate_score / reference_score < policy.aggregate_retention_min
                ):
                    issues.append(f"general quality category retention failed: {category}")
        computed_passed = len(issues) == issue_count_before
        if check_declared_status and entry.passed != computed_passed:
            issues.append(f"{profile.value} declared validation status differs from raw evidence")
    if len(dataset_digests) != 2:
        issues.append("agent-coding and general evaluation datasets are not distinct")
    if not index.release_ready:
        issues.append("direct release validation index is not release-ready")
    return issues


def build_direct_release_validation_index(
    request_path: str | Path,
) -> DirectReleaseValidationIndex:
    """Build the fail-closed direct-track quality index from raw BF16/candidate evidence."""

    request_source = Path(request_path).expanduser().resolve()
    request = load_model(request_source, DirectReleaseValidationRequest)
    base = request_source.parent
    source_path = required_file(
        base,
        request.source_checkpoint_manifest,
        "source checkpoint manifest",
    )
    candidate_manifest_path = required_file(
        base,
        request.candidate_artifact_manifest,
        "candidate artifact manifest",
    )
    coding_path = required_file(base, request.coding_suite_manifest, "coding suite manifest")
    general_overlap_path = required_file(
        base,
        request.general_calibration_overlap_report,
        "general calibration-overlap report",
    )
    source = load_model(source_path, SourceCheckpointManifest)
    candidate_manifest = load_model(candidate_manifest_path, ArtifactManifest)
    coding = load_model(coding_path, CodingSuiteManifest)

    entries = [
        DirectValidationEntry(
            profile=requested.profile,
            evaluation_manifest_file=requested.evaluation_manifest_file,
            evaluation_manifest_sha256=file_sha256(
                required_file(
                    base,
                    requested.evaluation_manifest_file,
                    f"{requested.profile.value} evaluation manifest",
                )
            ),
            reference_evaluation_file=requested.reference_evaluation_file,
            reference_evaluation_sha256=file_sha256(
                required_file(
                    base,
                    requested.reference_evaluation_file,
                    f"{requested.profile.value} reference quality evaluation",
                )
            ),
            candidate_evaluation_file=requested.candidate_evaluation_file,
            candidate_evaluation_sha256=file_sha256(
                required_file(
                    base,
                    requested.candidate_evaluation_file,
                    f"{requested.profile.value} candidate quality evaluation",
                )
            ),
            passed=True,
        )
        for requested in sorted(request.entries, key=lambda item: item.profile.value)
    ]
    provisional = DirectReleaseValidationIndex(
        entries=entries,
        general_calibration_overlap_report_file=request.general_calibration_overlap_report,
        general_calibration_overlap_report_sha256=file_sha256(general_overlap_path),
        release_ready=True,
    )
    evidence, _dependencies, manifest_paths = _validation_evidence(request_source, provisional)
    general_tasks = load_quality_tasks(manifest_paths[ProfileName.GENERAL])
    general_task_categories = {task.task_id: task.category for task in general_tasks}

    issues: list[str] = []
    if request.policy_sha256 != direct_policy_sha256():
        issues.append("validation request policy digest differs from the wheel-owned policy")
    if candidate_manifest.axquant_version != request.required_toolkit_version:
        issues.append("candidate artifact uses another AXQuant version")
    if candidate_manifest.source_model != source.source_model:
        issues.append("candidate artifact source differs from the immutable BF16 source")
    issues.extend(_coding_manifest_issues(coding_path, coding))
    issues.extend(
        coding_general_overlap_issues(
            coding_payloads=load_coding_payloads(coding_path, coding),
            general_tasks=general_tasks,
            similarity_threshold=coding.near_duplicate_threshold,
        )
    )
    overlap_path = bound_file(
        coding_path.parent,
        coding.calibration_overlap_report,
        coding.calibration_overlap_report_sha256,
        "coding calibration-overlap report",
    )
    overlap = load_model(overlap_path, CodingOverlapReport)
    general_overlap = load_model(general_overlap_path, CodingOverlapReport)
    if not overlap.passed or overlap.matches:
        issues.append("coding suite overlaps calibration")
    if overlap.calibration_dataset_sha256 != request.calibration_dataset_sha256:
        issues.append("coding overlap report uses another calibration dataset")
    if overlap.suite_dataset_sha256 != coding.dataset_sha256:
        issues.append("coding overlap report uses another coding suite")
    general_manifest_sha256 = file_sha256(manifest_paths[ProfileName.GENERAL])
    if not general_overlap.passed or general_overlap.matches:
        issues.append("general quality suite overlaps calibration")
    if general_overlap.calibration_dataset_sha256 != request.calibration_dataset_sha256:
        issues.append("general overlap report uses another calibration dataset")
    if general_overlap.suite_dataset_sha256 != general_manifest_sha256:
        issues.append("general overlap report uses another general quality suite")

    candidate_model = evidence[ProfileName.AGENT_CODING][1].model
    issues.extend(
        _quality_issues(
            provisional,
            evidence,
            suite=coding,
            calibration_dataset_sha256=request.calibration_dataset_sha256,
            reference_model=source.source_model,
            candidate_model=candidate_model,
            tokenizer_sha256=source.tokenizer_sha256,
            reference_model_artifact_sha256=file_sha256(source_path),
            candidate_model_artifact_sha256=file_sha256(candidate_manifest_path),
            suite_manifest_sha256=file_sha256(coding_path),
            toolkit_version=request.required_toolkit_version,
            general_task_categories=general_task_categories,
            check_declared_status=False,
        )
    )
    issues = list(dict.fromkeys(issues))
    if issues:
        entry_issues = ["validation is blocked; see the index-level recomputed issues"]
        entries = [
            entry.model_copy(update={"passed": False, "issues": entry_issues}) for entry in entries
        ]
    return DirectReleaseValidationIndex(
        entries=entries,
        general_calibration_overlap_report_file=request.general_calibration_overlap_report,
        general_calibration_overlap_report_sha256=file_sha256(general_overlap_path),
        release_ready=not issues,
        issues=issues,
    )


def _architecture_issues(
    *,
    inventory: Inventory,
    plan: QuantizationPlan,
    fingerprint: object,
) -> list[str]:
    issues: list[str] = []
    if getattr(fingerprint, "text_layer_count", None) != 48:
        issues.append("exact Qwen3-Coder-Next scope requires 48 text layers")
    if getattr(fingerprint, "full_attention_interval", None) != 4:
        issues.append("exact Qwen3-Coder-Next full-attention cadence differs")
    if getattr(fingerprint, "expert_count", None) != 512:
        issues.append("exact Qwen3-Coder-Next expert count differs")
    if getattr(fingerprint, "experts_per_token", None) != 10:
        issues.append("exact Qwen3-Coder-Next experts-per-token differs")
    quantizable_other = [
        tensor.name
        for tensor in inventory.tensors
        if tensor.quantizable and tensor.role is TensorRole.OTHER
    ]
    if quantizable_other:
        issues.append(f"inventory has unclassified quantizable tensors: {quantizable_other[:5]}")
    experts = [tensor for tensor in inventory.tensors if tensor.role is TensorRole.EXPERT]
    if not experts or not any(len(tensor.shape) == 3 for tensor in experts):
        issues.append("inventory does not prove fused 3-D expert coverage")
    routers = [tensor for tensor in inventory.tensors if tensor.role is TensorRole.ROUTER]
    if not routers:
        issues.append("inventory does not contain Qwen3-Next routers")
    assignments = {assignment.tensor: assignment for assignment in plan.assignments}
    for tensor in inventory.tensors:
        assignment = assignments.get(tensor.name)
        if assignment is None:
            issues.append(f"plan does not cover inventory tensor: {tensor.name}")
            continue
        minimum = {
            TensorRole.NORM: 16,
            TensorRole.LM_HEAD: plan.constraints.lm_head_min_bits,
            TensorRole.EMBEDDING: 8,
            TensorRole.ROUTER: 8,
        }.get(tensor.role)
        if minimum is not None and assignment.bits < minimum:
            issues.append(f"protected tensor is below its floor: {tensor.name}")
    return issues


def _refinement_issues(
    refinement: RefinementResult,
    measurements: DirectRefinementMeasurementSet,
    *,
    plan: QuantizationPlan,
    manifest_sha256: str,
    target_class: Qwen3NextTargetClass,
    candidate_model_id: str,
    quality_evidence_sha256: str,
    benchmark_evidence_sha256: str,
) -> list[str]:
    issues: list[str] = []
    if (
        refinement.selection_basis != "complete-model"
        or refinement.evidence_label != "holdout-bound"
    ):
        issues.append("refinement selection is not holdout-bound complete-model evidence")
    if refinement.selected_plan != plan or refinement.selected_plan_sha256 != stable_sha256(plan):
        issues.append("refinement result does not select the artifact plan")
    if measurements.refinement_sha256 != stable_sha256(refinement):
        issues.append("direct measurements do not bind the refinement result")
    if measurements.evaluator_version != f"axquant:{direct_policy().policy_id}":
        issues.append("direct measurements use another evaluator policy")
    if any(
        measurement.quality_evidence_sha256 != quality_evidence_sha256
        or measurement.benchmark_evidence_sha256 != benchmark_evidence_sha256
        for measurement in measurements.measurements
    ):
        issues.append("direct measurements do not bind the quality/benchmark evidence")
    selected = [
        measurement
        for measurement in measurements.measurements
        if measurement.candidate_id == measurements.selected_candidate_id
    ]
    if len(selected) != 1:
        issues.append("direct measurements do not contain one selected candidate")
        return issues
    candidate = selected[0]
    if (
        candidate.candidate_id != refinement.selected_candidate_id
        or candidate.plan_sha256 != stable_sha256(plan)
        or candidate.artifact_manifest_sha256 != manifest_sha256
        or candidate.target_class is not target_class
        or candidate.candidate_model.model_id != candidate_model_id
        or not candidate.validation_passed
    ):
        issues.append("selected direct measurement does not match the artifact candidate")
    if candidate.parent_candidate_id is None:
        issues.append("selected direct measurement has no measured parent/control")
    else:
        parents = [
            measurement
            for measurement in measurements.measurements
            if measurement.candidate_id == candidate.parent_candidate_id
            and measurement.target_class is target_class
        ]
        if len(parents) != 1:
            issues.append("selected direct measurement parent is missing or ambiguous")
        else:
            parent = parents[0]
            if (
                not parent.validation_passed
                or candidate.objective_loss >= parent.objective_loss
                or candidate.quality_retention < parent.quality_retention
            ):
                issues.append("selected candidate does not improve its valid parent/control")
    if any(
        measurement.target_class is not target_class for measurement in measurements.measurements
    ):
        issues.append("direct measurement set mixes 4-bit and 6-bit lineages")
    return issues


def build_direct_pareto(measurements: DirectRefinementMeasurementSet) -> DirectParetoReport:
    points: list[DirectParetoPoint] = []
    valid = [
        measurement for measurement in measurements.measurements if measurement.validation_passed
    ]
    for candidate in measurements.measurements:
        dominators: list[str] = []
        if candidate.validation_passed:
            for other in valid:
                if other.measurement_id == candidate.measurement_id:
                    continue
                weak = (
                    other.measured_bpw <= candidate.measured_bpw
                    and other.quality_retention >= candidate.quality_retention
                    and other.decode_tokens_per_second >= candidate.decode_tokens_per_second
                    and other.peak_memory_bytes <= candidate.peak_memory_bytes
                )
                strict = (
                    other.measured_bpw < candidate.measured_bpw
                    or other.quality_retention > candidate.quality_retention
                    or other.decode_tokens_per_second > candidate.decode_tokens_per_second
                    or other.peak_memory_bytes < candidate.peak_memory_bytes
                )
                if weak and strict:
                    dominators.append(other.candidate_id)
        points.append(
            DirectParetoPoint(
                candidate_id=candidate.candidate_id,
                measurement_id=candidate.measurement_id,
                target_class=candidate.target_class,
                measured_bpw=candidate.measured_bpw,
                quality_retention=candidate.quality_retention,
                decode_tokens_per_second=candidate.decode_tokens_per_second,
                peak_memory_bytes=candidate.peak_memory_bytes,
                validation_passed=candidate.validation_passed,
                frontier=candidate.validation_passed and not dominators,
                dominated_by=sorted(set(dominators)),
            )
        )
    return DirectParetoReport(
        measurement_set_sha256=stable_sha256(measurements),
        points=points,
        frontier_candidate_ids=sorted({point.candidate_id for point in points if point.frontier}),
    )


def _archive_issues(
    index_path: Path,
    index: EvidenceArchiveIndex,
    *,
    required_paths: list[Path],
) -> list[str]:
    issues: list[str] = []
    if not index.complete:
        issues.append("evidence archive index is incomplete")
    for record in index.records:
        path = required_file(
            index_path.parent, record.path, f"archived evidence {record.logical_name}"
        )
        if path.stat().st_size != record.size_bytes or file_sha256(path) != record.sha256:
            issues.append(f"archived evidence changed: {record.logical_name}")
        if ".internal/tmp" in record.durable_uri.replace("\\", "/"):
            issues.append(f"evidence is not durably archived: {record.logical_name}")
    archived_digests = {record.sha256 for record in index.records}
    missing = [path.name for path in required_paths if file_sha256(path) not in archived_digests]
    if missing:
        issues.append(f"load-bearing evidence is absent from the durable archive: {missing}")
    return issues


def _package_issues(
    artifact: Path,
    request_path: Path,
    request: Qwen3NextReleaseAuditRequest,
    policy_sha256: str,
) -> list[str]:
    issues: list[str] = []
    certification = artifact / "certification"
    expected = {
        "request.json": request_path,
        "coding_suite_manifest.json": required_file(
            request_path.parent, request.coding_suite_manifest, "coding suite manifest"
        ),
        "coding_suite_self_test.json": required_file(
            request_path.parent, request.coding_suite_self_test, "coding suite self-test"
        ),
        "benchmark_evidence_index.json": required_file(
            request_path.parent, request.benchmark_evidence_index, "benchmark evidence index"
        ),
        "release_validation_index.json": required_file(
            request_path.parent, request.release_validation_index, "release validation index"
        ),
        "refinement_measurements.json": required_file(
            request_path.parent, request.refinement_measurements, "refinement measurements"
        ),
        "hardware_profile_registry.json": required_file(
            request_path.parent, request.hardware_registry, "hardware registry"
        ),
        "compatibility_matrix.json": required_file(
            request_path.parent, request.compatibility_matrix, "compatibility matrix"
        ),
        "pareto_report.json": required_file(
            request_path.parent, request.pareto_report, "Pareto report"
        ),
        "reproduction_verification.json": required_file(
            request_path.parent, request.reproduction_verification, "reproduction verification"
        ),
        "evidence_archive_index.json": required_file(
            request_path.parent, request.evidence_archive_index, "evidence archive index"
        ),
    }
    for name, source in expected.items():
        packaged = certification / name
        if not packaged.is_file() or file_sha256(packaged) != file_sha256(source):
            issues.append(f"certification package is missing or stale: {name}")
    policy_path = certification / "policy.json"
    scope_path = certification / "exact_checkpoint_scope.json"
    if not policy_path.is_file():
        issues.append("certification package lacks policy.json")
    else:
        try:
            if stable_sha256(json.loads(policy_path.read_text(encoding="utf-8"))) != policy_sha256:
                issues.append("packaged certification policy differs from wheel policy")
        except (OSError, json.JSONDecodeError):
            issues.append("packaged certification policy is invalid")
    if not scope_path.is_file():
        issues.append("certification package lacks exact_checkpoint_scope.json")
    else:
        try:
            if stable_sha256(json.loads(scope_path.read_text(encoding="utf-8"))) != stable_sha256(
                request.certification_scope
            ):
                issues.append("packaged exact checkpoint scope differs from the request")
        except (OSError, json.JSONDecodeError):
            issues.append("packaged exact checkpoint scope is invalid")
    return issues


def _direct_wheel_issues(path: Path) -> list[str]:
    required = {
        "axquant/certification/dispatch.py",
        "axquant/certification/qwen3_next_direct.py",
        "axquant/certification/registry.py",
        "axquant/schema/certification.py",
        "axquant/schema/coding_suite.py",
    }
    try:
        with zipfile.ZipFile(path) as wheel:
            missing = sorted(required - set(wheel.namelist()))
    except (OSError, zipfile.BadZipFile) as exc:
        return [f"direct certification wheel cannot be inspected: {exc}"]
    if missing:
        return [f"toolkit wheel lacks direct certification modules: {missing}"]
    return []


def build_qwen3_next_release_audit(request_path: str | Path) -> Qwen3NextReleaseAudit:
    request_source = Path(request_path).expanduser().resolve()
    request = load_model(request_source, Qwen3NextReleaseAuditRequest)
    base = request_source.parent
    artifact = required_directory(base, request.artifact_directory, "release artifact")
    paths = {
        "inventory": required_file(base, request.source_inventory, "source inventory"),
        "source_manifest": required_file(
            base, request.source_checkpoint_manifest, "source checkpoint manifest"
        ),
        "feasibility": required_file(base, request.feasibility_report, "feasibility report"),
        "sensitivity": required_file(base, request.sensitivity_report, "sensitivity report"),
        "refinement": required_file(base, request.refinement_result, "refinement result"),
        "measurements": required_file(
            base, request.refinement_measurements, "refinement measurements"
        ),
        "validation": required_file(
            base, request.release_validation_index, "release validation index"
        ),
        "benchmark": required_file(
            base, request.benchmark_evidence_index, "benchmark evidence index"
        ),
        "coding": required_file(base, request.coding_suite_manifest, "coding suite manifest"),
        "coding_self_test": required_file(
            base, request.coding_suite_self_test, "coding suite self-test"
        ),
        "hardware": required_file(base, request.hardware_registry, "hardware registry"),
        "pareto": required_file(base, request.pareto_report, "Pareto report"),
        "compatibility": required_file(base, request.compatibility_matrix, "compatibility matrix"),
        "compatibility_request": required_file(
            base, request.compatibility_request, "compatibility request"
        ),
        "recipe": required_file(base, request.reproduction_recipe, "reproduction recipe"),
        "reproduction": required_file(
            base, request.reproduction_verification, "reproduction verification"
        ),
        "ax_manifest": required_file(
            base, request.ax_engine_manifest_check, "AX Engine manifest check"
        ),
        "ax_doctor": required_file(base, request.ax_engine_doctor_check, "AX Engine doctor check"),
        "ax_runtime": required_file(
            base, request.ax_engine_runtime_check, "AX Engine runtime check"
        ),
        "mlx_runtime": required_file(base, request.mlx_lm_runtime_check, "MLX-LM runtime check"),
        "archive": required_file(base, request.evidence_archive_index, "evidence archive index"),
        "wheel": required_file(base, request.toolkit_wheel, "toolkit wheel"),
    }
    lineage_paths = [
        required_file(base, value, "sensitivity lineage report")
        for value in request.sensitivity_lineage
    ]
    manifest_path = required_file(artifact, "axquant_manifest.json", "artifact manifest")
    plan_path = required_file(artifact, "axquant_plan.json", "artifact plan")

    inventory = load_model(paths["inventory"], Inventory)
    source_manifest = load_model(paths["source_manifest"], SourceCheckpointManifest)
    feasibility = load_model(paths["feasibility"], FeasibilityReport)
    sensitivity = load_model(paths["sensitivity"], SensitivityReport)
    sensitivity_lineage = [load_model(path, SensitivityReport) for path in lineage_paths]
    refinement = load_model(paths["refinement"], RefinementResult)
    measurements = load_model(paths["measurements"], DirectRefinementMeasurementSet)
    validation = load_model(paths["validation"], DirectReleaseValidationIndex)
    (
        validation_evidence,
        validation_dependency_paths,
        validation_manifest_paths,
    ) = _validation_evidence(paths["validation"], validation)
    general_overlap_path = bound_file(
        paths["validation"].parent,
        validation.general_calibration_overlap_report_file,
        validation.general_calibration_overlap_report_sha256,
        "general calibration-overlap report",
    )
    general_overlap = load_model(general_overlap_path, CodingOverlapReport)
    benchmark = load_model(paths["benchmark"], DirectBenchmarkEvidenceIndex)
    coding = load_model(paths["coding"], CodingSuiteManifest)
    coding_self_test = load_model(paths["coding_self_test"], CodingSuiteSelfTestReport)
    overlap_path = bound_file(
        paths["coding"].parent,
        coding.calibration_overlap_report,
        coding.calibration_overlap_report_sha256,
        "coding calibration-overlap report",
    )
    overlap = load_model(overlap_path, CodingOverlapReport)
    hardware = load_model(paths["hardware"], DirectHardwareProfileRegistry)
    pareto = load_model(paths["pareto"], DirectParetoReport)
    compatibility = load_model(paths["compatibility"], Qwen3NextCompatibilityMatrix)
    compatibility_request = load_model(
        paths["compatibility_request"], Qwen3NextCompatibilityRequest
    )
    recipe = load_model(paths["recipe"], ReproductionRecipe)
    reproduction = load_model(paths["reproduction"], ReproductionVerification)
    ax_manifest = load_model(paths["ax_manifest"], RuntimeCheck)
    ax_doctor = load_model(paths["ax_doctor"], RuntimeCheck)
    ax_runtime = load_model(paths["ax_runtime"], RuntimeCheck)
    mlx_runtime = load_model(paths["mlx_runtime"], RuntimeCheck)
    archive = load_model(paths["archive"], EvidenceArchiveIndex)
    manifest = load_model(manifest_path, ArtifactManifest)
    plan = load_model(plan_path, QuantizationPlan)

    if inventory.model.local_path is None:
        raise ArtifactError("source inventory does not bind a local immutable checkpoint")
    source_dir = Path(inventory.model.local_path).expanduser().resolve()
    if not source_dir.is_dir():
        raise ArtifactError(f"source inventory path does not exist: {source_dir}")
    eligibility = _eligibility_issues(
        source_dir=source_dir,
        inventory=inventory,
        plan=plan,
        manifest=manifest,
    )
    if eligibility:
        raise ArtifactError("qwen3-next-direct-v1 eligibility failed: " + "; ".join(eligibility))

    scope = request.certification_scope
    actual_fingerprint = architecture_fingerprint(source_dir, inventory=inventory)
    candidate_model = ax_runtime.model
    policy_sha256 = direct_policy_sha256()
    checks: list[Qwen3NextReleaseAuditCheck] = []

    n0_issues: list[str] = []
    if request.policy_sha256 != policy_sha256:
        n0_issues.append("request policy digest differs from the wheel-owned policy")
    if scope.architecture != actual_fingerprint:
        n0_issues.append("certification architecture fingerprint differs from the source")
    if not _same_source(scope.source_model, inventory.model):
        n0_issues.append("certification source identity differs from the inventory")
    provenance_path = source_dir / "axquant_source.json"
    if not provenance_path.is_file():
        n0_issues.append("immutable source lacks axquant_source.json")
    else:
        provenance = load_model(provenance_path, SourceConversionProvenance)
        if (
            provenance.source_model != scope.source_model.model_id
            or provenance.source_revision != scope.source_model.revision
        ):
            n0_issues.append("source conversion provenance differs from certification scope")
    if source_manifest.source_model != inventory.model:
        n0_issues.append("source checkpoint manifest identity differs from inventory")
    if (
        source_manifest.config_sha256 != actual_fingerprint.config_sha256
        or source_manifest.tokenizer_sha256 != actual_fingerprint.tokenizer_sha256
    ):
        n0_issues.append("source checkpoint manifest config/tokenizer binding differs")
    n0_issues.extend(source_manifest_issues(source_dir, source_manifest, inventory=inventory))
    n0_issues.extend(_feasibility_issues(feasibility, plan=plan, inventory=inventory))
    _add_check(
        checks,
        NonMtpGateId.N0,
        "Immutable technical feasibility",
        n0_issues,
        {
            "source_inventory": paths["inventory"],
            "source_checkpoint_manifest": paths["source_manifest"],
            "feasibility_report": paths["feasibility"],
        },
    )

    n1_issues = _artifact_issues(artifact, manifest)
    if file_sha256(manifest_path) != scope.artifact_manifest_sha256:
        n1_issues.append("certification scope binds another artifact manifest")
    if stable_sha256(plan) != manifest.plan_sha256:
        n1_issues.append("artifact manifest does not bind axquant_plan.json")
    if not _same_source(plan.source_model, scope.source_model) or not _same_source(
        manifest.source_model, scope.source_model
    ):
        n1_issues.append("plan/artifact source identity differs from certification scope")
    if manifest.profile != plan.profile or manifest.calibration != plan.calibration:
        n1_issues.append("artifact and plan profile/calibration provenance differ")
    if abs(manifest.measured_total_bpw - plan.effective_bpw) > (
        direct_policy().measured_plan_bpw_delta_max
    ):
        n1_issues.append("measured and planned BPW differ beyond policy")
    expected_target = _candidate_target_bpw(scope.target_class)
    if abs(plan.target_bpw - expected_target) > 1e-6:
        n1_issues.append("plan target BPW does not match the certification target class")
    source_weight_bytes = feasibility.source.weight_bytes if feasibility.source is not None else 0
    if source_weight_bytes <= 0 or (
        manifest.weight_file_size_bytes / source_weight_bytes
        > _artifact_weight_ratio_limit(scope.target_class)
    ):
        n1_issues.append("candidate artifact weight ratio exceeds policy")
    if not (artifact / "model-manifest.json").is_file():
        n1_issues.append("AX Engine native manifest is missing")
    n1_issues.extend(
        _runtime_check_issues(
            ax_manifest,
            runtime=RuntimeName.AX_ENGINE,
            kinds={"manifest", "static-compatibility"},
            artifact=artifact,
            candidate_model_id=candidate_model.model_id,
            candidate_revision=candidate_model.revision,
        )
    )
    n1_issues.extend(
        _runtime_check_issues(
            ax_doctor,
            runtime=RuntimeName.AX_ENGINE,
            kinds={"doctor"},
            artifact=artifact,
            candidate_model_id=candidate_model.model_id,
            candidate_revision=candidate_model.revision,
        )
    )
    n1_issues.extend(
        _runtime_check_issues(
            ax_runtime,
            runtime=RuntimeName.AX_ENGINE,
            kinds={"generation-smoke"},
            artifact=artifact,
            candidate_model_id=candidate_model.model_id,
            candidate_revision=candidate_model.revision,
        )
    )
    n1_issues.extend(
        _runtime_check_issues(
            mlx_runtime,
            runtime=RuntimeName.MLX_LM,
            kinds={"generation-smoke"},
            artifact=artifact,
            candidate_model_id=candidate_model.model_id,
            candidate_revision=candidate_model.revision,
        )
    )
    _add_check(
        checks,
        NonMtpGateId.N1,
        "Artifact integrity and dual-runtime vertical slice",
        n1_issues,
        {
            "artifact_manifest": manifest_path,
            "plan": plan_path,
            "ax_engine_manifest_check": paths["ax_manifest"],
            "ax_engine_doctor_check": paths["ax_doctor"],
            "ax_engine_runtime_check": paths["ax_runtime"],
            "mlx_lm_runtime_check": paths["mlx_runtime"],
        },
    )

    n2_issues = _benchmark_issues(
        benchmark,
        target_class=scope.target_class,
        artifact_manifest_sha256=file_sha256(manifest_path),
        hardware_scope_ids=set(scope.hardware_scope_ids),
    )
    _add_check(
        checks,
        NonMtpGateId.N2,
        "Matched direct-decode benchmark",
        n2_issues,
        {"benchmark_evidence_index": paths["benchmark"]},
    )

    n3_issues = _sensitivity_issues(sensitivity, inventory=inventory, plan=plan)
    lineage_by_digest = {stable_sha256(item): item for item in sensitivity_lineage}
    if len(lineage_by_digest) != len(sensitivity_lineage):
        n3_issues.append("sensitivity lineage contains duplicate reports")
    if sensitivity.calibration is not None:
        parent = sensitivity.calibration.metadata.get("base_sensitivity_sha256")
        if parent is not None and parent not in lineage_by_digest:
            n3_issues.append("sensitivity lineage parent is missing")
        calibration_path = artifact / "calibration_manifest.json"
        if not calibration_path.is_file():
            n3_issues.append("artifact lacks the bound calibration manifest")
        else:
            expected_calibration = sensitivity.calibration.metadata.get(
                "calibration_manifest_sha256"
            )
            calibration_manifest = load_model(calibration_path, CalibrationManifest)
            if not isinstance(expected_calibration, str) or not calibration_manifest_matches(
                calibration_path,
                calibration_manifest,
                expected_calibration,
            ):
                n3_issues.append("packaged calibration manifest differs from measured evidence")
        if any(key in sensitivity.calibration.metadata for key in CAPTURE_METADATA_KEYS):
            capture_path = artifact / "activation_capture_manifest.json"
            if not capture_path.is_file():
                n3_issues.append("artifact lacks the bound activation capture manifest")
            else:
                capture = load_model(capture_path, ActivationCaptureManifest)
                n3_issues.extend(
                    activation_capture_evidence_issues(
                        capture,
                        sensitivity.calibration.metadata,
                        model_id=plan.source_model.model_id,
                        revision=plan.source_model.revision,
                        dataset_id=sensitivity.calibration.dataset_id,
                    )
                )
    _add_check(
        checks,
        NonMtpGateId.N3,
        "Measured mixed-precision planner",
        n3_issues,
        {
            "sensitivity_report": paths["sensitivity"],
            **{
                f"sensitivity_lineage_{index:03d}": path for index, path in enumerate(lineage_paths)
            },
        },
    )

    coding_dependency_paths = [
        bound_file(
            paths["coding"].parent,
            relative_name,
            expected_sha256,
            f"coding suite shard {relative_name}",
        )
        for relative_name, expected_sha256 in coding.task_shards.items()
    ]
    coding_self_test_issues, coding_self_test_dependency_paths = _coding_self_test_issues(
        coding_self_test,
        report_path=paths["coding_self_test"],
        manifest=coding,
        manifest_path=paths["coding"],
    )
    n4_issues = _coding_manifest_issues(paths["coding"], coding)
    n4_issues.extend(coding_self_test_issues)
    general_tasks = load_quality_tasks(validation_manifest_paths[ProfileName.GENERAL])
    general_task_categories = {task.task_id: task.category for task in general_tasks}
    n4_issues.extend(
        coding_general_overlap_issues(
            coding_payloads=load_coding_payloads(paths["coding"], coding),
            general_tasks=general_tasks,
            similarity_threshold=coding.near_duplicate_threshold,
        )
    )
    if not overlap.passed or overlap.matches:
        n4_issues.append("coding suite overlaps calibration")
    if not general_overlap.passed or general_overlap.matches:
        n4_issues.append("general quality suite overlaps calibration")
    if overlap.similarity_threshold != coding.near_duplicate_threshold:
        n4_issues.append("coding overlap report uses another similarity threshold")
    if sensitivity.calibration is None:
        n4_issues.append("quality evidence cannot bind missing calibration provenance")
    else:
        if overlap.calibration_dataset_sha256 != sensitivity.calibration.dataset_sha256:
            n4_issues.append("coding overlap report uses another calibration dataset")
        if overlap.suite_dataset_sha256 != coding.dataset_sha256:
            n4_issues.append("coding overlap report uses another coding suite")
        if general_overlap.calibration_dataset_sha256 != sensitivity.calibration.dataset_sha256:
            n4_issues.append("general overlap report uses another calibration dataset")
        if general_overlap.suite_dataset_sha256 != file_sha256(
            validation_manifest_paths[ProfileName.GENERAL]
        ):
            n4_issues.append("general overlap report uses another general quality suite")
        n4_issues.extend(
            _quality_issues(
                validation,
                validation_evidence,
                suite=coding,
                calibration_dataset_sha256=sensitivity.calibration.dataset_sha256,
                reference_model=inventory.model,
                candidate_model=candidate_model,
                tokenizer_sha256=source_manifest.tokenizer_sha256,
                reference_model_artifact_sha256=file_sha256(paths["source_manifest"]),
                candidate_model_artifact_sha256=file_sha256(manifest_path),
                suite_manifest_sha256=file_sha256(paths["coding"]),
                toolkit_version=request.required_toolkit_version,
                general_task_categories=general_task_categories,
            )
        )
    _add_check(
        checks,
        NonMtpGateId.N4,
        "Coding and general quality",
        n4_issues,
        {
            "coding_suite_manifest": paths["coding"],
            "coding_suite_self_test": paths["coding_self_test"],
            "coding_overlap_report": overlap_path,
            "general_overlap_report": general_overlap_path,
            "release_validation_index": paths["validation"],
            "general_quality_manifest": validation_manifest_paths[ProfileName.GENERAL],
        },
    )

    n5_issues = _architecture_issues(
        inventory=inventory,
        plan=plan,
        fingerprint=actual_fingerprint,
    )
    _add_check(
        checks,
        NonMtpGateId.N5,
        "Exact Qwen3-Next architecture proof",
        n5_issues,
        {"source_inventory": paths["inventory"], "plan": plan_path},
    )

    n6_issues = _refinement_issues(
        refinement,
        measurements,
        plan=plan,
        manifest_sha256=file_sha256(manifest_path),
        target_class=scope.target_class,
        candidate_model_id=candidate_model.model_id,
        quality_evidence_sha256=file_sha256(paths["validation"]),
        benchmark_evidence_sha256=file_sha256(paths["benchmark"]),
    )
    _add_check(
        checks,
        NonMtpGateId.N6,
        "Complete candidate optimization",
        n6_issues,
        {
            "refinement_result": paths["refinement"],
            "refinement_measurements": paths["measurements"],
        },
    )

    n7_issues = list(hardware.issues)
    matching_hardware = [
        entry
        for entry in hardware.entries
        if entry.hardware_scope_id in scope.hardware_scope_ids
        and entry.artifact_manifest_sha256 == file_sha256(manifest_path)
        and entry.benchmark_evidence_sha256 == file_sha256(paths["benchmark"])
        and entry.candidate_id == refinement.selected_candidate_id
        and entry.release_ready
    ]
    if {entry.hardware_scope_id for entry in matching_hardware} != set(scope.hardware_scope_ids):
        n7_issues.append("hardware registry does not cover every certification scope")
    for entry in matching_hardware:
        if (
            entry.chip != direct_policy().formal_hardware_chip
            or entry.unified_memory_bytes != direct_policy().formal_hardware_memory_bytes
        ):
            n7_issues.append("hardware registry entry is outside the frozen M2 Ultra scope")
    rebuilt = build_direct_pareto(measurements)
    if rebuilt.model_dump(mode="json", exclude={"created_at"}) != pareto.model_dump(
        mode="json", exclude={"created_at"}
    ):
        n7_issues.append("direct Pareto report cannot be rebuilt from measurements")
    if refinement.selected_candidate_id not in pareto.frontier_candidate_ids:
        n7_issues.append("selected candidate is not on the direct Pareto frontier")
    if compatibility_request.source_model != scope.source_model:
        n7_issues.append("compatibility request source differs from certification scope")
    if compatibility_request.target_class is not scope.target_class:
        n7_issues.append("compatibility request target class differs")
    if (
        compatibility.source_model != compatibility_request.source_model
        or compatibility.target_class is not compatibility_request.target_class
        or compatibility.artifact_manifest_sha256 != file_sha256(manifest_path)
        or not compatibility.release_ready
    ):
        n7_issues.append("Qwen3-Next compatibility matrix is not release-ready for this artifact")
    if not reproduction.passed or reproduction.recipe_sha256 != stable_sha256(recipe):
        n7_issues.append("reproduction verification is not passing or recipe-bound")
    rerun_reproduction = verify_reproduction(
        recipe_path=paths["recipe"], artifact_dir=reproduction.artifact_path
    )
    if rerun_reproduction != reproduction:
        n7_issues.append("reproduction verification cannot be reproduced")
    n7_issues.extend(
        _archive_issues(
            paths["archive"],
            archive,
            required_paths=[path for key, path in paths.items() if key not in {"archive", "wheel"}]
            + [
                manifest_path,
                plan_path,
                *lineage_paths,
                overlap_path,
                *coding_dependency_paths,
                *coding_self_test_dependency_paths,
                *validation_dependency_paths,
            ],
        )
    )
    _add_check(
        checks,
        NonMtpGateId.N7,
        "Hardware-aware Pareto and reproduction",
        n7_issues,
        {
            "hardware_registry": paths["hardware"],
            "pareto_report": paths["pareto"],
            "compatibility_matrix": paths["compatibility"],
            "compatibility_request": paths["compatibility_request"],
            "reproduction_recipe": paths["recipe"],
            "reproduction_verification": paths["reproduction"],
            "evidence_archive_index": paths["archive"],
        },
    )

    toolkit_version, n8_issues = _wheel_identity(paths["wheel"])
    n8_issues.extend(_direct_wheel_issues(paths["wheel"]))
    if toolkit_version != request.required_toolkit_version:
        n8_issues.append("toolkit wheel version differs from the required release version")
    version_claims = {
        manifest.axquant_version,
        manifest.software_versions.axquant,
        plan.software_versions.axquant,
        recipe.axquant_version,
        recipe.software_versions.axquant,
    }
    if version_claims != {request.required_toolkit_version}:
        n8_issues.append("artifact/plan/recipe toolkit version binding differs")
    model_card = artifact / "README.md"
    if not model_card.is_file():
        n8_issues.append("release artifact lacks a model card")
    else:
        text = model_card.read_text(encoding="utf-8").lower()
        source_revision = scope.source_model.revision or ""
        if source_revision not in text or "non-mtp" not in text:
            n8_issues.append("model card lacks exact revision or explicit non-MTP scope")
        for claim in _FORBIDDEN_MODEL_CARD_CLAIMS:
            if claim in text:
                n8_issues.append(f"model card contains unsupported claim: {claim}")
    n8_issues.extend(_package_issues(artifact, request_source, request, policy_sha256))
    _add_check(
        checks,
        NonMtpGateId.N8,
        "Track-specific release package",
        n8_issues,
        {"toolkit_wheel": paths["wheel"], "request": request_source},
    )

    blockers = [f"{check.gate_id.value}: {issue}" for check in checks for issue in check.issues]
    return Qwen3NextReleaseAudit(
        certification_scope=scope,
        candidate_model=candidate_model,
        request_sha256=file_sha256(request_source),
        policy_sha256=policy_sha256,
        toolkit_version=toolkit_version,
        wheel_sha256=file_sha256(paths["wheel"]),
        checks=checks,
        blockers=blockers,
        release_ready=all(check.passed for check in checks),
    )
