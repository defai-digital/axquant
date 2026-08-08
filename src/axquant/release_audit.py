from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import zipfile
from email.parser import Parser
from math import isfinite
from pathlib import Path, PurePosixPath

from axquant.artifact_paths import (
    artifact_member_path,
    artifact_tree_files,
    artifact_tree_symlinks,
)
from axquant.calibration import calibration_manifest_matches
from axquant.capture_binding import (
    CAPTURE_METADATA_KEYS,
    activation_capture_evidence_issues,
)
from axquant.errors import ArtifactError, RefinementError, ValidationGateError
from axquant.identity import model_identity_key, same_model_identity
from axquant.mtp_sidecar import EXTERNAL_MTP_SIDECAR_FILENAMES
from axquant.pareto import build_pareto_report
from axquant.profiles import thresholds_for
from axquant.refinement import (
    COMPLETE_OBJECTIVE_VERSION,
    _is_monotonic_precision_refinement,
    build_complete_candidate_measurement,
)
from axquant.release_exceptions import (
    release_exception_allows_size,
    verify_release_exception,
)
from axquant.reproduction import verify_reproduction
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    PROTECTED_MIN_BITS,
    ActivationCaptureManifest,
    ArtifactManifest,
    BaselineKind,
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    CalibrationManifest,
    CompatibilityMatrix,
    CompatibilityMatrixRequest,
    CompleteCandidateMeasurement,
    EvaluationBundle,
    EvidenceKind,
    FeasibilityReport,
    HardwareProfileRegistry,
    ParetoReport,
    ProfileName,
    QualityComparisonReport,
    QuantizationPlan,
    QuantMethod,
    RefinementMeasurementSet,
    RefinementResult,
    ReleaseAudit,
    ReleaseAuditCheck,
    ReleaseAuditRequest,
    ReleaseException,
    ReleaseValidationIndex,
    ReproductionRecipe,
    ReproductionVerification,
    RuntimeCheck,
    RuntimeName,
    SensitivityReport,
    TensorRole,
    TensorSpec,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model, stable_sha256

_HARDWARE_EVIDENCE_FIELDS = (
    ("plan_file", "plan_file_sha256"),
    ("artifact_manifest_file", "artifact_manifest_sha256"),
    ("sensitivity_file", "sensitivity_sha256"),
    ("quality_comparison_file", "quality_comparison_sha256"),
    ("validation_file", "validation_sha256"),
    ("direct_evaluation_file", "direct_evaluation_sha256"),
    ("mtp_evaluation_file", "mtp_evaluation_sha256"),
    ("direct_benchmark_result_file", "direct_benchmark_result_sha256"),
    ("mtp_benchmark_result_file", "mtp_benchmark_result_sha256"),
    ("quantizer_execution_file", "quantizer_execution_sha256"),
)

# Release probes use the same role floors as planner protection policy.
_RELEASE_PROBE_MIN_BITS = PROTECTED_MIN_BITS
_ACTIVATION_REFINEMENT_METHODS = frozenset(
    {QuantMethod.AWQ, QuantMethod.GPTQ, QuantMethod.GPTQ_ACT}
)
_ACTIVATION_CAPTURE_MANIFEST = "activation_capture_manifest.json"
_REQUIRED_BENCHMARK_KINDS = {
    BenchmarkEvidenceKind.BF16,
    BenchmarkEvidenceKind.UNIFORM_4BIT,
    BenchmarkEvidenceKind.UNIFORM_6BIT,
    BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
    BenchmarkEvidenceKind.AXQUANT_MTP_ON,
}
_BENCHMARK_METADATA_FIELDS = (
    "prompt_count",
    "warmup_trials",
    "measured_trials",
    "successful_measured_trials",
    "failed_trials",
    "timed_out_trials",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "power_mode",
    "quantizer",
    "quantizer_version",
    "quality_dataset_sha256",
)
_BENCHMARK_CONTROL_FIELDS = (
    "prompt_count",
    "warmup_trials",
    "measured_trials",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "power_mode",
    "quality_dataset_sha256",
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
    supplied = Path(value).expanduser()
    supplied = supplied if supplied.is_absolute() else base / supplied
    if supplied.is_symlink():
        raise ArtifactError(f"{label} root is a symlink: {supplied}")
    path = _resolved(base, value)
    if not path.is_dir():
        raise ArtifactError(f"{label} does not exist: {path}")
    return path


def _bound_file(base: Path, value: str, expected_sha256: str, label: str) -> Path:
    path = _required_file(base, value, label)
    if file_sha256(path) != expected_sha256:
        raise ArtifactError(f"{label} checksum does not match its index: {path}")
    return path


def _artifact_issues(directory: Path, manifest: ArtifactManifest) -> list[str]:
    issues: list[str] = []
    symlinks = artifact_tree_symlinks(directory)
    if symlinks:
        issues.append(f"artifact tree contains symlinks: {symlinks}")
    record_paths = [record.path for record in manifest.files]
    if len(record_paths) != len(set(record_paths)):
        issues.append("artifact manifest contains duplicate file records")
    for record in manifest.files:
        try:
            path = artifact_member_path(directory, record.path)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        if not path.is_file():
            issues.append(f"artifact manifest file is missing: {record.path}")
        elif path.stat().st_size != record.size_bytes:
            issues.append(f"artifact manifest size changed: {record.path}")
        elif file_sha256(path) != record.sha256:
            issues.append(f"artifact manifest checksum changed: {record.path}")
    recorded_weight_files = {
        Path(record.path).as_posix()
        for record in manifest.files
        if Path(record.path).suffix.lower() == ".safetensors"
    }
    try:
        actual_files = artifact_tree_files(directory)
    except ValueError:
        actual_files = [
            path for path in directory.rglob("*") if path.is_file() and not path.is_symlink()
        ]
    actual_weight_files = {
        path.relative_to(directory).as_posix()
        for path in actual_files
        if path.suffix.lower() == ".safetensors"
    }
    if recorded_weight_files != actual_weight_files:
        missing = sorted(actual_weight_files - recorded_weight_files)
        extra = sorted(recorded_weight_files - actual_weight_files)
        issues.append(
            f"artifact manifest Safetensors coverage differs: missing={missing}, extra={extra}"
        )
    recorded_weight_bytes = sum(
        record.size_bytes
        for record in manifest.files
        if Path(record.path).suffix.lower() == ".safetensors"
    )
    if recorded_weight_bytes != manifest.weight_file_size_bytes:
        issues.append("artifact manifest Safetensors bytes do not match measured weight bytes")
    return issues


def _feasibility_issues(
    report: FeasibilityReport,
    plan: QuantizationPlan,
) -> list[str]:
    issues: list[str] = []
    if report.status != "ready-for-conversion":
        issues.append("feasibility status is not ready-for-conversion")
    source = report.source
    if source is None or not same_model_identity(source.model, plan.source_model):
        issues.append("feasibility source does not match the candidate plan")
        compared = list(report.baselines)
    else:
        compared = [source, *report.baselines]
        if source.kind != BaselineKind.BF16_SOURCE or not source.complete:
            issues.append("feasibility BF16 source is incomplete or has the wrong kind")
        if source.quantized is not False:
            issues.append("feasibility source is not an unquantized BF16 checkpoint")

    required_baselines = {
        BaselineKind.UNIFORM_4BIT,
        BaselineKind.UNIFORM_6BIT,
        BaselineKind.MIXED_PRECISION,
    }
    baseline_kinds = [baseline.kind for baseline in report.baselines]
    if (
        len(baseline_kinds) != len(set(baseline_kinds))
        or set(baseline_kinds) != required_baselines
        or any(not baseline.complete for baseline in report.baselines)
    ):
        issues.append("feasibility report does not contain one complete 4/6/mixed baseline")

    if compared:
        logical_parameters = {audit.logical_parameters for audit in compared}
        adapter_ids = {audit.adapter_id for audit in compared}
        scopes = {audit.optimization_scope for audit in compared}
        if len(logical_parameters) != 1 or next(iter(logical_parameters)) <= 0:
            issues.append("feasibility logical parameter counts are not equivalent")
        if adapter_ids != {plan.architecture_profile.adapter_id} or scopes != {
            plan.architecture_profile.optimization_scope
        }:
            issues.append("feasibility architecture profiles differ from the candidate plan")
        if any(not is_immutable_revision(audit.model.revision) for audit in compared):
            issues.append("feasibility checkpoint revisions are not all pinned")
        if any(audit.mtp_logical_parameters <= 0 for audit in compared):
            issues.append("feasibility evidence does not contain MTP parameters everywhere")
        if any(not audit.inspected or audit.issues for audit in compared):
            issues.append("feasibility contains an uninspected or issue-bearing checkpoint")

        for audit in compared:
            mlx_checks = [
                check
                for check in audit.runtime_checks
                if check.runtime == RuntimeName.MLX_LM
                and check.check_kind == "static-compatibility"
            ]
            if not mlx_checks or any(
                not check.available or not check.passed for check in mlx_checks
            ):
                issues.append(
                    f"feasibility MLX-LM compatibility did not pass for {audit.kind.value}"
                )
        for baseline in report.baselines:
            ax_checks = [
                check
                for check in baseline.runtime_checks
                if check.runtime == RuntimeName.AX_ENGINE and check.check_kind == "doctor"
            ]
            if not ax_checks or any(not check.available or not check.passed for check in ax_checks):
                issues.append(
                    f"feasibility AX Engine doctor did not pass for {baseline.kind.value}"
                )

    required_checks = {
        "required_baselines_complete",
        "logical_parameter_counts_match",
        "architecture_profiles_match",
        "mtp_tensors_present",
        "revisions_pinned",
        "source_bf16_available",
        "source_bf16_complete",
        "ax_engine_runtime_ready",
        "mlx_lm_static_compatible",
    }
    failed_checks = sorted(name for name in required_checks if report.checks.get(name) is not True)
    if failed_checks:
        issues.append(f"feasibility report has failed or missing checks: {failed_checks}")
    if not report.runtime_checks_requested:
        issues.append("feasibility report did not request baseline runtime checks")
    if report.blockers:
        issues.append(f"feasibility report has blockers: {report.blockers}")
    return issues


def _runtime_issues(
    check: RuntimeCheck,
    *,
    runtime: RuntimeName,
    artifact: Path,
    candidate_id: str,
    candidate_revision: str | None,
) -> list[str]:
    issues: list[str] = []
    expected_kind = "doctor" if runtime == RuntimeName.AX_ENGINE else "generation-smoke"
    if check.runtime != runtime or check.check_kind != expected_kind:
        issues.append(f"{runtime.value} runtime evidence has the wrong kind")
    if not check.available or not check.passed:
        issues.append(f"{runtime.value} runtime check did not pass")
    if check.model.model_id != candidate_id or check.model.revision != candidate_revision:
        issues.append(f"{runtime.value} runtime check identifies another candidate")
    if check.model.local_path is None or Path(check.model.local_path).resolve() != artifact:
        issues.append(f"{runtime.value} runtime check does not bind the artifact path")
    option = "--mlx-model-artifacts-dir" if runtime == RuntimeName.AX_ENGINE else "--model"
    try:
        command_target = Path(check.command[check.command.index(option) + 1]).resolve()
    except (ValueError, IndexError):
        issues.append(f"{runtime.value} runtime command has no artifact target")
    else:
        if command_target != artifact:
            issues.append(f"{runtime.value} runtime command targets another artifact")
    return issues


def _wheel_record_issues(
    wheel: zipfile.ZipFile,
    member_names: list[str],
    record_name: str,
) -> list[str]:
    issues: list[str] = []
    archive_members = set(member_names)
    if len(member_names) != len(archive_members):
        issues.append("toolkit wheel contains duplicate archive members")

    try:
        record_text = wheel.read(record_name).decode("utf-8")
        rows = list(csv.reader(io.StringIO(record_text)))
    except (OSError, UnicodeDecodeError, csv.Error, KeyError) as exc:
        return [f"toolkit wheel RECORD cannot be read: {exc}"]

    recorded_members: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 3:
            issues.append(f"toolkit wheel RECORD row {row_number} is malformed")
            continue
        member_name, encoded_hash, encoded_size = row
        member_path = PurePosixPath(member_name)
        if (
            not member_name
            or "\\" in member_name
            or member_path.is_absolute()
            or ".." in member_path.parts
        ):
            issues.append(f"toolkit wheel RECORD row {row_number} has an unsafe path")
            continue
        if member_name in recorded_members:
            issues.append(f"toolkit wheel RECORD lists {member_name!r} more than once")
            continue
        recorded_members.add(member_name)
        if member_name not in archive_members:
            issues.append(f"toolkit wheel RECORD lists missing member {member_name!r}")
            continue
        if member_name == record_name:
            if encoded_hash or encoded_size:
                issues.append("toolkit wheel RECORD must not hash or size itself")
            continue
        if not encoded_hash.startswith("sha256="):
            issues.append(f"toolkit wheel RECORD has no SHA-256 for {member_name!r}")
            continue
        digest_text = encoded_hash.removeprefix("sha256=")
        try:
            expected_digest = base64.urlsafe_b64decode(digest_text + "=" * (-len(digest_text) % 4))
            expected_size = int(encoded_size)
        except (ValueError, TypeError):
            issues.append(f"toolkit wheel RECORD metadata is invalid for {member_name!r}")
            continue
        member_data = wheel.read(member_name)
        if expected_size < 0 or expected_size != len(member_data):
            issues.append(f"toolkit wheel RECORD size mismatch for {member_name!r}")
        if expected_digest != hashlib.sha256(member_data).digest():
            issues.append(f"toolkit wheel RECORD hash mismatch for {member_name!r}")

    unrecorded = sorted(archive_members - recorded_members)
    if unrecorded:
        issues.append(f"toolkit wheel has unrecorded members: {unrecorded}")
    return issues


def _wheel_identity(path: Path) -> tuple[str | None, list[str]]:
    issues: list[str] = []
    try:
        with zipfile.ZipFile(path) as wheel:
            member_names = wheel.namelist()
            names = set(member_names)
            metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
            wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
            record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
            entry_point_names = [
                name for name in names if name.endswith(".dist-info/entry_points.txt")
            ]
            if (
                len(metadata_names) != 1
                or len(wheel_names) != 1
                or len(record_names) != 1
                or len(entry_point_names) != 1
            ):
                return None, ["toolkit wheel has incomplete distribution metadata"]
            issues.extend(_wheel_record_issues(wheel, member_names, record_names[0]))
            metadata = Parser().parsestr(wheel.read(metadata_names[0]).decode("utf-8"))
            version = metadata.get("Version")
            if metadata.get("Name", "").lower() != "axquant":
                issues.append("toolkit wheel distribution name is not axquant")
            classifiers = metadata.get_all("Classifier") or []
            if "Development Status :: 5 - Production/Stable" not in classifiers:
                issues.append("toolkit wheel is not classified as production/stable")
            if metadata.get("Requires-Python") != ">=3.11":
                issues.append("toolkit wheel does not require Python >=3.11")
            required_dependencies = {
                "huggingface-hub",
                "pydantic",
                "pyyaml",
                "safetensors",
                "structlog",
            }
            dependency_names = {
                re.split(r"[\s(<>=!~;]", requirement, maxsplit=1)[0].lower().replace("_", "-")
                for requirement in (metadata.get_all("Requires-Dist") or [])
            }
            missing_dependencies = sorted(required_dependencies - dependency_names)
            if missing_dependencies:
                issues.append(
                    f"toolkit wheel is missing runtime dependencies: {missing_dependencies}"
                )
            if metadata.get("License") != "MIT":
                issues.append("toolkit wheel metadata does not declare the MIT license")
            wheel_metadata = Parser().parsestr(wheel.read(wheel_names[0]).decode("utf-8"))
            if wheel_metadata.get("Root-Is-Purelib", "").lower() != "true":
                issues.append("toolkit wheel is not declared pure Python")
            if "py3-none-any" not in (wheel_metadata.get_all("Tag") or []):
                issues.append("toolkit wheel does not declare the py3-none-any tag")
            required_members = {
                "axquant/__init__.py",
                "axquant/cli/__init__.py",
                "axquant/schema/__init__.py",
                "axquant/release_audit.py",
                "axquant/release_exceptions.py",
                "axquant/hardware_registry.py",
                "axquant/reporting.py",
            }
            missing = sorted(required_members - names)
            if missing:
                issues.append(f"toolkit wheel is missing required modules: {missing}")
            if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
                issues.append("toolkit wheel does not contain its license")
            entry_points = wheel.read(entry_point_names[0]).decode("utf-8")
            if not re.search(
                r"(?m)^\s*axquant\s*=\s*axquant\.cli:entrypoint\s*$",
                entry_points,
            ):
                issues.append("toolkit wheel does not expose the axquant console entry point")
            if "axquant/__init__.py" in names:
                package_init = wheel.read("axquant/__init__.py").decode("utf-8")
                version_match = re.search(
                    r'(?m)^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
                    package_init,
                )
                if version_match is None or version_match.group(1) != version:
                    issues.append("toolkit package version differs from distribution metadata")
            return version, issues
    except (OSError, KeyError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        return None, [f"toolkit wheel cannot be inspected: {exc}"]


def _validation_evidence(
    index_path: Path,
    index: ReleaseValidationIndex,
) -> dict[ProfileName, tuple[ValidationReport, BenchmarkEvidenceIndex]]:
    result: dict[ProfileName, tuple[ValidationReport, BenchmarkEvidenceIndex]] = {}
    for entry in index.entries:
        validation_path = _bound_file(
            index_path.parent,
            entry.validation_file,
            entry.validation_sha256,
            f"{entry.profile.value} validation",
        )
        benchmark_path = _bound_file(
            index_path.parent,
            entry.benchmark_index_file,
            entry.benchmark_index_sha256,
            f"{entry.profile.value} benchmark index",
        )
        validation = load_model(validation_path, ValidationReport)
        benchmark = load_model(benchmark_path, BenchmarkEvidenceIndex)
        for benchmark_entry in benchmark.entries:
            if benchmark_entry.status != "available":
                continue
            if benchmark_entry.evaluation_file is None or benchmark_entry.evaluation_sha256 is None:
                raise ArtifactError(
                    f"{entry.profile.value} {benchmark_entry.kind.value} "
                    "available benchmark evidence is incomplete"
                )
            evaluation_path = _bound_file(
                benchmark_path.parent,
                benchmark_entry.evaluation_file,
                benchmark_entry.evaluation_sha256,
                f"{entry.profile.value} {benchmark_entry.kind.value} evaluation",
            )
            evaluation = load_model(evaluation_path, EvaluationBundle)
            if (
                benchmark_entry.model is None
                or not same_model_identity(evaluation.model, benchmark_entry.model)
                or evaluation.runtime != benchmark_entry.runtime
                or evaluation.mtp_enabled != benchmark_entry.mtp_enabled
                or evaluation.baseline_kind != benchmark_entry.kind.value
            ):
                raise ArtifactError(
                    f"{entry.profile.value} {benchmark_entry.kind.value} "
                    "evaluation differs from its index"
                )
        result[entry.profile] = (validation, benchmark)
    return result


def _benchmark_index_issues(
    index_path: Path,
    index: BenchmarkEvidenceIndex,
) -> list[str]:
    issues: list[str] = []
    bundles: dict[BenchmarkEvidenceKind, EvaluationBundle] = {}
    entries = {entry.kind: entry for entry in index.entries}
    for kind in _REQUIRED_BENCHMARK_KINDS:
        if entries[kind].status != "available":
            issues.append(f"required benchmark baseline is unavailable: {kind.value}")

    for entry in index.entries:
        if entry.status != "available":
            continue
        if (
            entry.evaluation_file is None
            or entry.evaluation_sha256 is None
            or entry.model is None
            or entry.runtime is None
            or entry.mtp_enabled is None
        ):
            issues.append(f"{entry.kind.value} available benchmark evidence is incomplete")
            continue
        try:
            path = _bound_file(
                index_path.parent,
                entry.evaluation_file,
                entry.evaluation_sha256,
                f"{entry.kind.value} benchmark evaluation",
            )
            bundle = load_model(path, EvaluationBundle)
        except (ArtifactError, OSError, ValueError) as exc:
            issues.append(f"{entry.kind.value} benchmark evaluation is invalid: {exc}")
            continue
        bundles[entry.kind] = bundle
        if (
            bundle.baseline_kind != entry.kind.value
            or not same_model_identity(bundle.model, entry.model)
            or bundle.runtime != entry.runtime
            or bundle.mtp_enabled != entry.mtp_enabled
        ):
            issues.append(f"{entry.kind.value} benchmark entry differs from its evaluation")
        if bundle.workload != index.profile.value:
            issues.append(f"{entry.kind.value} workload differs from the benchmark profile")
        if bundle.runtime != RuntimeName.AX_ENGINE:
            issues.append(f"{entry.kind.value} benchmark did not use AX Engine")
        if not is_immutable_revision(bundle.model.revision):
            issues.append(f"{entry.kind.value} benchmark model revision is not immutable")
        integrity = bundle.integrity
        if not (
            integrity.safetensors_valid
            and integrity.index_complete
            and integrity.config_valid
            and integrity.source_revision_pinned
        ):
            issues.append(f"{entry.kind.value} benchmark integrity is incomplete")
        versions = bundle.software_versions
        if any(
            not getattr(versions, name)
            for name in (
                "axquant",
                "python",
                "mlx",
                "mlx_lm",
                "ax_engine",
                "safetensors",
                "pydantic",
            )
        ):
            issues.append(f"{entry.kind.value} benchmark software provenance is incomplete")
        hardware = bundle.hardware
        if (
            not hardware.device_name
            or not hardware.chip
            or not hardware.unified_memory_bytes
            or not hardware.os_version
            or hardware.kernel_fallbacks != 0
        ):
            issues.append(f"{entry.kind.value} benchmark hardware provenance is incomplete")
        metadata = bundle.benchmark_metadata
        missing_metadata = [
            name for name in _BENCHMARK_METADATA_FIELDS if metadata.get(name) in (None, "")
        ]
        if missing_metadata:
            issues.append(f"{entry.kind.value} benchmark metadata is missing: {missing_metadata}")
        if (
            metadata.get("successful_measured_trials") != metadata.get("measured_trials")
            or metadata.get("failed_trials") != 0
            or metadata.get("timed_out_trials") != 0
        ):
            issues.append(f"{entry.kind.value} benchmark trials are incomplete")
        if entry.kind == BenchmarkEvidenceKind.AXQUANT_MTP_OFF and bundle.mtp_enabled:
            issues.append("axquant-mtp-off benchmark has MTP enabled")
        if entry.kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON and (
            not bundle.mtp_enabled or bundle.integrity.mtp_layout_valid is not True
        ):
            issues.append("axquant-mtp-on benchmark lacks enabled, valid MTP")

    if bundles:
        dataset_digests = {bundle.dataset_sha256 for bundle in bundles.values()}
        random_seeds = {bundle.random_seed for bundle in bundles.values()}
        software = {stable_sha256(bundle.software_versions) for bundle in bundles.values()}
        hardware_environments = {
            (
                bundle.hardware.device_name,
                bundle.hardware.chip,
                bundle.hardware.unified_memory_bytes,
                bundle.hardware.os_version,
            )
            for bundle in bundles.values()
        }
        controls = {
            stable_sha256(
                {name: bundle.benchmark_metadata.get(name) for name in _BENCHMARK_CONTROL_FIELDS}
            )
            for bundle in bundles.values()
        }
        if len(dataset_digests) != 1 or index.dataset_sha256 not in dataset_digests:
            issues.append("benchmark index dataset digest is inconsistent")
        if len(random_seeds) != 1 or index.random_seed not in random_seeds:
            issues.append("benchmark index random seed is inconsistent")
        if len(software) != 1:
            issues.append("benchmark evaluations use different software versions")
        if len(hardware_environments) != 1:
            issues.append("benchmark evaluations use different hardware")
        if len(controls) != 1:
            issues.append("benchmark evaluations use different controls")
    direct = bundles.get(BenchmarkEvidenceKind.AXQUANT_MTP_OFF)
    mtp = bundles.get(BenchmarkEvidenceKind.AXQUANT_MTP_ON)
    if direct is None or mtp is None or not same_model_identity(direct.model, mtp.model):
        issues.append("benchmark MTP-off/on evidence does not use one identical checkpoint")
    return issues


def _hardware_evidence_issues(
    registry_path: Path,
    registry: HardwareProfileRegistry,
) -> list[str]:
    issues: list[str] = []
    for entry in registry.entries:
        for file_field, sha_field in _HARDWARE_EVIDENCE_FIELDS:
            path = _resolved(registry_path.parent, str(getattr(entry, file_field)))
            if not path.is_file() or file_sha256(path) != getattr(entry, sha_field):
                issues.append(f"hardware evidence checksum mismatch: {entry.entry_id}/{file_field}")
    return issues


def _complete_measurement_evidence_issues(
    registry_path: Path,
    registry: HardwareProfileRegistry,
    measurements: RefinementMeasurementSet,
    *,
    expected_evaluator_version: str,
) -> list[str]:
    issues: list[str] = []
    if measurements.evaluator_version != expected_evaluator_version:
        issues.append("complete-model measurements use a different objective evaluator version")
    measurements_by_id = {
        measurement.measurement_id: measurement for measurement in measurements.measurements
    }
    entries_by_measurement = {entry.measurement_id: entry for entry in registry.entries}
    if len(entries_by_measurement) != len(registry.entries):
        issues.append("hardware registry repeats complete-model measurement IDs")
    if set(entries_by_measurement) != set(measurements_by_id):
        issues.append("hardware registry does not bind every complete-model measurement")

    for measurement_id, entry in entries_by_measurement.items():
        measurement = measurements_by_id.get(measurement_id)
        if measurement is None:
            continue
        try:
            plan_path = _bound_file(
                registry_path.parent,
                entry.plan_file,
                entry.plan_file_sha256,
                f"{entry.entry_id} candidate plan",
            )
            artifact_path = _bound_file(
                registry_path.parent,
                entry.artifact_manifest_file,
                entry.artifact_manifest_sha256,
                f"{entry.entry_id} artifact manifest",
            )
            quality_path = _bound_file(
                registry_path.parent,
                entry.quality_comparison_file,
                entry.quality_comparison_sha256,
                f"{entry.entry_id} quality comparison",
            )
            validation_path = _bound_file(
                registry_path.parent,
                entry.validation_file,
                entry.validation_sha256,
                f"{entry.entry_id} validation",
            )
            plan = load_model(plan_path, QuantizationPlan)
            artifact = load_model(artifact_path, ArtifactManifest)
            quality = load_model(quality_path, QualityComparisonReport)
            validation = load_model(validation_path, ValidationReport)
            rebuilt = build_complete_candidate_measurement(
                candidate_id=measurement.candidate_id,
                measurement_id=measurement.measurement_id,
                plan=plan,
                artifact=artifact,
                artifact_sha256=file_sha256(artifact_path),
                quality=quality,
                quality_sha256=file_sha256(quality_path),
                validation=validation,
                validation_sha256=file_sha256(validation_path),
            )
        except (ArtifactError, OSError, RefinementError, ValueError) as exc:
            issues.append(
                f"complete-model measurement evidence is invalid for {entry.entry_id}: {exc}"
            )
            continue
        if (
            entry.candidate_id != measurement.candidate_id
            or entry.plan_sha256 != measurement.plan_sha256
            or entry.artifact_manifest_sha256 != measurement.artifact_manifest_sha256
            or entry.quality_comparison_sha256 != measurement.quality_comparison_sha256
            or entry.validation_sha256 != measurement.validation_sha256
        ):
            issues.append(
                f"hardware registry and complete-model measurement differ for {entry.entry_id}"
            )
        if rebuilt != measurement:
            issues.append(f"complete-model objective cannot be reproduced for {entry.entry_id}")
    return issues


def _compatibility_request_issues(
    request_path: Path,
    matrix: CompatibilityMatrix,
) -> list[str]:
    request = load_model(request_path, CompatibilityMatrixRequest)
    base = request_path.parent
    issues: list[str] = []
    matrix_by_key = {
        (
            Path(entry.artifact_path).expanduser().resolve(),
            entry.profile,
        ): entry
        for entry in matrix.entries
    }
    if len(matrix_by_key) != len(matrix.entries):
        issues.append("compatibility matrix has duplicate artifact/profile evidence")
    if (
        matrix.scope_policy != request.scope_policy
        or matrix.official_catalog_url != request.official_catalog_url
        or matrix.catalog_verified_at != request.catalog_verified_at
        or matrix.required_dense_models != request.required_dense_models
        or matrix.required_profiles != request.required_profiles
        or matrix.required_dense_checkpoints != len(request.required_dense_models)
    ):
        issues.append("compatibility matrix release scope differs from its original request")

    request_keys: set[tuple[Path, ProfileName]] = set()
    for candidate in request.candidates:
        artifact = _resolved(base, candidate.artifact_directory)
        try:
            manifest_path = _required_file(
                artifact,
                "axquant_manifest.json",
                "compatibility artifact manifest",
            )
            plan_path = _required_file(
                artifact,
                "axquant_plan.json",
                "compatibility artifact plan",
            )
            ax_engine_path = _required_file(
                base,
                candidate.ax_engine_check,
                "compatibility AX Engine check",
            )
            mlx_lm_path = _required_file(
                base,
                candidate.mlx_lm_check,
                "compatibility MLX-LM check",
            )
            validation_path = _required_file(
                base,
                candidate.validation_report,
                "compatibility validation",
            )
            manifest = load_model(manifest_path, ArtifactManifest)
            plan = load_model(plan_path, QuantizationPlan)
            ax_engine = load_model(ax_engine_path, RuntimeCheck)
            mlx_lm = load_model(mlx_lm_path, RuntimeCheck)
            validation = load_model(validation_path, ValidationReport)
        except (ArtifactError, OSError, ValueError) as exc:
            issues.append(f"compatibility evidence is invalid for {artifact}: {exc}")
            continue
        request_key = (artifact, validation.profile)
        if request_key in request_keys:
            issues.append(
                f"compatibility request duplicates {artifact} for {validation.profile.value}"
            )
            continue
        request_keys.add(request_key)
        entry = matrix_by_key.get(request_key)
        if entry is None:
            issues.append(f"compatibility matrix omits {artifact} for {validation.profile.value}")
            continue

        expected_hashes = {
            "artifact manifest": (
                file_sha256(manifest_path),
                entry.artifact_manifest_sha256,
            ),
            "AX Engine check": (
                file_sha256(ax_engine_path),
                entry.ax_engine_check_sha256,
            ),
            "MLX-LM check": (
                file_sha256(mlx_lm_path),
                entry.mlx_lm_check_sha256,
            ),
            "validation": (
                file_sha256(validation_path),
                entry.validation_sha256,
            ),
        }
        for label, (actual, expected) in expected_hashes.items():
            if actual != expected:
                issues.append(f"compatibility {label} checksum changed for {artifact}")
        if stable_sha256(plan) != manifest.plan_sha256 or entry.plan_sha256 != manifest.plan_sha256:
            issues.append(f"compatibility plan binding changed for {artifact}")
        if (
            not same_model_identity(entry.source_model, manifest.source_model)
            or entry.profile != validation.profile
            or not same_model_identity(entry.candidate_model, validation.candidate_model)
        ):
            issues.append(f"compatibility identity/profile evidence changed for {artifact}")
        if not validation.passed or entry.validation_passed != validation.passed:
            issues.append(f"compatibility validation no longer passes for {artifact}")
        candidate_weight_bytes = validation.comparisons.get("artifact.candidate_weight_bytes")
        if candidate_weight_bytes != manifest.weight_file_size_bytes:
            issues.append(f"compatibility size evidence changed for {artifact}")
        supported_bits = sorted({assignment.bits for assignment in plan.assignments})
        architecture = plan.architecture_profile
        if (
            entry.adapter_id != architecture.adapter_id
            or entry.dense != (architecture.dense is True)
            or entry.text_layer_count != architecture.text_layer_count
            or entry.measured_total_bpw != manifest.measured_total_bpw
            or entry.mtp_present != manifest.mtp_present
            or entry.supported_bits != supported_bits
        ):
            issues.append(f"compatibility architecture/precision evidence changed for {artifact}")
        issues.extend(
            f"compatibility {issue}"
            for issue in _runtime_issues(
                ax_engine,
                runtime=RuntimeName.AX_ENGINE,
                artifact=artifact,
                candidate_id=validation.candidate_model.model_id,
                candidate_revision=validation.candidate_model.revision,
            )
        )
        issues.extend(
            f"compatibility {issue}"
            for issue in _runtime_issues(
                mlx_lm,
                runtime=RuntimeName.MLX_LM,
                artifact=artifact,
                candidate_id=validation.candidate_model.model_id,
                candidate_revision=validation.candidate_model.revision,
            )
        )
        if (
            entry.ax_engine_passed != ax_engine.passed
            or entry.mlx_lm_passed != mlx_lm.passed
            or not entry.compatible
        ):
            issues.append(f"compatibility status changed or failed for {artifact}")
    if request_keys != set(matrix_by_key):
        issues.append("compatibility matrix does not cover its original request")
    return issues


def _registry_measurements(
    registry_path: Path,
    registry: HardwareProfileRegistry,
) -> tuple[RefinementMeasurementSet | None, list[str]]:
    path = _resolved(registry_path.parent, registry.measurement_set_file)
    if not path.is_file():
        return None, [f"hardware measurement set is missing: {path}"]
    if file_sha256(path) != registry.measurement_set_file_sha256:
        return None, ["hardware measurement set checksum does not match the registry"]
    try:
        measurements = load_model(path, RefinementMeasurementSet)
    except (OSError, ValueError, ArtifactError) as exc:
        return None, [f"hardware measurement set is invalid: {exc}"]
    if stable_sha256(measurements) != registry.measurement_set_sha256:
        return None, ["hardware measurement set semantic digest does not match the registry"]
    return measurements, []


def _normalized_validation_index(index: ReleaseValidationIndex) -> dict[str, object]:
    payload = index.model_dump(mode="json")
    for entry in payload["entries"]:
        entry["validation_file"] = "<packaged>"
        entry["benchmark_index_file"] = "<packaged>"
        entry["benchmark_index_sha256"] = "<packaged>"
    return payload


def _normalized_benchmark_index(index: BenchmarkEvidenceIndex) -> dict[str, object]:
    payload = index.model_dump(mode="json")
    for entry in payload["entries"]:
        if entry["evaluation_file"] is not None:
            entry["evaluation_file"] = "<packaged>"
    return payload


def _normalized_hardware_registry(registry: HardwareProfileRegistry) -> dict[str, object]:
    payload = registry.model_dump(mode="json")
    payload["measurement_set_file"] = "<packaged>"
    for entry in payload["entries"]:
        for file_field, _sha_field in _HARDWARE_EVIDENCE_FIELDS:
            entry[file_field] = "<packaged>"
    return payload


def _packaged_release_issues(
    *,
    artifact: Path,
    plan: QuantizationPlan,
    recipe: ReproductionRecipe,
    validation_index: ReleaseValidationIndex,
    validation_evidence: dict[
        ProfileName,
        tuple[ValidationReport, BenchmarkEvidenceIndex],
    ],
    hardware: HardwareProfileRegistry,
    measurements: RefinementMeasurementSet | None,
    pareto: ParetoReport,
) -> list[str]:
    required_release_files = {
        "README.md",
        "release_validation_index.json",
        "hardware_profile_registry.json",
        "refinement_measurements.json",
        "pareto_report.json",
        "reproduction_recipe.yaml",
        "quantization_plan.json",
        "axquant_conversion_manifest.json",
    }
    missing = sorted(
        relative for relative in required_release_files if not (artifact / relative).is_file()
    )
    if missing:
        return [f"prepared release files are missing: {missing}"]

    issues: list[str] = []
    try:
        packaged_plan = load_model(artifact / "quantization_plan.json", QuantizationPlan)
        if packaged_plan != plan:
            issues.append("packaged quantization plan differs from the audited plan")

        packaged_recipe = load_model(artifact / "reproduction_recipe.yaml", ReproductionRecipe)
        if packaged_recipe != recipe:
            issues.append("packaged reproduction recipe differs from the audited recipe")

        conversion_manifest_path = artifact / "axquant_conversion_manifest.json"
        load_model(conversion_manifest_path, ArtifactManifest)
        if file_sha256(conversion_manifest_path) != recipe.conversion_manifest_sha256:
            issues.append("packaged immutable conversion manifest differs from the audited recipe")

        packaged_pareto = load_model(artifact / "pareto_report.json", ParetoReport)
        if packaged_pareto != pareto:
            issues.append("packaged Pareto report differs from the audited report")

        packaged_measurements_path = artifact / "refinement_measurements.json"
        packaged_measurements = load_model(
            packaged_measurements_path,
            RefinementMeasurementSet,
        )
        if (
            measurements is None
            or packaged_measurements != measurements
            or file_sha256(packaged_measurements_path) != hardware.measurement_set_file_sha256
        ):
            issues.append(
                "packaged refinement measurements differ from the audited hardware evidence"
            )

        packaged_hardware_path = artifact / "hardware_profile_registry.json"
        packaged_hardware = load_model(packaged_hardware_path, HardwareProfileRegistry)
        if _normalized_hardware_registry(packaged_hardware) != _normalized_hardware_registry(
            hardware
        ):
            issues.append("packaged hardware registry differs from the audited registry")
        packaged_hardware_measurements, packaged_hardware_issues = _registry_measurements(
            packaged_hardware_path,
            packaged_hardware,
        )
        issues.extend(
            f"packaged {issue}"
            for issue in _hardware_evidence_issues(
                packaged_hardware_path,
                packaged_hardware,
            )
        )
        if packaged_hardware_measurements is not None and measurements is not None:
            issues.extend(
                f"packaged {issue}"
                for issue in _complete_measurement_evidence_issues(
                    packaged_hardware_path,
                    packaged_hardware,
                    packaged_hardware_measurements,
                    expected_evaluator_version=measurements.evaluator_version,
                )
            )
        issues.extend(f"packaged {issue}" for issue in packaged_hardware_issues)
        if (
            packaged_hardware_measurements is not None
            and packaged_hardware_measurements != measurements
        ):
            issues.append("packaged hardware registry binds different measurements")

        packaged_validation_path = artifact / "release_validation_index.json"
        packaged_validation = load_model(
            packaged_validation_path,
            ReleaseValidationIndex,
        )
        packaged_validation_evidence = _validation_evidence(
            packaged_validation_path,
            packaged_validation,
        )
        if _normalized_validation_index(packaged_validation) != _normalized_validation_index(
            validation_index
        ):
            issues.append("packaged release validation index differs from the audited index")
        for profile, (validation, benchmark) in validation_evidence.items():
            packaged_pair = packaged_validation_evidence.get(profile)
            if packaged_pair is None:
                issues.append(f"packaged release evidence lacks {profile.value}")
                continue
            packaged_report, packaged_benchmark = packaged_pair
            if packaged_report != validation:
                issues.append(f"packaged {profile.value} validation differs from audited evidence")
            if _normalized_benchmark_index(packaged_benchmark) != _normalized_benchmark_index(
                benchmark
            ):
                issues.append(
                    f"packaged {profile.value} benchmark index differs from audited evidence"
                )
    except (ArtifactError, AssertionError, OSError, ValueError) as exc:
        issues.append(f"prepared release evidence is invalid: {exc}")
    return issues


def _required_sensitivity_bits(tensor: TensorSpec, plan: QuantizationPlan) -> set[int]:
    if not tensor.quantizable:
        return {16}
    if (
        tensor.role.is_mtp
        and Path(tensor.file).name.lower() in EXTERNAL_MTP_SIDECAR_FILENAMES
        and plan.mtp.preserve_external_sidecar
    ):
        return {16}

    allowed = set(plan.candidate_bits) & set(plan.hardware.supported_bits)
    if tensor.role.is_mtp and plan.mtp.mode != "disabled":
        allowed &= set(plan.mtp.candidate_bits)
        minimum_bits = plan.mtp.min_bits
    else:
        minimum_bits = _RELEASE_PROBE_MIN_BITS.get(tensor.role, min(plan.candidate_bits))
        if tensor.role == TensorRole.LM_HEAD:
            # AXQ-026: an explicitly lowered LM-head floor widens the choice
            # set, so every reachable precision must be measured.
            minimum_bits = plan.constraints.lm_head_min_bits
    return {bits for bits in allowed if bits >= minimum_bits}


def _calibration_issues(
    artifact: Path,
    sensitivity: SensitivityReport,
    plan: QuantizationPlan,
) -> list[str]:
    issues: list[str] = []
    evidence = plan.calibration
    if evidence is None or sensitivity.calibration is None:
        return ["measured calibration provenance is missing"]
    reference = Path(evidence.reference)
    if (
        reference.is_absolute()
        or ".." in reference.parts
        or reference.name != "calibration_manifest.json"
    ):
        return ["calibration evidence does not reference the packaged manifest"]
    path = artifact / reference
    expected_sha256 = evidence.metadata.get("calibration_manifest_sha256")
    if not path.is_file() or not isinstance(expected_sha256, str):
        return ["calibration manifest is missing or checksum-mismatched"]
    try:
        manifest = load_model(path, CalibrationManifest)
    except (OSError, ValueError, ArtifactError) as exc:
        return [f"calibration manifest is invalid: {exc}"]
    if not calibration_manifest_matches(path, manifest, expected_sha256):
        return ["calibration manifest is missing or checksum-mismatched"]
    if (
        not same_model_identity(manifest.model, plan.source_model)
        or manifest.profile != plan.profile
        or manifest.dataset_id != evidence.dataset_id
        or manifest.dataset_sha256 != evidence.dataset_sha256
        or manifest.samples != evidence.samples
        or set(manifest.domains) != set(evidence.domains)
        or manifest.sequence_length != evidence.sequence_length
    ):
        issues.append("calibration manifest differs from the measured plan provenance")
    calibration_seed = evidence.metadata.get("calibration_random_seed")
    if (
        not isinstance(calibration_seed, int)
        or isinstance(calibration_seed, bool)
        or manifest.random_seed != calibration_seed
    ):
        issues.append("calibration manifest random seed is absent or differs from provenance")
    if not manifest.calibration_evaluation_separation_attested:
        issues.append("calibration/evaluation separation is not attested")
    return issues


def _sensitivity_measurement_issues(
    sensitivity: SensitivityReport,
    plan: QuantizationPlan,
) -> list[str]:
    issues: list[str] = []
    if not {4, 6, 8, 16}.issubset(plan.candidate_bits):
        issues.append("release plan does not declare the 4/6/8/BF16 precision set")

    entry_names = [entry.tensor.name for entry in sensitivity.entries]
    assignment_names = [assignment.tensor for assignment in plan.assignments]
    if len(entry_names) != len(set(entry_names)):
        issues.append("sensitivity report contains duplicate tensor entries")
    if len(assignment_names) != len(set(assignment_names)):
        issues.append("release plan contains duplicate tensor assignments")
    if set(entry_names) != set(assignment_names):
        issues.append("sensitivity report and release plan have different tensor coverage")

    for entry in sensitivity.entries:
        required_bits = _required_sensitivity_bits(entry.tensor, plan)
        external_preserved = (
            entry.tensor.role.is_mtp
            and Path(entry.tensor.file).name.lower() in EXTERNAL_MTP_SIDECAR_FILENAMES
            and plan.mtp.preserve_external_sidecar
        )
        required_scope = "tensor" if entry.tensor.quantizable and not external_preserved else None
        measured_bits = {
            candidate.bits
            for candidate in entry.candidates
            if candidate.measured_tokens > 0
            and candidate.evidence_scope != "module-group"
            and (required_scope is None or candidate.evidence_scope == required_scope)
        }
        missing = sorted(required_bits - measured_bits)
        if missing:
            issues.append(
                f"{entry.tensor.name} lacks complete measured candidates at bits {missing}"
            )
        if any(
            not isfinite(float(value))
            for candidate in entry.candidates
            for value in candidate.metrics.model_dump().values()
        ):
            issues.append(f"{entry.tensor.name} contains non-finite sensitivity metrics")
    return issues


def _activation_capture_artifact_issues(
    artifact: Path,
    sensitivity: SensitivityReport,
    plan: QuantizationPlan,
) -> list[str]:
    """Verify AWQ/GPTQ lineage through the packaged capture manifest."""
    methods = {
        assignment.method
        for assignment in plan.assignments
        if assignment.bits < 16 and assignment.method in _ACTIVATION_REFINEMENT_METHODS
    }
    if not methods:
        return []
    if plan.calibration is None or sensitivity.calibration is None:
        return ["AWQ/GPTQ release evidence lacks activation-capture calibration provenance"]
    plan_metadata = plan.calibration.metadata
    sensitivity_metadata = sensitivity.calibration.metadata
    missing = [
        key
        for key in CAPTURE_METADATA_KEYS
        if not isinstance(plan_metadata.get(key), str) or not plan_metadata.get(key)
    ]
    issues: list[str] = []
    if missing:
        issues.append(f"AWQ/GPTQ release plan lacks activation-capture bindings: {missing}")
    changed = [
        key
        for key in CAPTURE_METADATA_KEYS
        if plan_metadata.get(key) != sensitivity_metadata.get(key)
    ]
    if changed:
        issues.append(f"sensitivity and plan activation-capture bindings differ: {changed}")
    capture_path = artifact / _ACTIVATION_CAPTURE_MANIFEST
    if not capture_path.is_file():
        issues.append("release artifact does not package activation_capture_manifest.json")
        return issues
    try:
        manifest = load_model(capture_path, ActivationCaptureManifest)
    except (ArtifactError, OSError, ValueError) as exc:
        issues.append(f"packaged activation capture manifest is invalid: {exc}")
        return issues
    issues.extend(
        activation_capture_evidence_issues(
            manifest,
            plan_metadata,
            model_id=plan.source_model.model_id,
            revision=plan.source_model.revision,
            dataset_id=plan.calibration.dataset_id,
        )
    )
    return issues


_SENSITIVITY_LINEAGE_PROTOCOL_FIELDS = (
    "cache_key_sha256",
    "sample_order_sha256",
    "tokenizer_sha256",
    "calibration_manifest_sha256",
    "domain_provenance",
    "token_budget_per_candidate",
    "measured_tokens_per_candidate",
    "packed_replay_sequences",
    "replay_batches",
    "replay_batch_size",
    "packing",
    "metric_positions_per_sample",
    "long_context_min_tokens",
    "warmup_replays",
    "capture_points",
    "module_group_probing",
)


def _sensitivity_lineage_link_issues(
    child: SensitivityReport,
    parent: SensitivityReport,
) -> list[str]:
    issues: list[str] = []
    child_calibration = child.calibration
    parent_calibration = parent.calibration
    if child_calibration is None or parent_calibration is None:
        return ["sensitivity lineage link lacks calibration provenance"]

    parent_sha256 = stable_sha256(parent)
    metadata = child_calibration.metadata
    if metadata.get("base_sensitivity_sha256") != parent_sha256:
        issues.append("sensitivity lineage parent semantic digest differs from provenance")
    if metadata.get("base_inventory_sha256") != parent.inventory_sha256:
        issues.append("sensitivity lineage parent inventory digest differs from provenance")
    if metadata.get("base_probe_backend") != parent_calibration.backend:
        issues.append("sensitivity lineage parent backend differs from provenance")
    if metadata.get("refinement_probe_backend") != child_calibration.backend:
        issues.append("sensitivity lineage refinement backend differs from provenance")
    if parent.evidence_kind != EvidenceKind.MEASURED:
        issues.append("sensitivity lineage parent is not measured release evidence")
    if not same_model_identity(child.model, parent.model) or child.profile != parent.profile:
        issues.append("sensitivity lineage model/profile changed")
    if child.architecture_profile.model_dump(exclude={"notes"}) != (
        parent.architecture_profile.model_dump(exclude={"notes"})
    ):
        issues.append("sensitivity lineage architecture contract changed")
    if (
        child_calibration.dataset_sha256 != parent_calibration.dataset_sha256
        or child_calibration.samples != parent_calibration.samples
        or set(child_calibration.domains) != set(parent_calibration.domains)
        or child_calibration.sequence_length != parent_calibration.sequence_length
    ):
        issues.append("sensitivity lineage calibration dataset/protocol shape changed")
    changed_protocol = sorted(
        field
        for field in _SENSITIVITY_LINEAGE_PROTOCOL_FIELDS
        if child_calibration.metadata.get(field) != parent_calibration.metadata.get(field)
    )
    if changed_protocol:
        issues.append(f"sensitivity lineage probe protocol changed: {changed_protocol}")

    parent_entries = {entry.tensor.name: entry for entry in parent.entries}
    child_entries = {entry.tensor.name: entry for entry in child.entries}
    if len(parent_entries) != len(parent.entries) or len(child_entries) != len(child.entries):
        issues.append("sensitivity lineage contains duplicate tensor entries")
        return issues
    if set(parent_entries) != set(child_entries):
        issues.append("sensitivity lineage tensor coverage changed")
        return issues

    methods_value = metadata.get("candidate_methods")
    allowed_methods = (
        {method.strip() for method in methods_value.split(",") if method.strip()}
        if isinstance(methods_value, str)
        else set()
    )
    if not allowed_methods:
        issues.append("sensitivity lineage does not declare refinement candidate methods")
    changed_tensors = 0
    for tensor_name, parent_entry in parent_entries.items():
        child_entry = child_entries[tensor_name]
        if child_entry.tensor != parent_entry.tensor:
            issues.append(f"sensitivity lineage tensor metadata changed: {tensor_name}")
            continue
        parent_candidates = {
            (candidate.bits, candidate.method, candidate.group_size): candidate
            for candidate in parent_entry.candidates
        }
        child_candidates = {
            (candidate.bits, candidate.method, candidate.group_size): candidate
            for candidate in child_entry.candidates
        }
        if len(parent_candidates) != len(parent_entry.candidates) or len(child_candidates) != len(
            child_entry.candidates
        ):
            issues.append(f"sensitivity lineage candidate keys are duplicated: {tensor_name}")
            continue
        missing_or_changed = [
            key
            for key, candidate in parent_candidates.items()
            if child_candidates.get(key) != candidate
        ]
        if missing_or_changed:
            issues.append(
                f"sensitivity lineage changed base candidates for {tensor_name}: "
                f"{len(missing_or_changed)}"
            )
        additions = [
            candidate for key, candidate in child_candidates.items() if key not in parent_candidates
        ]
        if additions:
            changed_tensors += 1
        if any(candidate.method.value not in allowed_methods for candidate in additions):
            issues.append(f"sensitivity lineage added an undeclared method for {tensor_name}")
        if any(
            candidate.measured_tokens <= 0 or candidate.evidence_scope != "tensor"
            for candidate in additions
        ):
            issues.append(f"sensitivity lineage added unmeasured candidates for {tensor_name}")

    target_count = metadata.get("target_tensor_count")
    if (
        not isinstance(target_count, int)
        or isinstance(target_count, bool)
        or target_count != changed_tensors
    ):
        issues.append(
            "sensitivity lineage target tensor count does not match the measured additions"
        )
    return issues


def _sensitivity_lineage_issues(
    sensitivity: SensitivityReport,
    lineage: list[SensitivityReport],
) -> list[str]:
    issues: list[str] = []
    by_sha256: dict[str, SensitivityReport] = {}
    for report in lineage:
        digest = stable_sha256(report)
        if digest in by_sha256:
            issues.append(f"sensitivity lineage repeats semantic digest {digest}")
        by_sha256[digest] = report

    current = sensitivity
    visited = {stable_sha256(current)}
    used: set[str] = set()
    while current.calibration is not None:
        parent_digest = current.calibration.metadata.get("base_sensitivity_sha256")
        if parent_digest is None:
            break
        if not isinstance(parent_digest, str) or not parent_digest:
            issues.append("sensitivity lineage parent digest is malformed")
            break
        if parent_digest in visited:
            issues.append("sensitivity lineage contains a cycle")
            break
        parent = by_sha256.get(parent_digest)
        if parent is None:
            issues.append(f"sensitivity lineage parent is missing: {parent_digest}")
            break
        used.add(parent_digest)
        visited.add(parent_digest)
        issues.extend(_sensitivity_lineage_link_issues(current, parent))
        current = parent

    unused = sorted(set(by_sha256) - used)
    if unused:
        issues.append(f"sensitivity lineage contains unused reports: {unused}")
    return issues


def build_release_audit(request_path: str | Path) -> ReleaseAudit:
    request_source = Path(request_path).expanduser().resolve()
    request = load_model(request_source, ReleaseAuditRequest)
    base = request_source.parent
    artifact = _required_directory(base, request.artifact_directory, "release artifact")
    paths = {
        "feasibility": _required_file(base, request.feasibility_report, "feasibility report"),
        "sensitivity": _required_file(base, request.sensitivity_report, "sensitivity report"),
        "refinement": _required_file(base, request.refinement_result, "refinement result"),
        "validation": _required_file(
            base, request.release_validation_index, "release validation index"
        ),
        "hardware": _required_file(base, request.hardware_registry, "hardware registry"),
        "pareto": _required_file(base, request.pareto_report, "Pareto report"),
        "compatibility": _required_file(base, request.compatibility_matrix, "compatibility matrix"),
        "compatibility_request": _required_file(
            base,
            request.compatibility_request,
            "compatibility request",
        ),
        "recipe": _required_file(base, request.reproduction_recipe, "reproduction recipe"),
        "reproduction": _required_file(
            base, request.reproduction_verification, "reproduction verification"
        ),
        "ax_engine": _required_file(base, request.ax_engine_check, "AX Engine check"),
        "mlx_lm": _required_file(base, request.mlx_lm_check, "MLX-LM check"),
        "wheel": _required_file(base, request.toolkit_wheel, "toolkit wheel"),
    }
    sensitivity_lineage_paths = [
        _required_file(base, value, "sensitivity lineage report")
        for value in request.sensitivity_lineage
    ]
    release_exception_paths = [
        _required_file(base, value, "release exception") for value in request.release_exceptions
    ]
    release_exception_evidence_paths = {
        name: _required_file(base, value, f"release exception evidence {name}")
        for name, value in request.release_exception_evidence.items()
    }
    manifest_path = _required_file(artifact, "axquant_manifest.json", "artifact manifest")
    plan_path = _required_file(artifact, "axquant_plan.json", "artifact plan")
    manifest = load_model(manifest_path, ArtifactManifest)
    plan = load_model(plan_path, QuantizationPlan)
    feasibility = load_model(paths["feasibility"], FeasibilityReport)
    sensitivity = load_model(paths["sensitivity"], SensitivityReport)
    sensitivity_lineage = [
        load_model(path, SensitivityReport) for path in sensitivity_lineage_paths
    ]
    refinement = load_model(paths["refinement"], RefinementResult)
    validation_index = load_model(paths["validation"], ReleaseValidationIndex)
    validation_evidence = _validation_evidence(paths["validation"], validation_index)
    hardware = load_model(paths["hardware"], HardwareProfileRegistry)
    measurements, measurement_issues = _registry_measurements(paths["hardware"], hardware)
    pareto = load_model(paths["pareto"], ParetoReport)
    compatibility = load_model(paths["compatibility"], CompatibilityMatrix)
    compatibility_request = load_model(
        paths["compatibility_request"],
        CompatibilityMatrixRequest,
    )
    recipe = load_model(paths["recipe"], ReproductionRecipe)
    reproduction = load_model(paths["reproduction"], ReproductionVerification)
    ax_engine = load_model(paths["ax_engine"], RuntimeCheck)
    mlx_lm = load_model(paths["mlx_lm"], RuntimeCheck)
    release_exceptions = [load_model(path, ReleaseException) for path in release_exception_paths]
    candidate_model = next(iter(validation_index.entries)).candidate_model

    checks: list[ReleaseAuditCheck] = []

    m0_issues = _feasibility_issues(feasibility, plan)
    checks.append(
        ReleaseAuditCheck(
            gate_id="M0",
            name="Technical feasibility",
            passed=not m0_issues,
            evidence_sha256={"feasibility_report": file_sha256(paths["feasibility"])},
            issues=m0_issues,
        )
    )

    m1_issues = _artifact_issues(artifact, manifest)
    if stable_sha256(plan) != manifest.plan_sha256:
        m1_issues.append("artifact manifest does not bind axquant_plan.json")
    if not same_model_identity(manifest.source_model, plan.source_model):
        m1_issues.append("artifact and plan source models differ")
    if (
        manifest.profile != plan.profile
        or manifest.target_class != plan.target_class
        or manifest.calibration != plan.calibration
    ):
        m1_issues.append("artifact and plan profile/calibration provenance differ")
    if (
        manifest.effective_bpw != plan.effective_bpw
        or manifest.weight_distribution != plan.weight_distribution
        or manifest.mtp_distribution != plan.mtp_distribution
        or manifest.mtp_policy != plan.mtp
    ):
        m1_issues.append("artifact and plan precision policy differ")
    if manifest.runtime.optimization_scope != plan.architecture_profile.optimization_scope:
        m1_issues.append("artifact runtime optimization scope differs from the plan")
    if manifest.runtime.primary_runtime.name != RuntimeName.AX_ENGINE:
        m1_issues.append("AX Engine is not the artifact primary runtime")
    if not manifest.mtp_present or not manifest.runtime.mtp.detected:
        m1_issues.append("release artifact does not contain declared MTP weights")
    m1_issues.extend(
        _runtime_issues(
            ax_engine,
            runtime=RuntimeName.AX_ENGINE,
            artifact=artifact,
            candidate_id=candidate_model.model_id,
            candidate_revision=candidate_model.revision,
        )
    )
    m1_issues.extend(
        _runtime_issues(
            mlx_lm,
            runtime=RuntimeName.MLX_LM,
            artifact=artifact,
            candidate_id=candidate_model.model_id,
            candidate_revision=candidate_model.revision,
        )
    )
    checks.append(
        ReleaseAuditCheck(
            gate_id="M1",
            name="AX Engine vertical slice and runtime compatibility",
            passed=not m1_issues,
            evidence_sha256={
                "artifact_manifest": file_sha256(manifest_path),
                "plan": file_sha256(plan_path),
                "ax_engine_check": file_sha256(paths["ax_engine"]),
                "mlx_lm_check": file_sha256(paths["mlx_lm"]),
            },
            issues=m1_issues,
        )
    )

    m2_issues: list[str] = []
    if not validation_index.release_ready:
        m2_issues.append("release validation index did not pass")
    candidate_models = {
        model_identity_key(validation.candidate_model)
        for validation, _benchmark in validation_evidence.values()
    }
    reference_models = {
        model_identity_key(validation.reference_model)
        for validation, _benchmark in validation_evidence.values()
    }
    dataset_digests = {
        benchmark.dataset_sha256 for _validation, benchmark in validation_evidence.values()
    }
    if len(candidate_models) != 1 or len(reference_models) != 1:
        m2_issues.append("release profiles do not use one candidate/reference pair")
    if None in dataset_digests or len(dataset_digests) != 2:
        m2_issues.append("release profile datasets are not distinct and complete")
    validation_entries_by_profile = {entry.profile: entry for entry in validation_index.entries}
    expected_size_reference_kind = "uniform-6bit" if plan.target_class == "6bit" else "uniform-4bit"
    for profile, (validation, _benchmark) in validation_evidence.items():
        index_entry = validation_entries_by_profile[profile]
        if (
            not same_model_identity(index_entry.reference_model, validation.reference_model)
            or not same_model_identity(index_entry.candidate_model, validation.candidate_model)
            or index_entry.dataset_sha256 != _benchmark.dataset_sha256
            or index_entry.passed != validation.passed
            or validation.profile != profile
            or _benchmark.profile != profile
        ):
            m2_issues.append(f"{profile.value} release validation entry is inconsistent")
        acceptance = validation.comparisons.get("mtp.acceptance_retention")
        speedup = validation.comparisons.get("hardware.effective_speedup")
        if not validation.passed:
            m2_issues.append(f"{profile.value} validation did not pass")
        if plan.target_class in {"4bit", "6bit"} and validation.target_class != plan.target_class:
            m2_issues.append(
                f"{profile.value} validation target class differs from the release plan"
            )
        recorded_size_reference_kind = validation.comparisons.get("artifact.size_reference_kind")
        if (
            plan.target_class == "6bit"
            and recorded_size_reference_kind != expected_size_reference_kind
        ) or (
            recorded_size_reference_kind is not None
            and recorded_size_reference_kind != expected_size_reference_kind
        ):
            m2_issues.append(
                f"{profile.value} validation does not use the "
                f"{expected_size_reference_kind} size policy"
            )
        if validation.thresholds != thresholds_for(profile):
            m2_issues.append(
                f"{profile.value} validation does not use the authoritative profile thresholds"
            )
        expected_validation_passed = not any(
            issue.severity == "error" for issue in validation.issues
        )
        if validation.passed != expected_validation_passed:
            m2_issues.append(
                f"{profile.value} validation pass status is inconsistent with its issues"
            )
        if not isinstance(acceptance, (int, float)) or float(acceptance) < (
            validation.thresholds.minimum_mtp_acceptance_retention
        ):
            m2_issues.append(f"{profile.value} MTP acceptance retention gate failed")
        if not isinstance(speedup, (int, float)) or float(speedup) < (
            validation.thresholds.min_effective_speedup
        ):
            m2_issues.append(f"{profile.value} MTP speed gate failed")
    for profile, (_validation, benchmark) in validation_evidence.items():
        benchmark_path = _resolved(
            paths["validation"].parent,
            validation_entries_by_profile[profile].benchmark_index_file,
        )
        m2_issues.extend(
            f"{profile.value}: {issue}"
            for issue in _benchmark_index_issues(benchmark_path, benchmark)
        )
    checks.append(
        ReleaseAuditCheck(
            gate_id="M2",
            name="Correct and repeatable MTP benchmark",
            passed=not m2_issues,
            evidence_sha256={"release_validation_index": file_sha256(paths["validation"])},
            issues=m2_issues,
        )
    )

    m3_issues: list[str] = []
    if sensitivity.evidence_kind != EvidenceKind.MEASURED:
        m3_issues.append("sensitivity report is not measured evidence")
    if sensitivity.calibration != plan.calibration:
        m3_issues.append("sensitivity and plan calibration evidence differ")
    m3_issues.extend(_calibration_issues(artifact, sensitivity, plan))
    if stable_sha256(sensitivity) != plan.analysis_sha256:
        m3_issues.append("plan does not bind the measured sensitivity report")
    if (
        not same_model_identity(sensitivity.model, plan.source_model)
        or sensitivity.profile != plan.profile
    ):
        m3_issues.append("sensitivity report and plan identity/profile differ")
    # The support tier is current registry policy, not recorded evidence: a plan
    # legitimately carries a newer tier than the report it was built from
    # (AXQ-017), so profile equality excludes it.
    if sensitivity.architecture_profile.model_dump(
        exclude={"support_tier"}
    ) != plan.architecture_profile.model_dump(exclude={"support_tier"}):
        m3_issues.append("sensitivity and plan architecture profiles differ")
    m3_issues.extend(_sensitivity_measurement_issues(sensitivity, plan))
    m3_issues.extend(_activation_capture_artifact_issues(artifact, sensitivity, plan))
    m3_issues.extend(_sensitivity_lineage_issues(sensitivity, sensitivity_lineage))
    m3_evidence_sha256 = {"sensitivity_report": file_sha256(paths["sensitivity"])}
    m3_evidence_sha256.update(
        {
            f"sensitivity_lineage_{index:03d}": file_sha256(path)
            for index, path in enumerate(sensitivity_lineage_paths)
        }
    )
    checks.append(
        ReleaseAuditCheck(
            gate_id="M3",
            name="Measured MTP-aware planner",
            passed=not m3_issues,
            evidence_sha256=m3_evidence_sha256,
            issues=m3_issues,
        )
    )

    m4_issues: list[str] = []
    learned_methods = {
        assignment.method
        for assignment in plan.assignments
        if assignment.method
        in {QuantMethod.AWQ, QuantMethod.DWQ, QuantMethod.GPTQ, QuantMethod.GPTQ_ACT}
    }
    if not learned_methods:
        m4_issues.append("selected plan uses no learned refinement method")
    agent_benchmark = validation_evidence[ProfileName.AGENT_CODING][1]
    benchmark_entries = {entry.kind: entry for entry in agent_benchmark.entries}
    method_baseline = {
        QuantMethod.AWQ: BenchmarkEvidenceKind.AWQ,
        QuantMethod.DWQ: BenchmarkEvidenceKind.DWQ,
        QuantMethod.GPTQ: BenchmarkEvidenceKind.GPTQ,
        # Act-order GPTQ compares against the same uniform GPTQ baseline pack;
        # the ordering changes refinement, not the baseline family.
        QuantMethod.GPTQ_ACT: BenchmarkEvidenceKind.GPTQ,
    }
    missing_method_baselines = sorted(
        method.value
        for method in learned_methods
        if benchmark_entries[method_baseline[method]].status != "available"
    )
    if missing_method_baselines:
        m4_issues.append(
            f"selected learned methods lack comparison baselines: {missing_method_baselines}"
        )
    if not all(
        validation.passed and benchmark.release_ready
        for validation, benchmark in validation_evidence.values()
    ):
        m4_issues.append("complete candidate comparison did not pass both profiles")
    m4_exception_evidence: dict[str, str] = {}
    packaged_exception_path = artifact / "release_exception.json"
    if release_exceptions:
        exception = release_exceptions[0]
        exception_digest = stable_sha256(exception)
        m4_exception_evidence["release_exception"] = file_sha256(release_exception_paths[0])
        m4_exception_evidence.update(
            {
                f"release_exception_{name}": file_sha256(path)
                for name, path in sorted(release_exception_evidence_paths.items())
            }
        )
        if not packaged_exception_path.is_file():
            m4_issues.append("release artifact does not package its release exception")
        else:
            try:
                packaged_exception = load_model(
                    packaged_exception_path,
                    ReleaseException,
                )
                if stable_sha256(packaged_exception) != exception_digest:
                    m4_issues.append(
                        "packaged release exception differs from the approved exception"
                    )
                m4_exception_evidence["packaged_release_exception"] = file_sha256(
                    packaged_exception_path
                )
            except (ArtifactError, OSError, ValueError) as exc:
                m4_issues.append(f"packaged release exception is invalid: {exc}")
        for profile, (validation, _benchmark) in validation_evidence.items():
            if (
                len(validation.release_exceptions) != 1
                or stable_sha256(validation.release_exceptions[0]) != exception_digest
            ):
                m4_issues.append(
                    f"{profile.value} validation does not bind the approved release exception"
                )
                continue
            try:
                verify_release_exception(
                    exception,
                    plan=plan,
                    validation=validation,
                    evidence_files=release_exception_evidence_paths,
                )
                release_exception_allows_size(validation, plan=plan)
            except (ArtifactError, OSError, ValidationGateError, ValueError) as exc:
                m4_issues.append(f"{profile.value} release exception verification failed: {exc}")
    else:
        if packaged_exception_path.exists():
            m4_issues.append("release artifact contains an unapproved release exception")
        for profile, (validation, _benchmark) in validation_evidence.items():
            size_ratio = validation.comparisons.get("artifact.weight_size_ratio")
            if validation.release_exceptions:
                m4_issues.append(
                    f"{profile.value} validation contains an unapproved release exception"
                )
            if (
                isinstance(size_ratio, (int, float))
                and float(size_ratio) > validation.thresholds.max_weight_size_ratio
            ):
                m4_issues.append(
                    f"{profile.value} artifact size exceeds the limit without an exception"
                )
    checks.append(
        ReleaseAuditCheck(
            gate_id="M4",
            name="Quality and learned-method refinement",
            passed=not m4_issues,
            evidence_sha256={
                "release_validation_index": file_sha256(paths["validation"]),
                "selected_plan": file_sha256(plan_path),
                **m4_exception_evidence,
            },
            issues=m4_issues,
        )
    )

    m5_issues: list[str] = []
    if not compatibility.release_ready:
        m5_issues.append("Qwen family compatibility matrix is not release-ready")
    dense_source_checkpoints = {
        (entry.source_model.model_id, entry.source_model.revision)
        for entry in compatibility.entries
        if entry.dense and entry.compatible
    }
    if compatibility.distinct_dense_source_checkpoints != len(dense_source_checkpoints):
        m5_issues.append("compatibility matrix dense-checkpoint count is inconsistent")
    if compatibility.required_dense_checkpoints != request.required_dense_checkpoints:
        m5_issues.append("compatibility matrix and release audit require different family sizes")
    if len(dense_source_checkpoints) != request.required_dense_checkpoints:
        m5_issues.append(
            f"expected exactly {request.required_dense_checkpoints} dense source checkpoints; "
            f"observed {len(dense_source_checkpoints)}"
        )
    required_dense_model_ids = {
        requirement.model_id for requirement in compatibility.required_dense_models
    }
    observed_dense_model_ids = {model_id for model_id, _revision in dense_source_checkpoints}
    if observed_dense_model_ids != required_dense_model_ids:
        m5_issues.append(
            "compatible dense source models do not match the official release-time scope"
        )
    for model_id in sorted(required_dense_model_ids):
        model_profiles = {
            entry.profile
            for entry in compatibility.entries
            if entry.dense and entry.compatible and entry.source_model.model_id == model_id
        }
        if model_profiles != set(compatibility.required_profiles):
            m5_issues.append(f"{model_id} does not pass every required compatibility profile")
    if (
        compatibility.required_dense_models != compatibility_request.required_dense_models
        or compatibility.required_profiles != compatibility_request.required_profiles
    ):
        m5_issues.append("compatibility matrix official scope binding changed")
    validation_sha256_by_profile = {
        entry.profile: entry.validation_sha256 for entry in validation_index.entries
    }
    matching_compatibility = [
        entry
        for entry in compatibility.entries
        if same_model_identity(entry.candidate_model, candidate_model)
        and same_model_identity(entry.source_model, plan.source_model)
        and entry.plan_sha256 == manifest.plan_sha256
        and entry.artifact_manifest_sha256 == file_sha256(manifest_path)
        and entry.ax_engine_check_sha256 == file_sha256(paths["ax_engine"])
        and entry.mlx_lm_check_sha256 == file_sha256(paths["mlx_lm"])
        and entry.validation_sha256 == validation_sha256_by_profile.get(entry.profile)
        and entry.compatible
    ]
    matching_profiles = {entry.profile for entry in matching_compatibility}
    if matching_profiles != set(compatibility.required_profiles):
        m5_issues.append(
            "compatibility matrix does not certify the release candidate in every required profile"
        )
    m5_issues.extend(
        _compatibility_request_issues(
            paths["compatibility_request"],
            compatibility,
        )
    )
    checks.append(
        ReleaseAuditCheck(
            gate_id="M5",
            name="Dense Qwen family proof",
            passed=not m5_issues,
            evidence_sha256={
                "compatibility_matrix": file_sha256(paths["compatibility"]),
                "compatibility_request": file_sha256(paths["compatibility_request"]),
            },
            issues=m5_issues,
        )
    )

    m6_issues: list[str] = []
    if refinement.selection_basis != "complete-model":
        m6_issues.append("refinement selection is not based on complete-model measurements")
    if refinement.selected_plan_sha256 != manifest.plan_sha256 or refinement.selected_plan != plan:
        m6_issues.append("refinement selection does not match the release plan")
    history = {entry.candidate_id: entry for entry in refinement.history}
    if measurements is None:
        m6_issues.append("complete-model refinement measurements are unavailable")
    else:
        m6_issues.extend(
            _complete_measurement_evidence_issues(
                paths["hardware"],
                hardware,
                measurements,
                expected_evaluator_version=(
                    f"{request.required_toolkit_version}:{COMPLETE_OBJECTIVE_VERSION}"
                ),
            )
        )
        measurements_by_candidate: dict[str, list[CompleteCandidateMeasurement]] = {}
        for measurement in measurements.measurements:
            measurements_by_candidate.setdefault(measurement.candidate_id, []).append(measurement)
        verified_improvements: list[str] = []
        for entry in refinement.history:
            if entry.parent_id is None or entry.measured_loss is None or entry.measured_bpw is None:
                continue
            parent = history.get(entry.parent_id)
            child_plan = refinement.candidate_plans.get(entry.candidate_id)
            parent_plan = refinement.candidate_plans.get(entry.parent_id)
            if (
                parent is None
                or child_plan is None
                or parent_plan is None
                or not _is_monotonic_precision_refinement(parent_plan, child_plan)
                or parent.measured_loss is None
                or parent.measured_bpw is None
                or entry.measured_loss >= parent.measured_loss
                or entry.plan_sha256 == parent.plan_sha256
            ):
                continue
            child_measurements = measurements_by_candidate.get(entry.candidate_id, [])
            parent_measurements = measurements_by_candidate.get(parent.candidate_id, [])
            if (
                not child_measurements
                or not parent_measurements
                or any(
                    measurement.plan_sha256 != entry.plan_sha256
                    or measurement.profile != plan.profile
                    or not measurement.validation_passed
                    for measurement in child_measurements
                )
                or any(
                    measurement.plan_sha256 != parent.plan_sha256
                    or measurement.profile != plan.profile
                    or not measurement.validation_passed
                    for measurement in parent_measurements
                )
                or max(measurement.objective_loss for measurement in child_measurements)
                != entry.measured_loss
                or max(measurement.measured_bpw for measurement in child_measurements)
                != entry.measured_bpw
                or max(measurement.objective_loss for measurement in parent_measurements)
                != parent.measured_loss
                or max(measurement.measured_bpw for measurement in parent_measurements)
                != parent.measured_bpw
            ):
                continue
            verified_improvements.append(entry.candidate_id)
        if not verified_improvements:
            m6_issues.append(
                "no measurement-bound interaction refinement improves its parent candidate"
            )
        selected_measurements = [
            measurement
            for measurement in measurements.measurements
            if measurement.candidate_id == refinement.selected_candidate_id
        ]
        selected_history = history.get(refinement.selected_candidate_id)
        if (
            not selected_measurements
            or selected_history is None
            or any(
                measurement.plan_sha256 != refinement.selected_plan_sha256
                or not same_model_identity(measurement.candidate_model, candidate_model)
                or measurement.profile != plan.profile
                or not measurement.validation_passed
                for measurement in selected_measurements
            )
            or max(measurement.measured_bpw for measurement in selected_measurements)
            != selected_history.measured_bpw
            or max(measurement.objective_loss for measurement in selected_measurements)
            != selected_history.measured_loss
        ):
            m6_issues.append(
                "selected refinement result does not match its complete-model measurement"
            )
    checks.append(
        ReleaseAuditCheck(
            gate_id="M6",
            name="Measured interaction optimization",
            passed=not m6_issues,
            evidence_sha256={"refinement_result": file_sha256(paths["refinement"])},
            issues=m6_issues,
        )
    )

    m7_issues: list[str] = []
    if not hardware.release_ready:
        m7_issues.append("hardware registry is not release-ready")
    if not pareto.frontier_candidate_ids:
        m7_issues.append("Pareto report has no validated frontier")
    if pareto.measurement_set_sha256 != hardware.measurement_set_sha256:
        m7_issues.append("hardware registry and Pareto report use different measurements")
    m7_issues.extend(measurement_issues)
    m7_issues.extend(_hardware_evidence_issues(paths["hardware"], hardware))
    if measurements is not None:
        rebuilt_pareto = build_pareto_report(measurements)
        if rebuilt_pareto.model_dump(
            mode="json",
            exclude={"created_at"},
        ) != pareto.model_dump(mode="json", exclude={"created_at"}):
            m7_issues.append("Pareto report cannot be rebuilt from the bound measurements")
    matching_hardware = [
        entry
        for entry in hardware.entries
        if entry.candidate_id == refinement.selected_candidate_id
        and same_model_identity(entry.candidate_model, candidate_model)
        and entry.profile == plan.profile
        and entry.plan_sha256 == manifest.plan_sha256
        and entry.release_ready
    ]
    matching_pareto = [
        point
        for point in pareto.points
        if point.candidate_id == refinement.selected_candidate_id
        and point.measurement_id in {entry.measurement_id for entry in matching_hardware}
        and same_model_identity(point.candidate_model, candidate_model)
        and point.plan_sha256 == manifest.plan_sha256
        and point.frontier
        and point.candidate_id in pareto.frontier_candidate_ids
    ]
    if not matching_hardware:
        m7_issues.append("hardware registry does not certify the release candidate")
    if not matching_pareto:
        m7_issues.append("release candidate is not on the measured Pareto frontier")
    checks.append(
        ReleaseAuditCheck(
            gate_id="M7",
            name="Hardware-aware Pareto release candidate",
            passed=not m7_issues,
            evidence_sha256={
                "hardware_registry": file_sha256(paths["hardware"]),
                "pareto_report": file_sha256(paths["pareto"]),
            },
            issues=m7_issues,
        )
    )

    toolkit_version, m8_issues = _wheel_identity(paths["wheel"])
    if toolkit_version != request.required_toolkit_version:
        m8_issues.append(
            f"toolkit version {toolkit_version!r} is not {request.required_toolkit_version!r}"
        )
    version_claims = {
        "artifact manifest": manifest.axquant_version,
        "artifact software provenance": manifest.software_versions.axquant,
        "quantization plan software provenance": plan.software_versions.axquant,
        "reproduction recipe": recipe.axquant_version,
        "reproduction software provenance": recipe.software_versions.axquant,
    }
    for label, claimed_version in version_claims.items():
        if claimed_version != toolkit_version:
            m8_issues.append(
                f"{label} AXQuant version {claimed_version!r} differs from toolkit "
                f"{toolkit_version!r}"
            )
    expected_wheel_name = f"axquant-{request.required_toolkit_version}-py3-none-any.whl"
    if paths["wheel"].name != expected_wheel_name:
        m8_issues.append(
            f"toolkit wheel filename {paths['wheel'].name!r} is not {expected_wheel_name!r}"
        )
    if not reproduction.passed:
        m8_issues.append("reproduction verification did not pass")
    if reproduction.recipe_sha256 != stable_sha256(recipe):
        m8_issues.append("reproduction verification does not bind the supplied recipe")
    rerun_reproduction = verify_reproduction(
        recipe_path=paths["recipe"],
        artifact_dir=reproduction.artifact_path,
    )
    if rerun_reproduction != reproduction:
        m8_issues.append("reproduction verification cannot be reproduced from current files")
    if recipe.output_repository != candidate_model.model_id:
        m8_issues.append("reproduction recipe targets another candidate repository")
    if recipe.plan_sha256 != manifest.plan_sha256:
        m8_issues.append("reproduction recipe does not bind the release plan")
    if not same_model_identity(recipe.source_model, plan.source_model):
        m8_issues.append("reproduction recipe source differs from the release plan")
    if (
        recipe.calibration != plan.calibration
        or recipe.profile != plan.profile
        or recipe.random_seed != plan.random_seed
        or recipe.primary_runtime != plan.primary_runtime
    ):
        m8_issues.append("reproduction recipe execution provenance differs from the release plan")
    m8_issues.extend(
        _packaged_release_issues(
            artifact=artifact,
            plan=plan,
            recipe=recipe,
            validation_index=validation_index,
            validation_evidence=validation_evidence,
            hardware=hardware,
            measurements=measurements,
            pareto=pareto,
        )
    )
    checks.append(
        ReleaseAuditCheck(
            gate_id="M8",
            name="AXQuant v1.0 release",
            passed=not m8_issues,
            evidence_sha256={
                "toolkit_wheel": file_sha256(paths["wheel"]),
                "reproduction_recipe": file_sha256(paths["recipe"]),
                "reproduction_verification": file_sha256(paths["reproduction"]),
            },
            issues=m8_issues,
        )
    )

    blockers = [f"{check.gate_id}: {issue}" for check in checks for issue in check.issues]
    return ReleaseAudit(
        request_sha256=file_sha256(request_source),
        candidate_model=candidate_model,
        source_model=plan.source_model,
        toolkit_version=toolkit_version,
        wheel_sha256=file_sha256(paths["wheel"]),
        checks=checks,
        release_ready=all(check.passed for check in checks),
        blockers=blockers,
    )
