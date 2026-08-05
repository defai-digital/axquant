from __future__ import annotations

import re
from pathlib import Path

from axquant.errors import ArtifactError
from axquant.identity import model_identity_key, same_model_identity
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    EvaluationBundle,
    ReleaseValidationEntry,
    ReleaseValidationIndex,
    ReleaseValidationRequest,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model

_REQUIRED_BENCHMARK_KINDS = {
    BenchmarkEvidenceKind.BF16,
    BenchmarkEvidenceKind.UNIFORM_4BIT,
    BenchmarkEvidenceKind.UNIFORM_6BIT,
    BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
    BenchmarkEvidenceKind.AXQUANT_MTP_ON,
}


def _resolved(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def build_release_validation_index(
    request_path: str | Path,
) -> ReleaseValidationIndex:
    request_source = Path(request_path).expanduser().resolve()
    request = load_model(request_source, ReleaseValidationRequest)
    entries: list[ReleaseValidationEntry] = []
    issues: list[str] = []

    for requested in sorted(request.entries, key=lambda entry: entry.profile.value):
        validation_path = _resolved(request_source.parent, requested.validation_file)
        benchmark_path = _resolved(request_source.parent, requested.benchmark_index_file)
        validation = load_model(validation_path, ValidationReport)
        benchmark = load_model(benchmark_path, BenchmarkEvidenceIndex)
        if validation.profile != requested.profile:
            issues.append(f"{requested.profile.value} validation declares a different profile")
        if benchmark.profile != requested.profile:
            issues.append(f"{requested.profile.value} benchmark index declares a different profile")
        if not validation.passed:
            issues.append(f"{requested.profile.value} validation did not pass")
        if not benchmark.release_ready:
            issues.append(f"{requested.profile.value} benchmark index is not release-ready")
        if benchmark.dataset_sha256 is None or not re.fullmatch(
            r"[0-9a-f]{64}", benchmark.dataset_sha256
        ):
            issues.append(f"{requested.profile.value} benchmark dataset digest is invalid")
        if benchmark.random_seed is None:
            issues.append(f"{requested.profile.value} benchmark random seed is missing")
        entry_by_kind = {entry.kind: entry for entry in benchmark.entries}
        for kind in sorted(_REQUIRED_BENCHMARK_KINDS, key=lambda item: item.value):
            if entry_by_kind[kind].status != "available":
                issues.append(
                    f"{requested.profile.value} required benchmark evidence is unavailable: "
                    f"{kind.value}"
                )

        for evidence in benchmark.entries:
            if evidence.status != "available":
                continue
            if evidence.evaluation_file is None or evidence.evaluation_sha256 is None:
                issues.append(
                    f"{requested.profile.value}/{evidence.kind.value} "
                    "available benchmark evidence is incomplete"
                )
                continue
            evaluation_path = _resolved(
                benchmark_path.parent,
                evidence.evaluation_file,
            )
            label = f"{requested.profile.value}/{evidence.kind.value}"
            if not evaluation_path.is_file():
                issues.append(f"{label} benchmark evaluation file is missing")
                continue
            if file_sha256(evaluation_path) != evidence.evaluation_sha256:
                issues.append(f"{label} benchmark evaluation checksum changed")
                continue
            try:
                evaluation = load_model(evaluation_path, EvaluationBundle)
            except (ArtifactError, OSError, ValueError) as exc:
                issues.append(f"{label} benchmark evaluation is invalid: {exc}")
                continue
            if (
                evidence.model is None
                or not same_model_identity(evaluation.model, evidence.model)
                or evaluation.runtime != evidence.runtime
                or evaluation.mtp_enabled != evidence.mtp_enabled
                or evaluation.baseline_kind != evidence.kind.value
            ):
                issues.append(f"{label} benchmark index fields differ from the evaluation")
            if evaluation.dataset_sha256 != benchmark.dataset_sha256:
                issues.append(f"{label} benchmark dataset digest differs from the index")
            if evaluation.random_seed != benchmark.random_seed:
                issues.append(f"{label} benchmark random seed differs from the index")
            if evaluation.workload != requested.profile.value:
                issues.append(f"{label} benchmark workload differs from the profile")

        reference = entry_by_kind[BenchmarkEvidenceKind.UNIFORM_6BIT].model
        direct = entry_by_kind[BenchmarkEvidenceKind.AXQUANT_MTP_OFF].model
        mtp = entry_by_kind[BenchmarkEvidenceKind.AXQUANT_MTP_ON].model
        if reference is None or not same_model_identity(reference, validation.reference_model):
            issues.append(f"{requested.profile.value} uniform-6 evidence model differs")
        if direct is None or mtp is None or not same_model_identity(direct, mtp):
            issues.append(f"{requested.profile.value} AXQuant MTP pair is not identical")
        elif not same_model_identity(direct, validation.candidate_model):
            issues.append(f"{requested.profile.value} candidate evidence model differs")
        if not is_immutable_revision(validation.candidate_model.revision):
            issues.append(f"{requested.profile.value} candidate revision is not immutable")
        if not is_immutable_revision(validation.reference_model.revision):
            issues.append(f"{requested.profile.value} reference revision is not immutable")

        entries.append(
            ReleaseValidationEntry(
                profile=requested.profile,
                validation_file=str(validation_path),
                validation_sha256=file_sha256(validation_path),
                benchmark_index_file=str(benchmark_path),
                benchmark_index_sha256=file_sha256(benchmark_path),
                reference_model=validation.reference_model,
                candidate_model=validation.candidate_model,
                dataset_sha256=benchmark.dataset_sha256 or "",
                passed=validation.passed and benchmark.release_ready,
            )
        )

    candidate_models = {model_identity_key(entry.candidate_model) for entry in entries}
    reference_models = {model_identity_key(entry.reference_model) for entry in entries}
    dataset_digests = {entry.dataset_sha256 for entry in entries if entry.dataset_sha256}
    if len(candidate_models) != 1:
        issues.append("required profiles validate different candidate models")
    if len(reference_models) != 1:
        issues.append("required profiles use different uniform-6 reference models")
    if len(dataset_digests) != len(entries):
        issues.append("required profiles must use distinct benchmark datasets")

    return ReleaseValidationIndex(
        entries=entries,
        release_ready=not issues,
        issues=issues,
    )
