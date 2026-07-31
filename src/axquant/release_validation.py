from __future__ import annotations

from pathlib import Path

from axquant.schema import (
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    ModelIdentity,
    ReleaseValidationEntry,
    ReleaseValidationIndex,
    ReleaseValidationRequest,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model


def _resolved(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _same_model(left: ModelIdentity, right: ModelIdentity) -> bool:
    return left == right


def _model_key(model: ModelIdentity) -> tuple[str, str | None, str, str | None, str | None]:
    return (
        model.model_id,
        model.revision,
        model.format,
        model.architecture,
        model.local_path,
    )


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
        if benchmark.dataset_sha256 is None:
            issues.append(f"{requested.profile.value} benchmark dataset digest is missing")

        benchmark_entries = {entry.kind: entry for entry in benchmark.entries}
        reference = benchmark_entries[BenchmarkEvidenceKind.UNIFORM_6BIT].model
        direct = benchmark_entries[BenchmarkEvidenceKind.AXQUANT_MTP_OFF].model
        mtp = benchmark_entries[BenchmarkEvidenceKind.AXQUANT_MTP_ON].model
        if reference is None or not _same_model(reference, validation.reference_model):
            issues.append(f"{requested.profile.value} uniform-6 evidence model differs")
        if direct is None or mtp is None or not _same_model(direct, mtp):
            issues.append(f"{requested.profile.value} AXQuant MTP pair is not identical")
        elif not _same_model(direct, validation.candidate_model):
            issues.append(f"{requested.profile.value} candidate evidence model differs")
        if not validation.candidate_model.revision:
            issues.append(f"{requested.profile.value} candidate revision is not immutable")

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

    candidate_models = {_model_key(entry.candidate_model) for entry in entries}
    reference_models = {_model_key(entry.reference_model) for entry in entries}
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
