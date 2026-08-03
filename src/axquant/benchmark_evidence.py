from __future__ import annotations

from pathlib import Path

from axquant.schema import (
    BenchmarkEvidenceEntry,
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    BenchmarkEvidenceRequest,
    EvaluationBundle,
    RuntimeName,
)
from axquant.serde import file_sha256, load_model, stable_sha256

_OPTIONAL_BASELINES = {
    BenchmarkEvidenceKind.MIXED_PRECISION,
    BenchmarkEvidenceKind.AWQ,
    BenchmarkEvidenceKind.DWQ,
    BenchmarkEvidenceKind.GPTQ,
}

_REQUIRED_METADATA = (
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

_CONTROL_METADATA = (
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


def build_benchmark_evidence_index(
    request_path: str | Path,
) -> BenchmarkEvidenceIndex:
    request_source = Path(request_path).expanduser().resolve()
    request = load_model(request_source, BenchmarkEvidenceRequest)
    inputs = {entry.kind: entry for entry in request.entries}
    entries: list[BenchmarkEvidenceEntry] = []
    issues: list[str] = []
    bundles: dict[BenchmarkEvidenceKind, EvaluationBundle] = {}

    for kind in BenchmarkEvidenceKind:
        evidence = inputs[kind]
        if evidence.status == "unavailable":
            entries.append(
                BenchmarkEvidenceEntry(
                    kind=kind,
                    status="unavailable",
                    unavailable_reason=evidence.unavailable_reason,
                )
            )
            if kind not in _OPTIONAL_BASELINES:
                issues.append(f"required benchmark baseline is unavailable: {kind.value}")
            continue

        assert evidence.evaluation_file is not None
        evaluation_path = _resolved(request_source.parent, evidence.evaluation_file)
        bundle = load_model(evaluation_path, EvaluationBundle)
        bundles[kind] = bundle
        if bundle.baseline_kind != kind.value:
            issues.append(f"{kind.value} evidence declares baseline kind {bundle.baseline_kind}")
        if bundle.workload != request.profile.value:
            issues.append(f"{kind.value} workload does not match profile {request.profile.value}")
        if bundle.runtime != RuntimeName.AX_ENGINE:
            issues.append(f"{kind.value} release evidence must use AX Engine")
        if not bundle.model.revision:
            issues.append(f"{kind.value} model revision is not immutable")
        integrity = bundle.integrity
        if not (
            integrity.safetensors_valid
            and integrity.index_complete
            and integrity.config_valid
            and integrity.source_revision_pinned
        ):
            issues.append(f"{kind.value} checkpoint integrity evidence is incomplete")
        versions = bundle.software_versions
        for version_name in (
            "axquant",
            "python",
            "mlx",
            "mlx_lm",
            "ax_engine",
            "safetensors",
            "pydantic",
        ):
            if not getattr(versions, version_name):
                issues.append(f"{kind.value} software version is missing: {version_name}")
        hardware = bundle.hardware
        required_hardware = {
            "device_name": hardware.device_name,
            "chip": hardware.chip,
            "unified_memory_bytes": hardware.unified_memory_bytes,
            "os_version": hardware.os_version,
        }
        for hardware_name, value in required_hardware.items():
            if value in {None, ""}:
                issues.append(f"{kind.value} hardware identity is missing: {hardware_name}")
        if hardware.unified_memory_bytes is not None and hardware.unified_memory_bytes <= 0:
            issues.append(f"{kind.value} unified memory must be positive")
        if hardware.kernel_fallbacks is None:
            issues.append(f"{kind.value} kernel fallback count is missing")
        elif hardware.kernel_fallbacks != 0:
            issues.append(f"{kind.value} benchmark used kernel fallbacks")
        metadata = bundle.benchmark_metadata
        missing_metadata = [
            field_name
            for field_name in _REQUIRED_METADATA
            if metadata.get(field_name) is None or metadata.get(field_name) == ""
        ]
        if missing_metadata:
            issues.append(f"{kind.value} benchmark metadata is missing: {missing_metadata}")
        measured_trials = metadata.get("measured_trials")
        if metadata.get("successful_measured_trials") != measured_trials:
            issues.append(f"{kind.value} benchmark did not complete every measured trial")
        if metadata.get("failed_trials") != 0 or metadata.get("timed_out_trials") != 0:
            issues.append(f"{kind.value} benchmark contains failed or timed-out trials")
        metadata_ax_engine = metadata.get("ax_engine_version")
        if metadata_ax_engine is not None and metadata_ax_engine != versions.ax_engine:
            issues.append(f"{kind.value} AX Engine metadata and software versions differ")
        if kind == BenchmarkEvidenceKind.AXQUANT_MTP_OFF and bundle.mtp_enabled:
            issues.append("axquant-mtp-off evidence has MTP enabled")
        if kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON and not bundle.mtp_enabled:
            issues.append("axquant-mtp-on evidence has MTP disabled")
        if (
            kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON
            and bundle.integrity.mtp_layout_valid is not True
        ):
            issues.append("axquant-mtp-on evidence has no valid MTP layout")
        entries.append(
            BenchmarkEvidenceEntry(
                kind=kind,
                status="available",
                evaluation_file=str(evaluation_path),
                evaluation_sha256=file_sha256(evaluation_path),
                model=bundle.model,
                runtime=bundle.runtime,
                mtp_enabled=bundle.mtp_enabled,
            )
        )

    dataset_digests = {bundle.dataset_sha256 for bundle in bundles.values()}
    random_seeds = {bundle.random_seed for bundle in bundles.values()}
    if len(dataset_digests) > 1:
        issues.append("available benchmark baselines use different datasets")
    if len(random_seeds) > 1:
        issues.append("available benchmark baselines use different random seeds")
    software_environments = {
        (
            bundle.software_versions.axquant,
            bundle.software_versions.python,
            bundle.software_versions.mlx,
            bundle.software_versions.mlx_lm,
            bundle.software_versions.ax_engine,
            bundle.software_versions.safetensors,
            bundle.software_versions.pydantic,
        )
        for bundle in bundles.values()
    }
    if len(software_environments) > 1:
        issues.append("available benchmark baselines use different software versions")
    hardware_environments = {
        (
            bundle.hardware.device_name,
            bundle.hardware.chip,
            bundle.hardware.unified_memory_bytes,
            bundle.hardware.os_version,
        )
        for bundle in bundles.values()
    }
    if len(hardware_environments) > 1:
        issues.append("available benchmark baselines use different hardware")
    controls = {
        stable_sha256(
            {
                field_name: bundle.benchmark_metadata.get(field_name)
                for field_name in _CONTROL_METADATA
            }
        )
        for bundle in bundles.values()
    }
    if len(controls) > 1:
        issues.append("available benchmark baselines use different benchmark controls")

    direct = bundles.get(BenchmarkEvidenceKind.AXQUANT_MTP_OFF)
    mtp = bundles.get(BenchmarkEvidenceKind.AXQUANT_MTP_ON)
    if direct is not None and mtp is not None and direct.model != mtp.model:
        issues.append("AXQuant MTP-off/on evidence does not use the identical checkpoint")

    return BenchmarkEvidenceIndex(
        profile=request.profile,
        dataset_sha256=next(iter(dataset_digests)) if len(dataset_digests) == 1 else None,
        random_seed=next(iter(random_seeds)) if len(random_seeds) == 1 else None,
        entries=entries,
        release_ready=not issues,
        issues=issues,
    )
