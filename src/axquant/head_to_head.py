"""Head-to-head comparison page rendering (AXQ-022).

Renders the public comparison page for one base model from a bound benchmark
evidence index. Every number comes from a checksum-verified evaluation bundle;
the mandatory comparison set is always listed (unavailable entries appear with
their recorded reason, never silently dropped), and results are rendered with
equal prominence whether AXQuant wins or loses.
"""

from __future__ import annotations

from pathlib import Path

from axquant.errors import ArtifactError
from axquant.identity import same_model_identity
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    BenchmarkEvidenceEntry,
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    EvaluationBundle,
    RuntimeName,
)
from axquant.serde import file_sha256, load_model

_EXTERNAL_KINDS = {
    BenchmarkEvidenceKind.MIXED_PRECISION,
    BenchmarkEvidenceKind.AWQ,
    BenchmarkEvidenceKind.DWQ,
    BenchmarkEvidenceKind.GPTQ,
}
_REQUIRED_RELEASE_KINDS = {
    BenchmarkEvidenceKind.BF16,
    BenchmarkEvidenceKind.UNIFORM_4BIT,
    BenchmarkEvidenceKind.UNIFORM_6BIT,
    BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
    BenchmarkEvidenceKind.AXQUANT_MTP_ON,
}


def _resolved_bundles(
    index_source: Path,
    index: BenchmarkEvidenceIndex,
) -> list[tuple[BenchmarkEvidenceEntry, EvaluationBundle]]:
    bundles: list[tuple[BenchmarkEvidenceEntry, EvaluationBundle]] = []
    for entry in index.entries:
        if entry.status != "available":
            continue
        if entry.evaluation_file is None or entry.evaluation_sha256 is None:
            raise ArtifactError(
                f"available benchmark index entry is incomplete: {entry.kind.value}"
            )
        path = Path(entry.evaluation_file)
        candidate = path if path.is_absolute() else index_source.parent / path
        if candidate.is_symlink():
            raise ArtifactError(
                f"evaluation bundle must not be a symlink for {entry.kind.value}: {candidate}"
            )
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise ArtifactError(f"evaluation bundle does not exist: {resolved}")
        if file_sha256(resolved) != entry.evaluation_sha256:
            raise ArtifactError(
                f"evaluation bundle checksum mismatch for {entry.kind.value}: {resolved}"
            )
        bundle = load_model(resolved, EvaluationBundle)
        mismatches: list[str] = []
        if entry.model is None or not same_model_identity(bundle.model, entry.model):
            mismatches.append("model identity")
        if bundle.runtime != entry.runtime:
            mismatches.append("runtime")
        if bundle.mtp_enabled != entry.mtp_enabled:
            mismatches.append("MTP mode")
        if bundle.baseline_kind != entry.kind.value:
            mismatches.append("baseline kind")
        if bundle.workload != index.profile.value:
            mismatches.append("workload profile")
        if index.dataset_sha256 is not None and bundle.dataset_sha256 != index.dataset_sha256:
            mismatches.append("dataset checksum")
        if index.random_seed is not None and bundle.random_seed != index.random_seed:
            mismatches.append("random seed")
        if mismatches:
            raise ArtifactError(
                f"evaluation bundle bindings differ for {entry.kind.value}: {', '.join(mismatches)}"
            )
        bundles.append((entry, bundle))
    return bundles


def _format_optional(value: float | None, template: str = "{:.4f}") -> str:
    return template.format(value) if value is not None else "not measured"


def _format_memory(value: int | None) -> str:
    if value is None:
        return "not measured"
    return f"{value / 1024**3:.2f} GiB"


def _task_average(bundle: EvaluationBundle) -> str:
    scores = bundle.quality.task_scores
    if not scores:
        return "not measured"
    return f"{sum(scores.values()) / len(scores):.4f} ({len(scores)} reported scores)"


def _validate_release_comparison(
    entries: list[tuple[BenchmarkEvidenceEntry, EvaluationBundle]],
) -> None:
    by_kind = {entry.kind: bundle for entry, bundle in entries}
    missing = sorted(kind.value for kind in _REQUIRED_RELEASE_KINDS - set(by_kind))
    if missing:
        raise ArtifactError(f"release-ready benchmark evidence omits required entries: {missing}")
    if any(bundle.runtime != RuntimeName.AX_ENGINE for bundle in by_kind.values()):
        raise ArtifactError("release-ready head-to-head evidence must use AX Engine")
    if any(not is_immutable_revision(bundle.model.revision) for bundle in by_kind.values()):
        raise ArtifactError("release-ready head-to-head evidence requires immutable revisions")
    direct = by_kind[BenchmarkEvidenceKind.AXQUANT_MTP_OFF]
    mtp = by_kind[BenchmarkEvidenceKind.AXQUANT_MTP_ON]
    if not same_model_identity(direct.model, mtp.model):
        raise ArtifactError("release-ready MTP-off/on evidence uses different checkpoints")
    hardware = {
        (
            bundle.hardware.device_name,
            bundle.hardware.chip,
            bundle.hardware.unified_memory_bytes,
            bundle.hardware.os_version,
        )
        for bundle in by_kind.values()
    }
    if len(hardware) != 1:
        raise ArtifactError("release-ready head-to-head evidence uses different hardware")
    software = {
        (
            bundle.software_versions.axquant,
            bundle.software_versions.python,
            bundle.software_versions.mlx,
            bundle.software_versions.mlx_lm,
            bundle.software_versions.ax_engine,
            bundle.software_versions.safetensors,
            bundle.software_versions.pydantic,
        )
        for bundle in by_kind.values()
    }
    if len(software) != 1:
        raise ArtifactError("release-ready head-to-head evidence uses different software")


def render_head_to_head(
    index_path: str | Path,
    *,
    title: str | None = None,
) -> str:
    """Render the AXQ-022 comparison page from a bound benchmark evidence index."""
    index_input = Path(index_path).expanduser()
    if index_input.is_symlink():
        raise ArtifactError(f"benchmark evidence index must not be a symlink: {index_input}")
    index_source = index_input.resolve()
    index = load_model(index_source, BenchmarkEvidenceIndex)
    if index.release_ready and (index.dataset_sha256 is None or index.random_seed is None):
        raise ArtifactError("release-ready benchmark evidence is missing dataset or seed bindings")
    bundles = _resolved_bundles(index_source, index)
    if not bundles:
        raise ArtifactError("the benchmark evidence index has no available entries to render")
    if index.release_ready:
        _validate_release_comparison(bundles)
    hosts = sorted(
        {
            f"{bundle.hardware.device_name or 'unknown-host'} "
            f"({bundle.hardware.chip or 'unknown chip'})"
            for _, bundle in bundles
        }
    )
    lines: list[str] = []
    lines.append(f"# {title or 'AXQuant head-to-head comparison'}")
    lines.append("")
    lines.append(
        "Every value below is loaded from a checksum-verified evaluation bundle bound by the "
        "benchmark evidence index. Results are published with equal prominence whether AXQuant "
        "wins or loses (AXQ-022)."
    )
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Profile | `{index.profile.value}` |")
    lines.append(f"| Evaluation dataset SHA-256 | `{index.dataset_sha256 or 'unrecorded'}` |")
    seed = "unrecorded" if index.random_seed is None else str(index.random_seed)
    lines.append(f"| Random seed | {seed} |")
    lines.append(f"| Hosts | {'; '.join(hosts)} |")
    lines.append(f"| Release ready | {'yes' if index.release_ready else 'no'} |")
    lines.append("")
    lines.append("## Measured comparison")
    lines.append("")
    lines.append(
        "| Checkpoint | Model | Revision | Runtime | MTP | Perplexity | Task score "
        "| JSON valid | Decode tok/s | Peak memory |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry, bundle in bundles:
        external = " (external, attributed)" if entry.kind in _EXTERNAL_KINDS else ""
        lines.append(
            f"| {entry.kind.value}{external} "
            f"| `{bundle.model.model_id}` "
            f"| `{bundle.model.revision or 'unrecorded'}` "
            f"| {bundle.runtime.value} "
            f"| {'on' if bundle.mtp_enabled else 'off'} "
            f"| {_format_optional(bundle.quality.perplexity)} "
            f"| {_task_average(bundle)} "
            f"| {_format_optional(bundle.quality.json_valid_rate, '{:.2%}')} "
            f"| {_format_optional(bundle.hardware.decode_tokens_per_second, '{:.1f}')} "
            f"| {_format_memory(bundle.hardware.peak_memory_bytes)} |"
        )
    unavailable = [entry for entry in index.entries if entry.status == "unavailable"]
    if unavailable:
        lines.append("")
        lines.append("## Unavailable comparison entries")
        lines.append("")
        lines.append("| Checkpoint | Reason |")
        lines.append("| --- | --- |")
        for entry in unavailable:
            lines.append(f"| {entry.kind.value} | {entry.unavailable_reason} |")
    lines.append("")
    lines.append(
        "External baselines are attributed public checkpoints compared through their standard "
        "load contract; AXQuant reuses none of their code, data, or metadata (AXQ-001)."
    )
    lines.append("")
    return "\n".join(lines)
