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
from axquant.schema import (
    BenchmarkEvidenceEntry,
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    EvaluationBundle,
)
from axquant.serde import file_sha256, load_model

_EXTERNAL_KINDS = {
    BenchmarkEvidenceKind.MIXED_PRECISION,
    BenchmarkEvidenceKind.AWQ,
    BenchmarkEvidenceKind.DWQ,
    BenchmarkEvidenceKind.GPTQ,
}


def _resolved_bundles(
    index_source: Path,
    index: BenchmarkEvidenceIndex,
) -> list[tuple[BenchmarkEvidenceEntry, EvaluationBundle]]:
    bundles: list[tuple[BenchmarkEvidenceEntry, EvaluationBundle]] = []
    for entry in index.entries:
        if entry.status != "available":
            continue
        assert entry.evaluation_file is not None
        assert entry.evaluation_sha256 is not None
        path = Path(entry.evaluation_file)
        resolved = path if path.is_absolute() else (index_source.parent / path).resolve()
        if not resolved.is_file():
            raise ArtifactError(f"evaluation bundle does not exist: {resolved}")
        if file_sha256(resolved) != entry.evaluation_sha256:
            raise ArtifactError(
                f"evaluation bundle checksum mismatch for {entry.kind.value}: {resolved}"
            )
        bundles.append((entry, load_model(resolved, EvaluationBundle)))
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
    return f"{sum(scores.values()) / len(scores):.4f} ({len(scores)} tasks)"


def render_head_to_head(
    index_path: str | Path,
    *,
    title: str | None = None,
) -> str:
    """Render the AXQ-022 comparison page from a bound benchmark evidence index."""
    index_source = Path(index_path).expanduser().resolve()
    index = load_model(index_source, BenchmarkEvidenceIndex)
    bundles = _resolved_bundles(index_source, index)
    if not bundles:
        raise ArtifactError("the benchmark evidence index has no available entries to render")
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
