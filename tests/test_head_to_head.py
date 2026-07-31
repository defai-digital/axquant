from __future__ import annotations

from pathlib import Path

import pytest

from axquant.cli import main
from axquant.errors import ArtifactError
from axquant.head_to_head import render_head_to_head
from axquant.schema import (
    BenchmarkEvidenceEntry,
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceKind,
    EvaluationBundle,
    HardwareMetrics,
    IntegrityMetrics,
    ModelIdentity,
    ProfileName,
    QualityMetrics,
    SoftwareVersions,
)
from axquant.serde import file_sha256, write_data


def _versions() -> SoftwareVersions:
    return SoftwareVersions(
        axquant="1.0.0",
        python="3.13",
        mlx="0.32",
        mlx_lm="0.31",
        ax_engine="6.12.1",
        safetensors="0.6",
        pydantic="2.11",
    )


def _bundle(kind: BenchmarkEvidenceKind, *, perplexity: float) -> EvaluationBundle:
    candidate = kind in {
        BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
        BenchmarkEvidenceKind.AXQUANT_MTP_ON,
    }
    return EvaluationBundle(
        model=ModelIdentity(
            model_id="AutomatosX/candidate" if candidate else f"external/{kind.value}",
            revision=f"{kind.value}-revision",
        ),
        mtp_enabled=kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON,
        baseline_kind=kind.value,
        quality=QualityMetrics(
            perplexity=perplexity,
            task_scores={"coding": 0.8, "json": 0.9},
            json_valid_rate=0.98,
        ),
        hardware=HardwareMetrics(
            peak_memory_bytes=20 * 1024**3,
            decode_tokens_per_second=42.5,
            device_name="Mac15,9",
            chip="Apple M3 Max",
        ),
        integrity=IntegrityMetrics(
            safetensors_valid=True,
            index_complete=True,
            config_valid=True,
            source_revision_pinned=True,
        ),
        workload=ProfileName.AGENT_CODING.value,
        dataset_sha256="d" * 64,
        software_versions=_versions(),
        random_seed=7,
    )


def _index(tmp_path: Path) -> Path:
    available = {
        BenchmarkEvidenceKind.BF16: 5.0,
        BenchmarkEvidenceKind.UNIFORM_4BIT: 5.4,
        BenchmarkEvidenceKind.UNIFORM_6BIT: 5.1,
        BenchmarkEvidenceKind.MIXED_PRECISION: 5.2,
        BenchmarkEvidenceKind.AXQUANT_MTP_OFF: 5.3,
        BenchmarkEvidenceKind.AXQUANT_MTP_ON: 5.3,
    }
    entries: list[BenchmarkEvidenceEntry] = []
    for kind, perplexity in available.items():
        bundle = _bundle(kind, perplexity=perplexity)
        bundle_path = tmp_path / f"{kind.value}.json"
        write_data(bundle_path, bundle)
        entries.append(
            BenchmarkEvidenceEntry(
                kind=kind,
                status="available",
                evaluation_file=bundle_path.name,
                evaluation_sha256=file_sha256(bundle_path),
                model=bundle.model,
                runtime=bundle.runtime,
                mtp_enabled=bundle.mtp_enabled,
            )
        )
    for kind in (BenchmarkEvidenceKind.AWQ, BenchmarkEvidenceKind.DWQ):
        entries.append(
            BenchmarkEvidenceEntry(
                kind=kind,
                status="unavailable",
                unavailable_reason=f"no public {kind.value} checkpoint for this base model",
            )
        )
    index = BenchmarkEvidenceIndex(
        profile=ProfileName.AGENT_CODING,
        dataset_sha256="d" * 64,
        random_seed=7,
        entries=entries,
        release_ready=True,
    )
    index_path = tmp_path / "benchmark-evidence-index.json"
    write_data(index_path, index)
    return index_path


def test_head_to_head_renders_all_entries_with_equal_prominence(tmp_path: Path) -> None:
    page = render_head_to_head(_index(tmp_path), title="AX-Qwen3.6-27B head-to-head")
    assert page.startswith("# AX-Qwen3.6-27B head-to-head")
    # AXQuant's worse-than-uniform-6 perplexity is rendered plainly.
    assert "5.3000" in page and "5.1000" in page
    assert "mixed-precision (external, attributed)" in page
    assert "`external/mixed-precision`" in page
    assert "`AutomatosX/candidate`" in page
    assert "Mac15,9 (Apple M3 Max)" in page
    # Unavailable mandatory entries are listed with their reasons, never dropped.
    assert "no public awq checkpoint for this base model" in page
    assert "no public dwq checkpoint for this base model" in page
    assert "AXQ-001" in page
    for kind in BenchmarkEvidenceKind:
        assert kind.value in page


def test_head_to_head_rejects_checksum_mismatch(tmp_path: Path) -> None:
    index_path = _index(tmp_path)
    bundle_path = tmp_path / "bf16.json"
    bundle_path.write_text(bundle_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        render_head_to_head(index_path)


def test_head_to_head_cli_writes_page(tmp_path: Path) -> None:
    index_path = _index(tmp_path)
    output = tmp_path / "page.md"
    assert (
        main(
            [
                "head-to-head",
                "--benchmark-index",
                str(index_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_text(encoding="utf-8").startswith("# AXQuant head-to-head comparison")
