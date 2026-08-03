from __future__ import annotations

from pathlib import Path

import pytest

from axquant.benchmark_evidence import build_benchmark_evidence_index
from axquant.cli import main
from axquant.schema import (
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceInput,
    BenchmarkEvidenceKind,
    BenchmarkEvidenceRequest,
    EvaluationBundle,
    HardwareMetrics,
    IntegrityMetrics,
    ModelIdentity,
    ProfileName,
    SoftwareVersions,
)
from axquant.serde import load_model, write_data


def _versions() -> SoftwareVersions:
    return SoftwareVersions(
        axquant="0.1.0a0",
        python="3.13",
        mlx="0.32",
        mlx_lm="0.31",
        ax_engine="6.11.1",
        safetensors="0.6",
        pydantic="2.11",
    )


def _bundle(kind: BenchmarkEvidenceKind, *, dataset_sha256: str = "a" * 64) -> EvaluationBundle:
    candidate = kind in {
        BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
        BenchmarkEvidenceKind.AXQUANT_MTP_ON,
    }
    return EvaluationBundle(
        model=ModelIdentity(
            model_id="AutomatosX/candidate" if candidate else f"AutomatosX/{kind.value}",
            revision="candidate-revision" if candidate else f"{kind.value}-revision",
            local_path="/models/candidate" if candidate else None,
        ),
        mtp_enabled=kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON,
        baseline_kind=kind.value,
        hardware=HardwareMetrics(
            kernel_fallbacks=0,
            device_name="Mac15,9",
            chip="Apple M3 Max",
            unified_memory_bytes=128 * 1024**3,
            os_version="macOS-test",
        ),
        integrity=IntegrityMetrics(
            safetensors_valid=True,
            index_complete=True,
            config_valid=True,
            mtp_layout_valid=True,
            source_revision_pinned=True,
        ),
        workload=ProfileName.AGENT_CODING.value,
        dataset_sha256=dataset_sha256,
        software_versions=_versions(),
        random_seed=7,
        benchmark_metadata={
            "prompt_count": 5,
            "warmup_trials": 2,
            "measured_trials": 5,
            "successful_measured_trials": 5,
            "failed_trials": 0,
            "timed_out_trials": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 512,
            "power_mode": "AC power",
            "quantizer": kind.value,
            "quantizer_version": "fixture-v1",
            "quality_dataset_sha256": "q" * 64,
            "ax_engine_version": "6.11.1",
        },
    )


def _request(tmp_path: Path) -> BenchmarkEvidenceRequest:
    required = {
        BenchmarkEvidenceKind.BF16,
        BenchmarkEvidenceKind.UNIFORM_4BIT,
        BenchmarkEvidenceKind.UNIFORM_6BIT,
        BenchmarkEvidenceKind.AXQUANT_MTP_OFF,
        BenchmarkEvidenceKind.AXQUANT_MTP_ON,
    }
    entries: list[BenchmarkEvidenceInput] = []
    for kind in BenchmarkEvidenceKind:
        if kind in required:
            path = tmp_path / f"{kind.value}.json"
            write_data(path, _bundle(kind))
            entries.append(
                BenchmarkEvidenceInput(
                    kind=kind,
                    status="available",
                    evaluation_file=path.name,
                )
            )
        else:
            entries.append(
                BenchmarkEvidenceInput(
                    kind=kind,
                    status="unavailable",
                    unavailable_reason=f"{kind.value} artifact is not available",
                )
            )
    return BenchmarkEvidenceRequest(
        profile=ProfileName.AGENT_CODING,
        entries=entries,
    )


def test_benchmark_evidence_index_is_complete_and_checksum_bound(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    write_data(request_path, _request(tmp_path))

    index = build_benchmark_evidence_index(request_path)

    assert index.release_ready
    assert len(index.entries) == len(BenchmarkEvidenceKind)
    assert index.dataset_sha256 == "a" * 64
    assert index.random_seed == 7
    available = [entry for entry in index.entries if entry.status == "available"]
    assert all(entry.evaluation_sha256 for entry in available)
    assert {entry.kind for entry in index.entries if entry.status == "unavailable"} == {
        BenchmarkEvidenceKind.MIXED_PRECISION,
        BenchmarkEvidenceKind.AWQ,
        BenchmarkEvidenceKind.DWQ,
        BenchmarkEvidenceKind.GPTQ,
    }


def test_benchmark_evidence_cli_returns_one_for_missing_required_baseline(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    request.entries = [
        (
            BenchmarkEvidenceInput(
                kind=BenchmarkEvidenceKind.BF16,
                status="unavailable",
                unavailable_reason="highest-precision source benchmark unavailable",
            )
            if entry.kind == BenchmarkEvidenceKind.BF16
            else entry
        )
        for entry in request.entries
    ]
    request_path = tmp_path / "request.json"
    output = tmp_path / "index.json"
    write_data(request_path, request)

    assert (
        main(
            [
                "benchmark-index",
                "--request",
                str(request_path),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    index = load_model(output, BenchmarkEvidenceIndex)
    assert not index.release_ready
    assert "required benchmark baseline is unavailable: bf16" in index.issues


def test_benchmark_evidence_rejects_mismatched_dataset(tmp_path: Path) -> None:
    request = _request(tmp_path)
    uniform4_path = tmp_path / f"{BenchmarkEvidenceKind.UNIFORM_4BIT.value}.json"
    write_data(
        uniform4_path,
        _bundle(BenchmarkEvidenceKind.UNIFORM_4BIT, dataset_sha256="b" * 64),
    )
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_benchmark_evidence_index(request_path)

    assert not index.release_ready
    assert "available benchmark baselines use different datasets" in index.issues


def test_benchmark_evidence_rejects_kernel_fallback(tmp_path: Path) -> None:
    request = _request(tmp_path)
    mtp_path = tmp_path / f"{BenchmarkEvidenceKind.AXQUANT_MTP_ON.value}.json"
    bundle = _bundle(BenchmarkEvidenceKind.AXQUANT_MTP_ON)
    bundle.hardware.kernel_fallbacks = 1
    write_data(mtp_path, bundle)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_benchmark_evidence_index(request_path)

    assert not index.release_ready
    assert "axquant-mtp-on benchmark used kernel fallbacks" in index.issues


def test_benchmark_request_cannot_silently_omit_optional_baseline(tmp_path: Path) -> None:
    request = _request(tmp_path)
    payload = request.model_dump(mode="json")
    payload["entries"] = [
        entry for entry in payload["entries"] if entry["kind"] != BenchmarkEvidenceKind.AWQ
    ]

    with pytest.raises(ValueError, match="explicitly list every baseline"):
        BenchmarkEvidenceRequest.model_validate(payload)
