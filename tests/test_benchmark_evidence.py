from __future__ import annotations

from pathlib import Path

import pytest

from axquant.benchmark_evidence import build_benchmark_evidence_index, formal_mtp_bundle_issues
from axquant.cli import main
from axquant.schema import (
    BenchmarkEvidenceIndex,
    BenchmarkEvidenceInput,
    BenchmarkEvidenceKind,
    BenchmarkEvidenceRequest,
    EvaluationBundle,
    FormalHostContract,
    HardwareMetrics,
    IntegrityMetrics,
    ModelIdentity,
    MtpMetrics,
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
            revision="c" * 40 if candidate else "b" * 40,
            local_path="/models/candidate" if candidate else None,
        ),
        mtp_enabled=kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON,
        baseline_kind=kind.value,
        mtp=(
            MtpMetrics(
                token_accuracy={"1": 0.8},
                average_accepted_tokens=0.8,
                acceptance_rate=0.8,
                rejection_rate=0.2,
                effective_tokens_per_forward=1.8,
                repetition_rate=0.01,
                divergence_rate=0.0,
            )
            if kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON
            else None
        ),
        hardware=HardwareMetrics(
            kernel_fallbacks=0,
            device_name="Mac15,9",
            chip="Apple M3 Max",
            unified_memory_bytes=128 * 1024**3,
            os_version="macOS-test",
            mtp_effective_tokens_per_second=12.0,
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
            "draft_depth": 1,
            "power_mode": "AC power",
            "quantizer": kind.value,
            "quantizer_version": "fixture-v1",
            "runtime_env": {},
            "quality_dataset_sha256": "q" * 64,
            "ax_engine_version": "6.11.1",
            "mtp_metrics_protocol": (
                "adjacent-token-repeat-v1;depth1-proposal-accuracy-v1;"
                "greedy-output-ab-divergence-v1"
                if kind == BenchmarkEvidenceKind.AXQUANT_MTP_ON
                else None
            ),
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


def test_benchmark_evidence_rejects_mutable_revision_alias(tmp_path: Path) -> None:
    request = _request(tmp_path)
    bf16 = next(entry for entry in request.entries if entry.kind == BenchmarkEvidenceKind.BF16)
    assert bf16.evaluation_file is not None
    bundle = load_model(tmp_path / bf16.evaluation_file, EvaluationBundle)
    bundle.model.revision = "main"
    write_data(tmp_path / bf16.evaluation_file, bundle)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_benchmark_evidence_index(request_path)

    assert not index.release_ready
    assert "bf16 model revision is not immutable" in index.issues


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


def test_benchmark_evidence_rejects_missing_kernel_telemetry(tmp_path: Path) -> None:
    request = _request(tmp_path)
    mtp_path = tmp_path / f"{BenchmarkEvidenceKind.AXQUANT_MTP_ON.value}.json"
    bundle = _bundle(BenchmarkEvidenceKind.AXQUANT_MTP_ON)
    bundle.hardware.kernel_fallbacks = None
    write_data(mtp_path, bundle)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_benchmark_evidence_index(request_path)

    assert not index.release_ready
    assert "axquant-mtp-on kernel fallback count is missing" in index.issues


def test_benchmark_evidence_requires_complete_mtp_metrics(tmp_path: Path) -> None:
    request = _request(tmp_path)
    mtp_path = tmp_path / f"{BenchmarkEvidenceKind.AXQUANT_MTP_ON.value}.json"
    bundle = _bundle(BenchmarkEvidenceKind.AXQUANT_MTP_ON)
    bundle.mtp = None
    write_data(mtp_path, bundle)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_benchmark_evidence_index(request_path)

    assert not index.release_ready
    assert "axquant-mtp-on evidence is missing MTP metrics" in index.issues


def test_benchmark_evidence_rejects_empty_mtp_metrics(tmp_path: Path) -> None:
    request = _request(tmp_path)
    mtp_path = tmp_path / f"{BenchmarkEvidenceKind.AXQUANT_MTP_ON.value}.json"
    bundle = _bundle(BenchmarkEvidenceKind.AXQUANT_MTP_ON)
    bundle.mtp = MtpMetrics()
    write_data(mtp_path, bundle)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_benchmark_evidence_index(request_path)

    assert not index.release_ready
    assert any(
        issue.startswith("axquant-mtp-on evidence has incomplete MTP metrics")
        for issue in index.issues
    )


def test_benchmark_evidence_requires_effective_mtp_throughput(tmp_path: Path) -> None:
    request = _request(tmp_path)
    mtp_path = tmp_path / f"{BenchmarkEvidenceKind.AXQUANT_MTP_ON.value}.json"
    bundle = _bundle(BenchmarkEvidenceKind.AXQUANT_MTP_ON)
    bundle.hardware.mtp_effective_tokens_per_second = None
    write_data(mtp_path, bundle)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_benchmark_evidence_index(request_path)

    assert not index.release_ready
    assert "axquant-mtp-on evidence is missing effective MTP throughput" in index.issues


@pytest.mark.parametrize(
    ("field_name", "different_value"),
    [
        ("draft_depth", 2),
        (
            "runtime_env",
            {"AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL": "0"},
        ),
        ("ax_engine_version", "6.12.0"),
    ],
)
def test_benchmark_evidence_binds_execution_controls(
    tmp_path: Path,
    field_name: str,
    different_value: object,
) -> None:
    request = _request(tmp_path)
    uniform_path = tmp_path / f"{BenchmarkEvidenceKind.UNIFORM_4BIT.value}.json"
    bundle = _bundle(BenchmarkEvidenceKind.UNIFORM_4BIT)
    bundle.benchmark_metadata[field_name] = different_value  # type: ignore[assignment]
    write_data(uniform_path, bundle)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_benchmark_evidence_index(request_path)

    assert not index.release_ready
    assert "available benchmark baselines use different benchmark controls" in index.issues


@pytest.mark.parametrize("field_name", ["draft_depth", "runtime_env", "ax_engine_version"])
def test_benchmark_evidence_requires_bound_execution_controls(
    tmp_path: Path,
    field_name: str,
) -> None:
    request = _request(tmp_path)
    uniform_path = tmp_path / f"{BenchmarkEvidenceKind.UNIFORM_4BIT.value}.json"
    bundle = _bundle(BenchmarkEvidenceKind.UNIFORM_4BIT)
    del bundle.benchmark_metadata[field_name]
    write_data(uniform_path, bundle)
    request_path = tmp_path / "request.json"
    write_data(request_path, request)

    index = build_benchmark_evidence_index(request_path)

    assert not index.release_ready
    assert any(
        issue.startswith("uniform-4bit benchmark metadata is missing") and field_name in issue
        for issue in index.issues
    )


def test_benchmark_request_cannot_silently_omit_optional_baseline(tmp_path: Path) -> None:
    request = _request(tmp_path)
    payload = request.model_dump(mode="json")
    payload["entries"] = [
        entry for entry in payload["entries"] if entry["kind"] != BenchmarkEvidenceKind.AWQ
    ]

    with pytest.raises(ValueError, match="explicitly list every baseline"):
        BenchmarkEvidenceRequest.model_validate(payload)


def _contract(**overrides: str) -> FormalHostContract:
    defaults = {
        "hardware_id": "df-macbookpro-m5/apple-m3-max/fixture",
        "os_version": "macOS-test",
        "power_mode": "AC power",
        "storage_contract": "internal-ssd",
        "thermal_protocol": "ambient-22c",
        "operator": "operator-a",
    }
    defaults.update(overrides)
    return FormalHostContract(**defaults)


def test_formal_mtp_pair_is_admissible_on_the_contract_host() -> None:
    issues = formal_mtp_bundle_issues(
        mtp_off=_bundle(BenchmarkEvidenceKind.AXQUANT_MTP_OFF),
        mtp_on=_bundle(BenchmarkEvidenceKind.AXQUANT_MTP_ON),
        contract=_contract(),
        authorized_device_name="Mac15,9",
        authorized_chip="Apple M3 Max",
    )
    assert issues == []


def test_formal_mtp_pair_rejects_an_unauthorized_host() -> None:
    issues = formal_mtp_bundle_issues(
        mtp_off=_bundle(BenchmarkEvidenceKind.AXQUANT_MTP_OFF),
        mtp_on=_bundle(BenchmarkEvidenceKind.AXQUANT_MTP_ON),
        contract=_contract(),
        authorized_device_name="Mac14,13",
        authorized_chip="Apple M2 Ultra",
    )
    assert any("not the authorized formal device" in issue for issue in issues)
    assert any("not the authorized formal chip" in issue for issue in issues)


def test_formal_mtp_pair_rejects_contract_os_and_power_drift() -> None:
    issues = formal_mtp_bundle_issues(
        mtp_off=_bundle(BenchmarkEvidenceKind.AXQUANT_MTP_OFF),
        mtp_on=_bundle(BenchmarkEvidenceKind.AXQUANT_MTP_ON),
        contract=_contract(os_version="macOS-frozen-other", power_mode="battery"),
        authorized_device_name="Mac15,9",
        authorized_chip="Apple M3 Max",
    )
    assert any("differs from the frozen contract OS" in issue for issue in issues)
    assert any("power mode differs" in issue for issue in issues)


def test_formal_mtp_pair_rejects_dataset_and_control_drift() -> None:
    drifted_on = _bundle(BenchmarkEvidenceKind.AXQUANT_MTP_ON, dataset_sha256="d" * 64)
    drifted_on = drifted_on.model_copy(
        update={
            "benchmark_metadata": {
                **drifted_on.benchmark_metadata,
                "max_tokens": 1024,
            }
        }
    )
    issues = formal_mtp_bundle_issues(
        mtp_off=_bundle(BenchmarkEvidenceKind.AXQUANT_MTP_OFF),
        mtp_on=drifted_on,
        contract=_contract(),
        authorized_device_name="Mac15,9",
        authorized_chip="Apple M3 Max",
    )
    assert any("different datasets" in issue for issue in issues)
    assert any("different benchmark controls" in issue for issue in issues)


def test_formal_mtp_pair_requires_mtp_metrics_and_zero_fallbacks() -> None:
    broken_on = _bundle(BenchmarkEvidenceKind.AXQUANT_MTP_ON).model_copy(update={"mtp": None})
    fallback_off = _bundle(BenchmarkEvidenceKind.AXQUANT_MTP_OFF)
    fallback_off = fallback_off.model_copy(
        update={"hardware": fallback_off.hardware.model_copy(update={"kernel_fallbacks": 2})}
    )
    issues = formal_mtp_bundle_issues(
        mtp_off=fallback_off,
        mtp_on=broken_on,
        contract=_contract(),
        authorized_device_name="Mac15,9",
        authorized_chip="Apple M3 Max",
    )
    assert any("missing MTP metrics" in issue for issue in issues)
    assert any("kernel fallbacks" in issue for issue in issues)
