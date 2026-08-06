"""Kernel-latency table, provider, and latency-aware planning (ADR-0003)."""

from __future__ import annotations

import pytest

from axquant.kernel_latency import decode_latency_provider, packing_equivalence_method
from axquant.planner import plan_quantization
from axquant.schema import (
    ArchitectureProfile,
    ArchitectureSupportLevel,
    CandidateMeasurement,
    EvidenceKind,
    HardwareProfile,
    KernelLatencyEntry,
    KernelLatencyTable,
    MetricVector,
    ModelIdentity,
    OptimizationScope,
    PlanRequest,
    ProfileName,
    QuantMethod,
    RuntimeName,
    SensitivityReport,
    TensorRole,
    TensorSensitivity,
    TensorSpec,
)
from axquant.serde import stable_sha256
from axquant.versioning import collect_versions


def _entry(
    *,
    bits: int,
    group_size: int | None,
    hidden_size: int,
    decode_us: float,
    method: QuantMethod = QuantMethod.AFFINE,
) -> KernelLatencyEntry:
    return KernelLatencyEntry(
        runtime=RuntimeName.MLX_LM,
        bits=bits,
        group_size=group_size,
        method=method,
        hidden_size=hidden_size,
        decode_median_us=decode_us,
        prefill_median_us=decode_us * 10.0,
        dispersion=0.05,
        iterations=20,
    )


def _table(entries: list[KernelLatencyEntry]) -> KernelLatencyTable:
    return KernelLatencyTable(
        host_id="macstudio-m2u",
        chip="Apple M2 Ultra",
        os_version="15.5",
        software_versions=collect_versions(),
        warmup_iterations=5,
        entries=entries,
    )


def test_decode_latency_interpolates_between_hidden_sizes_and_clamps() -> None:
    table = _table(
        [
            _entry(bits=4, group_size=64, hidden_size=1024, decode_us=100.0),
            _entry(bits=4, group_size=64, hidden_size=2048, decode_us=200.0),
        ]
    )
    lookup = decode_latency_provider(table)
    assert lookup(4, 64, QuantMethod.AFFINE, 1536) == pytest.approx(150.0)
    assert lookup(4, 64, QuantMethod.AFFINE, 512) == pytest.approx(100.0)
    assert lookup(4, 64, QuantMethod.AFFINE, 8192) == pytest.approx(200.0)
    # Unmeasured configurations return None — never a guess across packings.
    assert lookup(4, 32, QuantMethod.AFFINE, 1024) is None
    assert lookup(6, 64, QuantMethod.AFFINE, 1024) is None


def test_provider_collapses_methods_to_their_packing_class() -> None:
    table = _table([_entry(bits=4, group_size=64, hidden_size=1024, decode_us=42.0)])
    lookup = decode_latency_provider(table)
    # Every quantized method packs to the same portable affine layout, so one
    # measured (bits, group) entry serves them all.
    for method in (
        QuantMethod.AFFINE,
        QuantMethod.DWQ,
        QuantMethod.AWQ,
        QuantMethod.GPTQ,
        QuantMethod.GPTQ_ACT,
    ):
        assert packing_equivalence_method(method) == QuantMethod.AFFINE
        assert lookup(4, 64, method, 1024) == pytest.approx(42.0)
    assert packing_equivalence_method(QuantMethod.BF16) == QuantMethod.BF16


def _single_tensor_report() -> SensitivityReport:
    tensor = TensorSpec(
        name="model.layers.0.mlp.down_proj.weight",
        module_path="model.layers.0.mlp.down_proj",
        shape=(16, 1024),
        dtype="BF16",
        parameters=16384,
        role=TensorRole.MLP,
        quantizable=True,
        file="model.safetensors",
        current_precision="bf16",
    )
    candidates = [
        CandidateMeasurement(bits=16, method=QuantMethod.BF16, metrics=MetricVector()),
        # g32 is marginally better on quality; g64 is within the 2% epsilon.
        CandidateMeasurement(
            bits=4,
            method=QuantMethod.AFFINE,
            group_size=32,
            metrics=MetricVector(output_kl=0.100),
        ),
        CandidateMeasurement(
            bits=4,
            method=QuantMethod.AFFINE,
            group_size=64,
            metrics=MetricVector(output_kl=0.101),
        ),
    ]
    return SensitivityReport(
        model=ModelIdentity(model_id="org/model", revision="abc"),
        architecture_profile=ArchitectureProfile(
            support_level=ArchitectureSupportLevel.SUPPORTED,
            product_family="qwen3.6",
            optimization_scope=OptimizationScope.TEXT_PATH,
            adapter_id="qwen36-v1",
            text_layer_count=1,
        ),
        profile=ProfileName.GENERAL,
        evidence_kind=EvidenceKind.ARCHITECTURE_PRIOR,
        inventory_sha256="a" * 64,
        entries=[TensorSensitivity(tensor=tensor, candidates=candidates)],
    )


def _request() -> PlanRequest:
    return PlanRequest(
        profile=ProfileName.GENERAL,
        target_bpw=5.2,
        candidate_bits=(4, 16),
        candidate_group_sizes=(32, 64),
        allow_unmeasured=True,
        hardware=HardwareProfile(),
    )


def test_latency_polish_prefers_faster_group_within_quality_epsilon() -> None:
    report = _single_tensor_report()

    baseline = plan_quantization(report, _request())
    assert baseline.cost_model == "abstract-bpw"
    assert baseline.kernel_latency_sha256 is None
    baseline_mlp = baseline.assignments[0]
    # Without a table the budget upgrade lands on the finer, higher-quality g32.
    assert baseline_mlp.group_size == 32

    table = _table(
        [
            _entry(bits=4, group_size=32, hidden_size=1024, decode_us=50.0),
            _entry(bits=4, group_size=64, hidden_size=1024, decode_us=30.0),
        ]
    )
    latency_plan = plan_quantization(report, _request(), kernel_latency=table)
    polished = latency_plan.assignments[0]
    # g64 is within the quality window, needs no extra storage, and its kernel
    # is measured faster — the plan flips and records why.
    assert polished.group_size == 64
    assert "kernel-latency" in polished.reason
    assert latency_plan.cost_model == "kernel-latency"
    assert latency_plan.kernel_latency_sha256 == stable_sha256(table)
    assert latency_plan.kernel_latency_host_id == "macstudio-m2u"


def test_latency_table_never_overrides_quality_outside_epsilon() -> None:
    report = _single_tensor_report()
    entry = report.entries[0]
    for candidate in entry.candidates:
        if candidate.group_size == 64:
            # Push g64 far outside the epsilon window.
            entry.candidates[entry.candidates.index(candidate)] = candidate.model_copy(
                update={"metrics": MetricVector(output_kl=0.150)}
            )
    table = _table(
        [
            _entry(bits=4, group_size=32, hidden_size=1024, decode_us=50.0),
            _entry(bits=4, group_size=64, hidden_size=1024, decode_us=1.0),
        ]
    )
    plan = plan_quantization(report, _request(), kernel_latency=table)
    # Even a 50x faster kernel cannot buy a quality regression beyond epsilon.
    assert plan.assignments[0].group_size == 32
    assert plan.cost_model == "kernel-latency"


def test_benchmark_kernels_ingests_ax_engine_raw_document(tmp_path) -> None:
    import json

    from axquant.cli import main
    from axquant.serde import load_model

    raw = {
        "schema_version": "ax-engine.kernel-latency-raw.v1",
        "ax_engine_version": "6.13.1",
        "prefill_rows": 512,
        "warmup_iterations": 5,
        "entries": [
            {
                "method": "bf16",
                "bits": 16,
                "group_size": None,
                "hidden_size": 2048,
                "decode_median_us": 235.9,
                "prefill_median_us": 691.8,
                "dispersion": 0.14,
                "iterations": 20,
            },
            {
                "method": "affine",
                "bits": 4,
                "group_size": 64,
                "hidden_size": 2048,
                "decode_median_us": 152.8,
                "prefill_median_us": 560.0,
                "dispersion": 0.05,
                "iterations": 20,
            },
        ],
        "warnings": [],
    }
    raw_path = tmp_path / "engine_raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    output_path = tmp_path / "kernel_latency.json"

    assert (
        main(
            [
                "benchmark-kernels",
                "--host-id",
                "macstudio-m2u",
                "--chip",
                "Apple M2 Ultra",
                "--os-version",
                "15.5",
                "--from-ax-engine",
                str(raw_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    table = load_model(output_path, KernelLatencyTable)
    assert {entry.runtime for entry in table.entries} == {RuntimeName.AX_ENGINE}
    assert table.software_versions.ax_engine == "6.13.1"
    # The provider infers the single runtime, so an engine table plugs into
    # `plan --latency-table` without extra flags; methods collapse to the
    # affine packing class.
    lookup = decode_latency_provider(table)
    assert lookup(4, 64, QuantMethod.GPTQ_ACT, 2048) == pytest.approx(152.8)
    assert lookup(16, None, QuantMethod.BF16, 2048) == pytest.approx(235.9)


def test_benchmark_kernels_rejects_malformed_ax_engine_document(tmp_path) -> None:
    import json

    from axquant.cli import main

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "something-else"}), encoding="utf-8")
    assert (
        main(
            [
                "benchmark-kernels",
                "--host-id",
                "h",
                "--chip",
                "c",
                "--os-version",
                "o",
                "--from-ax-engine",
                str(bad),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 2
    )


def test_benchmark_kernels_rejects_unknown_engine_methods(tmp_path) -> None:
    import json

    from axquant.cli import main

    raw = {
        "schema_version": "ax-engine.kernel-latency-raw.v1",
        "ax_engine_version": "6.13.1",
        "warmup_iterations": 5,
        "entries": [
            {
                "method": "mxfp4",
                "bits": 4,
                "group_size": 32,
                "hidden_size": 2048,
                "decode_median_us": 100.0,
                "prefill_median_us": 500.0,
                "dispersion": 0.05,
                "iterations": 20,
            }
        ],
        "warnings": [],
    }
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    # An unrecognized packing method must never be silently relabeled affine.
    assert (
        main(
            [
                "benchmark-kernels",
                "--host-id",
                "h",
                "--chip",
                "c",
                "--os-version",
                "o",
                "--from-ax-engine",
                str(raw_path),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 2
    )
