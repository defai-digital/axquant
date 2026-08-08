from __future__ import annotations

from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.errors import ArtifactError, PlanningError
from axquant.planner import plan_quantization
from axquant.schema import (
    ArtifactSizeEvidence,
    Inventory,
    ModelIdentity,
    MtpAbComparison,
    MtpPolicy,
    PlanRequest,
    ProfileName,
    QualityComparisonReport,
    QualityScoreComparison,
    RuntimeName,
    SoftwareVersions,
    TensorRole,
    TensorSpec,
)
from axquant.scoreboard import build_scoreboard, require_scoreboard_inputs_for_certification
from axquant.serde import load_model, write_data


def _tensor(name: str, parameters: int, role: TensorRole) -> TensorSpec:
    return TensorSpec(
        name=name,
        module_path=name.removesuffix(".weight"),
        shape=(parameters, 1),
        dtype="BF16",
        parameters=parameters,
        role=role,
        quantizable=True,
        file="model.safetensors",
        current_precision="bf16",
    )


def _plan():
    inventory = Inventory(
        model=ModelIdentity(model_id="org/model", revision="abc"),
        tensors=[_tensor("model.layers.0.mlp.down_proj.weight", 10_000, TensorRole.MLP)],
        total_parameters=10_000,
        quantizable_parameters=10_000,
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
    )
    report = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    return plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=6.5,
            allow_unmeasured=True,
            mtp=MtpPolicy(mode="protected"),
        ),
    )


def _size_evidence(
    tmp_path: Path,
    name: str,
    *,
    weight_bytes: int,
    kind: str | None = None,
) -> Path:
    path = tmp_path / f"{name}.json"
    write_data(
        path,
        ArtifactSizeEvidence(
            kind=kind or ("candidate" if name == "candidate" else "uniform-4bit"),
            model=ModelIdentity(model_id="org/model", revision="abc"),
            logical_parameters=10_000,
            weight_bytes=weight_bytes,
            measured_bpw=weight_bytes * 8 / 10_000,
            source_sha256="a" * 64,
        ),
    )
    return path


def _quality_report(tmp_path: Path, *, retention: float) -> Path:
    path = tmp_path / "quality.json"
    write_data(
        path,
        QualityComparisonReport(
            reference_model=ModelIdentity(model_id="org/model", revision="ref"),
            candidate_model=ModelIdentity(model_id="org/model", revision="abc"),
            dataset_sha256="a" * 64,
            random_seed=0,
            aggregate=QualityScoreComparison(
                reference=1.0, candidate=retention, delta=retention - 1.0, retention=retention
            ),
            categories={},
            tasks=[],
            reference_errors=0,
            candidate_errors=0,
        ),
    )
    return path


def _mtp_ab(tmp_path: Path, *, exactness_pass: bool, speedup: float | None) -> Path:
    path = tmp_path / "mtp-ab.json"
    speedup_pass = speedup is not None and speedup >= 1.20
    write_data(
        path,
        MtpAbComparison(
            profile_name="benchmark-ab",
            model=ModelIdentity(model_id="org/model", revision="a" * 40),
            runtime=RuntimeName.AX_ENGINE,
            workload=ProfileName.AGENT_CODING.value,
            dataset_sha256="a" * 64,
            random_seed=0,
            generation_controls={"temperature": 0.0},
            runtime_env={"AX_TEST_MODE": "1"},
            draft_depth=2,
            exactness_pass=exactness_pass,
            divergent_trial_count=0 if exactness_pass else 3,
            measured_trial_count=10,
            direct_tokens_per_second_p50=100.0,
            mtp_tokens_per_second_p50=(100.0 * speedup if speedup is not None else None),
            speedup=speedup,
            speedup_pass=speedup_pass,
            release_ready=exactness_pass and speedup_pass,
            ax_engine_version="6.12.1",
            runtime_chip="Apple M4 Max",
            software_versions=SoftwareVersions(
                axquant="1.0.0",
                python="3.13",
                mlx="0.32",
                mlx_lm="0.31",
                ax_engine="6.12.1",
                safetensors="0.6",
                pydantic="2.11",
            ),
        ),
    )
    return path


def _weighted_mtp_ab(tmp_path: Path) -> Path:
    path = tmp_path / "weighted-mtp-ab.json"
    write_data(
        path,
        MtpAbComparison(
            profile_name="benchmark-ab",
            model=ModelIdentity(model_id="org/model", revision="a" * 40),
            runtime=RuntimeName.AX_ENGINE,
            workload=ProfileName.AGENT_CODING.value,
            dataset_sha256="a" * 64,
            random_seed=0,
            generation_controls={"temperature": 0.0},
            runtime_env={"AX_TEST_MODE": "1"},
            draft_depth=1,
            exactness_pass=True,
            measured_trial_count=5,
            direct_tokens_per_second_p50=100.0,
            mtp_tokens_per_second_p50=110.0,
            direct_token_weighted_decode_tps=100.0,
            mtp_token_weighted_decode_tps=125.0,
            prompt_median_speedup=1.1,
            token_weighted_decode_speedup=1.25,
            speedup_metric="token-weighted-decode-tps",
            speedup=1.25,
            minimum_speedup=1.20,
            minimum_prompt_median_speedup=1.10,
            prompt_median_speedup_pass=True,
            speedup_pass=True,
            release_ready=True,
            ax_engine_version="6.12.1",
            runtime_chip="Apple M4 Max",
            software_versions=SoftwareVersions(
                axquant="1.0.0",
                python="3.13",
                mlx="0.32",
                mlx_lm="0.31",
                ax_engine="6.12.1",
                safetensors="0.6",
                pydantic="2.11",
            ),
        ),
    )
    return path


def test_size_ratio_passes_at_or_below_the_threshold(tmp_path: Path) -> None:
    candidate = _size_evidence(tmp_path, "candidate", weight_bytes=1100)
    reference = _size_evidence(tmp_path, "reference", weight_bytes=1000)

    report = build_scoreboard(
        plan=_plan(),
        candidate_size=candidate,
        size_reference=reference,
        max_size_ratio_to_uniform4=1.10,
    )

    row = next(r for r in report.rows if r.metric_id == "size_ratio_vs_uniform4")
    assert row.status == "pass"
    assert row.value == 1.1


def test_size_ratio_fails_above_the_threshold(tmp_path: Path) -> None:
    candidate = _size_evidence(tmp_path, "candidate", weight_bytes=1101)
    reference = _size_evidence(tmp_path, "reference", weight_bytes=1000)

    report = build_scoreboard(
        plan=_plan(),
        candidate_size=candidate,
        size_reference=reference,
        max_size_ratio_to_uniform4=1.10,
    )

    row = next(r for r in report.rows if r.metric_id == "size_ratio_vs_uniform4")
    assert row.status == "fail"
    assert report.overall_status == "fail"


def test_six_bit_scoreboard_uses_uniform6_size_reference(tmp_path: Path) -> None:
    plan = _plan().model_copy(update={"target_class": "6bit"})
    candidate = _size_evidence(tmp_path, "candidate", weight_bytes=1000)
    reference = _size_evidence(
        tmp_path,
        "uniform6-reference",
        weight_bytes=1000,
        kind="uniform-6bit",
    )

    report = build_scoreboard(
        plan=plan,
        candidate_size=candidate,
        size_reference=reference,
        max_size_ratio_to_uniform4=1.10,
    )

    row = next(item for item in report.rows if item.metric_id == "size_ratio_vs_uniform6")
    assert row.status == "pass"
    assert "size_ratio_vs_uniform4" not in {item.metric_id for item in report.rows}


def test_six_bit_scoreboard_rejects_uniform4_size_reference(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="requires uniform-6bit"):
        build_scoreboard(
            plan=_plan().model_copy(update={"target_class": "6bit"}),
            candidate_size=_size_evidence(tmp_path, "candidate", weight_bytes=1000),
            size_reference=_size_evidence(tmp_path, "reference", weight_bytes=1000),
        )


def test_quality_retention_pass_and_fail_directions(tmp_path: Path) -> None:
    passing = build_scoreboard(
        plan=_plan(),
        quality_comparison=_quality_report(tmp_path, retention=0.99),
        minimum_quality_retention=0.98,
    )
    failing = build_scoreboard(
        plan=_plan(),
        quality_comparison=_quality_report(tmp_path, retention=0.90),
        minimum_quality_retention=0.98,
    )

    pass_row = next(r for r in passing.rows if r.metric_id == "quality_retention")
    fail_row = next(r for r in failing.rows if r.metric_id == "quality_retention")
    assert pass_row.status == "pass"
    assert fail_row.status == "fail"
    assert failing.overall_status == "fail"


def test_mtp_speedup_pass_and_fail_directions(tmp_path: Path) -> None:
    passing = build_scoreboard(
        plan=_plan(),
        mtp_ab=_mtp_ab(tmp_path, exactness_pass=True, speedup=1.25),
        minimum_mtp_speedup=1.20,
    )
    failing = build_scoreboard(
        plan=_plan(),
        mtp_ab=_mtp_ab(tmp_path, exactness_pass=True, speedup=1.05),
        minimum_mtp_speedup=1.20,
    )

    pass_row = next(r for r in passing.rows if r.metric_id == "mtp_speedup")
    fail_row = next(r for r in failing.rows if r.metric_id == "mtp_speedup")
    assert pass_row.status == "pass"
    assert fail_row.status == "fail"
    # The speed gate is AX Engine-owned, not a planner regression -- must not
    # silently pass through as "planner is fine" while still reporting fail.
    assert fail_row.owner == "ax-engine"


def test_token_weighted_mtp_speedup_keeps_prompt_guardrail_visible(tmp_path: Path) -> None:
    report = build_scoreboard(
        plan=_plan(),
        mtp_ab=_weighted_mtp_ab(tmp_path),
        minimum_mtp_speedup=1.20,
    )

    row = next(item for item in report.rows if item.metric_id == "mtp_speedup")
    assert row.status == "pass"
    assert "speedup_metric=token-weighted-decode-tps" in row.notes
    assert "prompt_median_speedup=1.1" in row.notes


def test_mtp_exactness_divergence_fails_regardless_of_speedup(tmp_path: Path) -> None:
    report = build_scoreboard(
        plan=_plan(),
        mtp_ab=_mtp_ab(tmp_path, exactness_pass=False, speedup=1.50),
    )

    exactness_row = next(r for r in report.rows if r.metric_id == "mtp_exactness")
    assert exactness_row.status == "fail"
    assert report.overall_status == "fail"


def test_missing_mandatory_evidence_is_incomplete_not_silently_passing(tmp_path: Path) -> None:
    report = build_scoreboard(plan=_plan())

    assert report.overall_status == "incomplete"
    assert "size_ratio_vs_uniform4" in report.missing_mandatory
    assert "quality_retention" in report.missing_mandatory
    assert "mtp_exactness" in report.missing_mandatory
    assert "mtp_speedup" in report.missing_mandatory


def test_any_failing_gate_outranks_incomplete_in_overall_status(tmp_path: Path) -> None:
    # size/quality evidence is missing (would be "incomplete" alone), but a
    # present-and-failing MTP gate must still make the overall verdict "fail".
    report = build_scoreboard(
        plan=_plan(),
        mtp_ab=_mtp_ab(tmp_path, exactness_pass=False, speedup=None),
    )

    assert report.overall_status == "fail"


def test_scoreboard_rejects_unbound_passing_mtp_evidence(tmp_path: Path) -> None:
    mtp_path = tmp_path / "unbound-mtp.json"
    write_data(
        mtp_path,
        MtpAbComparison(
            profile_name=ProfileName.AGENT_CODING.value,
            exactness_pass=True,
            measured_trial_count=10,
            speedup=1.5,
            speedup_pass=True,
        ),
    )

    report = build_scoreboard(plan=_plan(), mtp_ab=mtp_path)

    assert next(row for row in report.rows if row.metric_id == "mtp_exactness").status == (
        "unavailable"
    )
    assert next(row for row in report.rows if row.metric_id == "mtp_speedup").status == (
        "unavailable"
    )


def test_scoreboard_rejects_mismatched_candidate_identity(tmp_path: Path) -> None:
    quality_path = _quality_report(tmp_path, retention=0.99)
    quality = load_model(quality_path, QualityComparisonReport)
    quality.candidate_model = ModelIdentity(model_id="unrelated/model", revision="revision")
    write_data(quality_path, quality)

    with pytest.raises(ArtifactError, match="different candidates"):
        build_scoreboard(
            plan=_plan(),
            candidate_size=_size_evidence(tmp_path, "candidate", weight_bytes=1000),
            size_reference=_size_evidence(tmp_path, "reference", weight_bytes=1000),
            quality_comparison=quality_path,
        )


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), 0.0, -1.0])
def test_scoreboard_rejects_invalid_thresholds(threshold: float) -> None:
    with pytest.raises(PlanningError, match="positive and finite"):
        build_scoreboard(plan=_plan(), minimum_mtp_speedup=threshold)


def test_certification_rejects_architecture_prior_scoreboard() -> None:
    report = build_scoreboard(plan=_plan())

    with pytest.raises(PlanningError, match="not eligible"):
        require_scoreboard_inputs_for_certification(report)
