from __future__ import annotations

from pathlib import Path

from axquant.analyzer import architecture_prior_report
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
    TensorRole,
    TensorSpec,
)
from axquant.scoreboard import build_scoreboard
from axquant.serde import write_data


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


def _size_evidence(tmp_path: Path, name: str, *, weight_bytes: int) -> Path:
    path = tmp_path / f"{name}.json"
    write_data(
        path,
        ArtifactSizeEvidence(
            kind="candidate" if name == "candidate" else "uniform-4bit",
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
            candidate_model=ModelIdentity(model_id="org/model", revision="cand"),
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
    write_data(
        path,
        MtpAbComparison(
            profile_name="agent-coding",
            exactness_pass=exactness_pass,
            divergent_trial_count=0 if exactness_pass else 3,
            measured_trial_count=10,
            speedup=speedup,
            speedup_pass=speedup is not None and speedup >= 1.20,
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
