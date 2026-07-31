"""Tests for the MTP benchmark harness (v0.2)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from axquant.benchmark import (
    MTP_DIAGNOSTIC_PROFILES,
    _ax_engine_version,
    _kernel_fallback_count,
    _percentile,
    _runtime_environment,
    _standalone_ax_engine_version,
    compare_mtp_ab_results,
    parse_runtime_env_items,
    result_to_evaluation_bundle,
    run_benchmark,
    run_mtp_ab,
    run_mtp_diagnostics,
    validate_ab_invariant,
)
from axquant.errors import BackendUnavailableError, BenchmarkError, InvariantViolationError
from axquant.schema import (
    BenchmarkConfig,
    BenchmarkResult,
    ModelIdentity,
    TrialResult,
)

_TEST_EXECUTABLE = sys.executable


@pytest.fixture
def prompt_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "prompts.jsonl"
    lines = [
        json.dumps({"prompt": "Write a function to sort a list"}),
        json.dumps({"prompt": "Fix this Python code"}),
        json.dumps({"prompt": "Generate a JSON schema"}),
        json.dumps({"prompt": "Translate to Japanese"}),
        json.dumps({"prompt": "Explain recursion"}),
    ]
    dataset.write_text("\n".join(lines), encoding="utf-8")
    return dataset


@pytest.fixture
def base_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        model=ModelIdentity(model_id="test-model", revision="abc123", local_path="/tmp/model"),
        mtp_enabled=False,
        baseline_kind="axquant-mtp-off",
        workload="agent-coding",
        dataset_sha256="deadbeef" * 8,
        prompt_count=3,
        warmup_trials=1,
        measured_trials=3,
        power_mode="AC power",
        quantizer="axquant",
        quantizer_version="0.1.0a0",
        random_seed=42,
    )


def _fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Simulate a successful AX Engine generate call."""
    return subprocess.CompletedProcess(
        args=command,
        returncode=0,
        stdout=json.dumps(
            {
                "tokens_generated": 100,
                "decode_seconds": 2.0,
                "prefill_seconds": 0.5,
                "mtp_accepted_tokens": 80,
                "mtp_proposed_tokens": 100,
                "peak_memory_bytes": 8_000_000_000,
                "runtime": {
                    "host": {
                        "device_class": "Mac15,9",
                        "detected_soc": "Apple M3 Max",
                        "unified_memory_bytes": 128 * 1024**3,
                    }
                },
            }
        ),
        stderr="",
    )


def _failing_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Simulate a failed AX Engine call."""
    return subprocess.CompletedProcess(
        args=command,
        returncode=1,
        stdout="",
        stderr="model loading failed",
    )


def _timeout_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Simulate a timeout."""
    return subprocess.CompletedProcess(
        args=command,
        returncode=1,
        stdout="",
        stderr="request timeout exceeded",
    )


class TestBenchmarkConfigValidation:
    def test_valid_config(self, base_config: BenchmarkConfig) -> None:
        assert base_config.warmup_trials == 1
        assert base_config.measured_trials == 3
        assert base_config.random_seed == 42

    def test_invalid_prompt_count(self) -> None:
        with pytest.raises(ValueError, match="prompt_count"):
            BenchmarkConfig(
                model=ModelIdentity(model_id="m"),
                baseline_kind="candidate",
                workload="test",
                dataset_sha256="abc",
                prompt_count=0,
            )

    def test_invalid_temperature(self) -> None:
        with pytest.raises(ValueError):
            BenchmarkConfig(
                model=ModelIdentity(model_id="m"),
                baseline_kind="candidate",
                workload="test",
                dataset_sha256="abc",
                prompt_count=1,
                temperature=-1.0,
            )

    def test_invalid_baseline_kind(self) -> None:
        with pytest.raises(ValueError):
            BenchmarkConfig(
                model=ModelIdentity(model_id="m"),
                baseline_kind="invalid-kind",  # type: ignore[arg-type]
                workload="test",
                dataset_sha256="abc",
                prompt_count=1,
            )


class TestABInvariant:
    def test_valid_ab_pair(self, base_config: BenchmarkConfig) -> None:
        mtp_config = base_config.model_copy(
            update={"mtp_enabled": True, "baseline_kind": "axquant-mtp-on"}
        )
        validate_ab_invariant(base_config, mtp_config)

    def test_mismatched_model(self, base_config: BenchmarkConfig) -> None:
        mtp_config = base_config.model_copy(
            update={
                "mtp_enabled": True,
                "baseline_kind": "axquant-mtp-on",
                "model": ModelIdentity(model_id="different-model"),
            }
        )
        with pytest.raises(InvariantViolationError, match="model identity"):
            validate_ab_invariant(base_config, mtp_config)

    def test_mismatched_dataset(self, base_config: BenchmarkConfig) -> None:
        mtp_config = base_config.model_copy(
            update={
                "mtp_enabled": True,
                "baseline_kind": "axquant-mtp-on",
                "dataset_sha256": "different" * 8,
            }
        )
        with pytest.raises(InvariantViolationError, match="dataset digest"):
            validate_ab_invariant(base_config, mtp_config)

    def test_direct_must_not_have_mtp(self, base_config: BenchmarkConfig) -> None:
        bad_direct = base_config.model_copy(update={"mtp_enabled": True})
        mtp_config = base_config.model_copy(
            update={"mtp_enabled": True, "baseline_kind": "axquant-mtp-on"}
        )
        with pytest.raises(InvariantViolationError, match="direct config must have MTP disabled"):
            validate_ab_invariant(bad_direct, mtp_config)

    def test_mtp_must_be_enabled(self, base_config: BenchmarkConfig) -> None:
        mtp_config = base_config.model_copy(update={"baseline_kind": "axquant-mtp-on"})
        with pytest.raises(InvariantViolationError, match="MTP config must have MTP enabled"):
            validate_ab_invariant(base_config, mtp_config)

    def test_mismatched_draft_depth(self, base_config: BenchmarkConfig) -> None:
        direct = base_config.model_copy(update={"draft_depth": 1})
        mtp_config = base_config.model_copy(
            update={
                "mtp_enabled": True,
                "baseline_kind": "axquant-mtp-on",
                "draft_depth": 2,
            }
        )
        with pytest.raises(InvariantViolationError, match="draft_depth"):
            validate_ab_invariant(direct, mtp_config)

    def test_mismatched_power_mode(self, base_config: BenchmarkConfig) -> None:
        mtp_config = base_config.model_copy(
            update={
                "mtp_enabled": True,
                "baseline_kind": "axquant-mtp-on",
                "power_mode": "battery",
            }
        )
        with pytest.raises(InvariantViolationError, match="power_mode"):
            validate_ab_invariant(base_config, mtp_config)


class TestPercentile:
    def test_empty_list(self) -> None:
        assert _percentile([], 50) is None

    def test_single_value(self) -> None:
        assert _percentile([5.0], 50) == 5.0

    def test_p50(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 50) == 3.0

    def test_p90(self) -> None:
        values = list(range(1, 101))
        result = _percentile([float(v) for v in values], 90)
        assert result is not None
        assert 89.0 <= result <= 91.0

    def test_p99(self) -> None:
        values = [float(v) for v in range(1, 101)]
        result = _percentile(values, 99)
        assert result is not None
        assert result >= 99.0


def test_runtime_environment_records_mtp_depth(base_config: BenchmarkConfig) -> None:
    direct = base_config.model_copy(update={"draft_depth": 1})
    mtp = direct.model_copy(update={"mtp_enabled": True})
    assert _runtime_environment(direct) == ["AX_NO_SPEC=1", "AX_MLX_MTP_MAX_DEPTH=1"]
    assert _runtime_environment(mtp) == ["AX_MLX_MTP_MAX_DEPTH=1"]


def test_runtime_environment_includes_kill_switches(base_config: BenchmarkConfig) -> None:
    config = base_config.model_copy(
        update={
            "mtp_enabled": True,
            "draft_depth": 1,
            "runtime_env": {
                "AX_MLX_MTP_BYPASS_MIN_SAMPLES": "1000",
                "AX_MLX_MTP_BYPASS_THRESHOLD": "0",
                "AX_MLX_MTP_DRAFT_MIN_CONFIDENCE": "0.000001",
                "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL": "0",
                "AX_MLX_QWEN_GATED_DELTA_DECODE_METAL": "0",
                "AX_MLX_SPECULATIVE_INVARIANT_PROJECTIONS": "all",
                "AX_MLX_SPECULATIVE_ROW_EXACT_POST_INPUT": "1",
                "AX_MLX_SPECULATIVE_SPLIT_FFN": "1",
            },
        }
    )
    assert _runtime_environment(config) == [
        "AX_MLX_MTP_MAX_DEPTH=1",
        "AX_MLX_MTP_BYPASS_MIN_SAMPLES=1000",
        "AX_MLX_MTP_BYPASS_THRESHOLD=0",
        "AX_MLX_MTP_DRAFT_MIN_CONFIDENCE=0.000001",
        "AX_MLX_QWEN_GATED_DELTA_DECODE_METAL=0",
        "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL=0",
        "AX_MLX_SPECULATIVE_INVARIANT_PROJECTIONS=all",
        "AX_MLX_SPECULATIVE_ROW_EXACT_POST_INPUT=1",
        "AX_MLX_SPECULATIVE_SPLIT_FFN=1",
    ]


def test_parse_runtime_env_items_allowlist() -> None:
    parsed = parse_runtime_env_items(
        [
            "AX_MLX_MTP_BYPASS_MIN_SAMPLES=1000",
            "AX_MLX_MTP_BYPASS_THRESHOLD=0",
            "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL=0",
            "AX_MLX_MTP_DRAFT_MIN_CONFIDENCE=0.000001",
            "AX_MLX_SPECULATIVE_INVARIANT_PROJECTIONS=all",
            "AX_MLX_SPECULATIVE_ROW_EXACT_POST_INPUT=1",
            "AX_MLX_SPECULATIVE_SPLIT_FFN=1",
        ]
    )
    assert parsed == {
        "AX_MLX_MTP_BYPASS_MIN_SAMPLES": "1000",
        "AX_MLX_MTP_BYPASS_THRESHOLD": "0",
        "AX_MLX_MTP_DRAFT_MIN_CONFIDENCE": "0.000001",
        "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL": "0",
        "AX_MLX_SPECULATIVE_INVARIANT_PROJECTIONS": "all",
        "AX_MLX_SPECULATIVE_ROW_EXACT_POST_INPUT": "1",
        "AX_MLX_SPECULATIVE_SPLIT_FFN": "1",
    }
    with pytest.raises(Exception, match=r"not allowlisted|runtime_env"):
        parse_runtime_env_items(["PATH=/tmp"])
    with pytest.raises(BenchmarkError, match=r"KEY=VALUE"):
        parse_runtime_env_items(["NOEQUALS"])


def test_kernel_fallback_count_excludes_policy_fallback_steps() -> None:
    decisions = {
        "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks": 2,
        "ax_mlx_kv_paged_pool_exhaustion_fallbacks": 1,
        "ax_mlx_prefix_cache_disk_fallback_recompute": 4,
        "ax_mtp_direct_fallback_steps": 8,
        "ax_mtp_ngram_fallback_no_candidate_steps": 16,
    }
    assert _kernel_fallback_count(decisions) == 3


def test_standalone_ax_engine_version_accepts_versioned_runtime_layouts(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "6.12.1" / "ax-engine-bench"
    nested = tmp_path / "v6.12.2-rc.1" / "bin" / "ax-engine-bench"
    unversioned = tmp_path / "runtime" / "ax-engine-bench"
    for executable in (direct, nested, unversioned):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()

    assert _standalone_ax_engine_version(str(direct)) == "6.12.1"
    assert _standalone_ax_engine_version(str(nested)) == "6.12.2-rc.1"
    assert _standalone_ax_engine_version(str(unversioned)) is None


def test_ax_engine_version_falls_back_when_doctor_has_no_install_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "6.12.1" / "ax-engine-bench"
    executable.parent.mkdir(parents=True)
    executable.touch()

    def doctor_without_install(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[str(executable), "doctor", "--json"],
            returncode=0,
            stdout='{"status":"ready"}',
            stderr="",
        )

    monkeypatch.setattr("axquant.benchmark.subprocess.run", doctor_without_install)
    assert _ax_engine_version(str(executable)) == "6.12.1"


class TestRunBenchmark:
    def test_successful_run(self, base_config: BenchmarkConfig, prompt_dataset: Path) -> None:
        result = run_benchmark(
            base_config,
            dataset_path=prompt_dataset,
            executable=_TEST_EXECUTABLE,
            runner=_fake_runner,
        )
        assert result.measured_count == 3
        assert result.failed_count == 0
        assert result.latency_p50 is not None
        assert result.tokens_per_second_p50 is not None
        assert result.runtime_device_name == "Mac15,9"
        assert result.runtime_chip == "Apple M3 Max"
        assert result.unified_memory_bytes == 128 * 1024**3

    def test_warmup_exclusion(self, base_config: BenchmarkConfig, prompt_dataset: Path) -> None:
        result = run_benchmark(
            base_config,
            dataset_path=prompt_dataset,
            executable=_TEST_EXECUTABLE,
            runner=_fake_runner,
        )
        warmup_trials = [t for t in result.trials if t.is_warmup]
        measured_trials = [t for t in result.trials if not t.is_warmup]
        assert len(warmup_trials) == 1
        assert len(measured_trials) == 3

    def test_failed_trials(self, base_config: BenchmarkConfig, prompt_dataset: Path) -> None:
        result = run_benchmark(
            base_config,
            dataset_path=prompt_dataset,
            executable=_TEST_EXECUTABLE,
            runner=_failing_runner,
        )
        assert result.failed_count == 4  # 1 warmup + 3 measured
        assert result.measured_count == 0

    def test_timeout_detection(self, base_config: BenchmarkConfig, prompt_dataset: Path) -> None:
        result = run_benchmark(
            base_config,
            dataset_path=prompt_dataset,
            executable=_TEST_EXECUTABLE,
            runner=_timeout_runner,
        )
        assert result.timed_out_count == 4

    def test_backend_unavailable(self, base_config: BenchmarkConfig, prompt_dataset: Path) -> None:
        with pytest.raises(BackendUnavailableError):
            run_benchmark(
                base_config,
                dataset_path=prompt_dataset,
                executable="nonexistent-binary-xyz",
                runner=_fake_runner,
            )

    def test_insufficient_prompts(self, base_config: BenchmarkConfig, tmp_path: Path) -> None:
        small_dataset = tmp_path / "small.jsonl"
        small_dataset.write_text(json.dumps({"prompt": "hello"}), encoding="utf-8")
        config = base_config.model_copy(update={"prompt_count": 10})
        with pytest.raises(BenchmarkError, match="prompts"):
            run_benchmark(
                config,
                dataset_path=small_dataset,
                executable=_TEST_EXECUTABLE,
                runner=_fake_runner,
            )


class TestResultToEvaluationBundle:
    def test_mtp_off_bundle(self, base_config: BenchmarkConfig, prompt_dataset: Path) -> None:
        result = run_benchmark(
            base_config,
            dataset_path=prompt_dataset,
            executable=_TEST_EXECUTABLE,
            runner=_fake_runner,
        )
        bundle = result_to_evaluation_bundle(result)
        assert bundle.mtp_enabled is False
        assert bundle.baseline_kind == "axquant-mtp-off"
        assert bundle.mtp is None

    def test_mtp_on_bundle(self, base_config: BenchmarkConfig, prompt_dataset: Path) -> None:
        mtp_config = base_config.model_copy(
            update={
                "mtp_enabled": True,
                "baseline_kind": "axquant-mtp-on",
                "draft_depth": 1,
            }
        )
        result = BenchmarkResult(
            config=mtp_config,
            trials=[
                TrialResult(
                    trial_index=0,
                    output_token_ids=[1, 1, 2, 3],
                    tokens_generated=4,
                    tokens_per_second=10.0,
                    mtp_accepted_tokens=3,
                    mtp_proposed_tokens=4,
                    mtp_decode_steps=3,
                )
            ],
            measured_count=1,
        )
        bundle = result_to_evaluation_bundle(result)
        assert bundle.mtp_enabled is True
        assert bundle.baseline_kind == "axquant-mtp-on"
        assert bundle.mtp is not None
        assert bundle.mtp.acceptance_rate == pytest.approx(0.75)
        assert bundle.mtp.token_accuracy == {"1": pytest.approx(0.75)}
        assert bundle.mtp.repetition_rate == pytest.approx(1 / 3)


class TestRunMtpAb:
    def test_ab_run(self, base_config: BenchmarkConfig, prompt_dataset: Path) -> None:
        mtp_config = base_config.model_copy(
            update={"mtp_enabled": True, "baseline_kind": "axquant-mtp-on"}
        )
        direct_bundle, mtp_bundle = run_mtp_ab(
            base_config,
            mtp_config,
            dataset_path=prompt_dataset,
            executable=_TEST_EXECUTABLE,
            runner=_fake_runner,
        )
        assert direct_bundle.mtp_enabled is False
        assert mtp_bundle.mtp_enabled is True
        assert direct_bundle.dataset_sha256 == mtp_bundle.dataset_sha256
        assert direct_bundle.benchmark_metadata["runtime_env"] == {}
        assert mtp_bundle.mtp is not None
        assert mtp_bundle.mtp.divergence_rate == 0.0
        assert (
            mtp_bundle.benchmark_metadata["mtp_metrics_protocol"]
            == "adjacent-token-repeat-v1;depth1-proposal-accuracy-v1;"
            "greedy-output-ab-divergence-v1"
        )

    def test_ab_run_can_record_a_failed_speed_gate(
        self,
        base_config: BenchmarkConfig,
        prompt_dataset: Path,
        tmp_path: Path,
    ) -> None:
        mtp_config = base_config.model_copy(
            update={"mtp_enabled": True, "baseline_kind": "axquant-mtp-on"}
        )
        output_dir = tmp_path / "ab"

        direct_bundle, mtp_bundle = run_mtp_ab(
            base_config,
            mtp_config,
            dataset_path=prompt_dataset,
            executable=_TEST_EXECUTABLE,
            runner=_fake_runner,
            output_dir=output_dir,
            enforce_speedup=False,
            minimum_speedup=1.20,
        )

        assert direct_bundle.benchmark_metadata["tokens_per_second_p50"] == pytest.approx(50.0)
        assert mtp_bundle.benchmark_metadata["tokens_per_second_p50"] == pytest.approx(50.0)
        comparison = json.loads((output_dir / "mtp_ab_comparison.json").read_text())
        assert comparison["minimum_speedup"] == pytest.approx(1.20)
        assert comparison["speedup_pass"] is False
        assert comparison["release_ready"] is False

    def test_ab_run_writes_comparison_before_a_failed_speed_gate(
        self,
        base_config: BenchmarkConfig,
        prompt_dataset: Path,
        tmp_path: Path,
    ) -> None:
        mtp_config = base_config.model_copy(
            update={"mtp_enabled": True, "baseline_kind": "axquant-mtp-on"}
        )
        output_dir = tmp_path / "ab"

        with pytest.raises(BenchmarkError, match="below required"):
            run_mtp_ab(
                base_config,
                mtp_config,
                dataset_path=prompt_dataset,
                executable=_TEST_EXECUTABLE,
                runner=_fake_runner,
                output_dir=output_dir,
                minimum_speedup=1.20,
            )

        assert (output_dir / "mtp_ab_comparison.json").is_file()


class TestMtpDiagnostics:
    def _result_with_tokens(
        self,
        config: BenchmarkConfig,
        tokens_by_trial: dict[int, list[int]],
        *,
        tps: float = 10.0,
        mtp_active: bool | None = None,
    ) -> BenchmarkResult:
        trials: list[TrialResult] = []
        for trial_index, tokens in tokens_by_trial.items():
            trials.append(
                TrialResult(
                    trial_index=trial_index,
                    is_warmup=False,
                    success=True,
                    output_token_ids=tokens,
                    tokens_generated=len(tokens),
                    tokens_per_second=tps,
                    mtp_active=mtp_active if config.mtp_enabled else False,
                    mtp_proposed_tokens=4 if config.mtp_enabled else 0,
                    mtp_accepted_tokens=3 if config.mtp_enabled else 0,
                    mtp_rejected_tokens=1 if config.mtp_enabled else 0,
                )
            )
        return BenchmarkResult(
            config=config,
            trials=trials,
            measured_count=len(trials),
            tokens_per_second_p50=tps,
            ax_engine_version="6.11.1",
            runtime_chip="Apple M3 Max",
        )

    def test_compare_detects_divergence(self, base_config: BenchmarkConfig) -> None:
        direct_config = base_config
        mtp_config = base_config.model_copy(
            update={"mtp_enabled": True, "baseline_kind": "axquant-mtp-on"}
        )
        direct = self._result_with_tokens(
            direct_config,
            {0: [1, 2, 3, 1710], 1: [9, 9, 9]},
            tps=15.0,
        )
        mtp = self._result_with_tokens(
            mtp_config,
            {0: [1, 2, 3, 13], 1: [9, 9, 9]},
            tps=15.2,
            mtp_active=True,
        )
        comparison = compare_mtp_ab_results(
            direct, mtp, profile_name="baseline", minimum_speedup=1.20
        )
        assert comparison.exactness_pass is False
        assert comparison.divergent_trial_count == 1
        assert comparison.trial_comparisons[0].first_diff_index == 3
        assert comparison.speedup is not None
        assert comparison.speedup == pytest.approx(15.2 / 15.0)
        assert comparison.speedup_pass is False
        assert comparison.release_ready is False

    def test_compare_release_ready(self, base_config: BenchmarkConfig) -> None:
        direct_config = base_config
        mtp_config = base_config.model_copy(
            update={"mtp_enabled": True, "baseline_kind": "axquant-mtp-on"}
        )
        tokens = {0: [1, 2, 3], 1: [4, 5, 6]}
        direct = self._result_with_tokens(direct_config, tokens, tps=10.0)
        mtp = self._result_with_tokens(mtp_config, tokens, tps=13.0, mtp_active=True)
        comparison = compare_mtp_ab_results(
            direct, mtp, profile_name="disable-post-input-metal", minimum_speedup=1.20
        )
        assert comparison.exactness_pass is True
        assert comparison.speedup_pass is True
        assert comparison.release_ready is True

    def test_compare_summarizes_mtp_phase_timings(
        self,
        base_config: BenchmarkConfig,
    ) -> None:
        mtp_config = base_config.model_copy(
            update={"mtp_enabled": True, "baseline_kind": "axquant-mtp-on"}
        )
        tokens = list(range(10))
        direct_trial = TrialResult(
            trial_index=0,
            tokens_generated=10,
            output_token_ids=tokens,
            tokens_per_second=10.0,
            backend_report={"performance": {"generation_time_us": 1_000_000}},
        )
        mtp_trial = TrialResult(
            trial_index=0,
            tokens_generated=10,
            output_token_ids=tokens,
            tokens_per_second=9.0,
            mtp_active=True,
            mtp_proposed_tokens=5,
            mtp_accepted_tokens=4,
            backend_report={
                "performance": {"generation_time_us": 1_100_000},
                "route": {
                    "crossover_decisions": {
                        "ax_mtp_draft_wall_us": 100_000,
                        "ax_mtp_verify_forward_wall_us": 20_000,
                        "ax_mtp_verify_eval_wall_us": 700_000,
                        "ax_mtp_rollback_wall_us": 100_000,
                        "ax_mtp_cache_clone_wall_us": 100,
                        "ax_mtp_accept_wall_us": 50,
                        "ax_mtp_tail_sample_wall_us": 20,
                        "ax_mtp_decode_steps": 5,
                        "ax_mtp_direct_fallback_steps": 5,
                        "ax_mtp_emitted_tokens": 9,
                        "ax_mtp_correctness_mode_conflicts": 5,
                    }
                },
            },
        )
        direct = BenchmarkResult(
            config=base_config,
            trials=[direct_trial],
            measured_count=1,
            tokens_per_second_p50=10.0,
        )
        mtp = BenchmarkResult(
            config=mtp_config,
            trials=[mtp_trial],
            measured_count=1,
            tokens_per_second_p50=9.0,
        )

        comparison = compare_mtp_ab_results(direct, mtp, minimum_speedup=1.20)

        timing = comparison.phase_timing
        assert timing is not None
        assert timing.direct_generation_us_per_output_token == pytest.approx(100_000)
        assert timing.mtp_generation_us_per_output_token == pytest.approx(110_000)
        assert timing.target_mtp_us_per_output_token == pytest.approx(100_000 / 1.2)
        assert timing.required_savings_us_per_output_token == pytest.approx(110_000 - 100_000 / 1.2)
        assert timing.phase_accounted_us_per_output_token == pytest.approx(92_017)
        assert timing.active_step_fraction == pytest.approx(0.5)
        assert timing.proposed_tokens == 5
        assert timing.accepted_tokens == 4
        assert timing.correctness_mode_conflicts == 5

    def test_run_mtp_ab_soft_records_comparison(
        self, base_config: BenchmarkConfig, prompt_dataset: Path, tmp_path: Path
    ) -> None:
        mtp_config = base_config.model_copy(
            update={"mtp_enabled": True, "baseline_kind": "axquant-mtp-on"}
        )
        out = tmp_path / "ab"
        run_mtp_ab(
            base_config,
            mtp_config,
            dataset_path=prompt_dataset,
            executable=_TEST_EXECUTABLE,
            runner=_fake_runner,
            output_dir=out,
            enforce_exactness=False,
        )
        assert (out / "mtp_ab_comparison.json").is_file()

    def test_run_mtp_diagnostics_matrix(
        self, base_config: BenchmarkConfig, prompt_dataset: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "diag"
        report = run_mtp_diagnostics(
            base_config,
            dataset_path=prompt_dataset,
            executable=_TEST_EXECUTABLE,
            runner=_fake_runner,
            output_dir=out,
            profiles=["baseline", "disable-post-input-metal"],
            minimum_speedup=1.20,
        )
        assert report.schema_version == "axquant.mtp-diagnostic.v1"
        assert len(report.profiles) == 2
        assert {item.profile_name for item in report.profiles} == {
            "baseline",
            "disable-post-input-metal",
        }
        assert (out / "mtp_diagnostic_report.json").is_file()
        assert (out / "baseline" / "mtp_ab_comparison.json").is_file()
        assert report.recommended_next_step
        assert set(MTP_DIAGNOSTIC_PROFILES) >= {"baseline", "disable-la-decode-metal"}

    def test_mismatched_runtime_env_rejected(self, base_config: BenchmarkConfig) -> None:
        mtp_config = base_config.model_copy(
            update={
                "mtp_enabled": True,
                "baseline_kind": "axquant-mtp-on",
                "runtime_env": {
                    "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL": "0",
                },
            }
        )
        with pytest.raises(InvariantViolationError, match="runtime_env"):
            validate_ab_invariant(base_config, mtp_config)
