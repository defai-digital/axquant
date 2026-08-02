"""MTP benchmark harness for AX Engine evaluation.

Orchestrates MTP off/on evaluation runs, enforces A/B invariants, and emits
strict EvaluationBundle artifacts.  AX Engine interaction uses subprocess
calls behind a lazy resolution step so the module imports without the runtime.
"""

from __future__ import annotations

import json
import platform
import random
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import structlog

from axquant.errors import BackendUnavailableError, BenchmarkError, InvariantViolationError
from axquant.inspector import inspect_model
from axquant.schema import (
    BenchmarkConfig,
    BenchmarkResult,
    EvaluationBundle,
    HardwareMetrics,
    IntegrityMetrics,
    MtpAbComparison,
    MtpDiagnosticReport,
    MtpMetrics,
    MtpPhaseTimingSummary,
    MtpTrialComparison,
    QualityMetrics,
    SoftwareVersions,
    TrialResult,
)
from axquant.serde import stable_sha256, write_data
from axquant.versioning import collect_versions, standalone_executable_version

log = structlog.get_logger()

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )


def _resolve_executable(executable: str) -> str | None:
    path = Path(executable).expanduser()
    if path.parent != Path("."):
        return str(path.resolve()) if path.is_file() else None
    return shutil.which(executable)


def _standalone_ax_engine_version(executable: str) -> str | None:
    """Compatibility wrapper around the shared standalone version probe."""
    return standalone_executable_version(executable)


def _ax_engine_version(executable: str) -> str | None:
    try:
        completed = subprocess.run(
            [executable, "doctor", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        report = json.loads(completed.stdout) if completed.returncode == 0 else {}
        install = report.get("install", {}) if isinstance(report, dict) else {}
        version = install.get("version") if isinstance(install, dict) else None
        if version is not None:
            return str(version)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return standalone_executable_version(executable)


def _sysctl_value(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _percentile(values: list[float], pct: float) -> float | None:
    """Compute the given percentile from a sorted list of values."""
    if not values:
        return None
    ordered = sorted(values)
    index = (pct / 100.0) * (len(ordered) - 1)
    lower = int(index)
    upper = lower + 1
    if upper >= len(ordered):
        return ordered[-1]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _adjacent_token_repetition_rate(trials: Sequence[TrialResult]) -> float | None:
    """Measure adjacent-token repetition across complete generated outputs.

    This intentionally uses only emitted token IDs, so the metric is reproducible
    from the persisted raw benchmark result and does not depend on decoded text.
    """
    repeated = 0
    comparisons = 0
    for trial in trials:
        tokens = trial.output_token_ids
        if len(tokens) < 2:
            continue
        repeated += sum(left == right for left, right in pairwise(tokens))
        comparisons += len(tokens) - 1
    return repeated / comparisons if comparisons else None


def validate_ab_invariant(
    direct_config: BenchmarkConfig,
    mtp_config: BenchmarkConfig,
) -> None:
    """Validate that two benchmark configs satisfy the MTP A/B invariant.

    The MTP speed comparison MUST use:
    - the identical checkpoint (model identity)
    - the identical runtime
    - the identical prompt set and dataset digest
    - the identical workload and generation controls
    - MTP disabled for direct, enabled for candidate
    """
    if direct_config.model != mtp_config.model:
        raise InvariantViolationError(
            "A/B invariant violated: model identity differs between direct and MTP runs"
        )
    if direct_config.runtime != mtp_config.runtime:
        raise InvariantViolationError(
            "A/B invariant violated: runtime differs between direct and MTP runs"
        )
    if direct_config.dataset_sha256 != mtp_config.dataset_sha256:
        raise InvariantViolationError(
            "A/B invariant violated: dataset digest differs between direct and MTP runs"
        )
    if direct_config.workload != mtp_config.workload:
        raise InvariantViolationError(
            "A/B invariant violated: workload differs between direct and MTP runs"
        )
    if direct_config.random_seed != mtp_config.random_seed:
        raise InvariantViolationError(
            "A/B invariant violated: random seed differs between direct and MTP runs"
        )
    if direct_config.temperature != mtp_config.temperature:
        raise InvariantViolationError(
            "A/B invariant violated: temperature differs between direct and MTP runs"
        )
    if direct_config.max_tokens != mtp_config.max_tokens:
        raise InvariantViolationError(
            "A/B invariant violated: max_tokens differs between direct and MTP runs"
        )
    for field_name in (
        "top_p",
        "top_k",
        "prompt_count",
        "warmup_trials",
        "measured_trials",
        "timeout_seconds",
        "draft_depth",
        "power_mode",
        "quantizer",
        "quantizer_version",
        "runtime_env",
    ):
        if getattr(direct_config, field_name) != getattr(mtp_config, field_name):
            raise InvariantViolationError(
                f"A/B invariant violated: {field_name} differs between direct and MTP runs"
            )
    if direct_config.mtp_enabled:
        raise InvariantViolationError(
            "A/B invariant violated: direct config must have MTP disabled"
        )
    if not mtp_config.mtp_enabled:
        raise InvariantViolationError("A/B invariant violated: MTP config must have MTP enabled")


def _load_prompts(dataset_path: Path, seed: int) -> list[str]:
    """Load and deterministically shuffle prompts from a JSONL dataset."""
    prompts: list[str] = []
    try:
        with dataset_path.open(encoding="utf-8") as source:
            for line in source:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if isinstance(record, dict):
                    text = record.get("prompt") or record.get("text") or record.get("content", "")
                    if isinstance(text, str) and text:
                        prompts.append(text)
                elif isinstance(record, str) and record:
                    prompts.append(record)
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot load prompt dataset {dataset_path}: {exc}") from exc
    if not prompts:
        raise BenchmarkError(f"prompt dataset contains no valid prompts: {dataset_path}")
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def _tokenize_prompts(config: BenchmarkConfig, prompts: list[str]) -> list[list[int]]:
    source = config.model.local_path or config.model.model_id
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise BackendUnavailableError(
            "AX Engine prompt tokenization requires transformers; install axquant[mlx]"
        ) from None
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if config.model.local_path is not None:
        kwargs["local_files_only"] = True
    elif config.model.revision is not None:
        kwargs["revision"] = config.model.revision
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        return [
            [int(token) for token in tokenizer.encode(prompt, add_special_tokens=True)]
            for prompt in prompts
        ]
    except (OSError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"cannot tokenize benchmark prompts with {source}: {exc}") from exc


# Documented M2 diagnostic profiles for Qwen linear-attention exactness kill switches.
# Values match the AX Engine 6.11.1 investigation under .internal/tmp/.
MTP_DIAGNOSTIC_PROFILES: dict[str, dict[str, str]] = {
    "baseline": {},
    "disable-post-input-metal": {
        "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL": "0",
    },
    "disable-la-decode-metal": {
        "AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL": "0",
        "AX_MLX_QWEN_GATED_DELTA_DECODE_METAL": "0",
    },
}

# The complete Qwen 3.6 exact-MTP measurement contract used by the formal M5
# suite. The exact flag alone is NOT the contract on AX Engine 6.12.x: without
# the invariant-projection/row-exact/split-FFN companions the verifier falls
# off the validated graph and every cycle pays a many-fold rollback and
# verify-eval penalty (measured 2026-08-01: 0.8925x misconfigured versus
# 1.0969x under the full contract on the same artifact/binary/host).
QWEN36_EXACT_MTP_PROFILE_ENV: dict[str, str] = {
    "AX_MLX_QWEN_LINEAR_MTP_EXACT": "1",
    "AX_MLX_MTP_BYPASS_MIN_SAMPLES": "1000",
    "AX_MLX_MTP_DRAFT_MIN_CONFIDENCE": "0",
    "AX_MLX_MTP_LINEAR_EXACT_REPLAY": "0",
    "AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL": "0",
    "AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS": "0",
    "AX_MLX_SPECULATIVE_INVARIANT_PROJECTIONS": "all",
    "AX_MLX_SPECULATIVE_ROW_EXACT_POST_INPUT": "1",
    "AX_MLX_SPECULATIVE_SPLIT_FFN": "1",
}


def _runtime_environment(config: BenchmarkConfig) -> list[str]:
    """Build the ordered env assignments passed to AX Engine via /usr/bin/env.

    Caller-supplied runtime_env is applied after harness controls so explicit
    AX_MLX_MTP_MAX_DEPTH can override draft_depth when both are present.
    """
    environment: list[str] = []
    if not config.mtp_enabled:
        environment.append("AX_NO_SPEC=1")
    if config.draft_depth is not None:
        environment.append(f"AX_MLX_MTP_MAX_DEPTH={config.draft_depth}")
    for key, value in sorted(config.runtime_env.items()):
        if key == "AX_MLX_MTP_MAX_DEPTH" and config.draft_depth is not None:
            # Prefer the explicit runtime_env value when both are set.
            environment = [
                item for item in environment if not item.startswith("AX_MLX_MTP_MAX_DEPTH=")
            ]
        environment.append(f"{key}={value}")
    return environment


def _kernel_fallback_count(decisions: Mapping[str, Any]) -> int:
    """Sum AX MLX kernel fallback counters without counting policy fallback steps."""
    return sum(
        int(value)
        for key, value in decisions.items()
        if key.startswith("ax_mlx_") and key.endswith("_fallbacks") and isinstance(value, int)
    )


def parse_runtime_env_items(items: Sequence[str] | None) -> dict[str, str]:
    """Parse CLI KEY=VALUE items into a sorted runtime_env mapping."""
    if not items:
        return {}
    parsed: dict[str, str] = {}
    for item in items:
        text = str(item).strip()
        if not text or "=" not in text:
            raise BenchmarkError(f"runtime env must be KEY=VALUE, got {item!r}")
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise BenchmarkError(f"runtime env must be KEY=VALUE, got {item!r}")
        parsed[key] = value
    # Validate through BenchmarkConfig's allowlist by constructing a throwaway field path.
    return BenchmarkConfig.model_validate(
        {
            "model": {"model_id": "env-check"},
            "baseline_kind": "candidate",
            "workload": "env-check",
            "dataset_sha256": "0" * 64,
            "prompt_count": 1,
            "runtime_env": parsed,
        }
    ).runtime_env


def resolve_diagnostic_profiles(
    names: Sequence[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Resolve named diagnostic profiles; default is the full documented kill-switch matrix."""
    selected = list(names) if names else list(MTP_DIAGNOSTIC_PROFILES)
    resolved: dict[str, dict[str, str]] = {}
    for name in selected:
        key = name.strip()
        if key not in MTP_DIAGNOSTIC_PROFILES:
            known = ", ".join(sorted(MTP_DIAGNOSTIC_PROFILES))
            raise BenchmarkError(f"unknown MTP diagnostic profile {name!r}; known: {known}")
        resolved[key] = dict(MTP_DIAGNOSTIC_PROFILES[key])
    return resolved


def _run_single_trial(
    config: BenchmarkConfig,
    prompt: str,
    trial_index: int,
    is_warmup: bool,
    *,
    executable: str,
    runner: CommandRunner,
    prompt_tokens: list[int] | None = None,
) -> tuple[TrialResult, bool]:
    """Execute a single benchmark trial via AX Engine subprocess.

    Returns (trial_result, is_timeout).
    """
    resolved = _resolve_executable(executable)
    if resolved is None:
        raise BackendUnavailableError(f"AX Engine executable not found: {executable}")

    using_real_runner = runner is _run
    if using_real_runner and config.model.local_path is None:
        raise BenchmarkError("AX Engine MLX benchmarking requires a local model artifact path")
    if using_real_runner and config.temperature != 0.0:
        raise BenchmarkError("AX Engine deterministic MLX benchmarking requires temperature 0")
    if using_real_runner and (config.top_p != 1.0 or config.top_k != 0):
        raise BenchmarkError("AX Engine MLX benchmark does not expose top-p/top-k controls")
    command = [
        resolved,
        "generate",
        "--tokens" if prompt_tokens is not None else "--prompt",
        ",".join(str(token) for token in prompt_tokens) if prompt_tokens is not None else prompt,
        "--max-output-tokens",
        str(config.max_tokens),
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(config.model.local_path or config.model.model_id),
        "--json",
    ]
    if using_real_runner:
        environment = _runtime_environment(config)
        command = ["/usr/bin/time", "-l", "/usr/bin/env", *environment, *command]

    start = time.monotonic()
    try:
        if using_real_runner:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
            )
        else:
            completed = runner(command)
    except subprocess.TimeoutExpired:
        return TrialResult(
            trial_index=trial_index,
            is_warmup=is_warmup,
            success=False,
            command=command,
            latency_seconds=time.monotonic() - start,
            error=f"request timeout exceeded {config.timeout_seconds} seconds",
        ), True
    except OSError as exc:
        return TrialResult(
            trial_index=trial_index,
            is_warmup=is_warmup,
            success=False,
            command=command,
            error=f"subprocess error: {exc}",
        ), False
    elapsed = time.monotonic() - start

    if completed.returncode != 0:
        is_timeout = "timeout" in completed.stderr.lower() if completed.stderr else False
        return TrialResult(
            trial_index=trial_index,
            is_warmup=is_warmup,
            success=False,
            command=command,
            latency_seconds=elapsed,
            error=(
                completed.stderr[:500] if completed.stderr else f"exit code {completed.returncode}"
            ),
        ), is_timeout

    report: dict[str, Any] = {}
    if completed.stdout.strip():
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError:
            report = {}

    performance = report.get("performance")
    performance = performance if isinstance(performance, dict) else {}
    mtp_report = performance.get("mtp")
    mtp_report = mtp_report if isinstance(mtp_report, dict) else {}
    route = report.get("route")
    route = route if isinstance(route, dict) else {}
    decisions = route.get("crossover_decisions")
    decisions = decisions if isinstance(decisions, dict) else {}
    runtime = report.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    host = runtime.get("host")
    host = host if isinstance(host, dict) else {}

    output_token_ids = [
        int(token) for token in report.get("output_tokens", []) if isinstance(token, int)
    ]
    tokens_generated = int(report.get("tokens_generated", len(output_token_ids)))
    generation_us = performance.get("generation_time_us")
    decode_seconds = (
        float(generation_us) / 1_000_000
        if isinstance(generation_us, int | float)
        else float(report.get("decode_seconds", elapsed))
    )
    tps = tokens_generated / decode_seconds if decode_seconds > 0 else 0.0
    accepted_tokens = mtp_report.get("accepted_tokens", report.get("mtp_accepted_tokens"))
    proposed_tokens = mtp_report.get("draft_tokens", report.get("mtp_proposed_tokens"))
    peak_match = re.search(r"\b(\d+)\s+maximum resident set size", completed.stderr)
    peak_memory_bytes = (
        int(peak_match.group(1)) if peak_match is not None else report.get("peak_memory_bytes")
    )
    verification_us = (
        int(decisions.get("ax_mtp_verify_forward_wall_us", 0))
        + int(decisions.get("ax_mtp_verify_eval_wall_us", 0))
        + int(decisions.get("ax_mtp_accept_wall_us", 0))
        + int(decisions.get("ax_mtp_rollback_wall_us", 0))
    )
    device_name = host.get("device_class") or host.get("model")
    if device_name is None and using_real_runner:
        device_name = _sysctl_value("hw.model")
    unified_memory = host.get("unified_memory_bytes") or host.get("memory_bytes")
    if unified_memory is None and using_real_runner:
        unified_memory = _sysctl_value("hw.memsize")

    return TrialResult(
        trial_index=trial_index,
        is_warmup=is_warmup,
        success=True,
        command=command,
        prompt_tokens=len(report.get("prompt_tokens", prompt_tokens or [])),
        tokens_generated=tokens_generated,
        output_token_ids=output_token_ids,
        output_sha256=stable_sha256(output_token_ids) if output_token_ids else None,
        latency_seconds=elapsed,
        time_to_first_token_seconds=(
            float(performance["time_to_first_token_us"]) / 1_000_000
            if "time_to_first_token_us" in performance
            else None
        ),
        prefill_seconds=(
            float(performance["prompt_eval_time_us"]) / 1_000_000
            if "prompt_eval_time_us" in performance
            else report.get("prefill_seconds")
        ),
        decode_seconds=decode_seconds,
        tokens_per_second=tps,
        mtp_accepted_tokens=int(accepted_tokens) if accepted_tokens is not None else None,
        mtp_proposed_tokens=int(proposed_tokens) if proposed_tokens is not None else None,
        mtp_rejected_tokens=(
            max(0, int(proposed_tokens) - int(accepted_tokens))
            if proposed_tokens is not None and accepted_tokens is not None
            else None
        ),
        mtp_decode_steps=(
            int(mtp_report["decode_steps"]) if "decode_steps" in mtp_report else None
        ),
        mtp_active=bool(mtp_report["active"]) if "active" in mtp_report else None,
        verification_overhead_seconds=verification_us / 1_000_000,
        kernel_fallbacks=_kernel_fallback_count(decisions),
        peak_memory_bytes=peak_memory_bytes,
        runtime_device_name=str(device_name) if device_name is not None else None,
        runtime_chip=str(host["detected_soc"]) if "detected_soc" in host else None,
        unified_memory_bytes=int(unified_memory) if unified_memory is not None else None,
        os_version=platform.platform(),
        terminal_stop_reason=report.get("finish_reason"),
        backend_report=report,
        backend_stderr=completed.stderr[-4000:],
    ), False


def run_benchmark(
    config: BenchmarkConfig,
    *,
    dataset_path: str | Path,
    executable: str = "ax-engine-bench",
    runner: CommandRunner = _run,
    output_dir: str | Path | None = None,
) -> BenchmarkResult:
    """Run a complete benchmark evaluation and return the result.

    Executes warmup + measured trials, computes latency distributions,
    and optionally persists raw logs to output_dir.
    """
    prompts = _load_prompts(Path(dataset_path).expanduser().resolve(), config.random_seed)
    if len(prompts) < config.prompt_count:
        raise BenchmarkError(
            f"dataset has {len(prompts)} prompts but config requires {config.prompt_count}"
        )
    selected_prompts = prompts[: config.prompt_count]
    tokenized_prompts: list[list[int] | None]
    if runner is _run:
        tokenized_prompts = [
            list(token_ids) for token_ids in _tokenize_prompts(config, selected_prompts)
        ]
    else:
        tokenized_prompts = [None] * len(selected_prompts)

    resolved = _resolve_executable(executable)
    if resolved is None:
        raise BackendUnavailableError(f"AX Engine executable not found: {executable}")

    total_trials = config.warmup_trials + config.measured_trials
    trials: list[TrialResult] = []
    failed_count = 0
    timed_out_count = 0

    for trial_index in range(total_trials):
        is_warmup = trial_index < config.warmup_trials
        prompt = selected_prompts[trial_index % len(selected_prompts)]
        prompt_tokens = tokenized_prompts[trial_index % len(selected_prompts)]
        result, is_timeout = _run_single_trial(
            config,
            prompt,
            trial_index,
            is_warmup,
            executable=executable,
            runner=runner,
            prompt_tokens=prompt_tokens,
        )
        if not result.success:
            failed_count += 1
            if is_timeout:
                timed_out_count += 1
        trials.append(result)

    # Compute distributions from measured (non-warmup, successful) trials
    measured_trials = [t for t in trials if not t.is_warmup and t.success]
    latencies = [t.latency_seconds for t in measured_trials]
    tps_values = [t.tokens_per_second for t in measured_trials if t.tokens_per_second > 0]

    benchmark_result = BenchmarkResult(
        config=config,
        trials=trials,
        measured_count=len(measured_trials),
        failed_count=failed_count,
        timed_out_count=timed_out_count,
        latency_p50=_percentile(latencies, 50),
        latency_p90=_percentile(latencies, 90),
        latency_p99=_percentile(latencies, 99),
        tokens_per_second_p50=_percentile(tps_values, 50),
        tokens_per_second_p90=_percentile(tps_values, 90),
        tokens_per_second_p99=_percentile(tps_values, 99),
        runtime_device_name=next(
            (trial.runtime_device_name for trial in measured_trials if trial.runtime_device_name),
            None,
        ),
        runtime_chip=next(
            (trial.runtime_chip for trial in measured_trials if trial.runtime_chip),
            None,
        ),
        unified_memory_bytes=next(
            (
                trial.unified_memory_bytes
                for trial in measured_trials
                if trial.unified_memory_bytes is not None
            ),
            None,
        ),
        os_version=next(
            (trial.os_version for trial in measured_trials if trial.os_version),
            platform.platform(),
        ),
        ax_engine_version=_ax_engine_version(resolved) if runner is _run else None,
    )

    # Persist raw logs if output_dir specified
    if output_dir is not None:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "benchmark_raw_log.json"
        write_data(log_path, benchmark_result)
        log.info("benchmark_log_saved", path=str(log_path))

    return benchmark_result


def result_to_evaluation_bundle(
    result: BenchmarkResult,
    *,
    software_versions: SoftwareVersions | None = None,
) -> EvaluationBundle:
    """Convert a BenchmarkResult into a strict EvaluationBundle."""
    config = result.config
    measured = [t for t in result.trials if not t.is_warmup and t.success]
    if not measured:
        raise BenchmarkError("benchmark produced no successful measured trials")

    # Aggregate MTP metrics
    mtp_metrics: MtpMetrics | None = None
    if config.mtp_enabled and measured:
        total_accepted = sum(t.mtp_accepted_tokens or 0 for t in measured)
        total_proposed = sum(t.mtp_proposed_tokens or 0 for t in measured)
        acceptance_rate = total_accepted / total_proposed if total_proposed > 0 else None
        avg_accepted = total_accepted / len(measured) if measured else None
        total_decode_steps = sum(t.mtp_decode_steps or 0 for t in measured)
        effective_tpf = (
            sum(t.tokens_generated for t in measured) / total_decode_steps
            if total_decode_steps > 0
            else None
        )
        mtp_metrics = MtpMetrics(
            token_accuracy=(
                {"1": acceptance_rate}
                if config.draft_depth == 1 and acceptance_rate is not None
                else {}
            ),
            average_accepted_tokens=avg_accepted,
            acceptance_rate=acceptance_rate,
            rejection_rate=(1.0 - acceptance_rate) if acceptance_rate is not None else None,
            effective_tokens_per_forward=effective_tpf,
            repetition_rate=_adjacent_token_repetition_rate(measured),
        )

    # Aggregate hardware metrics
    peak_memories = [t.peak_memory_bytes for t in measured if t.peak_memory_bytes is not None]
    decode_tps = [t.tokens_per_second for t in measured if t.tokens_per_second > 0]
    total_prefill_tokens = sum(t.prompt_tokens for t in measured)
    total_prefill_seconds = sum(t.prefill_seconds or 0.0 for t in measured)
    kernel_fallbacks = sum(t.kernel_fallbacks or 0 for t in measured)

    hardware = HardwareMetrics(
        peak_memory_bytes=max(peak_memories) if peak_memories else None,
        prefill_tokens_per_second=(
            total_prefill_tokens / total_prefill_seconds if total_prefill_seconds > 0 else None
        ),
        decode_tokens_per_second=_percentile(decode_tps, 50),
        mtp_effective_tokens_per_second=(
            _percentile(decode_tps, 50) if config.mtp_enabled else None
        ),
        kernel_fallbacks=kernel_fallbacks,
        device_name=result.runtime_device_name,
        chip=result.runtime_chip,
        unified_memory_bytes=result.unified_memory_bytes,
        os_version=result.os_version,
    )

    versions = software_versions or collect_versions()
    if result.ax_engine_version is not None:
        versions = versions.model_copy(update={"ax_engine": result.ax_engine_version})
    safetensors_valid = False
    index_complete = False
    config_valid = False
    mtp_layout_valid: bool | None = None
    if config.model.local_path is not None:
        model_path = Path(config.model.local_path)
        try:
            inventory = inspect_model(
                model_path,
                model_id=config.model.model_id,
                revision=config.model.revision,
                allow_quantized=True,
            )
            safetensors_valid = True
            index_complete = True
            config_valid = True
            mtp_layout_valid = inventory.mtp_present if config.mtp_enabled else None
        except Exception as exc:
            log.warning("benchmark_integrity_check_failed", error=str(exc))

    return EvaluationBundle(
        model=config.model,
        runtime=config.runtime,
        mtp_enabled=config.mtp_enabled,
        baseline_kind=config.baseline_kind,
        quality=QualityMetrics(),
        mtp=mtp_metrics,
        hardware=hardware,
        integrity=IntegrityMetrics(
            safetensors_valid=safetensors_valid,
            index_complete=index_complete,
            config_valid=config_valid,
            mtp_layout_valid=mtp_layout_valid,
            source_revision_pinned=config.model.revision is not None,
        ),
        workload=config.workload,
        dataset_sha256=config.dataset_sha256,
        software_versions=versions,
        random_seed=config.random_seed,
        benchmark_metadata={
            "prompt_count": config.prompt_count,
            "warmup_trials": config.warmup_trials,
            "measured_trials": config.measured_trials,
            "successful_measured_trials": result.measured_count,
            "failed_trials": result.failed_count,
            "timed_out_trials": result.timed_out_count,
            "latency_p50": result.latency_p50,
            "latency_p90": result.latency_p90,
            "latency_p99": result.latency_p99,
            "tokens_per_second_p50": result.tokens_per_second_p50,
            "tokens_per_second_p90": result.tokens_per_second_p90,
            "tokens_per_second_p99": result.tokens_per_second_p99,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_tokens": config.max_tokens,
            "draft_depth": config.draft_depth,
            "power_mode": config.power_mode,
            "quantizer": config.quantizer,
            "quantizer_version": config.quantizer_version,
            "runtime_env": dict(config.runtime_env),
            "ax_engine_version": result.ax_engine_version,
            "mtp_metrics_protocol": (
                "adjacent-token-repeat-v1;depth1-proposal-accuracy-v1"
                if config.mtp_enabled
                else None
            ),
        },
    )


def _first_diff_index(left: list[int], right: list[int]) -> int | None:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    if len(left) != len(right):
        return limit
    return None


_MTP_PHASE_DECISION_KEYS: dict[str, str] = {
    "draft_wall_us": "ax_mtp_draft_wall_us",
    "verify_forward_wall_us": "ax_mtp_verify_forward_wall_us",
    "verify_eval_wall_us": "ax_mtp_verify_eval_wall_us",
    "rollback_wall_us": "ax_mtp_rollback_wall_us",
    "cache_clone_wall_us": "ax_mtp_cache_clone_wall_us",
    "accept_wall_us": "ax_mtp_accept_wall_us",
    "tail_sample_wall_us": "ax_mtp_tail_sample_wall_us",
    "mtp_decode_steps": "ax_mtp_decode_steps",
    "direct_fallback_steps": "ax_mtp_direct_fallback_steps",
    "mtp_emitted_tokens": "ax_mtp_emitted_tokens",
    "correctness_mode_conflicts": "ax_mtp_correctness_mode_conflicts",
}


def _trial_route_decisions(trial: TrialResult) -> Mapping[str, Any]:
    route = trial.backend_report.get("route")
    if not isinstance(route, Mapping):
        return {}
    decisions = route.get("crossover_decisions")
    return decisions if isinstance(decisions, Mapping) else {}


def _nonnegative_counter(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _trial_generation_wall_us(trial: TrialResult) -> int:
    performance = trial.backend_report.get("performance")
    if isinstance(performance, Mapping):
        value = performance.get("generation_time_us")
        if not isinstance(value, bool) and isinstance(value, int | float) and value >= 0:
            return int(value)
    if trial.decode_seconds is not None:
        return max(0, round(trial.decode_seconds * 1_000_000))
    return 0


def _trial_output_tokens(trial: TrialResult) -> int:
    return max(trial.tokens_generated, len(trial.output_token_ids))


def _mtp_phase_timing_summary(
    direct_trials: Sequence[TrialResult],
    mtp_trials: Sequence[TrialResult],
    minimum_speedup: float,
) -> MtpPhaseTimingSummary | None:
    mtp_decisions = [_trial_route_decisions(trial) for trial in mtp_trials]
    decision_names = set(_MTP_PHASE_DECISION_KEYS.values())
    if not any(any(name in decisions for name in decision_names) for decisions in mtp_decisions):
        return None
    direct_output_tokens = sum(_trial_output_tokens(trial) for trial in direct_trials)
    mtp_output_tokens = sum(_trial_output_tokens(trial) for trial in mtp_trials)
    if direct_output_tokens <= 0 or mtp_output_tokens <= 0:
        return None
    direct_generation_wall_us = sum(_trial_generation_wall_us(trial) for trial in direct_trials)
    mtp_generation_wall_us = sum(_trial_generation_wall_us(trial) for trial in mtp_trials)
    counters = {
        field: sum(_nonnegative_counter(decisions.get(name)) for decisions in mtp_decisions)
        for field, name in _MTP_PHASE_DECISION_KEYS.items()
    }
    proposed_tokens = sum(trial.mtp_proposed_tokens or 0 for trial in mtp_trials)
    accepted_tokens = sum(trial.mtp_accepted_tokens or 0 for trial in mtp_trials)
    direct_us_per_token = direct_generation_wall_us / direct_output_tokens
    mtp_us_per_token = mtp_generation_wall_us / mtp_output_tokens
    target_us_per_token = (
        direct_us_per_token / minimum_speedup if minimum_speedup > 0 else mtp_us_per_token
    )
    phase_wall_us = sum(
        counters[field]
        for field in (
            "draft_wall_us",
            "verify_forward_wall_us",
            "verify_eval_wall_us",
            "rollback_wall_us",
            "cache_clone_wall_us",
            "accept_wall_us",
            "tail_sample_wall_us",
        )
    )
    routed_steps = counters["mtp_decode_steps"] + counters["direct_fallback_steps"]
    return MtpPhaseTimingSummary(
        direct_output_tokens=direct_output_tokens,
        mtp_output_tokens=mtp_output_tokens,
        direct_generation_wall_us=direct_generation_wall_us,
        mtp_generation_wall_us=mtp_generation_wall_us,
        direct_generation_us_per_output_token=direct_us_per_token,
        mtp_generation_us_per_output_token=mtp_us_per_token,
        target_mtp_us_per_output_token=target_us_per_token,
        required_savings_us_per_output_token=max(0.0, mtp_us_per_token - target_us_per_token),
        draft_wall_us=counters["draft_wall_us"],
        verify_forward_wall_us=counters["verify_forward_wall_us"],
        verify_eval_wall_us=counters["verify_eval_wall_us"],
        rollback_wall_us=counters["rollback_wall_us"],
        cache_clone_wall_us=counters["cache_clone_wall_us"],
        accept_wall_us=counters["accept_wall_us"],
        tail_sample_wall_us=counters["tail_sample_wall_us"],
        phase_accounted_us_per_output_token=phase_wall_us / mtp_output_tokens,
        mtp_decode_steps=counters["mtp_decode_steps"],
        direct_fallback_steps=counters["direct_fallback_steps"],
        active_step_fraction=(
            counters["mtp_decode_steps"] / routed_steps if routed_steps > 0 else None
        ),
        mtp_emitted_tokens=counters["mtp_emitted_tokens"],
        proposed_tokens=proposed_tokens,
        accepted_tokens=accepted_tokens,
        correctness_mode_conflicts=counters["correctness_mode_conflicts"],
    )


def compare_mtp_ab_results(
    direct_result: BenchmarkResult,
    mtp_result: BenchmarkResult,
    *,
    profile_name: str = "baseline",
    minimum_speedup: float = 1.20,
) -> MtpAbComparison:
    """Soft-compare MTP-off/on results without raising on greedy divergence."""
    issues: list[str] = []
    if direct_result.config.runtime_env != mtp_result.config.runtime_env:
        issues.append("runtime_env differs between direct and MTP results")
    if direct_result.failed_count or mtp_result.failed_count:
        issues.append("one or more trials failed or timed out")
    if (
        direct_result.runtime_chip
        and mtp_result.runtime_chip
        and direct_result.runtime_chip != mtp_result.runtime_chip
    ):
        issues.append("hardware chip differs between direct and MTP results")
    if (
        direct_result.ax_engine_version
        and mtp_result.ax_engine_version
        and direct_result.ax_engine_version != mtp_result.ax_engine_version
    ):
        issues.append("AX Engine version differs between direct and MTP results")

    direct_measured = {
        trial.trial_index: trial
        for trial in direct_result.trials
        if not trial.is_warmup and trial.success
    }
    mtp_measured = {
        trial.trial_index: trial
        for trial in mtp_result.trials
        if not trial.is_warmup and trial.success
    }
    if direct_measured.keys() != mtp_measured.keys():
        issues.append("measured trial sets differ between direct and MTP results")

    trial_comparisons: list[MtpTrialComparison] = []
    divergent = 0
    for trial_index in sorted(set(direct_measured) | set(mtp_measured)):
        direct_trial = direct_measured.get(trial_index)
        mtp_trial = mtp_measured.get(trial_index)
        if direct_trial is None or mtp_trial is None:
            divergent += 1
            trial_comparisons.append(
                MtpTrialComparison(
                    trial_index=trial_index,
                    outputs_equal=False,
                    first_diff_index=None,
                    direct_output_sha256=(
                        direct_trial.output_sha256 if direct_trial is not None else None
                    ),
                    mtp_output_sha256=mtp_trial.output_sha256 if mtp_trial is not None else None,
                    direct_token_count=(
                        len(direct_trial.output_token_ids) if direct_trial is not None else 0
                    ),
                    mtp_token_count=len(mtp_trial.output_token_ids) if mtp_trial is not None else 0,
                    mtp_proposed_tokens=(
                        mtp_trial.mtp_proposed_tokens if mtp_trial is not None else None
                    ),
                    mtp_accepted_tokens=(
                        mtp_trial.mtp_accepted_tokens if mtp_trial is not None else None
                    ),
                    mtp_rejected_tokens=(
                        mtp_trial.mtp_rejected_tokens if mtp_trial is not None else None
                    ),
                    mtp_active=mtp_trial.mtp_active if mtp_trial is not None else None,
                    direct_tokens_per_second=(
                        direct_trial.tokens_per_second if direct_trial is not None else None
                    ),
                    mtp_tokens_per_second=(
                        mtp_trial.tokens_per_second if mtp_trial is not None else None
                    ),
                )
            )
            continue
        if mtp_trial.mtp_active is False:
            issues.append(f"MTP was not active for measured trial {trial_index}")
        direct_tokens = list(direct_trial.output_token_ids)
        mtp_tokens = list(mtp_trial.output_token_ids)
        if not direct_tokens and not mtp_tokens:
            # Unit-test runners may omit tokens; treat as non-comparable success.
            equal = True
            first_diff: int | None = None
        elif not direct_tokens or not mtp_tokens:
            equal = False
            first_diff = 0
            divergent += 1
            issues.append(f"missing output tokens for trial {trial_index}")
        else:
            equal = direct_tokens == mtp_tokens
            first_diff = None if equal else _first_diff_index(direct_tokens, mtp_tokens)
            if not equal:
                divergent += 1
                issues.append(f"greedy outputs differ for trial {trial_index}")
        trial_comparisons.append(
            MtpTrialComparison(
                trial_index=trial_index,
                outputs_equal=equal,
                first_diff_index=first_diff,
                direct_output_sha256=direct_trial.output_sha256,
                mtp_output_sha256=mtp_trial.output_sha256,
                direct_token_count=len(direct_tokens),
                mtp_token_count=len(mtp_tokens),
                mtp_proposed_tokens=mtp_trial.mtp_proposed_tokens,
                mtp_accepted_tokens=mtp_trial.mtp_accepted_tokens,
                mtp_rejected_tokens=mtp_trial.mtp_rejected_tokens,
                mtp_active=mtp_trial.mtp_active,
                direct_tokens_per_second=direct_trial.tokens_per_second,
                mtp_tokens_per_second=mtp_trial.tokens_per_second,
            )
        )

    direct_tps = direct_result.tokens_per_second_p50
    mtp_tps = mtp_result.tokens_per_second_p50
    speedup: float | None = None
    if direct_tps is not None and mtp_tps is not None and direct_tps > 0:
        speedup = mtp_tps / direct_tps
    exactness_pass = divergent == 0 and not any(
        issue.startswith("one or more trials failed")
        or issue.startswith("measured trial sets differ")
        or issue.startswith("MTP was not active")
        for issue in issues
    )
    speedup_pass = speedup is not None and speedup >= minimum_speedup
    release_ready = (
        exactness_pass
        and speedup_pass
        and not any(
            issue.startswith("hardware chip differs")
            or issue.startswith("AX Engine version differs")
            or issue.startswith("runtime_env differs")
            for issue in issues
        )
    )
    phase_timing = _mtp_phase_timing_summary(
        list(direct_measured.values()),
        list(mtp_measured.values()),
        minimum_speedup,
    )
    return MtpAbComparison(
        profile_name=profile_name,
        runtime_env=dict(direct_result.config.runtime_env),
        draft_depth=direct_result.config.draft_depth,
        exactness_pass=exactness_pass,
        divergent_trial_count=divergent,
        measured_trial_count=len(direct_measured),
        failed_trial_count=direct_result.failed_count + mtp_result.failed_count,
        direct_tokens_per_second_p50=direct_tps,
        mtp_tokens_per_second_p50=mtp_tps,
        speedup=speedup,
        minimum_speedup=minimum_speedup,
        speedup_pass=speedup_pass,
        release_ready=release_ready,
        ax_engine_version=direct_result.ax_engine_version or mtp_result.ax_engine_version,
        runtime_chip=direct_result.runtime_chip or mtp_result.runtime_chip,
        phase_timing=phase_timing,
        trial_comparisons=trial_comparisons,
        issues=issues,
    )


def run_mtp_ab(
    config_direct: BenchmarkConfig,
    config_mtp: BenchmarkConfig,
    *,
    dataset_path: str | Path,
    executable: str = "ax-engine-bench",
    runner: CommandRunner = _run,
    output_dir: str | Path | None = None,
    enforce_exactness: bool = True,
    enforce_speedup: bool = True,
    minimum_speedup: float | None = None,
) -> tuple[EvaluationBundle, EvaluationBundle]:
    """Run MTP-off and MTP-on benchmarks with A/B invariant validation.

    When ``enforce_exactness`` is true (default), greedy token identity is required.
    Set it false for diagnostic sweeps that must record divergence without aborting
    the overall matrix early; use :func:`compare_mtp_ab_results` or
    :func:`run_mtp_diagnostics` for structured reports. Set ``enforce_speedup`` false
    to retain complete evaluation evidence for a measured speed-gate failure; exactness
    and all other A/B invariants remain enforced.
    """
    validate_ab_invariant(config_direct, config_mtp)

    direct_result = run_benchmark(
        config_direct,
        dataset_path=dataset_path,
        executable=executable,
        runner=runner,
        output_dir=Path(output_dir) / "mtp-off" if output_dir else None,
    )
    mtp_result = run_benchmark(
        config_mtp,
        dataset_path=dataset_path,
        executable=executable,
        runner=runner,
        output_dir=Path(output_dir) / "mtp-on" if output_dir else None,
    )
    comparison = compare_mtp_ab_results(
        direct_result,
        mtp_result,
        profile_name="benchmark-ab",
        minimum_speedup=minimum_speedup if minimum_speedup is not None else 0.0,
    )
    if output_dir is not None:
        write_data(Path(output_dir) / "mtp_ab_comparison.json", comparison)
    if direct_result.failed_count or mtp_result.failed_count:
        raise BenchmarkError("MTP A/B evidence contains failed or timed-out trials")
    if enforce_exactness:
        for issue in comparison.issues:
            if issue.startswith("greedy outputs differ"):
                trial_index = int(issue.rsplit(" ", 1)[-1])
                raise InvariantViolationError(
                    f"A/B invariant violated: greedy outputs differ for trial {trial_index}"
                )
            if issue.startswith("MTP was not active"):
                raise BenchmarkError(issue)
            if issue.startswith("measured trial sets differ"):
                raise InvariantViolationError("A/B invariant violated: measured trial sets differ")
            if issue.startswith("hardware chip differs"):
                raise InvariantViolationError("A/B invariant violated: hardware chip differs")
            if issue.startswith("AX Engine version differs"):
                raise InvariantViolationError("A/B invariant violated: AX Engine version differs")
        if enforce_speedup and minimum_speedup is not None and not comparison.speedup_pass:
            raise BenchmarkError(
                f"MTP speedup {comparison.speedup} is below required {minimum_speedup}"
            )

    direct_bundle = result_to_evaluation_bundle(direct_result)
    mtp_bundle = result_to_evaluation_bundle(mtp_result)
    if mtp_bundle.mtp is None:
        raise BenchmarkError("MTP A/B result did not produce MTP metrics")
    comparison_count = len(comparison.trial_comparisons)
    divergence_rate = (
        comparison.divergent_trial_count / comparison_count if comparison_count else None
    )
    mtp_bundle.mtp.divergence_rate = divergence_rate
    mtp_bundle.benchmark_metadata["mtp_metrics_protocol"] = (
        "adjacent-token-repeat-v1;depth1-proposal-accuracy-v1;greedy-output-ab-divergence-v1"
    )
    return direct_bundle, mtp_bundle


def run_mtp_diagnostics(
    base_config: BenchmarkConfig,
    *,
    dataset_path: str | Path,
    executable: str = "ax-engine-bench",
    runner: CommandRunner = _run,
    output_dir: str | Path | None = None,
    profiles: Sequence[str] | None = None,
    minimum_speedup: float = 1.20,
) -> MtpDiagnosticReport:
    """Run the documented M2 kill-switch matrix and emit a diagnostic report.

    Each profile is soft-compared (exactness failures are recorded, not raised) so the
    full matrix completes. Release gates remain fail-closed: ``any_release_ready`` is
    true only when a profile is both exact and meets ``minimum_speedup``.
    """
    if base_config.mtp_enabled:
        raise BenchmarkError("diagnostic base config must have mtp_enabled=false")
    resolved_profiles = resolve_diagnostic_profiles(profiles)
    comparisons: list[MtpAbComparison] = []
    out_root = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    for profile_name, runtime_env in resolved_profiles.items():
        profile_env = {**base_config.runtime_env, **runtime_env}
        direct_config = base_config.model_copy(
            update={
                "mtp_enabled": False,
                "baseline_kind": "axquant-mtp-off",
                "runtime_env": profile_env,
            }
        )
        mtp_config = base_config.model_copy(
            update={
                "mtp_enabled": True,
                "baseline_kind": "axquant-mtp-on",
                "runtime_env": profile_env,
            }
        )
        profile_dir = out_root / profile_name if out_root is not None else None
        direct_result = run_benchmark(
            direct_config,
            dataset_path=dataset_path,
            executable=executable,
            runner=runner,
            output_dir=profile_dir / "mtp-off" if profile_dir is not None else None,
        )
        mtp_result = run_benchmark(
            mtp_config,
            dataset_path=dataset_path,
            executable=executable,
            runner=runner,
            output_dir=profile_dir / "mtp-on" if profile_dir is not None else None,
        )
        comparison = compare_mtp_ab_results(
            direct_result,
            mtp_result,
            profile_name=profile_name,
            minimum_speedup=minimum_speedup,
        )
        comparisons.append(comparison)
        if profile_dir is not None:
            write_data(profile_dir / "mtp_ab_comparison.json", comparison)
            try:
                write_data(
                    profile_dir / "evaluation_mtp_off.json",
                    result_to_evaluation_bundle(direct_result),
                )
                write_data(
                    profile_dir / "evaluation_mtp_on.json",
                    result_to_evaluation_bundle(mtp_result),
                )
            except BenchmarkError as exc:
                log.warning(
                    "mtp_diagnostic_bundle_skipped",
                    profile=profile_name,
                    error=str(exc),
                )
        log.info(
            "mtp_diagnostic_profile_completed",
            profile=profile_name,
            exactness_pass=comparison.exactness_pass,
            speedup=comparison.speedup,
            release_ready=comparison.release_ready,
            issues=len(comparison.issues),
        )

    any_exact = any(item.exactness_pass for item in comparisons)
    any_ready = any(item.release_ready for item in comparisons)
    if any_ready:
        next_step = (
            "At least one kill-switch profile is exact and meets the speedup floor. "
            "Re-run full release A/B on a supported host with that runtime_env recorded, "
            "then bind the evidence into benchmark-index / validation-index."
        )
    elif any_exact:
        next_step = (
            "Exactness recovered under a kill-switch profile but speedup is still below "
            f"{minimum_speedup:.2f}x. Measure depth/host alternatives or pursue an AX Engine "
            "performance fix; do not claim MTP acceleration."
        )
    else:
        next_step = (
            "No profile restored greedy exactness. Keep fail-closed. Next: instrument the "
            "failing rejection cycle in AX Engine (direct vs verifier logits and linear-attn "
            "state checksums) or wait for a runtime fix; do not waive M2."
        )

    report = MtpDiagnosticReport(
        model=base_config.model,
        workload=base_config.workload,
        dataset_sha256=base_config.dataset_sha256,
        minimum_speedup=minimum_speedup,
        profiles=comparisons,
        any_exactness_pass=any_exact,
        any_release_ready=any_ready,
        recommended_next_step=next_step,
    )
    if out_root is not None:
        write_data(out_root / "mtp_diagnostic_report.json", report)
    return report
