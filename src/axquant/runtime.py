from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError, BackendUnavailableError
from axquant.schema import (
    AxEngineOptimizationMetadata,
    ModelIdentity,
    MtpRuntimeMetadata,
    OptimizationScope,
    QuantizationPlan,
    RuntimeCheck,
    RuntimeMetadata,
    RuntimeName,
    RuntimeProfile,
    RuntimeSupportLevel,
)

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


def _json_report(stdout: str) -> dict[str, Any]:
    if not stdout.strip():
        return {}
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return {"raw_stdout": stdout.strip()}
    return value if isinstance(value, dict) else {"value": value}


def _model_identity(
    directory: Path,
    supplied: ModelIdentity | None,
) -> ModelIdentity:
    return supplied or ModelIdentity(
        model_id=str(directory),
        local_path=str(directory),
    )


def generate_ax_engine_manifest(
    model_dir: str | Path,
    *,
    executable: str = "ax-engine-bench",
    runner: CommandRunner = _run,
    model_identity: ModelIdentity | None = None,
) -> RuntimeCheck:
    directory = Path(model_dir).expanduser().resolve()
    model = _model_identity(directory, model_identity)
    resolved = _resolve_executable(executable)
    if resolved is None:
        return RuntimeCheck(
            model=model,
            runtime=RuntimeName.AX_ENGINE,
            check_kind="manifest",
            available=False,
            passed=False,
            stderr=f"executable not found: {executable}",
        )
    command = [
        resolved,
        "generate-manifest",
        "--json",
        "--validate",
        str(directory),
    ]
    completed = runner(command)
    manifest_exists = (directory / "model-manifest.json").is_file()
    diagnostics = f"{completed.stdout}\n{completed.stderr}".casefold()
    validation_failed = any(
        marker in diagnostics
        for marker in (
            "validation failed",
            "invalid native model manifest",
            '"status":"failed"',
            '"status": "failed"',
        )
    )
    return RuntimeCheck(
        model=model,
        runtime=RuntimeName.AX_ENGINE,
        check_kind="manifest",
        available=True,
        passed=completed.returncode == 0 and manifest_exists and not validation_failed,
        command=command,
        exit_code=completed.returncode,
        report=_json_report(completed.stdout),
        stderr=completed.stderr or ("manifest validation failed" if validation_failed else ""),
    )


def require_ax_engine_manifest(
    model_dir: str | Path,
    *,
    executable: str = "ax-engine-bench",
    runner: CommandRunner = _run,
    model_identity: ModelIdentity | None = None,
) -> RuntimeCheck:
    result = generate_ax_engine_manifest(
        model_dir,
        executable=executable,
        runner=runner,
        model_identity=model_identity,
    )
    if not result.available:
        raise BackendUnavailableError(result.stderr)
    if not result.passed:
        raise ArtifactError(
            f"AX Engine manifest generation or validation failed: {result.stderr or result.report}"
        )
    return result


def check_ax_engine(
    model_dir: str | Path,
    *,
    executable: str = "ax-engine",
    runner: CommandRunner = _run,
    model_identity: ModelIdentity | None = None,
) -> RuntimeCheck:
    directory = Path(model_dir).expanduser().resolve()
    model = _model_identity(directory, model_identity)
    resolved = _resolve_executable(executable)
    if resolved is None:
        return RuntimeCheck(
            model=model,
            runtime=RuntimeName.AX_ENGINE,
            check_kind="doctor",
            available=False,
            passed=False,
            stderr=f"executable not found: {executable}",
        )
    command = [
        resolved,
        "doctor",
        "--json",
        "--mlx-model-artifacts-dir",
        str(directory),
    ]
    completed = runner(command)
    report = _json_report(completed.stdout)
    declared_statuses = [report[key] for key in ("result", "status") if key in report]
    return RuntimeCheck(
        model=model,
        runtime=RuntimeName.AX_ENGINE,
        check_kind="doctor",
        available=True,
        passed=(
            completed.returncode == 0
            and bool(declared_statuses)
            and all(status == "ready" for status in declared_statuses)
        ),
        command=command,
        exit_code=completed.returncode,
        report=report,
        stderr=completed.stderr,
    )


def check_mlx_lm_static(
    model_dir: str | Path,
    *,
    model_identity: ModelIdentity | None = None,
) -> RuntimeCheck:
    directory = Path(model_dir).expanduser().resolve()
    model = _model_identity(directory, model_identity)
    config_present = (directory / "config.json").is_file()
    weights_present = any(directory.glob("*.safetensors"))
    importable = importlib.util.find_spec("mlx_lm") is not None
    executable = shutil.which("mlx_lm.generate") or shutil.which("mlx_lm.convert")
    installed = importable or executable is not None
    report = {
        "config_present": config_present,
        "weights_present": weights_present,
        "mlx_lm_installed": installed,
        "mlx_lm_importable": importable,
        "mlx_lm_executable": executable,
        "scope": "static-only",
        "mtp": "runtime-dependent",
    }
    return RuntimeCheck(
        model=model,
        runtime=RuntimeName.MLX_LM,
        check_kind="static-compatibility",
        available=installed,
        passed=installed and config_present and weights_present,
        report=report,
        stderr="" if installed else "mlx-lm is not installed",
    )


def check_mlx_lm_generation(
    model_dir: str | Path,
    *,
    executable: str = "mlx_lm.generate",
    runner: CommandRunner = _run,
    model_identity: ModelIdentity | None = None,
) -> RuntimeCheck:
    directory = Path(model_dir).expanduser().resolve()
    model = _model_identity(directory, model_identity)
    static = check_mlx_lm_static(directory, model_identity=model)
    resolved = _resolve_executable(executable)
    if resolved is None:
        return RuntimeCheck(
            model=model,
            runtime=RuntimeName.MLX_LM,
            check_kind="generation-smoke",
            available=False,
            passed=False,
            report={"static_check_passed": static.passed},
            stderr=f"executable not found: {executable}",
        )
    if not static.passed:
        return RuntimeCheck(
            model=model,
            runtime=RuntimeName.MLX_LM,
            check_kind="generation-smoke",
            available=True,
            passed=False,
            report={"static_check_passed": False},
            stderr="static MLX-LM compatibility check failed",
        )
    command = [
        resolved,
        "--model",
        str(directory),
        "--prompt",
        "Reply with OK.",
        "--max-tokens",
        "2",
        "--temp",
        "0",
        "--verbose",
        "false",
    ]
    completed = runner(command)
    output = completed.stdout.strip()
    return RuntimeCheck(
        model=model,
        runtime=RuntimeName.MLX_LM,
        check_kind="generation-smoke",
        available=True,
        passed=completed.returncode == 0 and bool(output),
        command=command,
        exit_code=completed.returncode,
        report={
            "scope": "load-and-generation",
            "standard_inference": completed.returncode == 0 and bool(output),
            "output_characters": len(output),
            "mtp": "runtime-dependent",
        },
        stderr=completed.stderr,
    )


def build_runtime_metadata(
    plan: QuantizationPlan,
    output_dir: str | Path,
) -> RuntimeMetadata:
    directory = Path(output_dir).expanduser().resolve()
    sidecar = directory / "mtp.safetensors"
    runtime_contract = directory / "mtplx_runtime.json"
    contract: dict[str, Any] = {}
    if runtime_contract.is_file():
        try:
            value = json.loads(runtime_contract.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {}
        if isinstance(value, dict):
            contract = value
    depth = contract.get("mtp_depth_max")
    draft_tokens = int(depth) if isinstance(depth, int) and depth > 0 else None
    precision = None
    sidecar_description = contract.get("mtp_sidecar")
    if isinstance(sidecar_description, str):
        match = re.search(r"INT(4|6|8|16)", sidecar_description.upper())
        if match:
            precision = f"{match.group(1)}bit"
    mtp_detected = sidecar.is_file() or bool(plan.mtp_distribution)
    return RuntimeMetadata(
        primary_runtime=RuntimeProfile(
            name=RuntimeName.AX_ENGINE,
            compatibility_level="A",
            support_level=RuntimeSupportLevel.OPTIMIZED,
            standard_inference=True,
            mtp_support="native",
            manifest="model-manifest.json",
            notes=["Runtime claims require a passing AX Engine doctor and benchmark report."],
        ),
        compatible_runtimes=[
            RuntimeProfile(
                name=RuntimeName.MLX_LM,
                compatibility_level="B",
                support_level=RuntimeSupportLevel.STANDARD_INFERENCE,
                standard_inference=True,
                mtp_support="runtime-dependent",
                manifest="config.json",
                notes=[
                    "Standard backbone inference is the compatibility target.",
                    "AXQuant MTP metadata may be ignored by MLX-LM.",
                ],
            )
        ],
        optimization_scope=plan.architecture_profile.optimization_scope,
        mtp=MtpRuntimeMetadata(
            detected=mtp_detected,
            sidecar_file="mtp.safetensors" if sidecar.is_file() else None,
            optimized=False,
            enabled_by_default=sidecar.is_file() and bool(contract),
            draft_tokens=draft_tokens,
            verification_mode="runtime-default" if sidecar.is_file() else None,
            head_precision=precision,
        ),
        ax_engine=AxEngineOptimizationMetadata(
            preferred_group_size=plan.group_size,
            fused_mtp=None,
            decode_kernel=None,
            kernel_evidence="unmeasured",
        ),
        memory_policy={
            "kv_cache_precision": "runtime-default",
            "prefix_cache": "runtime-managed",
            "mtp_buffers": "preallocate-when-enabled" if mtp_detected else "not-required",
            "unified_memory_safety_margin": "benchmark-required",
        },
    )


def assert_qwen36_conversion_scope(plan: QuantizationPlan) -> None:
    profile = plan.architecture_profile
    if profile.product_family != "qwen3.6":
        raise ArtifactError("AXQuant conversion is restricted to Qwen 3.6")
    if profile.optimization_scope != OptimizationScope.TEXT_PATH:
        raise ArtifactError("this Qwen 3.6 checkpoint is inventory-only and cannot be converted")
