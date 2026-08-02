from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError, BackendUnavailableError
from axquant.schema import (
    AxEngineOptimizationMetadata,
    KvCacheRuntimeMetadata,
    ModelIdentity,
    MtpRuntimeMetadata,
    OptimizationScope,
    QuantizationPlan,
    RuntimeCheck,
    RuntimeMetadata,
    RuntimeName,
    RuntimeProfile,
    RuntimeSupportLevel,
    SupportTier,
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
    command = [resolved]
    if Path(resolved).name == "mlx_lm":
        command.append("generate")
    command.extend(
        [
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
    )
    kv_execution = _advisory_kv_execution(directory)
    if kv_execution is not None:
        command.extend(
            [
                "--kv-bits",
                str(kv_execution[0]),
                "--kv-group-size",
                str(kv_execution[1]),
            ]
        )
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
            "kv_cache_execution": (
                {
                    "kv_bits": kv_execution[0],
                    "kv_group_size": kv_execution[1],
                    "source": "axquant_runtime.json advisory values",
                }
                if kv_execution is not None
                else "runtime-default"
            ),
        },
        stderr=completed.stderr,
    )


def _advisory_kv_execution(directory: Path) -> tuple[int, int] | None:
    """Read the artifact's advisory MLX-LM KV quantization, if planned.

    Artifacts converted with a per-layer KV-cache plan record advisory
    global values for MLX-LM (`advisory_mlx_lm_kv_bits`/`_group_size`).
    Passing them to `mlx_lm.generate` executes the plan's KV quantization
    through the public `QuantizedKVCache` path, so the smoke exercises the
    same KV precision the plan declared instead of silently ignoring it.
    BF16 advisories (16-bit) keep the runtime default.
    """
    runtime_path = directory / "axquant_runtime.json"
    if not runtime_path.is_file():
        return None
    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    kv = payload.get("kv_cache")
    if not isinstance(kv, dict):
        return None
    bits = kv.get("advisory_mlx_lm_kv_bits")
    group = kv.get("advisory_mlx_lm_kv_group_size")
    if not isinstance(bits, int) or not isinstance(group, int) or bits >= 16:
        return None
    return bits, group


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
    kv_metadata: KvCacheRuntimeMetadata | None = None
    if plan.kv_cache is not None:
        kv = plan.kv_cache
        ordered = sorted(kv.layers, key=lambda layer: layer.layer_index)
        bit_values = [layer.bits for layer in ordered]
        group_values = [layer.group_size for layer in ordered]
        kv_metadata = KvCacheRuntimeMetadata(
            allocation_basis=kv.allocation_basis,
            layer_bits=bit_values,
            layer_group_sizes=group_values,
            advisory_mlx_lm_kv_bits=max(set(bit_values), key=bit_values.count),
            advisory_mlx_lm_kv_group_size=max(set(group_values), key=group_values.count),
        )
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
        kv_cache=kv_metadata,
        memory_policy={
            "kv_cache_precision": ("planned-per-layer" if kv_metadata else "runtime-default"),
            "prefix_cache": "runtime-managed",
            "mtp_buffers": "preallocate-when-enabled" if mtp_detected else "not-required",
            "unified_memory_safety_margin": "benchmark-required",
        },
    )


def assert_conversion_scope(plan: QuantizationPlan) -> None:
    """Fail closed unless the family's tier and scope permit conversion.

    The tier system (AXQ-017) is the conversion gate: `convertible` requires
    recorded promotion evidence and `certified` requires the release audit,
    so a hard-coded family allowlist would only duplicate — and drift from —
    the registry's declared tiers.
    """
    profile = plan.architecture_profile
    if profile.support_tier is SupportTier.INSPECT_ONLY:
        raise ArtifactError(
            f"the {profile.product_family} family is inspect-only; conversion requires the "
            "convertible or certified tier and its promotion evidence (AXQ-017)"
        )
    if profile.optimization_scope != OptimizationScope.TEXT_PATH:
        raise ArtifactError(
            f"this {profile.product_family} checkpoint is inventory-only and cannot be converted"
        )
    if plan.kv_cache is not None:
        if plan.kv_cache.allocation_basis == "measured" and not plan.kv_cache.sensitivity_sha256:
            raise ArtifactError(
                "a measured KV-cache plan must bind its sensitivity report digest (AXQ-024)"
            )
        if (
            profile.text_layer_count is not None
            and len(plan.kv_cache.layers) != profile.text_layer_count
        ):
            raise ArtifactError("the KV-cache plan does not cover the text layer count")


def check_mlx_lm_kv_layered(
    model_dir: str | Path,
    *,
    runner: CommandRunner = _run,
    model_identity: ModelIdentity | None = None,
) -> RuntimeCheck:
    """Execute the artifact's planned per-layer KV precisions through MLX-LM.

    Runs ``python -m axquant.kv_exec``, which builds one cache object per
    layer from the planned table (``QuantizedKVCache`` for quantized layers,
    the model's own cache otherwise) and generates through the public
    ``prompt_cache`` API — the compatibility runtime's true per-layer KV
    execution path. Families whose attention implementation rejects
    quantized caches fail closed with the runtime's own error.
    """
    directory = Path(model_dir).expanduser().resolve()
    model = _model_identity(directory, model_identity)
    command = [
        sys.executable,
        "-m",
        "axquant.kv_exec",
        "--model",
        str(directory),
        "--max-tokens",
        "16",
    ]
    completed = runner(command)
    report: dict[str, Any] = {"scope": "per-layer-kv-generation"}
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        if isinstance(payload, dict):
            report.update(payload)
    except (json.JSONDecodeError, IndexError):
        report["error"] = "kv_exec produced no parseable report"
    passed = completed.returncode == 0 and bool(report.get("ok"))
    return RuntimeCheck(
        model=model,
        runtime=RuntimeName.MLX_LM,
        check_kind="kv-layered-generation-smoke",
        available=True,
        passed=passed,
        command=command,
        exit_code=completed.returncode,
        report=report,
        stderr=completed.stderr,
    )
