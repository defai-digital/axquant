from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from safetensors import SafetensorError, safe_open

from axquant.errors import ArtifactError, BackendUnavailableError
from axquant.mtp_sidecar import EXTERNAL_MTP_SIDECAR_FILENAMES
from axquant.schema import (
    AX_ENGINE_EXECUTABLE_BITS,
    AX_ENGINE_EXECUTABLE_GROUP_SIZES,
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
from axquant.serde import load_model, read_data

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
        # Preserve a virtual-environment Python symlink: resolving it to the
        # Homebrew framework binary drops the venv's site-packages.
        return str(path.absolute()) if path.is_file() else None
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


def _reports_ready(report: dict[str, Any]) -> bool:
    declared_statuses = [report[key] for key in ("result", "status") if key in report]
    return bool(declared_statuses) and all(status == "ready" for status in declared_statuses)


def _safetensors_has_payload(path: Path) -> bool:
    try:
        with safe_open(path, framework="numpy") as tensors:
            return bool(tensors.keys())
    except (OSError, SafetensorError):
        return False


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
    report = _json_report(completed.stdout)
    declared_ready = _reports_ready(report)
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
    failure = ""
    if validation_failed:
        failure = "manifest validation failed"
    elif not declared_ready:
        failure = "manifest generator did not report a parseable ready status"
    elif not manifest_exists:
        failure = "manifest generator did not create model-manifest.json"
    return RuntimeCheck(
        model=model,
        runtime=RuntimeName.AX_ENGINE,
        check_kind="manifest",
        available=True,
        passed=(
            completed.returncode == 0
            and manifest_exists
            and declared_ready
            and not validation_failed
        ),
        command=command,
        exit_code=completed.returncode,
        report=report,
        stderr=completed.stderr or failure,
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
    return RuntimeCheck(
        model=model,
        runtime=RuntimeName.AX_ENGINE,
        check_kind="doctor",
        available=True,
        passed=completed.returncode == 0 and _reports_ready(report),
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
    config_path = directory / "config.json"
    config_present = config_path.is_file()
    config_valid = False
    if config_present:
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config_payload = None
        config_valid = isinstance(config_payload, dict)
    sidecar_names = {
        *(name.casefold() for name in EXTERNAL_MTP_SIDECAR_FILENAMES),
        "vision.safetensors",
    }
    main_weight_paths = sorted(
        (
            path
            for path in directory.glob("*.safetensors")
            if path.is_file() and path.name.casefold() not in sidecar_names
        ),
        key=lambda path: path.name,
    )
    valid_main_weight_files = [
        path.name for path in main_weight_paths if _safetensors_has_payload(path)
    ]
    invalid_main_weight_files = [
        path.name for path in main_weight_paths if path.name not in valid_main_weight_files
    ]
    main_weight_files = [path.name for path in main_weight_paths]
    weights_present = bool(main_weight_files)
    weights_valid = weights_present and not invalid_main_weight_files
    importable = importlib.util.find_spec("mlx_lm") is not None
    executable = shutil.which("mlx_lm.generate") or shutil.which("mlx_lm.convert")
    installed = importable or executable is not None
    report = {
        "config_present": config_present,
        "config_valid": config_valid,
        "weights_present": weights_present,
        "weights_valid": weights_valid,
        "main_weight_files": main_weight_files,
        "valid_main_weight_files": valid_main_weight_files,
        "invalid_main_weight_files": invalid_main_weight_files,
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
        passed=installed and config_valid and weights_valid,
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
    # Fail closed on advisory KV / runtime metadata before install or runner gates so
    # invalid artifacts raise even when mlx-lm is not importable (non-MLX CI path).
    kv_execution = _advisory_kv_execution(directory)
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
    # Generation-smoke uses an external executable + runner. Require artifact
    # readiness (config + weights), not Python package install — that would block
    # non-MLX hosts that still exercise CLI generation via a resolved binary.
    config_valid = bool(static.report.get("config_valid"))
    weights_valid = bool(static.report.get("weights_valid"))
    artifact_ready = config_valid and weights_valid
    if not artifact_ready:
        return RuntimeCheck(
            model=model,
            runtime=RuntimeName.MLX_LM,
            check_kind="generation-smoke",
            available=True,
            passed=False,
            report={
                "static_check_passed": False,
                "config_valid": config_valid,
                "weights_valid": weights_valid,
            },
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


def _check_multimodal_generation(
    model_dir: str | Path,
    *,
    runtime: RuntimeName,
    check_kind: Literal["transcription-smoke", "vision-generation-smoke"],
    executable: str,
    module: str,
    media_flag: str,
    media_path: str | Path,
    runner: CommandRunner,
    model_identity: ModelIdentity | None,
) -> RuntimeCheck:
    directory = Path(model_dir).expanduser().resolve()
    model = _model_identity(directory, model_identity)
    media = Path(media_path).expanduser().resolve()
    resolved = _resolve_executable(executable)
    if resolved is None:
        return RuntimeCheck(
            model=model,
            runtime=runtime,
            check_kind=check_kind,
            available=False,
            passed=False,
            stderr=f"executable not found: {executable}",
        )
    if not media.is_file():
        return RuntimeCheck(
            model=model,
            runtime=runtime,
            check_kind=check_kind,
            available=True,
            passed=False,
            stderr=f"QA media input not found: {media}",
        )
    command = [resolved, "-m", module]
    command.extend(["--model", str(directory), media_flag, str(media)])
    if runtime is RuntimeName.MLX_AUDIO:
        temporary = tempfile.TemporaryDirectory(prefix="axquant-asr-smoke-")
        output_stem = Path(temporary.name) / "transcript"
        transcript = output_stem.with_suffix(".txt")
        command.extend(
            [
                "--output-path",
                str(output_stem),
                "--format",
                "txt",
                "--max-tokens",
                "32",
            ]
        )
    else:
        temporary = None
        transcript = None
        command.extend(
            [
                "--prompt",
                "Read the image and answer briefly.",
                "--max-tokens",
                "16",
                "--temperature",
                "0",
                "--no-verbose",
            ]
        )
    try:
        completed = runner(command)
        if transcript is not None:
            output = transcript.read_text(encoding="utf-8").strip() if transcript.is_file() else ""
        else:
            output = completed.stdout.strip()
    finally:
        if temporary is not None:
            temporary.cleanup()
    return RuntimeCheck(
        model=model,
        runtime=runtime,
        check_kind=check_kind,
        available=True,
        passed=completed.returncode == 0 and bool(output),
        command=command,
        exit_code=completed.returncode,
        report={
            "scope": "load-and-multimodal-generation",
            "output_characters": len(output),
            "media_input": media.name,
        },
        stderr=completed.stderr,
    )


def check_mlx_audio_transcription(
    model_dir: str | Path,
    *,
    audio: str | Path,
    executable: str = "python3",
    runner: CommandRunner = _run,
    model_identity: ModelIdentity | None = None,
) -> RuntimeCheck:
    return _check_multimodal_generation(
        model_dir,
        runtime=RuntimeName.MLX_AUDIO,
        check_kind="transcription-smoke",
        executable=executable,
        module="mlx_audio.stt.generate",
        media_flag="--audio",
        media_path=audio,
        runner=runner,
        model_identity=model_identity,
    )


def check_mlx_vlm_generation(
    model_dir: str | Path,
    *,
    image: str | Path,
    executable: str = "python3",
    runner: CommandRunner = _run,
    model_identity: ModelIdentity | None = None,
) -> RuntimeCheck:
    return _check_multimodal_generation(
        model_dir,
        runtime=RuntimeName.MLX_VLM,
        check_kind="vision-generation-smoke",
        executable=executable,
        module="mlx_vlm.generate",
        media_flag="--image",
        media_path=image,
        runner=runner,
        model_identity=model_identity,
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
    from pydantic import ValidationError

    runtime_path = directory / "axquant_runtime.json"
    if not runtime_path.is_file():
        return None
    try:
        payload = read_data(runtime_path)
        if not isinstance(payload, dict):
            raise ArtifactError("runtime metadata must be a JSON object")
        raw_kv = payload.get("kv_cache")
        if raw_kv is not None:
            if not isinstance(raw_kv, dict):
                raise ArtifactError("kv_cache must be a JSON object or null")
            raw_bits = raw_kv.get("advisory_mlx_lm_kv_bits")
            raw_group = raw_kv.get("advisory_mlx_lm_kv_group_size")
            if type(raw_bits) is not int or raw_bits not in AX_ENGINE_EXECUTABLE_BITS:
                raise ArtifactError(
                    "kv_cache.advisory_mlx_lm_kv_bits is not executable by AX Engine"
                )
            if type(raw_group) is not int or raw_group not in AX_ENGINE_EXECUTABLE_GROUP_SIZES:
                raise ArtifactError(
                    "kv_cache.advisory_mlx_lm_kv_group_size is not executable by AX Engine"
                )
            if raw_kv.get("advisory") is not True:
                raise ArtifactError("kv_cache.advisory must be true")
        metadata = load_model(runtime_path, RuntimeMetadata)
    except (ArtifactError, ValueError, ValidationError, TypeError) as exc:
        raise ArtifactError(f"axquant_runtime.json is invalid: {exc}") from exc
    kv = metadata.kv_cache
    if kv is None:
        return None
    bits = kv.advisory_mlx_lm_kv_bits
    group = kv.advisory_mlx_lm_kv_group_size
    if bits >= 16:
        return None
    return bits, group


def _load_mtp_runtime_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"mtplx_runtime.json is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError("mtplx_runtime.json must be a JSON object")
    return value


def _resolve_mtp_sidecar(directory: Path) -> Path | None:
    existing = [
        directory / name for name in EXTERNAL_MTP_SIDECAR_FILENAMES if (directory / name).is_file()
    ]
    invalid = [path.name for path in existing if not _safetensors_has_payload(path)]
    if invalid:
        raise ArtifactError(f"invalid MTP Safetensors sidecar(s): {invalid}")
    return existing[0] if existing else None


def _mtp_contract_values(contract: dict[str, Any]) -> tuple[int | None, str | None]:
    depth = contract.get("mtp_depth_max")
    if depth is not None and (type(depth) is not int or depth <= 0):
        raise ArtifactError("mtplx_runtime.json mtp_depth_max must be a positive integer")
    draft_tokens = depth if isinstance(depth, int) else None

    precision_bits: int | None = None
    if "mtp_sidecar_bits" in contract:
        structured_bits = contract["mtp_sidecar_bits"]
        if type(structured_bits) is not int or structured_bits not in {4, 6, 8, 16}:
            raise ArtifactError("mtplx_runtime.json mtp_sidecar_bits must be one of 4, 6, 8, 16")
        precision_bits = structured_bits
    else:
        sidecar_description = contract.get("mtp_sidecar")
        if sidecar_description is not None and not isinstance(sidecar_description, str):
            raise ArtifactError("mtplx_runtime.json mtp_sidecar must be a string")
        if isinstance(sidecar_description, str):
            match = re.search(r"INT(4|6|8|16)", sidecar_description.upper())
            if match:
                precision_bits = int(match.group(1))
    precision = f"{precision_bits}bit" if precision_bits is not None else None
    return draft_tokens, precision


def _advisory_kv_pair(plan: QuantizationPlan) -> tuple[int, int] | None:
    if plan.kv_cache is None:
        return None
    pairs = [(layer.bits, layer.group_size) for layer in plan.kv_cache.layers]
    counts = Counter(pairs)
    return min(
        counts,
        key=lambda pair: (
            -counts[pair],
            -pair[0],
            pair[1],
        ),
    )


def build_runtime_metadata(
    plan: QuantizationPlan,
    output_dir: str | Path,
) -> RuntimeMetadata:
    directory = Path(output_dir).expanduser().resolve()
    sidecar = _resolve_mtp_sidecar(directory)
    runtime_contract = directory / "mtplx_runtime.json"
    contract = _load_mtp_runtime_contract(runtime_contract)
    draft_tokens, precision = _mtp_contract_values(contract)
    mtp_detected = sidecar is not None
    mtp_capable = mtp_detected
    kv_metadata: KvCacheRuntimeMetadata | None = None
    if plan.kv_cache is not None:
        kv = plan.kv_cache
        ordered = sorted(kv.layers, key=lambda layer: layer.layer_index)
        bit_values = [layer.bits for layer in ordered]
        group_values = [layer.group_size for layer in ordered]
        advisory_pair = _advisory_kv_pair(plan)
        if advisory_pair is None:
            raise ArtifactError("KV-cache plan requires at least one layer")
        kv_metadata = KvCacheRuntimeMetadata(
            allocation_basis=kv.allocation_basis,
            layer_bits=bit_values,
            layer_group_sizes=group_values,
            advisory_mlx_lm_kv_bits=advisory_pair[0],
            advisory_mlx_lm_kv_group_size=advisory_pair[1],
        )
    adapter_id = plan.architecture_profile.adapter_id
    modality_runtime = (
        RuntimeName.MLX_AUDIO
        if adapter_id == "qwen3-asr-v1"
        else RuntimeName.MLX_VLM
        if adapter_id == "qwen3-vl-v1"
        else None
    )
    if modality_runtime is None:
        primary_runtime = RuntimeProfile(
            name=RuntimeName.AX_ENGINE,
            compatibility_level="A",
            support_level=RuntimeSupportLevel.OPTIMIZED,
            standard_inference=True,
            mtp_support="native" if mtp_capable else "none",
            manifest="model-manifest.json",
            notes=["Runtime claims require a passing AX Engine doctor and benchmark report."],
        )
        compatible_runtimes = [
            RuntimeProfile(
                name=RuntimeName.MLX_LM,
                compatibility_level="B",
                support_level=RuntimeSupportLevel.STANDARD_INFERENCE,
                standard_inference=True,
                mtp_support="runtime-dependent" if mtp_capable else "none",
                manifest="config.json",
                notes=[
                    "Standard backbone inference is the compatibility target.",
                    "AXQuant MTP metadata may be ignored by MLX-LM.",
                ],
            )
        ]
    else:
        runtime_label = "MLX-Audio" if modality_runtime is RuntimeName.MLX_AUDIO else "MLX-VLM"
        primary_runtime = RuntimeProfile(
            name=modality_runtime,
            compatibility_level="A",
            support_level=RuntimeSupportLevel.STANDARD_INFERENCE,
            standard_inference=True,
            mtp_support="none",
            manifest="config.json",
            notes=[f"{runtime_label} loads the protected modality tower and AXQ language decoder."],
        )
        compatible_runtimes = []
    return RuntimeMetadata(
        primary_runtime=primary_runtime,
        compatible_runtimes=compatible_runtimes,
        optimization_scope=plan.architecture_profile.optimization_scope,
        mtp=MtpRuntimeMetadata(
            detected=mtp_detected,
            sidecar_file=sidecar.name if sidecar is not None else None,
            optimized=False,
            enabled_by_default=sidecar is not None and bool(contract),
            draft_tokens=draft_tokens if mtp_detected else None,
            verification_mode="runtime-default" if mtp_detected else None,
            head_precision=precision if mtp_detected else None,
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
