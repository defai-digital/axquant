from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import shutil
import struct
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import structlog

from axquant.artifact_paths import artifact_tree_files
from axquant.calibration import calibration_manifest_matches
from axquant.capture_binding import (
    LoadedActivationCapture,
    activation_capture_evidence_issues,
)
from axquant.errors import ArtifactError, BackendUnavailableError, PlanningError
from axquant.inspector import inspect_model, resolve_model_dir
from axquant.module_paths import fused_expert_tensor_target, mlx_tensor_binding_groups
from axquant.mtp_sidecar import EXTERNAL_MTP_SIDECAR_FILENAMES, prepare_qwen36_mtp_sidecar
from axquant.multimodal_backend import (
    conversion_backend,
    convert_multimodal,
    preflight_multimodal,
)
from axquant.predicate import PlanPredicate, build_quant_predicate
from axquant.runtime import (
    assert_conversion_scope,
    build_runtime_metadata,
    generate_ax_engine_manifest,
    require_ax_engine_manifest,
)
from axquant.schema import (
    ArtifactFile,
    ArtifactManifest,
    CalibrationManifest,
    KvSensitivityReport,
    MtpSidecarLayout,
    ProtectedTensorSidecarManifest,
    QuantizationPlan,
    QuantizerExecutionManifest,
    QuantizerExecutionRecord,
    QuantMethod,
    TensorRole,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data
from axquant.source_prep import prepare_conversion_source

_LOG = structlog.get_logger()
_ACTIVATION_REFINEMENT_METHODS = frozenset(
    {QuantMethod.AWQ, QuantMethod.GPTQ, QuantMethod.GPTQ_ACT}
)
_CAPTURE_MANIFEST_NAME = "activation_capture_manifest.json"


def _maybe_force_mlx_cpu() -> None:
    """Honor ``AXQUANT_FORCE_CPU=1`` for large re-packs when Metal times out."""

    flag = os.environ.get("AXQUANT_FORCE_CPU", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    try:
        mx = importlib.import_module("mlx.core")
        mx.set_default_device(mx.cpu)
        _LOG.info("mlx_force_cpu", device=str(mx.default_device()))
    except Exception as exc:  # pragma: no cover - optional backend path
        _LOG.warning("mlx_force_cpu_failed", error=str(exc))


def _mlx_api() -> tuple[Any, Any]:
    try:
        mlx_lm = importlib.import_module("mlx_lm")
    except ModuleNotFoundError as exc:
        raise BackendUnavailableError(
            "the MLX backend is not installed; install axquant[mlx]"
        ) from exc
    _maybe_force_mlx_cpu()
    return mlx_lm.convert, mlx_lm.load


def _dequantize_quantized_multilinear(model: Any) -> Any:
    """Dequantize MLX-LM ``QuantizedMultiLinear`` modules left by FP8 loads.

    Public ``mlx_lm.utils.dequantize_model`` only handles QuantizedLinear,
    QuantizedEmbedding, and QuantizedSwitchLinear. DeepSeek V4 attention
    ``wo_a`` is a ``MultiLinear`` that loads as ``QuantizedMultiLinear`` and
    would otherwise remain 8-bit through a 16-bit plan allocation.
    """

    try:
        mx = importlib.import_module("mlx.core")
        tree_unflatten = importlib.import_module("mlx.utils").tree_unflatten
        mla = importlib.import_module("mlx_lm.models.mla")
    except ModuleNotFoundError:
        return model
    quantized_cls = getattr(mla, "QuantizedMultiLinear", None)
    multilinear_cls = getattr(mla, "MultiLinear", None)
    if quantized_cls is None or multilinear_cls is None:
        return model
    replacements: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, quantized_cls):
            continue
        weight = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            module.group_size,
            module.bits,
            module.mode,
        )
        num_heads, output_dims, input_dims = weight.shape
        restored = multilinear_cls(input_dims, output_dims, num_heads)
        restored.weight = weight
        replacements.append((name, restored))
    if replacements:
        model.update_modules(tree_unflatten(replacements))
        _LOG.info("dequantized_quantized_multilinear", count=len(replacements))
    return model


def _mlx_convert_with_optional_dequant(
    model_ref: str,
    *,
    mlx_path: str,
    quantize: bool,
    q_group_size: int,
    q_bits: int,
    quant_predicate: Any,
    revision: str | None,
) -> None:
    """Convert via MLX-LM, dequantizing mixed-precision sources first when needed.

    DeepSeek V4 Flash ships FP4/FP8 experts that load as already-quantized MLX
    modules without ``to_quantized``. Re-packing requires dequant then affine
    quant under the plan predicate (lazy load keeps peak memory manageable).
    """
    convert, load = _mlx_api()
    try:
        utils = importlib.import_module("mlx_lm.utils")
    except ModuleNotFoundError:
        convert(
            model_ref,
            mlx_path=mlx_path,
            quantize=quantize,
            q_group_size=q_group_size,
            q_bits=q_bits,
            quant_predicate=quant_predicate,
            revision=revision,
        )
        return
    config_path = Path(model_ref).expanduser() / "config.json"
    needs_dequant = False
    if config_path.is_file():
        try:
            config_obj = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config_obj = {}
        if isinstance(config_obj, dict):
            needs_dequant = bool(
                config_obj.get("quantization") or config_obj.get("quantization_config")
            )
    if not needs_dequant or not quantize:
        convert(
            model_ref,
            mlx_path=mlx_path,
            quantize=quantize,
            q_group_size=q_group_size,
            q_bits=q_bits,
            quant_predicate=quant_predicate,
            revision=revision,
        )
        return
    _LOG.info("conversion_dequant_requant_started", model=model_ref)
    model, tokenizer, config = load(
        model_ref,
        revision=revision,
        return_config=True,
        lazy=True,
    )
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    model = utils.dequantize_model(model)
    model = _dequantize_quantized_multilinear(model)
    model, config = utils.quantize_model(
        model,
        config,
        q_group_size,
        q_bits,
        mode="affine",
        quant_predicate=quant_predicate,
    )
    utils.save(mlx_path, model_ref, model, tokenizer, config)
    _LOG.info("conversion_dequant_requant_completed", model=model_ref)


def _preflight_coverage(model: str, revision: str | None, predicate: PlanPredicate) -> None:
    _, load = _mlx_api()
    try:
        loaded = load(model, revision=revision, lazy=True, return_config=True)
    except Exception as exc:
        raise ArtifactError(f"cannot load model structure for conversion preflight: {exc}") from exc
    mlx_model = loaded[0]
    try:
        for path, module in mlx_model.named_modules():
            if predicate.lookup(path) is not None:
                predicate(path, module)
    finally:
        del mlx_model, loaded
    unmatched = predicate.unmatched_quantized_modules()
    if unmatched:
        preview = sorted(unmatched)[:10]
        suffix = "" if len(unmatched) <= 10 else f" and {len(unmatched) - 10} more"
        raise PlanningError(f"plan modules do not match the MLX model: {preview}{suffix}")


def _copy_verified(source: Path, destination: Path) -> None:
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise ArtifactError(f"conversion output contains a different {destination.name}")
        return
    shutil.copy2(source, destination)
    if file_sha256(destination) != file_sha256(source):
        raise ArtifactError(f"{source.name} checksum changed during copy")


def _validate_mtp_sidecar_provenance(source: Path) -> None:
    """Fail closed on supplied provenance that does not prove a raw sidecar copy.

    Legacy source sidecars without provenance remain usable for development conversion. They
    cannot pass the release artifact audit, which separately requires provenance. Once a
    provenance file is supplied, it must bind the exact payload and explicitly reject transforms;
    otherwise a transformed sidecar could be silently copied as if it were raw.
    """

    provenance_path = source.parent / "ax_mtp_sidecar_manifest.json"
    if not provenance_path.is_file():
        return
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"MTP sidecar provenance is unreadable: {provenance_path}") from exc
    if not isinstance(provenance, dict):
        raise ArtifactError("MTP sidecar provenance must be a JSON object")
    transform = provenance.get("transform")
    if not isinstance(transform, dict) or transform.get("mode") != "byte_preserved":
        raise ArtifactError("MTP sidecar provenance must declare transform.mode=byte_preserved")
    output = provenance.get("output")
    mtp = output.get("mtp") if isinstance(output, dict) else None
    if not isinstance(mtp, dict):
        raise ArtifactError("MTP sidecar provenance does not bind an output sidecar")
    expected_path = mtp.get("path")
    if not isinstance(expected_path, str) or Path(expected_path).name != source.name:
        raise ArtifactError("MTP sidecar provenance binds a different sidecar path")
    expected_size = mtp.get("size_bytes")
    expected_sha256 = mtp.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or source.stat().st_size != expected_size
    ):
        raise ArtifactError("MTP sidecar provenance size does not match the sidecar")
    if not isinstance(expected_sha256, str) or file_sha256(source) != expected_sha256.lower():
        raise ArtifactError("MTP sidecar provenance checksum does not match the sidecar")


def _source_has_external_mtp_sidecar(model: str | Path) -> bool:
    """True when the source checkpoint ships MTP weights as a root sidecar file.

    Families like Qwen 3.6 externalise the MTP head into ``mtp.safetensors``,
    which byte-preservation must copy explicitly. Families like Qwen 3.5 store
    the MTP tensors inside the indexed shards, where the planner's 16-bit
    logical preservation carries them through conversion with no sidecar input.
    A non-directory source (hub reference) keeps the fail-closed requirement.
    """
    root = Path(model).expanduser()
    if not root.is_dir():
        return True
    return any((root / name).is_file() for name in EXTERNAL_MTP_SIDECAR_FILENAMES)


def _resolve_bound_conversion_source(
    model: str,
    plan: QuantizationPlan,
    revision: str | None,
) -> Path:
    """Resolve the exact checkpoint identity carried by ``plan``.

    Conversion must never fall back from an unrelated or unresolved ``model``
    argument to the plan's local path.  Doing so lets a caller execute the
    predicate against one checkpoint while the artifact continues to claim
    another model and revision.
    """

    planned_revision = plan.source_model.revision
    if revision is not None and revision != planned_revision:
        raise PlanningError(
            "conversion revision does not match the plan source revision: "
            f"{revision!r} != {planned_revision!r}"
        )
    effective_revision = revision if revision is not None else planned_revision
    expected_path = (
        Path(plan.source_model.local_path).expanduser().resolve()
        if plan.source_model.local_path
        else None
    )
    supplied_path = Path(model).expanduser()
    if supplied_path.is_dir():
        resolved = supplied_path.resolve()
        if expected_path is None:
            raise PlanningError(
                "a local conversion source requires a plan-bound source_model.local_path"
            )
        if resolved != expected_path:
            raise PlanningError(
                "local conversion source does not match the plan source path: "
                f"{resolved} != {expected_path}"
            )
        return resolved

    if model != plan.source_model.model_id:
        raise PlanningError(
            "conversion model does not match the plan source model: "
            f"{model!r} != {plan.source_model.model_id!r}"
        )
    try:
        resolved = resolve_model_dir(
            plan.source_model.model_id,
            revision=effective_revision,
            allow_download=False,
        )
    except ArtifactError:
        if expected_path is None or not expected_path.is_dir():
            raise
        resolved = expected_path
    if expected_path is not None and resolved != expected_path:
        raise PlanningError(
            "resolved conversion checkpoint does not match the plan-bound local source"
        )
    return resolved


def _resolve_external_mtp_sidecar_file(sidecar: Path) -> Path:
    if not sidecar.is_dir():
        return sidecar
    for name in EXTERNAL_MTP_SIDECAR_FILENAMES:
        candidate = sidecar / name
        if candidate.is_file():
            return candidate
    raise ArtifactError(
        f"MTP sidecar directory {sidecar} does not contain any of "
        f"{sorted(EXTERNAL_MTP_SIDECAR_FILENAMES)}"
    )


def _copy_external_mtp_bundle(sidecar: Path, output_dir: Path) -> None:
    source = _resolve_external_mtp_sidecar_file(sidecar)
    if not source.is_file():
        raise ArtifactError(f"MTP sidecar does not exist: {source}")
    _validate_mtp_sidecar_provenance(source)
    _copy_verified(source, output_dir / "mtp.safetensors")
    for companion_name in ("mtplx_runtime.json", "ax_mtp_sidecar_manifest.json"):
        companion = source.parent / companion_name
        if companion.is_file():
            _copy_verified(companion, output_dir / companion_name)
    _declare_raw_mtp_norm_layout(output_dir)


def _declare_raw_mtp_norm_layout(output_dir: Path) -> None:
    """Declare the byte-preserved sidecar's norm representation for AX Engine.

    Byte preservation keeps the raw HF zero-centred norm deltas. AX Engine
    reads ``mtp_norm_layout`` from ``mtplx_runtime.json`` and applies the
    ``+1.0`` HF-delta conversion to every norm at load time; without the
    declaration it must guess from tensor statistics. Only the sidecar tensor
    payloads are byte-preserved; the runtime contract is AXQuant metadata, so
    adding the declaration does not touch preserved bytes. An explicit layout
    declaration copied from the source bundle always wins.
    """
    runtime_path = output_dir / "mtplx_runtime.json"
    contract: dict[str, Any] = {
        "schema_version": "axquant.mtp-runtime.v1",
        "mtp_depth_max": 1,
    }
    if runtime_path.is_file():
        try:
            value = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"copied mtplx_runtime.json is unreadable: {runtime_path}") from exc
        if not isinstance(value, dict):
            raise ArtifactError("copied mtplx_runtime.json must be a JSON object")
        contract = value
    changed = False
    if "mtp_norm_layout" not in contract:
        contract["mtp_norm_layout"] = "raw_hf_delta"
        changed = True
    if "mtp_sidecar_bits" not in contract:
        sidecar_description = contract.get("mtp_sidecar")
        if isinstance(sidecar_description, str):
            match = re.search(r"INT(16|8|6|4)(?![0-9])", sidecar_description.upper())
            if match:
                # AX Engine prefers this structured field over its free-text
                # heuristic, which guesses 4-bit for anything non-INT8.
                contract["mtp_sidecar_bits"] = int(match.group(1))
                changed = True
    if changed:
        write_data(runtime_path, contract)


def _validated_calibration_source(
    plan: QuantizationPlan,
    calibration_manifest: str | Path | None,
) -> Path | None:
    evidence = plan.calibration
    if evidence is None:
        if calibration_manifest is not None:
            raise PlanningError("plan has no calibration evidence to bind the supplied manifest")
        return None
    if calibration_manifest is None:
        if plan.evidence_kind.release_quality:
            raise PlanningError("measured conversion requires --calibration-manifest")
        return None
    source = Path(calibration_manifest).expanduser().resolve()
    if not source.is_file():
        raise ArtifactError(f"calibration manifest does not exist: {source}")
    expected_sha256 = evidence.metadata.get("calibration_manifest_sha256")
    if not isinstance(expected_sha256, str) or expected_sha256 in {"", "unknown"}:
        raise PlanningError("plan calibration evidence has no manifest checksum")
    manifest = load_model(source, CalibrationManifest)
    # Current calibration evidence uses AXQuant's canonical artifact identity,
    # while early development artifacts stored the byte-level file checksum.
    # Accept the latter for backwards compatibility, but always recognize the
    # canonical hash emitted by the probe so a measured plan can bind its
    # manifest regardless of its timestamp or JSON formatting.
    if not calibration_manifest_matches(source, manifest, expected_sha256):
        raise PlanningError("calibration manifest checksum does not match the plan")
    same_model_identity = (
        manifest.model.model_id == plan.source_model.model_id
        and manifest.model.revision == plan.source_model.revision
        and manifest.model.format == plan.source_model.format
    )
    architectures_conflict = (
        manifest.model.architecture is not None
        and plan.source_model.architecture is not None
        and manifest.model.architecture != plan.source_model.architecture
    )
    if not same_model_identity or architectures_conflict:
        raise PlanningError("calibration manifest source model does not match the plan")
    if manifest.profile != plan.profile:
        raise PlanningError("calibration manifest profile does not match the plan")
    cache_directory_reference = evidence.dataset_id == str(source.parent)
    if (
        (manifest.dataset_id != evidence.dataset_id and not cache_directory_reference)
        or manifest.dataset_sha256 != evidence.dataset_sha256
        or manifest.samples != evidence.samples
        or set(manifest.domains) != set(evidence.domains)
        or manifest.sequence_length != evidence.sequence_length
    ):
        raise PlanningError("calibration manifest contents do not match the plan evidence")
    if not manifest.calibration_evaluation_separation_attested:
        raise PlanningError("calibration manifest lacks evaluation-separation attestation")
    return source


def _validated_activation_capture(
    plan: QuantizationPlan,
    calibration_activations: Mapping[str, Any] | None,
) -> LoadedActivationCapture | None:
    """Bind AWQ/GPTQ conversion to the exact capture used by analysis.

    Architecture-prior/manual development plans have no calibration evidence
    to carry a digest, but still require a verified capture wrapper.  Every
    measured plan must carry and match the full capture binding.
    """
    methods = {
        allocation.method
        for allocation in plan.assignments
        if allocation.bits < 16 and allocation.method in _ACTIVATION_REFINEMENT_METHODS
    }
    if not methods:
        return None
    if calibration_activations is None:
        raise PlanningError(
            "AWQ/GPTQ conversion requires calibration activations from capture-activations"
        )
    if not isinstance(calibration_activations, LoadedActivationCapture):
        raise PlanningError(
            "AWQ/GPTQ conversion requires a checksum-bound capture loaded with "
            "load_capture_activations; an unbound activation mapping is not evidence"
        )
    capture = calibration_activations
    evidence = plan.calibration
    if evidence is not None:
        issues = activation_capture_evidence_issues(
            capture.manifest,
            evidence.metadata,
            model_id=plan.source_model.model_id,
            revision=plan.source_model.revision,
            dataset_id=evidence.dataset_id,
        )
        if issues:
            raise PlanningError(f"activation capture does not match the plan: {issues}")
    elif plan.evidence_kind.release_quality:
        raise PlanningError("measured AWQ/GPTQ plan has no activation-capture provenance")
    elif (
        capture.manifest.model != plan.source_model.model_id
        or capture.manifest.revision != plan.source_model.revision
    ):
        raise PlanningError("activation capture source model does not match the plan")
    return capture


def _validated_kv_sensitivity_source(
    plan: QuantizationPlan,
    kv_sensitivity: str | Path | None,
) -> Path | None:
    kv = plan.kv_cache
    if kv is None or kv.allocation_basis != "measured":
        if kv_sensitivity is not None:
            raise PlanningError(
                "the plan has no measured KV-cache section to bind the supplied report"
            )
        return None
    if kv_sensitivity is None:
        raise PlanningError(
            "a measured KV-cache plan requires --kv-sensitivity with its bound report (AXQ-025)"
        )
    source = Path(kv_sensitivity).expanduser().resolve()
    if not source.is_file():
        raise ArtifactError(f"KV sensitivity report does not exist: {source}")
    report = load_model(source, KvSensitivityReport)
    if stable_sha256(report) != kv.sensitivity_sha256:
        raise PlanningError("KV sensitivity report digest does not match the plan binding")
    if (
        report.model.model_id != plan.source_model.model_id
        or report.model.revision != plan.source_model.revision
    ):
        raise PlanningError("KV sensitivity report source model does not match the plan")
    return source


def _safetensor_header(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        with path.open("rb") as source:
            header_size_bytes = source.read(8)
            if len(header_size_bytes) != 8:
                raise ArtifactError(f"invalid Safetensors header: {path}")
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header = json.loads(source.read(header_size))
    except (OSError, json.JSONDecodeError, struct.error) as exc:
        raise ArtifactError(f"cannot read Safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ArtifactError(f"invalid Safetensors header object: {path}")
    return 8 + header_size, header


def _source_weight_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read source weight index: {exc}") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            raise ArtifactError("source weight index has no weight_map")
        result: dict[str, Path] = {}
        for tensor_name, relative_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(relative_name, str):
                raise ArtifactError("source weight index contains a non-string entry")
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ArtifactError(f"source weight index contains unsafe path {relative_name}")
            result[tensor_name] = model_dir / relative
        return result
    result = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        if shard.name in {"mtp.safetensors", "vision.safetensors"}:
            continue
        _, header = _safetensor_header(shard)
        for tensor_name in header:
            if tensor_name != "__metadata__":
                if tensor_name in result:
                    raise ArtifactError(f"duplicate source tensor {tensor_name}")
                result[tensor_name] = shard
    return result


def _restore_protected_vision_config(model_dir: Path, output_dir: Path) -> None:
    """Restore upstream vision metadata removed by text-weight MLX conversion.

    MLX-LM converts the language trunk and may omit ``vision_config`` even when
    AXQuant subsequently restores the protected vision tensors into their own
    sidecar.  AX Engine 6.12+ indexes that sidecar and therefore needs the
    original vision contract to interpret it.  Copy only the immutable
    architecture/token fields from the revision-pinned source and fail closed
    if the converted config already contains conflicting values.
    """

    source_path = model_dir / "config.json"
    output_path = output_dir / "config.json"
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        converted = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot restore protected vision config: {exc}") from exc
    if not isinstance(source, dict) or not isinstance(converted, dict):
        raise ArtifactError("protected vision config files must contain JSON objects")
    source_vision = source.get("vision_config")
    if not isinstance(source_vision, dict) or not source_vision:
        raise ArtifactError("protected vision tensors require a non-empty source vision_config")

    protected_fields = (
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "language_model_only",
    )
    changed = False
    for field_name in protected_fields:
        if field_name not in source:
            continue
        source_value = source[field_name]
        if field_name not in converted:
            converted[field_name] = source_value
            changed = True
        elif converted[field_name] != source_value:
            if field_name == "vision_config" and converted[field_name] is None:
                converted[field_name] = source_value
                changed = True
            else:
                raise ArtifactError(
                    f"converted config conflicts with protected source field {field_name}"
                )
    if changed:
        write_data(output_path, converted)


def _extract_protected_vision(
    model_dir: Path,
    plan: QuantizationPlan,
    output_dir: Path,
) -> ProtectedTensorSidecarManifest | None:
    return _extract_protected_sidecar(
        model_dir,
        plan,
        output_dir,
        role_label="vision",
        output_name="vision.safetensors",
        manifest_name="axquant_vision_sidecar_manifest.json",
        allocations=[
            allocation for allocation in plan.assignments if allocation.role == TensorRole.VISION
        ],
    )


def _extract_protected_integrated_mtp(
    model_dir: Path,
    plan: QuantizationPlan,
    output_dir: Path,
) -> ProtectedTensorSidecarManifest | None:
    """Extract integrated MTP tensors into the canonical external sidecar.

    Families like Qwen 3.5 store the MTP head inside the indexed shards; the
    MLX-LM text-model mapping does not carry those tensors, so without this
    extraction the fail-closed parameter-coverage check aborts conversion.
    Byte-copying them into ``mtp.safetensors`` preserves the raw HF payloads
    exactly (the same contract as a byte-preserved external sidecar) and gives
    every converted artifact one MTP layout regardless of how the source
    packaged the head.
    """
    manifest = _extract_protected_sidecar(
        model_dir,
        plan,
        output_dir,
        role_label="mtp",
        output_name="mtp.safetensors",
        manifest_name="axquant_mtp_sidecar_manifest.json",
        allocations=[allocation for allocation in plan.assignments if allocation.role.is_mtp],
    )
    if manifest is not None:
        _declare_raw_mtp_norm_layout(output_dir)
    return manifest


def _extract_protected_sidecar(
    model_dir: Path,
    plan: QuantizationPlan,
    output_dir: Path,
    *,
    role_label: str,
    output_name: str,
    manifest_name: str,
    allocations: list[Any],
) -> ProtectedTensorSidecarManifest | None:
    if not allocations:
        return None
    if any(allocation.bits != 16 for allocation in allocations):
        raise PlanningError(f"protected {role_label} tensors must remain at reference precision")
    weight_map = _source_weight_map(model_dir)
    selected: list[tuple[str, Path, str, tuple[int, ...], int, int, int]] = []
    headers: dict[Path, tuple[int, dict[str, Any]]] = {}
    for allocation in sorted(allocations, key=lambda item: item.tensor):
        source_file = weight_map.get(allocation.tensor)
        if source_file is None or not source_file.is_file():
            raise ArtifactError(
                f"protected {role_label} tensor is missing from source: {allocation.tensor}"
            )
        if source_file not in headers:
            headers[source_file] = _safetensor_header(source_file)
        data_base, header = headers[source_file]
        entry = header.get(allocation.tensor)
        if not isinstance(entry, dict):
            raise ArtifactError(f"invalid source metadata for {allocation.tensor}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
        ):
            raise ArtifactError(f"invalid source tensor entry for {allocation.tensor}")
        selected.append(
            (
                allocation.tensor,
                source_file,
                dtype,
                tuple(shape),
                data_base,
                offsets[0],
                offsets[1],
            )
        )

    output_header: dict[str, Any] = {
        "__metadata__": {
            "format": "mlx",
            "axquant_role": f"protected-{role_label}",
            "source_revision": plan.source_model.revision or "unknown",
        }
    }
    output_offset = 0
    for tensor_name, _source_file, dtype, shape, _base, start, end in selected:
        length = end - start
        if length <= 0:
            raise ArtifactError(f"invalid source data offsets for {tensor_name}")
        output_header[tensor_name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [output_offset, output_offset + length],
        }
        output_offset += length
    header_bytes = json.dumps(
        output_header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    output_path = output_dir / output_name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        dir=output_dir,
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(struct.pack("<Q", len(header_bytes)))
            destination.write(header_bytes)
            for _name, source_file, _dtype, _shape, data_base, start, end in selected:
                remaining = end - start
                with source_file.open("rb") as source:
                    source.seek(data_base + start)
                    while remaining:
                        chunk = source.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise ArtifactError(
                                f"unexpected end of source tensor data in {source_file}"
                            )
                        destination.write(chunk)
                        remaining -= len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise

    _, verified_header = _safetensor_header(output_path)
    verified_names = sorted(name for name in verified_header if name != "__metadata__")
    expected_names = [entry[0] for entry in selected]
    if verified_names != expected_names:
        raise ArtifactError(f"protected {role_label} sidecar tensor coverage mismatch")
    source_files = sorted({entry[1] for entry in selected})
    manifest = ProtectedTensorSidecarManifest(
        source_model=plan.source_model,
        role=role_label,
        tensor_count=len(selected),
        parameters=sum(allocation.parameters for allocation in allocations),
        dtypes=tuple(sorted({entry[2] for entry in selected})),
        tensor_names_sha256=stable_sha256(expected_names),
        source_files=[
            ArtifactFile(
                path=source_file.name,
                size_bytes=source_file.stat().st_size,
                sha256=file_sha256(source_file),
            )
            for source_file in source_files
        ],
        output=ArtifactFile(
            path=output_path.name,
            size_bytes=output_path.stat().st_size,
            sha256=file_sha256(output_path),
        ),
    )
    write_data(output_dir / manifest_name, manifest)
    return manifest


def _artifact_files(output_dir: Path) -> list[ArtifactFile]:
    files: list[ArtifactFile] = []
    try:
        artifact_files = artifact_tree_files(output_dir)
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc
    for path in artifact_files:
        if path.name == "axquant_manifest.json":
            continue
        files.append(
            ArtifactFile(
                path=path.relative_to(output_dir).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
            )
        )
    return files


def _is_converted_quant_sidecar_name(tensor_path: str) -> bool:
    """True when a plan tensor is quant metadata rather than a logical weight.

    Mixed-precision exports use singular ``.scale`` / ``.bias`` sidecars next to
    weight bodies. HyperConnection learnable scales and MoE router gate biases
    are real parameters and must remain in converted coverage.
    """

    if tensor_path.endswith((".scales", ".biases")):
        return True
    if tensor_path.endswith(
        (".attn_hc.scale", ".ffn_hc.scale", ".hc_head.scale", ".hc_attn_scale", ".hc_ffn_scale")
    ) or tensor_path in {"hc_head_scale", "model.hc_head.scale"}:
        return False
    if tensor_path.endswith(".scale"):
        return True
    if tensor_path.endswith(".ffn.gate.bias") or tensor_path.endswith(
        ".ffn.gate.e_score_correction_bias"
    ):
        return False
    return tensor_path.endswith(".bias")


def _validated_plan_source_tensors(
    source_dir: Path,
    plan: QuantizationPlan,
) -> dict[str, Any]:
    """Verify that the bound source still has the exact logical plan coverage."""

    inventory = inspect_model(
        source_dir,
        model_id=plan.source_model.model_id,
        revision=plan.source_model.revision,
        # Plans may bind mixed-precision exports (e.g. DeepSeek V4 Flash FP4+FP8)
        # when produced with --allow-quantized re-pack inventory.
        allow_quantized=True,
    )
    metadata_names = {tensor.name for tensor in inventory.tensors if tensor.quantization_metadata}
    # Scale/bias sidecars of mixed-precision exports are plan bookkeeping only;
    # convert coverage compares logical weight tensors.
    expected = {
        allocation.tensor: allocation
        for allocation in plan.assignments
        if allocation.tensor not in metadata_names
    }
    actual = {
        tensor.name: tensor for tensor in inventory.tensors if not tensor.quantization_metadata
    }
    if set(actual) != set(expected):
        raise PlanningError(
            "conversion source tensor coverage does not match the plan: "
            f"missing={sorted(set(expected) - set(actual))[:10]}, "
            f"extra={sorted(set(actual) - set(expected))[:10]}"
        )
    for tensor_name, allocation in expected.items():
        source_tensor = actual[tensor_name]
        if (
            source_tensor.module_path != allocation.module_path
            or source_tensor.role is not allocation.role
            or source_tensor.parameters != allocation.parameters
        ):
            raise PlanningError(
                f"conversion source tensor {tensor_name} no longer matches its plan allocation"
            )
    if inventory.architecture_profile != plan.architecture_profile:
        raise PlanningError("conversion source architecture profile does not match the plan")
    return actual


def _bind_converted_tensors(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, tuple[Any, ...]]:
    """Bind plan tensors to every strictly declared converted output component."""

    bound: dict[str, tuple[Any, ...]] = {}
    used: set[str] = set()
    missing: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    actual_names = set(actual)
    fused_groups = _fused_expected_groups(expected)
    fused_members = {member for members in fused_groups.values() for member in members}
    for expected_name in expected:
        if expected_name in fused_members:
            continue
        component_names: list[str] = []
        invalid_matches: list[str] = []
        binding_groups = mlx_tensor_binding_groups(expected_name)
        for aliases in binding_groups:
            matches = sorted(set(aliases) & actual_names)
            if len(matches) != 1:
                invalid_matches.extend(matches)
                continue
            component_names.append(matches[0])
        if len(component_names) != len(binding_groups):
            missing.append(expected_name)
            continue
        if invalid_matches or len(set(component_names)) != len(component_names):
            ambiguous[expected_name] = sorted(set(invalid_matches + component_names))
            continue
        collisions = sorted(set(component_names) & used)
        if collisions:
            ambiguous[expected_name] = collisions
            continue
        used.update(component_names)
        bound[expected_name] = tuple(actual[name] for name in component_names)

    for target, members in fused_groups.items():
        aliases = mlx_tensor_binding_groups(members[0])[0]
        matches = sorted(set(aliases) & actual_names)
        if len(matches) == 0:
            missing.extend(members)
            continue
        if len(matches) != 1:
            ambiguous[target] = matches
            continue
        output_name = matches[0]
        if output_name in used:
            ambiguous[target] = [output_name]
            continue
        used.add(output_name)
        components = (actual[output_name],)
        for member in members:
            bound[member] = components

    extra = sorted(set(actual) - used)
    if missing or ambiguous or extra:
        raise ArtifactError(
            "converted checkpoint tensor coverage mismatch: "
            f"missing={sorted(missing)[:10]}, "
            f"ambiguous={dict(sorted(ambiguous.items())[:10])}, "
            f"extra={extra[:10]}"
        )
    return bound


def _fused_expected_groups(expected: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Group a complete contiguous set of indexed source experts by MLX stack target."""

    indexed: dict[str, list[tuple[int, str]]] = {}
    for tensor_name in expected:
        binding = fused_expert_tensor_target(tensor_name)
        if binding is not None:
            indexed.setdefault(binding[0], []).append((binding[1], tensor_name))

    groups: dict[str, tuple[str, ...]] = {}
    for output_target, members in indexed.items():
        ordered = sorted(members)
        indices = [index for index, _ in ordered]
        if indices != list(range(len(ordered))):
            # DeepSeek V4 FusedSwitchGLU stacks both w1 (gate) and w3 (up) into
            # switch_mlp.gate_proj, so each expert index appears twice.
            multiplicity: dict[int, int] = {}
            for index, _ in ordered:
                multiplicity[index] = multiplicity.get(index, 0) + 1
            unique = sorted(multiplicity)
            counts = set(multiplicity.values())
            if not (
                unique == list(range(len(unique))) and len(counts) == 1 and next(iter(counts)) > 1
            ):
                raise ArtifactError(
                    f"converted expert fusion {output_target} does not have contiguous "
                    f"source indices: {indices[:10]}"
                )
        groups[output_target] = tuple(name for _, name in ordered)
    return groups


def _logical_source_shape(source_tensor: Any) -> tuple[int, ...]:
    """Return the logical weight shape for re-pack shape checks.

    Mixed-precision exports store U32 packed columns. Parameter counts already
    use the logical width; convert expected shapes must as well, or re-packing
    from MXFP4/affine sources under-predicts the output pack width (GPT-OSS).
    """

    shape = tuple(int(dim) for dim in source_tensor.shape)
    if not shape:
        return shape
    current_bits = getattr(source_tensor, "current_bits", None)
    dtype = str(getattr(source_tensor, "dtype", "") or "")
    name = str(getattr(source_tensor, "name", "") or "")
    if (
        dtype == "U32"
        and isinstance(current_bits, int)
        and not isinstance(current_bits, bool)
        and current_bits > 0
        and current_bits < 16
        and name.endswith((".weight", "_blocks"))
    ):
        packed_last = shape[-1]
        logical_last = packed_last * 32 // current_bits
        if logical_last > 0 and packed_last * 32 == logical_last * current_bits:
            return (*shape[:-1], logical_last)
    return shape


def _expected_converted_shapes(
    tensor_name: str,
    source_shape: tuple[int, ...],
    bits: int,
) -> tuple[tuple[int, ...], ...]:
    """Derive exact output shapes, including MLX-LM's packed gate/up split."""

    component_count = len(mlx_tensor_binding_groups(tensor_name))
    if bits < 16:
        if not source_shape or source_shape[-1] * bits % 32:
            raise ArtifactError(
                f"source tensor {tensor_name} cannot be packed exactly at {bits} bits"
            )
        final_dimension = source_shape[-1] * bits // 32
    else:
        final_dimension = source_shape[-1]

    if component_count == 1:
        return ((*source_shape[:-1], final_dimension),)
    if component_count == 2 and len(source_shape) >= 2 and source_shape[-2] % 2 == 0:
        component_shape = (
            *source_shape[:-2],
            source_shape[-2] // 2,
            final_dimension,
        )
        return (component_shape, component_shape)
    raise ArtifactError(
        f"converted tensor {tensor_name} has no valid shape-conserving split "
        f"for source shape {source_shape}"
    )


def _fused_member_multiplicity(members: tuple[str, ...]) -> int:
    """How many plan tensors map to each expert index in a fused group.

    DeepSeek V4 FusedSwitchGLU binds both ``w1`` and ``w3`` to
    ``switch_mlp.gate_proj``, so each index appears twice. Qwen/Nemotron gate
    experts bind once per index even when the expert count is even — do not
    infer dual-concat from even group size alone.
    """

    counts: dict[int, int] = {}
    for member in members:
        binding = fused_expert_tensor_target(member)
        if binding is None:
            continue
        index = binding[1]
        counts[index] = counts.get(index, 0) + 1
    if not counts:
        return 1
    values = set(counts.values())
    if len(values) != 1:
        raise ArtifactError(
            "converted expert fusion has inconsistent per-index membership counts: "
            f"{sorted(counts.items())[:10]}"
        )
    return next(iter(values))


def _expected_fused_converted_shapes(
    tensor_name: str,
    source_shapes: tuple[tuple[int, ...], ...],
    bits: int,
    *,
    multiplicity: int = 1,
) -> tuple[tuple[int, ...], ...]:
    """Derive the exact leading-axis stack shape for indexed expert weights."""

    if not source_shapes or len(set(source_shapes)) != 1:
        raise ArtifactError(f"converted expert fusion {tensor_name} mixes source shapes")
    if multiplicity < 1:
        raise ArtifactError(f"converted expert fusion {tensor_name} has invalid multiplicity")
    base_shape = source_shapes[0]
    member_count = len(source_shapes)
    if member_count % multiplicity != 0:
        raise ArtifactError(
            f"converted expert fusion {tensor_name} member count {member_count} "
            f"is not divisible by multiplicity {multiplicity}"
        )
    expert_count = member_count // multiplicity
    fused_base = base_shape
    # DeepSeek V4 FusedSwitchGLU concatenates w1+w3 output rows into gate_proj
    # (axis 0 of each 2-D expert weight) when multiplicity is 2.
    if multiplicity > 1 and len(base_shape) >= 1:
        fused_base = (base_shape[0] * multiplicity, *base_shape[1:])
    component_shapes = _expected_converted_shapes(tensor_name, fused_base, bits)
    if len(component_shapes) != 1:
        raise ArtifactError(f"converted expert fusion {tensor_name} has multiple output components")
    return ((expert_count, *component_shapes[0]),)


def _shape_element_count(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _protected_shape_matches(
    plan: QuantizationPlan,
    tensor_name: str,
    source_shape: tuple[int, ...],
    actual_shape: tuple[int, ...],
) -> bool:
    if actual_shape == source_shape:
        return True
    model_type = plan.architecture_profile.config_model_type
    # deepseek_v4.sanitize reshapes 2-D Linear wo_a weights into MultiLinear 3-D
    # ``(o_groups, o_lora_rank, -1)`` without changing element count.
    if (
        model_type == "deepseek_v4"
        and tensor_name.endswith(".attn.wo_a.weight")
        and len(source_shape) == 2
        and len(actual_shape) == 3
        and actual_shape[-1] == source_shape[-1]
        and _shape_element_count(source_shape) == _shape_element_count(actual_shape)
    ):
        _LOG.info(
            "converted_shape_transform_verified",
            tensor=tensor_name,
            source_shape=source_shape,
            actual_shape=actual_shape,
            transform="mlx-lm-deepseek-v4-wo_a-multilinear-reshape",
        )
        return True
    qwen_hybrid_conv1d = model_type in {
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_next",
    } and tensor_name.endswith(".linear_attn.conv1d.weight")
    nemotron_conv1d = model_type == "nemotron_h" and tensor_name.endswith(".mixer.conv1d.weight")
    if (
        (qwen_hybrid_conv1d or nemotron_conv1d)
        and len(source_shape) == 3
        and source_shape[-1] != 1
        and actual_shape == (source_shape[0], source_shape[2], source_shape[1])
    ):
        _LOG.info(
            "converted_shape_transform_verified",
            tensor=tensor_name,
            source_shape=source_shape,
            actual_shape=actual_shape,
            transform=f"mlx-lm-{model_type}-conv1d-moveaxis-2-1",
        )
        return True
    qwen3_vl_patch_embed = (
        model_type == "qwen3_vl"
        and tensor_name.endswith(".visual.patch_embed.proj.weight")
        and len(source_shape) == 5
        and actual_shape
        == (
            source_shape[0],
            source_shape[2],
            source_shape[3],
            source_shape[4],
            source_shape[1],
        )
    )
    if qwen3_vl_patch_embed:
        _LOG.info(
            "converted_shape_transform_verified",
            tensor=tensor_name,
            source_shape=source_shape,
            actual_shape=actual_shape,
            transform="mlx-vlm-qwen3-vl-conv3d-out-dhw-in",
        )
        return True
    return False


def _verify_converted_weights(
    staging_dir: Path,
    plan: QuantizationPlan,
    *,
    source_tensors: Mapping[str, Any] | None = None,
) -> tuple[int, int, int, int, int, int, float, float]:
    if source_tensors is None:
        if not plan.source_model.local_path:
            raise ArtifactError(
                "converted weight verification requires the plan-bound local source path"
            )
        source_tensors = _validated_plan_source_tensors(
            Path(plan.source_model.local_path).expanduser().resolve(),
            plan,
        )
    inventory = inspect_model(
        staging_dir,
        model_id=plan.source_model.model_id,
        revision=plan.source_model.revision,
        allow_quantized=True,
    )
    expected_parameters = sum(allocation.parameters for allocation in plan.assignments)
    expected_mtp_parameters = sum(
        allocation.parameters for allocation in plan.assignments if allocation.role.is_mtp
    )
    actual_mtp_parameters = sum(
        tensor.parameters for tensor in inventory.tensors if tensor.role.is_mtp
    )
    expected_vision_parameters = sum(
        allocation.parameters
        for allocation in plan.assignments
        if allocation.role == TensorRole.VISION
    )
    actual_vision_parameters = sum(
        tensor.parameters for tensor in inventory.tensors if tensor.role == TensorRole.VISION
    )
    expected_audio_parameters = sum(
        allocation.parameters
        for allocation in plan.assignments
        if allocation.role == TensorRole.AUDIO
    )
    actual_audio_parameters = sum(
        tensor.parameters for tensor in inventory.tensors if tensor.role == TensorRole.AUDIO
    )
    # Scale/bias sidecars of mixed-precision sources are not logical converted
    # weight tensors after affine re-pack (they become MLX quant metadata).
    # MoE router ``*.ffn.gate.bias`` is a real parameter (DeepSeek renames it to
    # ``e_score_correction_bias``); keep it in expected coverage.
    expected_tensors = {
        allocation.tensor: allocation
        for allocation in plan.assignments
        if allocation.parameters > 0 and not _is_converted_quant_sidecar_name(allocation.tensor)
    }
    # Exclude the same quant-sidecar name classes from the converted side that
    # were dropped from expected_tensors (``.bias`` Linear biases stay float
    # under nn.quantize; GPT-OSS expert bias is 2-D BF16 and is not plan-bound).
    output_tensors = {
        tensor.name: tensor
        for tensor in inventory.tensors
        if not tensor.quantization_metadata and not _is_converted_quant_sidecar_name(tensor.name)
    }
    actual_tensors = _bind_converted_tensors(expected_tensors, output_tensors)
    fused_groups = _fused_expected_groups(expected_tensors)
    fused_by_member = {
        member: (target, members) for target, members in fused_groups.items() for member in members
    }
    verified_groups: set[str] = set()
    metadata_names = {tensor.name for tensor in inventory.tensors if tensor.quantization_metadata}
    for tensor_name, allocation in expected_tensors.items():
        fused = fused_by_member.get(tensor_name)
        verification_name: str
        members: tuple[str, ...]
        if fused is None:
            verification_name = tensor_name
            members = (tensor_name,)
        else:
            verification_name, members = fused
        if verification_name in verified_groups:
            continue
        verified_groups.add(verification_name)

        allocations = tuple(expected_tensors[member] for member in members)
        packing = {(item.bits, item.group_size, item.method, item.role) for item in allocations}
        if len(packing) != 1 or (fused is not None and allocation.role is not TensorRole.EXPERT):
            raise ArtifactError(
                f"converted expert fusion {verification_name} mixes plan packing or roles"
            )
        allocation = allocations[0]
        actual_components = actual_tensors[members[0]]
        if any(
            len(actual_tensors[member]) != len(actual_components)
            or any(
                member_component is not actual_component
                for member_component, actual_component in zip(
                    actual_tensors[member],
                    actual_components,
                    strict=True,
                )
            )
            for member in members[1:]
        ):
            raise ArtifactError(
                f"converted expert fusion {verification_name} has inconsistent output bindings"
            )
        actual_parameters = sum(component.parameters for component in actual_components)
        expected_group_parameters = sum(item.parameters for item in allocations)
        if actual_parameters != expected_group_parameters:
            raise ArtifactError(
                f"converted tensor {verification_name} parameter count does not match the plan: "
                f"{actual_parameters} != {expected_group_parameters}"
            )
        source_shapes = tuple(
            _logical_source_shape(source_tensors[member]) for member in members
        )
        source_shape = source_shapes[0]
        if fused is not None:
            expected_shapes = _expected_fused_converted_shapes(
                tensor_name,
                source_shapes,
                allocation.bits,
                multiplicity=_fused_member_multiplicity(members),
            )
        else:
            expected_shapes = _expected_converted_shapes(
                tensor_name,
                source_shape,
                allocation.bits,
            )
        actual_shapes = tuple(component.shape for component in actual_components)
        if allocation.bits < 16:
            if actual_shapes != expected_shapes or any(
                component.current_bits != allocation.bits
                or component.current_group_size != allocation.group_size
                or component.current_method is not QuantMethod.AFFINE
                for component in actual_components
            ):
                raise ArtifactError(
                    f"converted tensor {verification_name} packing does not match the plan: "
                    f"expected shapes {expected_shapes}, affine {allocation.bits}-bit "
                    f"group {allocation.group_size}; found shapes {actual_shapes}"
                )
            required_metadata = {
                f"{component.module_path}.{suffix}"
                for component in actual_components
                for suffix in ("scales", "biases")
            }
            missing_metadata = required_metadata - metadata_names
            if missing_metadata:
                raise ArtifactError(
                    f"converted tensor {verification_name} lacks affine metadata: "
                    f"{sorted(missing_metadata)}"
                )
        else:
            shape_matches = actual_shapes == expected_shapes
            if len(actual_components) == 1 and not shape_matches:
                shape_matches = _protected_shape_matches(
                    plan,
                    tensor_name,
                    source_shape,
                    actual_components[0].shape,
                )
            if not shape_matches:
                raise ArtifactError(
                    f"protected tensor {verification_name} shape changed during conversion: "
                    f"{actual_shapes} != {expected_shapes}"
                )
            # Byte-preserved MTP sidecars keep the source's native packing
            # (DeepSeek V4 Flash experts stay FP4/I8). Only non-MTP protected
            # tensors must remain at reference (≥16-bit) precision.
            if not allocation.role.is_mtp and any(
                component.current_bits is not None and component.current_bits < 16
                for component in actual_components
            ):
                raise ArtifactError(
                    f"protected tensor {verification_name} was packed below its "
                    f"{allocation.bits}-bit plan"
                )
    if inventory.total_parameters != expected_parameters:
        raise ArtifactError(
            "converted checkpoint logical parameter coverage mismatch: "
            f"expected {expected_parameters}, found {inventory.total_parameters}"
        )
    if actual_mtp_parameters != expected_mtp_parameters:
        raise ArtifactError(
            "converted checkpoint MTP parameter coverage mismatch: "
            f"expected {expected_mtp_parameters}, found {actual_mtp_parameters}"
        )
    if actual_vision_parameters != expected_vision_parameters:
        raise ArtifactError(
            "converted checkpoint vision parameter coverage mismatch: "
            f"expected {expected_vision_parameters}, found {actual_vision_parameters}"
        )
    if actual_audio_parameters != expected_audio_parameters:
        raise ArtifactError(
            "converted checkpoint audio parameter coverage mismatch: "
            f"expected {expected_audio_parameters}, found {actual_audio_parameters}"
        )

    weight_files = [staging_dir / relative for relative in inventory.source_files]
    if not weight_files:
        raise ArtifactError("converted checkpoint contains no Safetensors weight files")
    try:
        actual_weight_files = {
            path.relative_to(staging_dir).as_posix()
            for path in artifact_tree_files(staging_dir)
            if path.suffix.lower() == ".safetensors"
        }
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc
    inventoried_weight_files = {path.relative_to(staging_dir).as_posix() for path in weight_files}
    if actual_weight_files != inventoried_weight_files:
        raise ArtifactError(
            "converted checkpoint Safetensors coverage mismatch: "
            f"missing={sorted(actual_weight_files - inventoried_weight_files)}, "
            f"extra={sorted(inventoried_weight_files - actual_weight_files)}"
        )
    mtp_files = [path for path in weight_files if path.name == "mtp.safetensors"]
    protected_files = [path for path in weight_files if path.name == "vision.safetensors"]
    main_files = [path for path in weight_files if path.name != "mtp.safetensors"]
    weight_file_size_bytes = sum(path.stat().st_size for path in weight_files)
    main_weight_file_size_bytes = sum(path.stat().st_size for path in main_files)
    mtp_weight_file_size_bytes = sum(path.stat().st_size for path in mtp_files)
    protected_weight_file_size_bytes = sum(path.stat().st_size for path in protected_files)
    main_logical_parameters = expected_parameters - expected_mtp_parameters
    if main_logical_parameters <= 0:
        raise ArtifactError("converted checkpoint has no main-model logical parameters")
    measured_total_bpw = 8.0 * weight_file_size_bytes / expected_parameters
    measured_main_bpw = 8.0 * main_weight_file_size_bytes / main_logical_parameters
    return (
        expected_parameters,
        main_logical_parameters,
        weight_file_size_bytes,
        main_weight_file_size_bytes,
        mtp_weight_file_size_bytes,
        protected_weight_file_size_bytes,
        measured_total_bpw,
        measured_main_bpw,
    )


def convert_model(
    *,
    model: str,
    plan: QuantizationPlan,
    output: str | Path,
    revision: str | None = None,
    mtp_sidecar: str | Path | None = None,
    mtp_layout: MtpSidecarLayout = MtpSidecarLayout.BYTE_PRESERVED,
    calibration_manifest: str | Path | None = None,
    kv_sensitivity: str | Path | None = None,
    calibration_activations: Mapping[str, Any] | None = None,
    allow_unmeasured: bool = False,
    ax_engine_manifest: Literal["required", "if-available", "skip"] = "required",
    ax_engine_bench: str = "ax-engine-bench",
) -> ArtifactManifest:
    if not plan.evidence_kind.release_quality and not allow_unmeasured:
        raise PlanningError(
            "conversion requires measured evidence; pass --allow-unmeasured only for dry runs"
        )
    calibration_source = _validated_calibration_source(plan, calibration_manifest)
    bound_capture = _validated_activation_capture(plan, calibration_activations)
    assert_conversion_scope(plan)
    kv_sensitivity_source = _validated_kv_sensitivity_source(plan, kv_sensitivity)
    quantized_allocations = [allocation for allocation in plan.assignments if allocation.bits < 16]
    if not quantized_allocations:
        raise PlanningError("conversion plan contains no quantized assignments")
    mtp_allocations = [allocation for allocation in plan.assignments if allocation.role.is_mtp]
    if plan.architecture_profile.mtp_declared and not mtp_allocations:
        raise PlanningError(
            "the architecture declares MTP but the plan contains no MTP tensor allocations"
        )
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists():
        raise ArtifactError(f"conversion output already exists: {output_dir}")
    # Resolve the physical source early so architecture prep (e.g. gemma4_unified
    # → gemma4 text path) can stage beside the output rather than on /tmp.
    original_source_dir = _resolve_bound_conversion_source(model, plan, revision)
    source_tensors = _validated_plan_source_tensors(original_source_dir, plan)
    if (
        plan.mtp.preserve_external_sidecar
        and mtp_allocations
        and mtp_sidecar is None
        and _source_has_external_mtp_sidecar(original_source_dir)
    ):
        raise PlanningError("MTP sidecar preservation requires --mtp-sidecar")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    staging_dir = temporary_root / "artifact"
    prep_dir = temporary_root / "prepared-source"
    convert_model_ref = str(original_source_dir)
    convert_revision = None
    try:
        prepared = (
            prepare_conversion_source(
                original_source_dir,
                work_dir=prep_dir,
                model_id=plan.source_model.model_id,
            )
            if original_source_dir.is_dir()
            else None
        )
        if prepared is not None:
            convert_model_ref = str(prepared)
            convert_revision = None  # local prepared tree; no Hub revision
            _LOG.info(
                "conversion_source_prepared",
                original=str(original_source_dir),
                prepared=convert_model_ref,
            )
        backend = conversion_backend(plan)
        if backend == "mlx-lm" and any(
            allocation.role == TensorRole.AUDIO for allocation in plan.assignments
        ):
            # The mlx-lm backend has no protected-audio delivery path (the
            # prepared text view drops audio towers and nothing restores
            # them), so the post-conversion coverage check could never pass;
            # fail before the expensive conversion with an accurate reason.
            raise PlanningError(
                "plan contains protected audio tensors the mlx-lm backend cannot "
                "deliver; convert this checkpoint through its multimodal backend"
            )
        predicate = build_quant_predicate(
            plan,
            execute_refinement=False,
            calibration_activations=calibration_activations,
        )
        _LOG.info("conversion_preflight_started", model=convert_model_ref)
        if backend == "mlx-lm":
            _preflight_coverage(convert_model_ref, convert_revision, predicate)
        else:
            preflight_multimodal(Path(convert_model_ref), plan, predicate)
        predicate = build_quant_predicate(plan, calibration_activations=calibration_activations)
        default_quantized_bits = min(allocation.bits for allocation in quantized_allocations)
        try:
            if backend == "mlx-lm":
                _mlx_convert_with_optional_dequant(
                    convert_model_ref,
                    mlx_path=str(staging_dir),
                    quantize=True,
                    q_group_size=plan.group_size,
                    q_bits=default_quantized_bits,
                    quant_predicate=predicate,
                    revision=convert_revision,
                )
            else:
                convert_multimodal(
                    Path(convert_model_ref),
                    staging_dir,
                    plan,
                    predicate,
                    default_quantized_bits,
                )
        except Exception as exc:
            raise ArtifactError(f"{backend} conversion failed: {exc}") from exc
        unmatched = predicate.unmatched_quantized_modules()
        if unmatched:
            raise PlanningError(
                f"{backend} did not visit planned modules: {sorted(unmatched)[:10]}"
            )
        execution_records = [
            QuantizerExecutionRecord(
                method=allocation.method,
                module_path=allocation.module_path,
                bits=allocation.bits,
                group_size=allocation.group_size,
                success=allocation.module_path in predicate.matched,
                note=(
                    "DWQ percentile clipping followed by portable affine packing"
                    if allocation.method.value == "dwq"
                    else (
                        "AWQ activation scaling followed by portable affine packing"
                        if allocation.method.value == "awq"
                        else (
                            "GPTQ Hessian error compensation followed by portable affine packing"
                            if allocation.method.value == "gptq"
                            else (
                                "group-preserving act-order GPTQ Hessian error compensation "
                                "followed by portable affine packing"
                                if allocation.method.value == "gptq-act"
                                else f"{backend} affine packing"
                            )
                        )
                    )
                ),
                metadata=(
                    predicate.method_metadata.get(allocation.module_path, {})
                    if allocation.method.value in ("awq", "gptq", "gptq-act")
                    else predicate.dwq_metadata.get(allocation.module_path, {})
                ),
            )
            for allocation in quantized_allocations
        ]
        write_data(
            staging_dir / "axquant_quantizer_execution.json",
            QuantizerExecutionManifest(
                plan_sha256=stable_sha256(plan),
                records=execution_records,
            ),
        )
        # Always extract protected tensors from the original checkpoint, not the
        # prepared MLX text-path view (which filters multimodal weights).
        fallback_dirs = [original_source_dir, Path(model).expanduser()]
        if plan.source_model.local_path:
            fallback_dirs.append(Path(plan.source_model.local_path).expanduser())
        source_model_dir: Path | None = None
        for candidate_dir in fallback_dirs:
            if candidate_dir.is_dir():
                source_model_dir = candidate_dir.resolve()
                break
        if backend == "mlx-lm" and any(
            allocation.role == TensorRole.VISION for allocation in plan.assignments
        ):
            if source_model_dir is None:
                raise ArtifactError(
                    "protected vision extraction requires a resolved local source checkpoint"
                )
            _restore_protected_vision_config(source_model_dir, staging_dir)
            _extract_protected_vision(source_model_dir, plan, staging_dir)
        if mtp_sidecar is not None:
            if not plan.mtp.preserve_external_sidecar:
                raise PlanningError("the plan does not permit an external MTP sidecar")
            if mtp_layout == MtpSidecarLayout.BYTE_PRESERVED:
                _copy_external_mtp_bundle(
                    Path(mtp_sidecar).expanduser().resolve(),
                    staging_dir,
                )
            elif mtp_layout == MtpSidecarLayout.AX_ENGINE_QWEN36_V1:
                prepare_qwen36_mtp_sidecar(
                    mtp_sidecar,
                    staging_dir,
                    source_model=plan.source_model,
                )
            else:
                raise PlanningError(f"unsupported MTP sidecar layout: {mtp_layout}")
        elif (
            plan.mtp.preserve_external_sidecar
            and any(allocation.role.is_mtp for allocation in plan.assignments)
            and source_model_dir is not None
            and not _source_has_external_mtp_sidecar(source_model_dir)
        ):
            # Integrated MTP (e.g. Qwen 3.5): the MLX-LM text mapping drops the
            # in-shard MTP tensors, so byte-copy them into the canonical
            # external sidecar before the parameter-coverage check runs.
            _extract_protected_integrated_mtp(source_model_dir, plan, staging_dir)
        if calibration_source is not None:
            _copy_verified(calibration_source, staging_dir / "calibration_manifest.json")
        if bound_capture is not None:
            write_data(staging_dir / _CAPTURE_MANIFEST_NAME, bound_capture.manifest)
        if kv_sensitivity_source is not None:
            _copy_verified(kv_sensitivity_source, staging_dir / "kv_sensitivity.json")
        write_data(staging_dir / "axquant_plan.json", plan)
        if ax_engine_manifest == "required":
            require_ax_engine_manifest(staging_dir, executable=ax_engine_bench)
        elif ax_engine_manifest == "if-available":
            result = generate_ax_engine_manifest(staging_dir, executable=ax_engine_bench)
            if result.available and not result.passed:
                raise ArtifactError(
                    f"AX Engine manifest generation failed: {result.stderr or result.report}"
                )
            if not result.available:
                _LOG.warning("ax_engine_manifest_skipped", reason=result.stderr)
        runtime = build_runtime_metadata(plan, staging_dir)
        write_data(staging_dir / "axquant_runtime.json", runtime)
        (
            logical_parameters,
            main_logical_parameters,
            weight_file_size_bytes,
            main_weight_file_size_bytes,
            mtp_weight_file_size_bytes,
            protected_weight_file_size_bytes,
            measured_total_bpw,
            measured_main_bpw,
        ) = _verify_converted_weights(
            staging_dir,
            plan,
            source_tensors=source_tensors,
        )
        manifest = ArtifactManifest(
            axquant_version=plan.software_versions.axquant,
            source_model=plan.source_model,
            plan_sha256=stable_sha256(plan),
            calibration=plan.calibration,
            profile=plan.profile,
            target_class=plan.target_class,
            effective_bpw=plan.effective_bpw,
            logical_parameters=logical_parameters,
            main_logical_parameters=main_logical_parameters,
            weight_file_size_bytes=weight_file_size_bytes,
            main_weight_file_size_bytes=main_weight_file_size_bytes,
            mtp_weight_file_size_bytes=mtp_weight_file_size_bytes,
            protected_weight_file_size_bytes=protected_weight_file_size_bytes,
            measured_total_bpw=measured_total_bpw,
            measured_main_bpw=measured_main_bpw,
            weight_distribution=plan.weight_distribution,
            mtp_distribution=plan.mtp_distribution,
            mtp_present=bool(plan.mtp_distribution) or mtp_sidecar is not None,
            mtp_policy=plan.mtp,
            runtime=runtime,
            software_versions=plan.software_versions,
            files=_artifact_files(staging_dir),
        )
        write_data(staging_dir / "axquant_manifest.json", manifest)
        if output_dir.exists():
            raise ArtifactError(f"conversion output appeared during conversion: {output_dir}")
        staging_dir.rename(output_dir)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    _LOG.info(
        "conversion_completed",
        output=str(output_dir),
        measured_total_bpw=manifest.measured_total_bpw,
        files=len(manifest.files),
    )
    return manifest
