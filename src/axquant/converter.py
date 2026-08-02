from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import struct
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import structlog

from axquant.errors import ArtifactError, BackendUnavailableError, PlanningError
from axquant.inspector import inspect_model, resolve_model_dir
from axquant.mtp_sidecar import EXTERNAL_MTP_SIDECAR_FILENAMES, prepare_qwen36_mtp_sidecar
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
    TensorRole,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data
from axquant.source_prep import prepare_conversion_source

_LOG = structlog.get_logger()


def _mlx_api() -> tuple[Any, Any]:
    try:
        mlx_lm = importlib.import_module("mlx_lm")
    except ModuleNotFoundError as exc:
        raise BackendUnavailableError(
            "the MLX backend is not installed; install axquant[mlx]"
        ) from exc
    return mlx_lm.convert, mlx_lm.load


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
    if not isinstance(expected_size, int) or source.stat().st_size != expected_size:
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
    if "mtp_norm_layout" not in contract:
        contract["mtp_norm_layout"] = "raw_hf_delta"
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
    canonical_sha256 = stable_sha256(manifest.model_dump(mode="json", exclude={"created_at"}))
    # Current calibration evidence uses AXQuant's canonical artifact identity,
    # while early development artifacts stored the byte-level file checksum.
    # Accept the latter for backwards compatibility, but always recognize the
    # canonical hash emitted by the probe so a measured plan can bind its
    # manifest regardless of its timestamp or JSON formatting.
    if expected_sha256 not in {file_sha256(source), canonical_sha256}:
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
            or not all(isinstance(value, int) for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
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
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
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


def _verify_converted_weights(
    staging_dir: Path,
    plan: QuantizationPlan,
) -> tuple[int, int, int, int, int, int, float, float]:
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

    weight_files = [staging_dir / relative for relative in inventory.source_files]
    if not weight_files:
        raise ArtifactError("converted checkpoint contains no Safetensors weight files")
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
    awq_activations: Mapping[str, Any] | None = None,
    allow_unmeasured: bool = False,
    ax_engine_manifest: Literal["required", "if-available", "skip"] = "required",
    ax_engine_bench: str = "ax-engine-bench",
) -> ArtifactManifest:
    if not plan.evidence_kind.release_quality and not allow_unmeasured:
        raise PlanningError(
            "conversion requires measured evidence; pass --allow-unmeasured only for dry runs"
        )
    calibration_source = _validated_calibration_source(plan, calibration_manifest)
    assert_conversion_scope(plan)
    kv_sensitivity_source = _validated_kv_sensitivity_source(plan, kv_sensitivity)
    quantized_allocations = [allocation for allocation in plan.assignments if allocation.bits < 16]
    if not quantized_allocations:
        raise PlanningError("conversion plan contains no quantized assignments")
    if (
        plan.mtp.preserve_external_sidecar
        and any(allocation.role.is_mtp for allocation in plan.assignments)
        and mtp_sidecar is None
        and _source_has_external_mtp_sidecar(model)
    ):
        raise PlanningError("MTP sidecar preservation requires --mtp-sidecar")
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists():
        raise ArtifactError(f"conversion output already exists: {output_dir}")
    # Resolve the physical source early so architecture prep (e.g. gemma4_unified
    # → gemma4 text path) can stage beside the output rather than on /tmp.
    try:
        original_source_dir = resolve_model_dir(model, revision=revision, allow_download=False)
    except Exception:
        original_source_dir = Path(model).expanduser()
        if not original_source_dir.is_dir():
            original_source_dir = Path(plan.source_model.local_path or "").expanduser()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    staging_dir = temporary_root / "artifact"
    prep_dir = temporary_root / "prepared-source"
    convert_model_ref = model
    convert_revision = revision
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
        predicate = build_quant_predicate(
            plan,
            execute_refinement=False,
            awq_activations=awq_activations,
        )
        _LOG.info("conversion_preflight_started", model=convert_model_ref)
        _preflight_coverage(convert_model_ref, convert_revision, predicate)
        convert, _ = _mlx_api()
        predicate = build_quant_predicate(plan, awq_activations=awq_activations)
        default_quantized_bits = min(allocation.bits for allocation in quantized_allocations)
        try:
            convert(
                convert_model_ref,
                mlx_path=str(staging_dir),
                quantize=True,
                q_group_size=plan.group_size,
                q_bits=default_quantized_bits,
                quant_predicate=predicate,
                revision=convert_revision,
            )
        except Exception as exc:
            raise ArtifactError(f"MLX-LM conversion failed: {exc}") from exc
        unmatched = predicate.unmatched_quantized_modules()
        if unmatched:
            raise PlanningError(f"MLX-LM did not visit planned modules: {sorted(unmatched)[:10]}")
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
                        else "MLX-LM affine packing"
                    )
                ),
                metadata=(
                    predicate.awq_metadata.get(allocation.module_path, {})
                    if allocation.method.value == "awq"
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
        source_model_dir = original_source_dir.resolve() if original_source_dir.is_dir() else Path()
        if not source_model_dir.is_dir():
            source_model_dir = Path(model).expanduser().resolve()
        if not source_model_dir.is_dir():
            source_model_dir = Path(plan.source_model.local_path or "").expanduser().resolve()
        if any(allocation.role == TensorRole.VISION for allocation in plan.assignments):
            if not source_model_dir.is_dir():
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
            and source_model_dir.is_dir()
            and not _source_has_external_mtp_sidecar(source_model_dir)
        ):
            # Integrated MTP (e.g. Qwen 3.5): the MLX-LM text mapping drops the
            # in-shard MTP tensors, so byte-copy them into the canonical
            # external sidecar before the parameter-coverage check runs.
            _extract_protected_integrated_mtp(source_model_dir, plan, staging_dir)
        if calibration_source is not None:
            _copy_verified(calibration_source, staging_dir / "calibration_manifest.json")
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
        ) = _verify_converted_weights(staging_dir, plan)
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
