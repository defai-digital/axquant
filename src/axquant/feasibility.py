from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axquant.errors import AxquantError
from axquant.inspector import inspect_model, resolve_model_dir
from axquant.runtime import check_ax_engine, check_mlx_lm_static
from axquant.schema import (
    ArchitectureSupportLevel,
    ArtifactIntegrity,
    BaselineAudit,
    BaselineKind,
    FeasibilityReport,
    ModelIdentity,
    OptimizationScope,
    RuntimeName,
)
from axquant.serde import file_sha256, stable_sha256

_REVISION = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ArtifactTarget:
    model: str | Path
    kind: BaselineKind
    model_id: str | None = None
    revision: str | None = None


def _read_json_object(path: Path) -> tuple[bool, dict[str, Any] | None]:
    if not path.is_file():
        return False, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True, None
    return True, value if isinstance(value, dict) else None


def _inferred_revision(directory: Path, revision: str | None) -> str | None:
    if revision:
        return revision
    return directory.name if _REVISION.fullmatch(directory.name) else None


def _index_integrity(directory: Path) -> tuple[bool, bool, list[str]]:
    index_path = directory / "model.safetensors.index.json"
    present, index = _read_json_object(index_path)
    if not present:
        return False, (directory / "model.safetensors").is_file(), []
    if index is None:
        return True, False, ["model.safetensors.index.json is not a valid JSON object"]
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return True, False, ["model.safetensors.index.json has no non-empty weight_map"]
    referenced: set[str] = set()
    issues: list[str] = []
    for value in weight_map.values():
        if not isinstance(value, str):
            issues.append("model.safetensors.index.json contains a non-string shard reference")
            continue
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            issues.append(f"model.safetensors.index.json contains an unsafe path: {value}")
            continue
        referenced.add(relative.as_posix())
    missing = sorted(name for name in referenced if not (directory / name).is_file())
    if missing:
        issues.append(f"model.safetensors.index.json references missing shards: {missing}")
    root_shards = {
        path.name for path in directory.glob("model-*-of-*.safetensors") if path.is_file()
    }
    unindexed = sorted(root_shards - referenced)
    if unindexed:
        issues.append(f"model shards are not referenced by the index: {unindexed}")
    return True, not issues, issues


def _native_manifest_integrity(directory: Path) -> tuple[bool, bool]:
    present, manifest = _read_json_object(directory / "model-manifest.json")
    if manifest is None:
        return present, False
    return (
        present,
        isinstance(manifest.get("schema_version"), str)
        and isinstance(manifest.get("tensors"), list)
        and bool(manifest["tensors"]),
    )


def _mtp_runtime_integrity(directory: Path) -> tuple[bool, bool, dict[str, Any] | None]:
    present, runtime = _read_json_object(directory / "mtplx_runtime.json")
    if runtime is None:
        return present, False, None
    valid = (
        isinstance(runtime.get("arch_id"), str)
        and isinstance(runtime.get("mtp_depth_max"), int)
        and runtime["mtp_depth_max"] > 0
        and isinstance(runtime.get("mtp_tensor_count"), int)
        and runtime["mtp_tensor_count"] > 0
    )
    return present, valid, runtime


def _mtp_provenance_integrity(directory: Path) -> tuple[bool, bool]:
    present, provenance = _read_json_object(directory / "ax_mtp_sidecar_manifest.json")
    if provenance is None:
        return present, False
    output = provenance.get("output")
    mtp = output.get("mtp") if isinstance(output, dict) else None
    if not isinstance(mtp, dict):
        return present, False
    relative_value = mtp.get("path")
    size_value = mtp.get("size_bytes")
    digest_value = mtp.get("sha256")
    if not isinstance(relative_value, str):
        return present, False
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        return present, False
    sidecar = directory / relative
    if not sidecar.is_file():
        return present, False
    if not isinstance(size_value, int) or sidecar.stat().st_size != size_value:
        return present, False
    return (
        present,
        isinstance(digest_value, str)
        and len(digest_value) == 64
        and file_sha256(sidecar) == digest_value.lower(),
    )


def _artifact_integrity(
    directory: Path,
) -> tuple[ArtifactIntegrity, list[str], dict[str, Any] | None]:
    config_present, config = _read_json_object(directory / "config.json")
    index_present, index_complete, issues = _index_integrity(directory)
    native_present, native_valid = _native_manifest_integrity(directory)
    runtime_present, runtime_valid, runtime = _mtp_runtime_integrity(directory)
    provenance_present, provenance_valid = _mtp_provenance_integrity(directory)
    safetensors_present = any(directory.glob("*.safetensors"))
    tokenizer_present = any(
        (directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    )
    if not config_present:
        issues.append("config.json is missing")
    elif config is None:
        issues.append("config.json is not a valid JSON object")
    if not safetensors_present:
        issues.append("no Safetensors weights were found")
    if not index_complete:
        issues.append("the model weight index is incomplete")
    if not tokenizer_present:
        issues.append("tokenizer files are missing")
    return (
        ArtifactIntegrity(
            config_valid=config is not None,
            safetensors_present=safetensors_present,
            index_present=index_present,
            index_complete=index_complete,
            native_manifest_present=native_present,
            native_manifest_valid=native_valid,
            tokenizer_present=tokenizer_present,
            mtp_sidecar_present=(directory / "mtp.safetensors").is_file(),
            mtp_runtime_present=runtime_present,
            mtp_runtime_valid=runtime_valid,
            mtp_provenance_present=provenance_present,
            mtp_provenance_valid=provenance_valid,
        ),
        issues,
        runtime,
    )


def _empty_integrity() -> ArtifactIntegrity:
    return ArtifactIntegrity(
        config_valid=False,
        safetensors_present=False,
        index_present=False,
        index_complete=False,
        native_manifest_present=False,
        native_manifest_valid=False,
        tokenizer_present=False,
        mtp_sidecar_present=False,
        mtp_runtime_present=False,
        mtp_runtime_valid=False,
        mtp_provenance_present=False,
        mtp_provenance_valid=False,
    )


def _failed_audit(target: ArtifactTarget, issue: str) -> BaselineAudit:
    local = Path(target.model).expanduser()
    return BaselineAudit(
        kind=target.kind,
        model=ModelIdentity(
            model_id=target.model_id or str(target.model),
            revision=target.revision,
            local_path=str(local.resolve()) if local.exists() else None,
        ),
        inspected=False,
        logical_parameters=0,
        mtp_logical_parameters=0,
        weight_bytes=0,
        main_weight_bytes=0,
        mtp_weight_bytes=0,
        effective_bpw=0.0,
        main_effective_bpw=0.0,
        precision_parameters={},
        precision_fractions={},
        integrity=_empty_integrity(),
        complete=False,
        issues=[issue],
    )


def _kind_issues(audit: BaselineAudit) -> list[str]:
    issues: list[str] = []
    if audit.kind == BaselineKind.BF16_SOURCE:
        if audit.quantized:
            issues.append("the BF16 source is already quantized")
        if audit.precision_parameters.get("bf16", 0) <= 0:
            issues.append("the BF16 source contains no BF16 tensor parameters")
        return issues
    if not audit.quantized:
        issues.append(f"{audit.kind.value} is not marked as a quantized MLX checkpoint")
    expected = {
        BaselineKind.UNIFORM_4BIT: "4bit",
        BaselineKind.UNIFORM_6BIT: "6bit",
    }.get(audit.kind)
    if expected is not None and audit.precision_parameters.get(expected, 0) <= 0:
        issues.append(f"{audit.kind.value} contains no {expected} tensor parameters")
    if audit.kind == BaselineKind.MIXED_PRECISION:
        quantized_precisions = {
            precision
            for precision, parameters in audit.precision_parameters.items()
            if precision in {"4bit", "6bit", "8bit"} and parameters > 0
        }
        if len(quantized_precisions) < 2:
            issues.append("mixed-precision baseline does not contain two quantized precisions")
    return issues


def audit_artifact(
    target: ArtifactTarget,
    *,
    run_runtime_checks: bool = False,
    ax_engine: str = "ax-engine",
) -> BaselineAudit:
    try:
        directory = resolve_model_dir(target.model)
    except (AxquantError, OSError, ValueError) as exc:
        return _failed_audit(target, str(exc))
    revision = _inferred_revision(directory, target.revision)
    integrity, integrity_issues, mtp_runtime = _artifact_integrity(directory)
    try:
        inventory = inspect_model(
            directory,
            model_id=target.model_id or str(target.model),
            revision=revision,
            allow_quantized=True,
        )
    except (AxquantError, OSError, ValueError) as exc:
        failed = _failed_audit(target, str(exc))
        failed.model.local_path = str(directory)
        failed.model.revision = revision
        failed.integrity = integrity
        failed.issues = [*integrity_issues, str(exc)]
        return failed

    mtp_parameters = sum(tensor.parameters for tensor in inventory.tensors if tensor.role.is_mtp)
    main_parameters = max(inventory.total_parameters - mtp_parameters, 0)
    main_weight_bytes = max(inventory.weight_bytes - inventory.mtp_weight_bytes, 0)
    precision_fractions = {
        precision: parameters / inventory.total_parameters
        for precision, parameters in sorted(inventory.precision_parameters.items())
        if inventory.total_parameters
    }
    runtime_checks = [check_mlx_lm_static(directory)]
    if run_runtime_checks and target.kind != BaselineKind.BF16_SOURCE:
        runtime_checks.insert(0, check_ax_engine(directory, executable=ax_engine))
    audit = BaselineAudit(
        kind=target.kind,
        model=inventory.model,
        inspected=True,
        inventory_sha256=stable_sha256(inventory.model_dump(mode="json", exclude={"created_at"})),
        adapter_id=inventory.architecture_profile.adapter_id,
        optimization_scope=inventory.architecture_profile.optimization_scope,
        quantized=inventory.quantized_source,
        logical_parameters=inventory.total_parameters,
        mtp_logical_parameters=mtp_parameters,
        weight_bytes=inventory.weight_bytes,
        main_weight_bytes=main_weight_bytes,
        mtp_weight_bytes=inventory.mtp_weight_bytes,
        effective_bpw=(
            inventory.weight_bytes * 8 / inventory.total_parameters
            if inventory.total_parameters
            else 0.0
        ),
        main_effective_bpw=(main_weight_bytes * 8 / main_parameters if main_parameters else 0.0),
        precision_parameters=dict(sorted(inventory.precision_parameters.items())),
        precision_fractions=precision_fractions,
        integrity=integrity,
        runtime_checks=runtime_checks,
        complete=False,
        issues=list(integrity_issues),
    )
    if inventory.architecture_profile.support_level != ArchitectureSupportLevel.SUPPORTED:
        audit.issues.append("the checkpoint does not match the supported Qwen 3.6 adapter")
    if inventory.architecture_profile.optimization_scope != OptimizationScope.TEXT_PATH:
        audit.issues.append("the checkpoint is not supported for Qwen 3.6 text-path conversion")
    if revision is None:
        audit.issues.append("the checkpoint revision is not pinned")
    if not inventory.mtp_present or mtp_parameters <= 0:
        audit.issues.append("MTP tensors were not found")
    audit.issues.extend(_kind_issues(audit))

    if target.kind != BaselineKind.BF16_SOURCE:
        required = {
            "AX Engine native manifest": integrity.native_manifest_valid,
            "MTP sidecar": integrity.mtp_sidecar_present,
            "MTP runtime contract": integrity.mtp_runtime_valid,
            "MTP provenance": integrity.mtp_provenance_valid,
        }
        audit.issues.extend(
            f"{name} is missing or invalid" for name, valid in required.items() if not valid
        )
        expected_count = mtp_runtime.get("mtp_tensor_count") if mtp_runtime else None
        actual_count = sum(
            1
            for tensor in inventory.tensors
            if Path(tensor.file).name == "mtp.safetensors" and not tensor.quantization_metadata
        )
        if isinstance(expected_count, int) and expected_count != actual_count:
            audit.issues.append(
                "MTP runtime tensor count does not match mtp.safetensors "
                f"({expected_count} declared, {actual_count} found)"
            )
    audit.issues = list(dict.fromkeys(audit.issues))
    audit.complete = not audit.issues
    return audit


def _matching(audits: list[BaselineAudit], field: str) -> bool:
    if not audits or any(not audit.inspected for audit in audits):
        return False
    values = {getattr(audit, field) for audit in audits}
    return len(values) == 1


def assess_feasibility(
    *,
    reference_4bit: ArtifactTarget,
    reference_6bit: ArtifactTarget,
    source_bf16: ArtifactTarget | None = None,
    mixed_baseline: ArtifactTarget | None = None,
    run_runtime_checks: bool = False,
    ax_engine: str = "ax-engine",
) -> FeasibilityReport:
    baselines = [
        audit_artifact(
            reference_4bit,
            run_runtime_checks=run_runtime_checks,
            ax_engine=ax_engine,
        ),
        audit_artifact(
            reference_6bit,
            run_runtime_checks=run_runtime_checks,
            ax_engine=ax_engine,
        ),
    ]
    if mixed_baseline is not None:
        baselines.append(
            audit_artifact(
                mixed_baseline,
                run_runtime_checks=run_runtime_checks,
                ax_engine=ax_engine,
            )
        )
    source = (
        audit_artifact(source_bf16, run_runtime_checks=False, ax_engine=ax_engine)
        if source_bf16 is not None
        else None
    )
    compared = [*baselines, *([source] if source is not None else [])]
    structural_complete = all(audit.complete for audit in baselines)
    parameter_count_match = _matching(compared, "logical_parameters")
    architecture_match = _matching(compared, "adapter_id") and _matching(
        compared, "optimization_scope"
    )
    mtp_present = all(audit.mtp_logical_parameters > 0 for audit in compared)
    revisions_pinned = all(audit.model.revision is not None for audit in compared)
    ax_checks = [
        check
        for audit in baselines
        for check in audit.runtime_checks
        if check.runtime == RuntimeName.AX_ENGINE
    ]
    mlx_checks = [
        check
        for audit in compared
        for check in audit.runtime_checks
        if check.runtime == RuntimeName.MLX_LM
    ]
    ax_engine_ready = bool(ax_checks) and all(check.passed for check in ax_checks)
    mlx_lm_compatible = bool(mlx_checks) and all(check.passed for check in mlx_checks)
    source_ready = source is not None and source.complete
    checks = {
        "required_baselines_complete": structural_complete,
        "logical_parameter_counts_match": parameter_count_match,
        "architecture_profiles_match": architecture_match,
        "mtp_tensors_present": mtp_present,
        "revisions_pinned": revisions_pinned,
        "source_bf16_available": source is not None,
        "source_bf16_complete": source_ready,
        "ax_engine_runtime_ready": ax_engine_ready,
        "mlx_lm_static_compatible": mlx_lm_compatible,
    }
    blockers: list[str] = []
    for audit in baselines:
        blockers.extend(f"{audit.kind.value}: {issue}" for issue in audit.issues)
    if source is not None:
        blockers.extend(f"{source.kind.value}: {issue}" for issue in source.issues)
    if not parameter_count_match:
        blockers.append("logical parameter counts differ across supplied checkpoints")
    if not architecture_match:
        blockers.append("architecture profiles differ across supplied checkpoints")
    if not mtp_present:
        blockers.append("one or more supplied checkpoints do not contain MTP tensors")
    if not revisions_pinned:
        blockers.append("one or more supplied checkpoint revisions are not pinned")
    if run_runtime_checks and not ax_engine_ready:
        blockers.append("one or more quantized baselines failed AX Engine doctor")

    hard_blocked = bool(blockers)
    if not hard_blocked and source is None:
        status = "baseline-ready"
        blockers.append("a complete BF16 source checkpoint is required before conversion")
    elif hard_blocked:
        status = "blocked"
    else:
        status = "ready-for-conversion"
    warnings: list[str] = []
    if mixed_baseline is None:
        warnings.append("no mixed-precision comparison baseline was supplied")
    if not run_runtime_checks:
        warnings.append("AX Engine doctor was not run")
    if not mlx_lm_compatible:
        warnings.append("MLX-LM static compatibility was not confirmed in the current environment")
    return FeasibilityReport(
        status=status,
        source=source,
        baselines=baselines,
        runtime_checks_requested=run_runtime_checks,
        checks=checks,
        blockers=list(dict.fromkeys(blockers)),
        warnings=warnings,
    )


def feasibility_markdown(report: FeasibilityReport) -> str:
    artifacts = [*report.baselines, *([report.source] if report.source is not None else [])]
    rows = "\n".join(
        "| "
        f"{audit.kind.value} | `{audit.model.model_id}` | "
        f"`{audit.model.revision or 'unrecorded'}` | "
        f"{audit.logical_parameters:,} | {audit.weight_bytes / (1024**3):.3f} | "
        f"{audit.effective_bpw:.4f} | {audit.mtp_logical_parameters:,} | "
        f"{'PASS' if audit.complete else 'FAIL'} |"
        for audit in artifacts
    )
    checks = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in sorted(report.checks.items())
    )
    blockers = "\n".join(f"- {item}" for item in report.blockers) or "- None."
    warnings = "\n".join(f"- {item}" for item in report.warnings) or "- None."
    checkpoint_header = (
        "| Kind | Model | Revision | Logical parameters | Weight GiB | Effective BPW "
        "| MTP parameters | Integrity |"
    )
    return f"""# AXQuant Feasibility Report

| Property | Value |
| --- | --- |
| Schema | `{report.schema_version}` |
| Status | `{report.status}` |
| Runtime checks requested | `{report.runtime_checks_requested}` |

## Checkpoints

{checkpoint_header}
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
{rows}

## Checks

| Check | Result |
| --- | --- |
{checks}

## Blockers

{blockers}

## Warnings

{warnings}

`baseline-ready` means the comparison checkpoints are auditable, but conversion cannot begin
until a complete, revision-pinned BF16 source is available. Runtime and performance claims still
require the dedicated AX Engine benchmark pipeline.
"""
