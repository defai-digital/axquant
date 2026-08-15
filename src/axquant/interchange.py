from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError
from safetensors import safe_open

from axquant.artifact_paths import artifact_tree_files
from axquant.errors import ArtifactError
from axquant.inspector import module_path_for
from axquant.schema import (
    ArtifactManifest,
    QuantizationPlan,
    QuantizerExecutionManifest,
    QuantMethod,
)
from axquant.serde import load_model, read_data, stable_sha256

_METADATA_SUFFIXES = (".scales", ".biases", "_scales")


def _config_mapping(directory: Path, issues: list[str]) -> dict[str, Any]:
    config_path = directory / "config.json"
    if not config_path.is_file():
        issues.append("pack does not contain config.json")
        return {}
    payload = read_data(config_path)
    if not isinstance(payload, dict):
        issues.append("config.json root must be an object")
        return {}
    return payload


def _quantization_entries(
    config: dict[str, Any],
    issues: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    raw = config.get("quantization")
    if raw is None:
        raw = config.get("quantization_config")
    if raw is None:
        issues.append("config.json does not declare quantization metadata")
        return {}, None
    if not isinstance(raw, dict):
        issues.append("config quantization metadata must be an object")
        return {}, None
    global_entry = raw if any(key in raw for key in ("bits", "group_size", "mode")) else None
    entries = {
        key: value
        for key, value in raw.items()
        if isinstance(key, str)
        and isinstance(value, dict)
        and any(item in value for item in ("bits", "group_size", "mode", "quant_method"))
    }
    method = raw.get("mode", raw.get("quant_method"))
    if method is not None and str(method).lower() != "affine":
        issues.append(f"config declares non-affine quantized method: {method}")
    for module_path, entry in sorted(entries.items()):
        entry_method = entry.get("mode", entry.get("quant_method", "affine"))
        if str(entry_method).lower() != "affine":
            issues.append(
                f"quantized module {module_path} declares non-affine method: {entry_method}"
            )
        bits = entry.get("bits")
        group_size = entry.get("group_size")
        if type(bits) is not int or not 2 <= bits < 16:
            issues.append(f"quantized module {module_path} has invalid packed bits: {bits!r}")
        if type(group_size) is not int or group_size < 1:
            issues.append(
                f"quantized module {module_path} has invalid affine group size: {group_size!r}"
            )
    if global_entry is not None:
        bits = global_entry.get("bits")
        group_size = global_entry.get("group_size")
        if type(bits) is not int or not 2 <= bits < 16:
            issues.append(f"global quantization metadata has invalid packed bits: {bits!r}")
        if type(group_size) is not int or group_size < 1:
            issues.append(
                f"global quantization metadata has invalid affine group size: {group_size!r}"
            )
    return entries, global_entry


def _tensor_dtypes(directory: Path, issues: list[str]) -> dict[str, str]:
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        issues.append("pack does not contain Safetensors weights")
        return {}
    dtypes: dict[str, str] = {}
    for path in files:
        try:
            with safe_open(path, framework="numpy") as handle:
                for name in list(handle.keys()):
                    if name in dtypes:
                        issues.append(f"duplicate tensor across Safetensors shards: {name}")
                        continue
                    dtypes[name] = str(handle.get_slice(name).get_dtype())
        except (OSError, ValueError) as exc:
            issues.append(f"cannot inspect {path.name}: {exc}")
    return dtypes


def _optional_model(
    path: Path,
    model_type: type[ArtifactManifest] | type[QuantizationPlan] | type[QuantizerExecutionManifest],
    issues: list[str],
) -> ArtifactManifest | QuantizationPlan | QuantizerExecutionManifest | None:
    if not path.is_file():
        return None
    try:
        return load_model(path, model_type)
    except (ArtifactError, ValidationError, ValueError) as exc:
        issues.append(f"{path.name} is invalid: {exc}")
        return None


def check_affine_u32_pack(directory: str | Path) -> list[str]:
    """Return conformance issues for the frozen ``axq-affine-u32-v1`` contract.

    The check reads JSON and Safetensors metadata only. It never imports MLX or
    invokes a runtime.
    """

    source = Path(directory).expanduser()
    if source.is_symlink():
        raise ArtifactError(f"interchange pack must not be a symlink: {source}")
    root = source.resolve()
    if not root.is_dir():
        raise ArtifactError(f"interchange pack directory does not exist: {root}")
    try:
        artifact_tree_files(root)
    except ValueError as exc:
        raise ArtifactError(f"interchange pack tree is unsafe: {exc}") from exc

    issues: list[str] = []
    config = _config_mapping(root, issues)
    quantization_entries, global_quantization = _quantization_entries(config, issues)
    dtypes = _tensor_dtypes(root, issues)
    tensor_names = set(dtypes)
    packed_weights = {
        name: dtype
        for name, dtype in dtypes.items()
        if dtype == "U32" and not name.endswith(_METADATA_SUFFIXES)
    }
    if not packed_weights:
        issues.append("pack contains no U32 affine-packed weight tensors")

    packed_modules: set[str] = set()
    for name in sorted(packed_weights):
        module_path = module_path_for(name)
        packed_modules.add(module_path)
        missing = [
            f"{module_path}.{suffix}"
            for suffix in ("scales", "biases")
            if f"{module_path}.{suffix}" not in tensor_names
        ]
        if missing:
            issues.append(f"affine-packed tensor {name} lacks metadata: {missing}")
        declaration = quantization_entries.get(module_path, global_quantization)
        if declaration is None:
            issues.append(f"affine-packed tensor {name} has no quantization declaration")
            continue
        method = declaration.get("mode", declaration.get("quant_method", "affine"))
        if str(method).lower() != "affine":
            issues.append(f"affine-packed tensor {name} is declared as non-affine {method}")
    for module_path in sorted(quantization_entries):
        if module_path not in packed_modules:
            issues.append(f"quantized module {module_path} is declared without a U32 packed weight")

    plan_value = _optional_model(root / "axquant_plan.json", QuantizationPlan, issues)
    plan = plan_value if isinstance(plan_value, QuantizationPlan) else None
    manifest_value = _optional_model(root / "axquant_manifest.json", ArtifactManifest, issues)
    manifest = manifest_value if isinstance(manifest_value, ArtifactManifest) else None
    execution_value = _optional_model(
        root / "axquant_quantizer_execution.json",
        QuantizerExecutionManifest,
        issues,
    )
    execution = execution_value if isinstance(execution_value, QuantizerExecutionManifest) else None

    if plan is not None:
        quantized_allocations = [
            allocation for allocation in plan.assignments if allocation.bits < 16
        ]
        for allocation in quantized_allocations:
            if allocation.method is QuantMethod.BF16:
                issues.append(
                    f"quantized plan allocation uses BF16 method: {allocation.module_path}"
                )
            if allocation.module_path not in packed_modules:
                issues.append(
                    "planned quantized module has no affine U32 pack declaration: "
                    f"{allocation.module_path}"
                )
        if manifest is not None and manifest.plan_sha256 != stable_sha256(plan):
            issues.append("artifact manifest does not bind the packaged quantization plan")

    if execution is not None:
        if plan is None:
            issues.append("quantizer execution manifest is present without a valid packaged plan")
        elif execution.plan_sha256 != stable_sha256(plan):
            issues.append("quantizer execution manifest does not bind the packaged plan")
        for record in execution.records:
            if record.bits >= 16:
                continue
            if not record.success or record.fallback:
                issues.append(f"quantizer execution did not complete cleanly: {record.module_path}")
            if record.method is not QuantMethod.AFFINE and (
                record.note is None or "affine packing" not in record.note.lower()
            ):
                issues.append(
                    "refined quantizer execution does not attest final affine packing: "
                    f"{record.module_path}"
                )

    return sorted(set(issues))
