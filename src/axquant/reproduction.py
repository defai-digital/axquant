from __future__ import annotations

from pathlib import Path

from axquant.errors import ArtifactError
from axquant.schema import (
    ArtifactManifest,
    QuantizationPlan,
    ReproductionRecipe,
    ReproductionVerification,
)
from axquant.serde import file_sha256, load_model, stable_sha256


def _safe_relative_path(root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactError(f"unsafe reproduction path: {relative_name}")
    return root / relative


def _check_bound_file(
    *,
    root: Path,
    relative_name: str,
    expected_sha256: str,
    label: str,
    issues: list[str],
) -> Path:
    path = _safe_relative_path(root, relative_name)
    if not path.is_file():
        issues.append(f"{label} is missing: {relative_name}")
    elif file_sha256(path) != expected_sha256:
        issues.append(f"{label} checksum does not match the recipe: {relative_name}")
    return path


def verify_reproduction(
    *,
    recipe_path: str | Path,
    artifact_dir: str | Path,
) -> ReproductionVerification:
    recipe_source = Path(recipe_path).expanduser().resolve()
    artifact = Path(artifact_dir).expanduser().resolve()
    if not recipe_source.is_file():
        raise ArtifactError(f"reproduction recipe does not exist: {recipe_source}")
    if not artifact.is_dir():
        raise ArtifactError(f"reproduced artifact directory does not exist: {artifact}")

    recipe = load_model(recipe_source, ReproductionRecipe)
    issues: list[str] = []
    recipe_root = recipe_source.parent

    plan_path = _check_bound_file(
        root=recipe_root,
        relative_name=recipe.plan_file,
        expected_sha256=recipe.plan_file_sha256,
        label="quantization plan",
        issues=issues,
    )
    _check_bound_file(
        root=recipe_root,
        relative_name=recipe.calibration_file,
        expected_sha256=recipe.calibration_file_sha256,
        label="calibration manifest",
        issues=issues,
    )
    _check_bound_file(
        root=recipe_root,
        relative_name=recipe.conversion_manifest_file,
        expected_sha256=recipe.conversion_manifest_sha256,
        label="immutable conversion manifest",
        issues=issues,
    )
    if recipe.mtp_sidecar_file is not None and recipe.mtp_sidecar_sha256 is not None:
        _check_bound_file(
            root=recipe_root,
            relative_name=recipe.mtp_sidecar_file,
            expected_sha256=recipe.mtp_sidecar_sha256,
            label="MTP sidecar",
            issues=issues,
        )
    for companion in recipe.mtp_companion_files:
        companion_path = _check_bound_file(
            root=recipe_root,
            relative_name=companion.path,
            expected_sha256=companion.sha256,
            label="MTP companion",
            issues=issues,
        )
        if companion_path.is_file() and companion_path.stat().st_size != companion.size_bytes:
            issues.append(f"MTP companion size does not match the recipe: {companion.path}")

    if plan_path.is_file():
        try:
            plan = load_model(plan_path, QuantizationPlan)
        except (ArtifactError, ValueError) as exc:
            issues.append(f"quantization plan cannot be validated: {exc}")
        else:
            if stable_sha256(plan) != recipe.plan_sha256:
                issues.append("quantization plan semantic checksum does not match the recipe")
            if plan.source_model != recipe.source_model:
                issues.append("quantization plan source model does not match the recipe")

    artifact_manifest_path = artifact / "axquant_manifest.json"
    actual_logical_parameters = 0
    actual_weight_bytes = 0
    if not artifact_manifest_path.is_file():
        issues.append("reproduced artifact has no axquant_manifest.json")
    else:
        try:
            manifest = load_model(artifact_manifest_path, ArtifactManifest)
        except (ArtifactError, ValueError) as exc:
            issues.append(f"reproduced artifact manifest cannot be validated: {exc}")
        else:
            actual_logical_parameters = manifest.logical_parameters
            actual_weight_bytes = manifest.weight_file_size_bytes
            if manifest.source_model != recipe.source_model:
                issues.append("reproduced artifact source model does not match the recipe")
            if manifest.plan_sha256 != recipe.plan_sha256:
                issues.append("reproduced artifact plan checksum does not match the recipe")
            if manifest.logical_parameters != recipe.expected_logical_parameters:
                issues.append("reproduced artifact logical parameter count changed")
            if manifest.weight_file_size_bytes != recipe.expected_weight_file_size_bytes:
                issues.append("reproduced artifact weight-file byte count changed")

    verified_weight_files: list[str] = []
    expected_paths = {record.path for record in recipe.expected_weight_files}
    actual_paths = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*.safetensors")
        if path.is_file()
    }
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing:
        issues.append(f"reproduced artifact is missing expected weight files: {missing}")
    if unexpected:
        issues.append(f"reproduced artifact has unexpected weight files: {unexpected}")
    for record in recipe.expected_weight_files:
        path = _safe_relative_path(artifact, record.path)
        if not path.is_file():
            continue
        if path.stat().st_size != record.size_bytes:
            issues.append(f"reproduced weight file size changed: {record.path}")
            continue
        if file_sha256(path) != record.sha256:
            issues.append(f"reproduced weight file checksum changed: {record.path}")
            continue
        verified_weight_files.append(record.path)

    return ReproductionVerification(
        recipe_sha256=stable_sha256(recipe),
        artifact_path=str(artifact),
        passed=not issues,
        issues=issues,
        verified_weight_files=sorted(verified_weight_files),
        expected_logical_parameters=recipe.expected_logical_parameters,
        actual_logical_parameters=actual_logical_parameters,
        expected_weight_file_size_bytes=recipe.expected_weight_file_size_bytes,
        actual_weight_file_size_bytes=actual_weight_bytes,
    )
