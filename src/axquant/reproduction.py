from __future__ import annotations

from pathlib import Path

from axquant.artifact_paths import artifact_member_path, artifact_tree_files
from axquant.errors import ArtifactError
from axquant.schema import (
    ArtifactManifest,
    CalibrationManifest,
    QuantizationPlan,
    ReproductionRecipe,
    ReproductionVerification,
)
from axquant.serde import file_sha256, load_model, stable_sha256


def _safe_relative_path(root: Path, relative_name: str) -> Path:
    try:
        return artifact_member_path(root, relative_name)
    except ValueError as exc:
        raise ArtifactError(f"unsafe reproduction path: {relative_name}") from exc


def _check_bound_file(
    *,
    root: Path,
    relative_name: str,
    expected_sha256: str,
    label: str,
    issues: list[str],
) -> Path | None:
    try:
        path = _safe_relative_path(root, relative_name)
    except ArtifactError:
        issues.append(f"{label} uses an unsafe path: {relative_name}")
        return None
    if not path.is_file():
        issues.append(f"{label} is missing: {relative_name}")
    elif file_sha256(path) != expected_sha256:
        issues.append(f"{label} checksum does not match the recipe: {relative_name}")
    return path


def _calibration_issues(
    manifest: CalibrationManifest,
    recipe: ReproductionRecipe,
) -> list[str]:
    expected = recipe.calibration
    issues: list[str] = []
    if manifest.model != recipe.source_model:
        issues.append("calibration manifest source model does not match the recipe")
    if manifest.profile != recipe.profile:
        issues.append("calibration manifest profile does not match the recipe")
    if manifest.random_seed != recipe.random_seed:
        issues.append("calibration manifest random seed does not match the recipe")
    bindings = (
        ("dataset ID", manifest.dataset_id, expected.dataset_id),
        ("dataset checksum", manifest.dataset_sha256, expected.dataset_sha256),
        ("sample count", manifest.samples, expected.samples),
        ("domains", manifest.domains, expected.domains),
        ("sequence length", manifest.sequence_length, expected.sequence_length),
    )
    for label, actual, recorded in bindings:
        if actual != recorded:
            issues.append(f"calibration manifest {label} does not match the recipe")
    if not manifest.calibration_evaluation_separation_attested:
        issues.append("calibration manifest does not attest evaluation separation")
    return issues


def _manifest_weight_records(manifest: ArtifactManifest) -> list[tuple[str, int, str]]:
    return sorted(
        (record.path, record.size_bytes, record.sha256)
        for record in manifest.files
        if Path(record.path).suffix.lower() == ".safetensors"
    )


def _recipe_weight_records(recipe: ReproductionRecipe) -> list[tuple[str, int, str]]:
    return sorted(
        (record.path, record.size_bytes, record.sha256) for record in recipe.expected_weight_files
    )


def verify_reproduction(
    *,
    recipe_path: str | Path,
    artifact_dir: str | Path,
) -> ReproductionVerification:
    recipe_input = Path(recipe_path).expanduser()
    artifact_input = Path(artifact_dir).expanduser()
    if recipe_input.is_symlink():
        raise ArtifactError(f"reproduction recipe must not be a symlink: {recipe_input}")
    if artifact_input.is_symlink():
        raise ArtifactError(f"reproduced artifact root must not be a symlink: {artifact_input}")
    recipe_source = recipe_input.resolve()
    artifact = artifact_input.resolve()
    if not recipe_source.is_file():
        raise ArtifactError(f"reproduction recipe does not exist: {recipe_source}")
    if not artifact.is_dir():
        raise ArtifactError(f"reproduced artifact directory does not exist: {artifact}")
    try:
        artifact_files = artifact_tree_files(artifact)
    except ValueError as exc:
        raise ArtifactError(f"reproduced artifact tree is unsafe: {exc}") from exc

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
    calibration_path = _check_bound_file(
        root=recipe_root,
        relative_name=recipe.calibration_file,
        expected_sha256=recipe.calibration_file_sha256,
        label="calibration manifest",
        issues=issues,
    )
    conversion_manifest_path = _check_bound_file(
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
        if (
            companion_path is not None
            and companion_path.is_file()
            and companion_path.stat().st_size != companion.size_bytes
        ):
            issues.append(f"MTP companion size does not match the recipe: {companion.path}")

    if plan_path is not None and plan_path.is_file():
        try:
            plan = load_model(plan_path, QuantizationPlan)
        except (ArtifactError, ValueError) as exc:
            issues.append(f"quantization plan cannot be validated: {exc}")
        else:
            if stable_sha256(plan) != recipe.plan_sha256:
                issues.append("quantization plan semantic checksum does not match the recipe")
            if plan.source_model != recipe.source_model:
                issues.append("quantization plan source model does not match the recipe")
            if plan.profile != recipe.profile:
                issues.append("quantization plan profile does not match the recipe")
            if plan.random_seed != recipe.random_seed:
                issues.append("quantization plan random seed does not match the recipe")
            if plan.calibration != recipe.calibration:
                issues.append("quantization plan calibration evidence does not match the recipe")

    if calibration_path is not None and calibration_path.is_file():
        try:
            calibration = load_model(calibration_path, CalibrationManifest)
        except (ArtifactError, ValueError) as exc:
            issues.append(f"calibration manifest cannot be validated: {exc}")
        else:
            issues.extend(_calibration_issues(calibration, recipe))

    expected_weight_records = _recipe_weight_records(recipe)
    if sum(record[1] for record in expected_weight_records) != (
        recipe.expected_weight_file_size_bytes
    ):
        issues.append("expected weight-file records do not sum to the recipe byte count")

    if conversion_manifest_path is not None and conversion_manifest_path.is_file():
        try:
            conversion_manifest = load_model(conversion_manifest_path, ArtifactManifest)
        except (ArtifactError, ValueError) as exc:
            issues.append(f"immutable conversion manifest cannot be validated: {exc}")
        else:
            if conversion_manifest.source_model != recipe.source_model:
                issues.append("immutable conversion manifest source model does not match recipe")
            if conversion_manifest.plan_sha256 != recipe.plan_sha256:
                issues.append("immutable conversion manifest plan checksum does not match recipe")
            if conversion_manifest.profile != recipe.profile:
                issues.append("immutable conversion manifest profile does not match recipe")
            if conversion_manifest.axquant_version != recipe.axquant_version:
                issues.append("immutable conversion manifest AXQuant version does not match recipe")
            if conversion_manifest.software_versions != recipe.software_versions:
                issues.append("immutable conversion manifest software versions do not match recipe")
            if conversion_manifest.runtime.primary_runtime.name != recipe.primary_runtime:
                issues.append("immutable conversion manifest runtime does not match recipe")
            if conversion_manifest.logical_parameters != recipe.expected_logical_parameters:
                issues.append("immutable conversion manifest logical parameter count changed")
            if conversion_manifest.weight_file_size_bytes != recipe.expected_weight_file_size_bytes:
                issues.append("immutable conversion manifest weight-file byte count changed")
            if _manifest_weight_records(conversion_manifest) != expected_weight_records:
                issues.append(
                    "immutable conversion manifest weight records do not match the recipe"
                )

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
            if manifest.profile != recipe.profile:
                issues.append("reproduced artifact profile does not match the recipe")
            if manifest.axquant_version != recipe.axquant_version:
                issues.append("reproduced artifact AXQuant version does not match the recipe")
            if manifest.software_versions != recipe.software_versions:
                issues.append("reproduced artifact software versions do not match the recipe")
            if manifest.runtime.primary_runtime.name != recipe.primary_runtime:
                issues.append("reproduced artifact runtime does not match the recipe")
            if manifest.logical_parameters != recipe.expected_logical_parameters:
                issues.append("reproduced artifact logical parameter count changed")
            if manifest.weight_file_size_bytes != recipe.expected_weight_file_size_bytes:
                issues.append("reproduced artifact weight-file byte count changed")
            if _manifest_weight_records(manifest) != expected_weight_records:
                issues.append("reproduced artifact weight records do not match the recipe")

    verified_weight_files: list[str] = []
    expected_paths = {record.path for record in recipe.expected_weight_files}
    actual_paths = {
        path.relative_to(artifact).as_posix()
        for path in artifact_files
        if path.suffix.lower() == ".safetensors"
    }
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing:
        issues.append(f"reproduced artifact is missing expected weight files: {missing}")
    if unexpected:
        issues.append(f"reproduced artifact has unexpected weight files: {unexpected}")
    for record in recipe.expected_weight_files:
        try:
            path = _safe_relative_path(artifact, record.path)
        except ArtifactError:
            issues.append(f"reproduced weight file uses an unsafe path: {record.path}")
            continue
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
