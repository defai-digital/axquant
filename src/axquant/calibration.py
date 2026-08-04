from __future__ import annotations

import json
from pathlib import Path

from axquant.errors import ArtifactError
from axquant.schema import CalibrationManifest, ModelIdentity, ProfileName
from axquant.serde import file_sha256, load_model, stable_sha256, write_data


def calibration_manifest_sha256(manifest: CalibrationManifest) -> str:
    """Return the semantic identity used to bind calibration evidence.

    ``created_at`` records when a reusable manifest was first materialized; it
    is not an input to calibration and therefore is not part of the binding.
    """

    return stable_sha256(manifest.model_dump(mode="json", exclude={"created_at"}))


def calibration_manifest_matches(
    path: Path,
    manifest: CalibrationManifest,
    expected_sha256: str,
) -> bool:
    """Match current semantic bindings and legacy byte-level bindings."""

    return expected_sha256 in {calibration_manifest_sha256(manifest), file_sha256(path)}


def _count_jsonl(path: Path) -> int:
    samples = 0
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ArtifactError(f"{path}:{line_number} must contain a JSON object")
                samples += 1
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid JSONL in {path}:{exc.lineno}: {exc.msg}") from exc
    except OSError as exc:
        raise ArtifactError(f"cannot read calibration dataset {path}: {exc}") from exc
    if samples == 0:
        raise ArtifactError("calibration dataset contains no samples")
    return samples


def prepare_calibration(
    *,
    model: ModelIdentity,
    dataset: str | Path,
    output_dir: str | Path,
    profile: ProfileName,
    domains: list[str],
    sequence_length: int,
    random_seed: int,
    tokenizer_revision: str | None = None,
    separation_attested: bool,
) -> CalibrationManifest:
    dataset_path = Path(dataset).expanduser().resolve()
    if not dataset_path.is_file():
        raise ArtifactError(f"calibration dataset does not exist: {dataset_path}")
    manifest = CalibrationManifest(
        model=model,
        profile=profile,
        dataset_id=str(dataset_path),
        dataset_sha256=file_sha256(dataset_path),
        samples=_count_jsonl(dataset_path),
        domains=domains,
        sequence_length=sequence_length,
        random_seed=random_seed,
        tokenizer_revision=tokenizer_revision or model.revision,
        calibration_evaluation_separation_attested=separation_attested,
    )
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "calibration_manifest.json"
    if manifest_path.exists():
        existing = load_model(manifest_path, CalibrationManifest)
        existing_identity = existing.model_dump(mode="json", exclude={"created_at"})
        manifest_identity = manifest.model_dump(mode="json", exclude={"created_at"})
        if stable_sha256(existing_identity) != stable_sha256(manifest_identity):
            raise ArtifactError(f"calibration cache already exists with different inputs: {output}")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    write_data(manifest_path, manifest)
    return manifest
