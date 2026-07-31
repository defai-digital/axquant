"""Recipe bundles: checksummed, publishable planning artifacts (AXQ-020).

A bundle binds a plan or manual recipe to a pinned source model identity so a
user conversion can reuse published planning evidence. Resolution is
fail-closed: payload checksum, model identity, and evidence-kind consistency
are all verified before a plan is produced, and a bundle never upgrades the
evidence kind of its payload.
"""

from __future__ import annotations

import posixpath
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from axquant import __version__
from axquant.errors import ArtifactError
from axquant.manual import manual_quantization_plan
from axquant.schema import (
    Inventory,
    ManualPlanRecipe,
    QuantizationPlan,
    RecipeBundle,
)
from axquant.serde import file_sha256, load_model, write_data

RECIPE_BUNDLE_FILE = "axquant_recipe_bundle.json"
REMOTE_SCHEME = "hf://"


def _parse_remote_reference(reference: str) -> tuple[str, str, str]:
    """Split ``hf://OWNER/REPO@REVISION[/PATH]`` into repo, revision, and record path."""
    body = reference.removeprefix(REMOTE_SCHEME)
    repo_id, separator, rest = body.partition("@")
    if not separator or not rest:
        raise ArtifactError(f"remote recipe reference must pin a revision (AXQ-023): {reference}")
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise ArtifactError(f"remote recipe reference must use hf://OWNER/REPO: {reference}")
    revision, _, path = rest.partition("/")
    if not revision:
        raise ArtifactError(f"remote recipe reference must pin a revision (AXQ-023): {reference}")
    return repo_id, revision, path or RECIPE_BUNDLE_FILE


def _download_remote_file(repo_id: str, revision: str, filename: str, reference: str) -> Path:
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
    except Exception as exc:
        raise ArtifactError(f"remote recipe download failed for {reference}: {exc}") from exc


def _remote_bundle(reference: str) -> tuple[RecipeBundle, Path]:
    repo_id, revision, record_name = _parse_remote_reference(reference)
    record_path = _download_remote_file(repo_id, revision, record_name, reference)
    record = load_model(record_path, RecipeBundle)
    payload_name = posixpath.normpath(
        posixpath.join(posixpath.dirname(record_name), record.payload_file)
    )
    if payload_name.startswith(".."):
        raise ArtifactError(
            f"recipe bundle {record.bundle_id} payload escapes the repository: {payload_name}"
        )
    payload = _download_remote_file(repo_id, revision, payload_name, reference)
    return record, payload


def _verify_payload(record: RecipeBundle, payload: Path) -> None:
    if not payload.is_file():
        raise ArtifactError(f"recipe bundle payload does not exist: {payload}")
    digest = file_sha256(payload)
    if digest != record.payload_sha256:
        raise ArtifactError(
            f"recipe bundle payload checksum mismatch for {record.bundle_id}: "
            f"expected {record.payload_sha256}, found {digest}"
        )


def load_recipe_bundle(bundle: str | Path) -> tuple[RecipeBundle, Path]:
    """Load a local or ``hf://`` bundle and verify its payload checksum."""
    if isinstance(bundle, str) and bundle.startswith(REMOTE_SCHEME):
        record, payload = _remote_bundle(bundle)
    else:
        bundle_path = Path(bundle).expanduser().resolve()
        if bundle_path.is_dir():
            bundle_path = bundle_path / RECIPE_BUNDLE_FILE
        record = load_model(bundle_path, RecipeBundle)
        payload = (bundle_path.parent / record.payload_file).resolve()
    _verify_payload(record, payload)
    return record, payload


def resolve_recipe_plan(
    bundle: str | Path,
    *,
    inventory: Inventory,
) -> tuple[RecipeBundle, QuantizationPlan]:
    """Verify a bundle against the target inventory and produce its plan."""
    record, payload = load_recipe_bundle(bundle)
    target = inventory.model
    if record.source_model.model_id != target.model_id:
        raise ArtifactError(
            f"recipe bundle {record.bundle_id} targets {record.source_model.model_id}, "
            f"not {target.model_id}"
        )
    if target.revision and record.source_model.revision != target.revision:
        raise ArtifactError(
            f"recipe bundle {record.bundle_id} pins revision "
            f"{record.source_model.revision}, not {target.revision}"
        )
    if record.payload_kind == "plan":
        plan = load_model(payload, QuantizationPlan)
    else:
        recipe = load_model(payload, ManualPlanRecipe)
        plan = manual_quantization_plan(inventory, recipe)
    if plan.evidence_kind != record.evidence_kind:
        raise ArtifactError(
            f"recipe bundle {record.bundle_id} declares {record.evidence_kind.value} evidence "
            f"but its payload produces {plan.evidence_kind.value}"
        )
    return record, plan


def export_recipe_bundle(
    *,
    plan: str | Path,
    output_dir: str | Path,
    bundle_id: str,
    lineage: dict[str, str] | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Export a plan file as a recipe bundle directory."""
    plan_path = Path(plan).expanduser().resolve()
    loaded = load_model(plan_path, QuantizationPlan)
    if not loaded.source_model.revision:
        raise ArtifactError("a recipe bundle requires a revision-pinned plan")
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload_name = "plan.json"
    destination = directory / payload_name
    if destination.exists():
        raise ArtifactError(f"recipe bundle payload already exists: {destination}")
    shutil.copyfile(plan_path, destination)
    record = RecipeBundle(
        bundle_id=bundle_id,
        source_model=loaded.source_model,
        evidence_kind=loaded.evidence_kind,
        payload_kind="plan",
        payload_file=payload_name,
        payload_sha256=file_sha256(destination),
        lineage=dict(lineage or {}),
        axquant_version=__version__,
        notes=list(notes or []),
    )
    bundle_path = directory / RECIPE_BUNDLE_FILE
    if bundle_path.exists():
        raise ArtifactError(f"recipe bundle record already exists: {bundle_path}")
    write_data(bundle_path, record)
    return bundle_path
