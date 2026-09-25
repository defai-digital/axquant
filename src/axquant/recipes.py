"""Recipe bundles: checksummed, publishable planning artifacts (AXQ-020).

A bundle binds a plan or manual recipe to a pinned source model identity so a
user conversion can reuse published planning evidence. Resolution is
fail-closed: payload checksum, model identity, and evidence-kind consistency
are all verified before a plan is produced, and a bundle never upgrades the
evidence kind of its payload.
"""

from __future__ import annotations

import posixpath
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from axquant import __version__
from axquant.errors import ArtifactError
from axquant.manual import manual_quantization_plan
from axquant.revisions import is_immutable_revision
from axquant.schema import (
    Inventory,
    QuantizationPlan,
    RecipeBundle,
    SourcePlanBinding,
)
from axquant.schema.loading import (
    load_manual_plan_recipe,
    load_quantization_plan,
)
from axquant.serde import file_sha256, load_model, write_data
from axquant.source_binding import (
    BINDING_NAME,
    binding_path_beside,
    load_source_plan_binding,
    source_binding_plan_issues,
)

RECIPE_BUNDLE_FILE = "axquant_recipe_bundle.json"
REMOTE_SCHEME = "hf://"
# Records the bundle's producer-origin source binding in the bundle lineage map,
# so the binding file is tamper-evident and discoverable without changing the
# immutable recipe-bundle envelope.
SOURCE_BINDING_LINEAGE_KEY = "source_binding_sha256"
_LINEAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _safe_relative_path(value: str, *, label: str) -> str:
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if (
        not value
        or "\\" in value
        or normalized.startswith(("/", "~/"))
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ArtifactError(f"{label} must be a safe normalized relative path: {value!r}")
    return normalized


def _validated_lineage(lineage: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, digest in lineage.items():
        if _LINEAGE_NAME.fullmatch(name) is None:
            raise ArtifactError(
                f"recipe lineage name must be a safe lowercase identifier: {name!r}"
            )
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ArtifactError(
                f"recipe lineage digest for {name!r} must be 64 lowercase hexadecimal digits"
            )
        result[name] = digest
    return result


def _parse_remote_reference(reference: str) -> tuple[str, str, str]:
    """Split ``hf://OWNER/REPO@REVISION[/PATH]`` into repo, revision, and record path."""
    body = reference.removeprefix(REMOTE_SCHEME)
    repo_id, separator, rest = body.partition("@")
    if not separator or not rest:
        raise ArtifactError(f"remote recipe reference must pin a revision (AXQ-023): {reference}")
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise ArtifactError(f"remote recipe reference must use hf://OWNER/REPO: {reference}")
    revision, _, path = rest.partition("/")
    if not is_immutable_revision(revision):
        raise ArtifactError(f"remote recipe reference must pin a revision (AXQ-023): {reference}")
    record_path = _safe_relative_path(
        path or RECIPE_BUNDLE_FILE,
        label="remote recipe record path",
    )
    return repo_id, revision, record_path


def _download_remote_file(repo_id: str, revision: str, filename: str, reference: str) -> Path:
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
    except Exception as exc:
        raise ArtifactError(f"remote recipe download failed for {reference}: {exc}") from exc


def _remote_bundle(reference: str) -> tuple[RecipeBundle, Path]:
    repo_id, revision, record_name = _parse_remote_reference(reference)
    record_path = _download_remote_file(repo_id, revision, record_name, reference)
    record = load_model(record_path, RecipeBundle)
    payload_file = _safe_relative_path(
        record.payload_file,
        label=f"recipe bundle {record.bundle_id} payload path",
    )
    payload_name = posixpath.normpath(posixpath.join(posixpath.dirname(record_name), payload_file))
    _safe_relative_path(
        payload_name,
        label=f"recipe bundle {record.bundle_id} repository payload path",
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
        payload_file = _safe_relative_path(
            record.payload_file,
            label=f"recipe bundle {record.bundle_id} payload path",
        )
        bundle_root = bundle_path.parent.resolve()
        payload = (bundle_root / payload_file).resolve()
        if not payload.is_relative_to(bundle_root):
            raise ArtifactError(
                f"recipe bundle {record.bundle_id} payload escapes its bundle directory"
            )
    _validated_lineage(record.lineage)
    _verify_payload(record, payload)
    return record, payload


@dataclass(frozen=True, slots=True)
class ResolvedRecipePlan:
    """A recipe bundle resolved against a target inventory (AXQ-048).

    ``source_binding`` is the *producer's* binding, carried by the bundle. It is
    never minted here: a binding built from the consumer's own directory could
    not say anything about the checkpoint the producer planned from.
    """

    record: RecipeBundle
    plan: QuantizationPlan
    source_binding: SourcePlanBinding | None


def _bundle_source_binding(
    bundle: str | Path,
    record: RecipeBundle,
) -> SourcePlanBinding | None:
    """Load the binding a bundle carries, if its lineage declares one."""

    expected = record.lineage.get(SOURCE_BINDING_LINEAGE_KEY)
    if expected is None:
        return None
    if isinstance(bundle, str) and bundle.startswith(REMOTE_SCHEME):
        repo_id, revision, record_name = _parse_remote_reference(bundle)
        name = posixpath.normpath(posixpath.join(posixpath.dirname(record_name), BINDING_NAME))
        path = _download_remote_file(repo_id, revision, name, bundle)
    else:
        bundle_path = Path(bundle).expanduser().resolve()
        root = bundle_path if bundle_path.is_dir() else bundle_path.parent
        path = root / BINDING_NAME
        if not path.is_file():
            raise ArtifactError(
                f"recipe bundle {record.bundle_id} declares "
                f"{SOURCE_BINDING_LINEAGE_KEY} but carries no {BINDING_NAME}"
            )
    if file_sha256(path) != expected:
        raise ArtifactError(f"recipe bundle {record.bundle_id} source binding checksum mismatch")
    return load_source_plan_binding(path)


def resolve_recipe_plan(
    bundle: str | Path,
    *,
    inventory: Inventory,
) -> ResolvedRecipePlan:
    """Verify a bundle against the target inventory and produce its plan."""
    record, payload = load_recipe_bundle(bundle)
    source_binding = _bundle_source_binding(bundle, record)
    target = inventory.model
    if not is_immutable_revision(target.revision):
        raise ArtifactError(
            f"recipe bundle {record.bundle_id} cannot be resolved against an unpinned inventory"
        )
    if record.source_model.model_id != target.model_id:
        raise ArtifactError(
            f"recipe bundle {record.bundle_id} targets {record.source_model.model_id}, "
            f"not {target.model_id}"
        )
    if record.source_model.revision != target.revision:
        raise ArtifactError(
            f"recipe bundle {record.bundle_id} pins revision "
            f"{record.source_model.revision}, not {target.revision}"
        )
    if record.payload_kind == "plan":
        plan = load_quantization_plan(payload)
        if (
            plan.source_model.model_id != record.source_model.model_id
            or plan.source_model.revision != record.source_model.revision
        ):
            raise ArtifactError(
                f"recipe bundle {record.bundle_id} plan source identity does not match "
                "the bundle record"
            )
        if source_binding is not None:
            binding_issues = source_binding_plan_issues(binding=source_binding, plan=plan)
            if binding_issues:
                raise ArtifactError(
                    f"recipe bundle {record.bundle_id} source binding does not match its "
                    "plan: " + "; ".join(binding_issues)
                )
        if plan.source_model.local_path is not None:
            # Legacy path-bearing payload: the producer's checkpoint path is not
            # portable to another machine, and this payload carries no binding to
            # verify instead, so rebind the executable copy to the target
            # inventory without altering the checksummed bundle payload.
            plan = plan.model_copy(
                update={
                    "source_model": plan.source_model.model_copy(
                        update={"local_path": target.local_path}
                    )
                }
            )
        # A path-neutral payload stays path-neutral: its carried binding is what
        # verifies the consumer's checkpoint, and rebinding it would put the
        # consumer's path back into evidence — and would make the check compare
        # the consumer's directory against itself. Without a binding it can still
        # convert by hub id; convert fails closed for a local directory.
    else:
        recipe = load_manual_plan_recipe(payload)
        plan = manual_quantization_plan(inventory, recipe)
        if source_binding is not None:
            raise ArtifactError(
                f"recipe bundle {record.bundle_id} carries a plan source binding but its "
                "payload is a manual recipe"
            )
    if plan.evidence_kind != record.evidence_kind:
        raise ArtifactError(
            f"recipe bundle {record.bundle_id} declares {record.evidence_kind.value} evidence "
            f"but its payload produces {plan.evidence_kind.value}"
        )
    return ResolvedRecipePlan(record=record, plan=plan, source_binding=source_binding)


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
    loaded = load_quantization_plan(plan_path)
    if not is_immutable_revision(loaded.source_model.revision):
        raise ArtifactError("a recipe bundle requires a revision-pinned plan")
    validated_lineage = _validated_lineage(dict(lineage or {}))
    supplied_directory = Path(output_dir).expanduser()
    if supplied_directory.is_symlink():
        raise ArtifactError("recipe bundle output directory cannot be a symbolic link")
    directory = supplied_directory.resolve()
    payload_name = "plan.json"
    destination = directory / payload_name
    bundle_path = directory / RECIPE_BUNDLE_FILE
    if destination.exists():
        raise ArtifactError(f"recipe bundle payload already exists: {destination}")
    if bundle_path.exists():
        raise ArtifactError(f"recipe bundle record already exists: {bundle_path}")
    record = RecipeBundle(
        bundle_id=bundle_id,
        source_model=loaded.source_model,
        evidence_kind=loaded.evidence_kind,
        payload_kind="plan",
        payload_file=payload_name,
        payload_sha256=file_sha256(plan_path),
        lineage=validated_lineage,
        axquant_version=__version__,
        notes=list(notes or []),
    )
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plan_path, destination)
    # Carry the producer's source binding so a consumer on another machine can
    # verify the checkpoint it converts against the one the producer planned
    # from — a binding minted by the consumer could not say that (AXQ-048).
    binding_source = binding_path_beside(plan_path)
    if binding_source.is_file():
        binding = load_source_plan_binding(binding_source)
        binding_issues = source_binding_plan_issues(binding=binding, plan=loaded)
        if binding_issues:
            raise ArtifactError(
                f"source binding beside {plan_path.name} does not match its plan: "
                + "; ".join(binding_issues)
            )
        binding_target = directory / BINDING_NAME
        shutil.copyfile(binding_source, binding_target)
        record = record.model_copy(
            update={
                "lineage": {
                    **record.lineage,
                    SOURCE_BINDING_LINEAGE_KEY: file_sha256(binding_target),
                }
            }
        )
    write_data(bundle_path, record)
    return bundle_path
