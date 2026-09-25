"""Build, write, load, and verify the source-plan binding sidecar (AXQ-048).

The sidecar is written next to the plan by ``plan`` and copied into the artifact
by convert, so the published tree records what the conversion was bound to
without recording where it happened. Verification recomputes the fingerprint from
the checkpoint actually being opened.
"""

from __future__ import annotations

from pathlib import Path

from axquant.errors import ArtifactError
from axquant.identity import semantic_model_identity, semantic_plan_sha256
from axquant.schema import (
    ModelIdentity,
    QuantizationPlan,
    SourcePlanBinding,
    SourcePlanBindingMember,
)
from axquant.schema.loading import load_versioned
from axquant.serde import file_sha256, read_data, write_data

BINDING_NAME = "axquant_source_binding.json"
_INDEX_NAME = "model.safetensors.index.json"


def source_binding_fingerprint(
    directory: str | Path,
) -> tuple[str, str | None, list[tuple[str, int]]]:
    """Structural fingerprint of a checkpoint directory.

    Returns ``(config_sha256, index_sha256_or_None, members)`` where members are
    sorted ``(relative_path, size_bytes)`` for every Safetensors file. Only
    ``config.json`` and the index are hashed; weight files are stat'ed.
    """

    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ArtifactError(f"source binding requires a checkpoint directory: {root}")
    config = root / "config.json"
    if not config.is_file():
        raise ArtifactError(f"source binding requires a checkpoint config: {config}")
    index = root / _INDEX_NAME
    members = sorted(
        (
            (path.relative_to(root).as_posix(), path.stat().st_size)
            for path in root.rglob("*.safetensors")
            if path.is_file()
        ),
    )
    if not members:
        raise ArtifactError(f"source binding found no Safetensors members under {root}")
    return (
        file_sha256(config),
        file_sha256(index) if index.is_file() else None,
        members,
    )


def build_source_plan_binding(
    plan: QuantizationPlan,
    source_dir: str | Path,
) -> SourcePlanBinding:
    """Bind ``plan`` to the structural fingerprint of ``source_dir``."""

    config_sha256, index_sha256, members = source_binding_fingerprint(source_dir)
    return SourcePlanBinding(
        plan_sha256=semantic_plan_sha256(plan),
        source_model=semantic_model_identity(plan.source_model),
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        members=[SourcePlanBindingMember(path=path, size_bytes=size) for path, size in members],
    )


def write_source_plan_binding(
    directory: str | Path,
    plan: QuantizationPlan,
    source_dir: str | Path,
) -> Path:
    """Write the binding beside the plan and return its path."""

    target = Path(directory).expanduser() / BINDING_NAME
    write_data(target, build_source_plan_binding(plan, source_dir))
    return target


def load_source_plan_binding(path: str | Path) -> SourcePlanBinding:
    return load_versioned(path, SourcePlanBinding)


def binding_path_beside(path: str | Path) -> Path:
    """The binding that belongs to the artifact at ``path``."""

    resolved = Path(path).expanduser()
    return (resolved if resolved.is_dir() else resolved.parent) / BINDING_NAME


def source_binding_plan_issues(
    *,
    binding: SourcePlanBinding,
    plan: QuantizationPlan,
) -> list[str]:
    """Binding issues that need only the plan, not the source directory.

    A bundle carries the producer's binding and the consumer holds a different
    copy of the same revision, so the plan half must be checkable on its own.
    """

    issues: list[str] = []
    if binding.plan_sha256 != semantic_plan_sha256(plan):
        issues.append("source binding belongs to another plan")
    if binding.source_model != semantic_model_identity(plan.source_model):
        issues.append("source binding identifies another source model")
    return issues


def source_binding_issues(
    *,
    binding: SourcePlanBinding,
    plan: QuantizationPlan,
    source_dir: str | Path,
) -> list[str]:
    """Differences between a binding and the checkpoint actually supplied."""

    issues = source_binding_plan_issues(binding=binding, plan=plan)
    try:
        config_sha256, index_sha256, members = source_binding_fingerprint(source_dir)
    except ArtifactError as exc:
        return [*issues, str(exc)]
    if binding.config_sha256 != config_sha256:
        issues.append("source checkpoint config differs from the bound source")
    if binding.index_sha256 != index_sha256:
        issues.append("source checkpoint Safetensors index differs from the bound source")
    bound_members = [(member.path, member.size_bytes) for member in binding.members]
    if bound_members != members:
        missing = sorted(set(bound_members) - set(members))
        extra = sorted(set(members) - set(bound_members))
        issues.append(
            f"source checkpoint members differ from the bound source: {missing=} {extra=}"
        )
    return issues


def path_shaped_model_id(value: str) -> bool:
    """True when an identity value is a filesystem location, not a name.

    A locally sourced run records its source as ``--model-id`` when given one and
    falls back to the argument otherwise, so a bare path can end up as
    ``model_id``. Such a value may not be published: convert from a hub id or
    re-run the pipeline that produced the evidence with an explicit
    ``--model-id``.
    """

    if value.startswith("file:"):
        return True
    return value.startswith(("/", "~")) or "\\" in value


def path_shaped_identity_issues(model: ModelIdentity) -> list[str]:
    """Identity values that may not be published because they are filesystem paths."""

    return ["model_id is a filesystem path"] if path_shaped_model_id(model.model_id) else []


def _identity_values(node: object, path: str = "$") -> list[tuple[str, str]]:
    """Every ``model``/``model_id`` string in a payload, with its location.

    These are the fields a locally sourced run fills with the ``--model``
    argument, so they are the ones that can carry a filesystem path into
    published evidence.
    """

    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"model", "model_id"} and isinstance(value, str):
                found.append((f"{path}.{key}", value))
                continue
            found.extend(_identity_values(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_identity_values(item, f"{path}[{index}]"))
    return found


def tree_identity_issues(directory: str | Path) -> dict[str, list[str]]:
    """Path-shaped model identities found anywhere in a publication tree.

    The plan and manifest are denied outright by ``artifact_identity_issues``.
    The rest are reported so an operator can see what a stricter gate would
    reject — the same content the publication privacy scan flags for absolute
    paths, extended to relative ones and ``file:`` URLs.
    """

    root = Path(directory).expanduser()
    reported: dict[str, list[str]] = {}
    try:
        members = sorted(path for path in root.rglob("*") if path.is_file())
    except OSError:
        return reported
    for path in members:
        if path.suffix.casefold() not in {".json", ".yaml", ".yml"}:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"axquant_plan.json", "axquant_manifest.json"}:
            continue
        try:
            payload = read_data(path)
        except ArtifactError:
            continue
        issues = [
            f"{location} is a filesystem path ({value!r})"
            for location, value in _identity_values(payload)
            if path_shaped_model_id(value)
        ]
        if issues:
            reported[relative] = issues
    return reported


def artifact_identity_issues(directory: str | Path) -> list[str]:
    """Path-shaped identities recorded in an artifact that is about to be published."""

    root = Path(directory).expanduser()
    issues: list[str] = []
    for name in ("axquant_plan.json", "axquant_manifest.json"):
        path = root / name
        if not path.is_file():
            continue
        payload = read_data(path)
        source_model = payload.get("source_model") if isinstance(payload, dict) else None
        value = source_model.get("model_id") if isinstance(source_model, dict) else None
        if isinstance(value, str) and path_shaped_model_id(value):
            issues.append(
                f"{name}: source_model.model_id is a filesystem path ({value!r}); "
                "re-run the pipeline with an explicit --model-id so the published "
                "artifact records a name instead of a location"
            )
    return issues
