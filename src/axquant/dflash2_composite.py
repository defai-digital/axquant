"""DFlash2 composite target + drafter packs (AXQ-067 DF-5, local pilot).

A composite is a new directory holding the target pack's files unchanged plus
the admitted upstream drafter's ``config.json`` and ``model.safetensors``
under ``dflash2/``, where AX Engine's bundled discovery looks first. Nothing
is requantized: the drafter bytes must equal the mirrored pin and the target
bytes must equal the target's manifest. ``axquant_dflash2_companion.json``
binds both. The target's certificates do not transfer, and the composite is
development metadata; route, speed and quality need their own evidence.
Publication of a ``-DFlash2`` repository needs a separate owner direction
(AXQ-071).
"""

from __future__ import annotations

import os
import re
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from axquant.artifact_paths import artifact_tree_files
from axquant.dflash2_target import (
    DECLARATION_NAME,
    admitted_targets,
    require_publication_declaration,
)
from axquant.errors import ArtifactError, AxquantError
from axquant.naming import DFLASH2_COMPANION_NAME
from axquant.schema.dflash2 import (
    Dflash2Companion,
    Dflash2CompanionFile,
    Dflash2DrafterPin,
    Dflash2TargetDeclaration,
)
from axquant.serde import file_sha256, load_model, read_data, write_data

DRAFTER_DIR = "dflash2"
_DRAFTER_FILES = ("config.json", "model.safetensors")
_LICENSE_LINE = re.compile(r"^license:\s*([A-Za-z0-9_.+-]+)\s*$", re.MULTILINE)


def _composer_version() -> str:
    try:
        return f"axquant {version('axquant')}"
    except PackageNotFoundError:
        return "axquant unknown"


def _target_manifest_issues(target: Path) -> list[str]:
    payload = read_data(target / "axquant_manifest.json")
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        return ["target axquant_manifest.json has no file list"]
    issues: list[str] = []
    for record in files:
        if not isinstance(record, dict):
            issues.append("target manifest file record is not an object")
            continue
        name, size, digest = record.get("path"), record.get("size_bytes"), record.get("sha256")
        path = target / str(name)
        if not isinstance(name, str) or not path.is_file():
            issues.append(f"target manifest file is missing: {name}")
        elif path.stat().st_size != size:
            issues.append(f"target manifest size mismatch: {name}")
        elif file_sha256(path) != digest:
            issues.append(f"target manifest digest mismatch: {name}")
    return issues


def _drafter_license(drafter_dir: Path) -> str:
    readme = drafter_dir / "README.md"
    if not readme.is_file():
        raise ArtifactError("drafter README.md is required to record its license")
    text = readme.read_text(encoding="utf-8")
    front = text.split("---", 2)[1] if text.startswith("---") and text.count("---") >= 2 else ""
    match = _LICENSE_LINE.search(front)
    if match is None:
        raise ArtifactError("drafter README.md front matter declares no license")
    return match.group(1)


def _admitted_pin(declaration: Dflash2TargetDeclaration) -> Dflash2DrafterPin:
    for target in admitted_targets().targets:
        if target.source_model_id == declaration.admitted_source_model_id:
            return Dflash2DrafterPin(
                repo=target.drafter_repo,
                revision=target.drafter_revision,
                config_sha256=target.drafter_config_sha256,
                weights_sha256=target.drafter_weights_sha256,
            )
    raise ArtifactError(f"{declaration.admitted_source_model_id} is not an admitted target")


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    real = source.resolve()
    try:
        os.link(real, target)
    except OSError:
        shutil.copy2(real, target)


def _notice(pin: Dflash2DrafterPin, license_id: str) -> str:
    return (
        f"The files in this directory are the DFlash2 drafter {pin.repo} at revision\n"
        f"{pin.revision}, redistributed unmodified under its upstream license\n"
        f"({license_id}). README.md is the upstream model card. AXQuant did not train,\n"
        "fine-tune or requantize these files; AX Engine quantizes the drafter at load.\n"
    )


def compose_dflash2_composite(
    *, target_dir: str | Path, drafter_dir: str | Path, output_dir: str | Path
) -> Dflash2Companion:
    """Build a composite in a new directory; fail closed on any digest mismatch."""

    target = Path(target_dir).expanduser().resolve()
    drafter = Path(drafter_dir).expanduser()
    output = Path(output_dir).expanduser().absolute()
    if output.exists():
        raise ArtifactError(f"composite output already exists: {output}")
    if not output.parent.is_dir():
        raise ArtifactError(f"composite output parent does not exist: {output.parent}")
    try:
        output.relative_to(target)
    except ValueError:
        pass
    else:
        raise ArtifactError("composite output must be outside the target pack")
    try:
        declaration = require_publication_declaration(target)
    except (AxquantError, OSError, ValueError) as exc:
        raise ArtifactError(f"target DFlash2 declaration check failed: {exc}") from exc
    if declaration is None or declaration.status != "target" or declaration.drafter is None:
        raise ArtifactError("composites are built only from a declared DFlash2 target pack")
    pin = _admitted_pin(declaration)
    if declaration.drafter != pin:
        raise ArtifactError("target declaration drafter pin differs from the admitted mirror")
    assert declaration.source_model_id is not None and declaration.source_revision is not None
    members = artifact_tree_files(target)
    names = [path.relative_to(target).as_posix() for path in members]
    if DFLASH2_COMPANION_NAME in names or any(name.startswith(f"{DRAFTER_DIR}/") for name in names):
        raise ArtifactError("target pack already bundles a DFlash2 drafter")
    issues = _target_manifest_issues(target)
    if issues:
        raise ArtifactError("target pack is not manifest-bound: " + "; ".join(issues))
    expected = {"config.json": pin.config_sha256, "model.safetensors": pin.weights_sha256}
    for name in _DRAFTER_FILES:
        path = drafter / name
        if not path.is_file():
            raise ArtifactError(f"drafter file is missing: {name}")
        if file_sha256(path) != expected[name]:
            raise ArtifactError(f"drafter {name} does not match the admitted pin")
    license_id = _drafter_license(drafter)

    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        for path, name in zip(members, names, strict=True):
            _link_or_copy(path, staging / name)
        for name in (*_DRAFTER_FILES, "README.md"):
            _link_or_copy(drafter / name, staging / DRAFTER_DIR / name)
        (staging / DRAFTER_DIR / "NOTICE").write_text(_notice(pin, license_id), encoding="utf-8")
        records = [
            Dflash2CompanionFile(
                path=f"{DRAFTER_DIR}/{path.name}",
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
            )
            for path in sorted((staging / DRAFTER_DIR).iterdir())
        ]
        companion = Dflash2Companion(
            target_artifact_manifest_sha256=file_sha256(target / "axquant_manifest.json"),
            target_declaration_sha256=file_sha256(target / DECLARATION_NAME),
            source_model_id=declaration.source_model_id,
            source_revision=declaration.source_revision,
            drafter=pin,
            drafter_license=license_id,
            files=records,
            composer_version=_composer_version(),
        )
        write_data(staging / DFLASH2_COMPANION_NAME, companion)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return companion


def composite_issues(directory: str | Path) -> list[str]:
    """Return issues unless a composite still matches its companion record."""

    root = Path(directory).expanduser().resolve()
    path = root / DFLASH2_COMPANION_NAME
    if not path.is_file():
        return [f"{DFLASH2_COMPANION_NAME} is missing"]
    try:
        companion = load_model(path, Dflash2Companion)
    except (AxquantError, OSError, ValueError) as exc:
        return [f"{DFLASH2_COMPANION_NAME} is invalid: {exc}"]
    issues: list[str] = []
    if file_sha256(root / "axquant_manifest.json") != companion.target_artifact_manifest_sha256:
        issues.append("target axquant_manifest.json differs from the companion binding")
    declaration = root / DECLARATION_NAME
    if not declaration.is_file() or file_sha256(declaration) != companion.target_declaration_sha256:
        issues.append(f"{DECLARATION_NAME} differs from the companion binding")
    issues += _target_manifest_issues(root)
    recorded = {item.path: item for item in companion.files}
    present = {
        p.relative_to(root).as_posix()
        for p in artifact_tree_files(root)
        if p.relative_to(root).as_posix().startswith(f"{DRAFTER_DIR}/")
    }
    for name in sorted(present ^ set(recorded)):
        issues.append(f"drafter directory differs from the companion record: {name}")
    for name in sorted(present & set(recorded)):
        if file_sha256(root / name) != recorded[name].sha256:
            issues.append(f"drafter file digest mismatch: {name}")
    try:
        pin = _admitted_pin(load_model(declaration, Dflash2TargetDeclaration))
    except (AxquantError, OSError, ValueError) as exc:
        issues.append(f"admitted drafter pin unavailable: {exc}")
    else:
        if pin != companion.drafter:
            issues.append("companion drafter pin differs from the admitted mirror")
    return issues


def composite_summary(companion: Dflash2Companion) -> dict[str, Any]:
    return {
        "drafter": f"{companion.drafter.repo}@{companion.drafter.revision}",
        "drafter_license": companion.drafter_license,
        "source": f"{companion.source_model_id}@{companion.source_revision}",
        "files": [item.path for item in companion.files],
    }
