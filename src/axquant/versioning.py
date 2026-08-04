from __future__ import annotations

import platform
import re
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from axquant import __version__
from axquant.schema import SoftwareVersions

_VERSION_DIRECTORY = re.compile(r"^v?(?P<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z][0-9A-Za-z.-]*)?)$")


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def standalone_executable_version(executable: str | Path) -> str | None:
    """Infer a version only from an explicitly versioned standalone install directory.

    Accepts either ``.../<version>/<binary>`` or ``.../<version>/bin/<binary>``.
    Unversioned paths return None so release provenance never invents a version.
    """
    resolved = Path(executable).expanduser().resolve()
    if not resolved.is_file():
        return None
    version_directory = resolved.parent.parent if resolved.parent.name == "bin" else resolved.parent
    match = _VERSION_DIRECTORY.fullmatch(version_directory.name)
    return match.group("version") if match is not None else None


def executable_distribution_version(executable: str, distribution: str) -> str | None:
    """Return a distribution version for a resolved executable when available.

    Prefers Homebrew Cellar layout, then an explicitly versioned standalone path.
    """
    resolved_name = shutil.which(executable)
    if resolved_name is None:
        path = Path(executable).expanduser()
        if path.parent != Path(".") and path.is_file():
            resolved_name = str(path.resolve())
        else:
            return None
    resolved = Path(resolved_name).resolve()
    parts = resolved.parts
    for index, part in enumerate(parts[:-2]):
        if part == distribution and parts[index - 1 : index] == ("Cellar",):
            match = _VERSION_DIRECTORY.fullmatch(parts[index + 1])
            return match.group("version") if match is not None else None
    return standalone_executable_version(resolved)


def collect_versions(*, ax_engine_executable: str = "ax-engine-bench") -> SoftwareVersions:
    safetensors = _version("safetensors")
    pydantic = _version("pydantic")
    if safetensors is None or pydantic is None:
        raise RuntimeError("required package version metadata is unavailable")
    return SoftwareVersions(
        axquant=__version__,
        python=platform.python_version(),
        mlx=_version("mlx"),
        mlx_lm=_version("mlx-lm"),
        ax_engine=executable_distribution_version(ax_engine_executable, "ax-engine"),
        safetensors=safetensors,
        pydantic=pydantic,
    )
