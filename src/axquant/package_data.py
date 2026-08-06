"""Load packaged data files shipped under ``axquant.data``.

YAML and other static tables live next to ``reference_calibration.jsonl`` so
wheels include them without extra hatch configuration. Fail closed when a
resource is missing or invalid — same class of failure as a missing constant.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import Any

import yaml

from axquant.errors import ArtifactError


@cache
def load_package_yaml(name: str) -> Any:
    """Load a YAML document from the ``axquant.data`` package by file name.

    ``name`` is a file name only (for example ``profiles.yaml``), not a path.
    """
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ArtifactError(f"invalid package data name: {name!r}")
    resource = files("axquant.data").joinpath(name)
    try:
        text = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, TypeError) as exc:
        raise ArtifactError(f"missing package data {name!r}: {exc}") from exc
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ArtifactError(f"invalid package data {name!r}: {exc}") from exc
    if payload is None:
        raise ArtifactError(f"package data {name!r} is empty")
    return payload


def message_template(section: str, key: str) -> str:
    """Return a stable message template from ``messages.yaml``."""
    payload = load_package_yaml("messages.yaml")
    if not isinstance(payload, dict):
        raise ArtifactError("messages.yaml must be a mapping")
    section_value = payload.get(section)
    if not isinstance(section_value, dict) or key not in section_value:
        raise ArtifactError(f"messages.yaml missing {section}.{key}")
    value = section_value[key]
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"messages.yaml {section}.{key} must be a non-empty string")
    return value
