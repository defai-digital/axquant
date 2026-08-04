from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from axquant.errors import ArtifactError

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_data(path: str | Path) -> Any:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"cannot read {source}: {exc}") from exc
    try:
        if source.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ArtifactError(f"invalid structured data in {source}: {exc}") from exc
    _reject_non_finite(payload)
    return payload


def load_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    payload = read_data(path)
    _require_schema_version(path, model_type, payload)
    return model_type.model_validate(payload)


def _require_schema_version(path: str | Path, model_type: type[ModelT], payload: Any) -> None:
    """Fail closed when a persisted artifact omits its ``schema_version`` key.

    Every AXQuant model with a ``schema_version`` field declares it as
    ``Literal["...vN"] = "...vN"`` so in-repo code can construct instances
    without repeating the literal. That convenience also means Pydantic
    accepts a JSON payload with the key missing exactly as happily as one
    with it present -- silently treating an unversioned artifact as the
    current schema. This re-asserts "every artifact carries a
    schema_version" specifically for data loaded from disk (the only place
    the guarantee matters) without touching the many in-repo constructor
    call sites that rely on the default for construction convenience.
    """
    if "schema_version" not in model_type.model_fields:
        return
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ArtifactError(
            f"{path}: missing required 'schema_version' field for {model_type.__name__}"
        )


def _serializable(value: BaseModel | dict[str, Any] | list[Any]) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _strip_created_at(value: Any) -> Any:
    """Drop ``created_at`` at every nesting depth before hashing.

    Artifact models stamp ``created_at`` with a fresh timestamp on every
    construction, so two objects with identical semantic content produce
    different hashes unless creation time is excluded. Several call sites
    already work around this manually before calling `stable_sha256`; doing
    it here makes that guarantee hold for every caller, including nested
    sub-models exposed through `model_dump(mode="json")`.
    """
    if isinstance(value, dict):
        return {key: _strip_created_at(item) for key, item in value.items() if key != "created_at"}
    if isinstance(value, list):
        return [_strip_created_at(item) for item in value]
    return value


def stable_sha256(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    payload = json.dumps(
        _strip_created_at(_serializable(value)),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_data(path: str | Path, value: BaseModel | dict[str, Any] | list[Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _serializable(value)
    _reject_non_finite(payload)
    if destination.suffix.lower() in {".yaml", ".yml"}:
        rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    else:
        rendered = (
            json.dumps(
                payload,
                allow_nan=False,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    write_text(destination, rendered)


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactError("structured artifacts cannot contain non-finite numbers")
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_non_finite(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)


def write_text(path: str | Path, rendered: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
