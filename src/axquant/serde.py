from __future__ import annotations

import hashlib
import json
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
            return yaml.safe_load(text)
        return json.loads(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ArtifactError(f"invalid structured data in {source}: {exc}") from exc


def load_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(read_data(path))


def _serializable(value: BaseModel | dict[str, Any] | list[Any]) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def stable_sha256(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    payload = json.dumps(
        _serializable(value),
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
    if destination.suffix.lower() in {".yaml", ".yml"}:
        rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    else:
        rendered = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    write_text(destination, rendered)


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
