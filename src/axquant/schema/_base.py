from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SoftwareVersions(StrictModel):
    axquant: str
    python: str
    mlx: str | None = None
    mlx_lm: str | None = None
    ax_engine: str | None = None
    safetensors: str
    pydantic: str
