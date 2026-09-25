"""Source-plan binding sidecar model (AXQ-048).

Conversion must prove that the checkpoint it opens is the one the plan was built
from. That proof used to be a local-path equality on ``plan.source_model``, which
forced published evidence to record an operator path. This sidecar carries a
structural fingerprint of the source instead: two small file digests plus the
Safetensors member list. Weight bytes are never hashed, so the cost is O(files),
not O(bytes).

The fingerprint identifies the checkpoint *layout and metadata*, not the weight
values: two checkpoints with the same member sizes and the same index can share
it. It is a conversion-time guard and an auditable record, not a content hash of
the tensor payloads — do not present it as one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.inventory import ModelIdentity

_SHA256 = r"^[0-9a-f]{64}$"


class SourcePlanBindingMember(StrictModel):
    """One Safetensors member of the bound source checkpoint."""

    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


class SourcePlanBinding(StrictModel):
    schema_version: Literal["axquant.source-plan-binding.v1"] = "axquant.source-plan-binding.v1"
    # Binds the sidecar to one plan, so it cannot be paired with another.
    plan_sha256: str = Field(pattern=_SHA256)
    source_model: ModelIdentity
    config_sha256: str = Field(pattern=_SHA256)
    # None when the checkpoint ships no Safetensors index (single-file export).
    index_sha256: str | None = Field(default=None, pattern=_SHA256)
    members: list[SourcePlanBindingMember] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
