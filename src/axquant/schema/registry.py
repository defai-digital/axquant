"""Schema version registry for freeze-class contracts.

AXQ-042 freezes public certification schema versions first. Broader toolkit
artifact snapshots (Codex CG1/CG3) can extend this registry without changing
the public-cert entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from axquant.schema._base import StrictModel
from axquant.schema.expert_stream import ExpertStreamManifest
from axquant.schema.public_certification import (
    CHECKPOINT_SCHEMA_VERSION,
    MTP_SCHEMA_VERSION,
    PublicCheckpointCertification,
    PublicMtpAccelerationCertification,
)

CompatibilityClass = Literal[
    "public-certification",
    "evidence",
    "release",
    "operational",
]
FreezePolicy = Literal["immutable-envelope", "additive-ok"]


@dataclass(frozen=True, slots=True)
class SchemaRegistryEntry:
    schema_version: str
    model: type[StrictModel]
    compatibility_class: CompatibilityClass
    freeze_policy: FreezePolicy
    description: str


_PUBLIC_CERT_ENTRIES: Final[tuple[SchemaRegistryEntry, ...]] = (
    SchemaRegistryEntry(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        model=PublicCheckpointCertification,
        compatibility_class="public-certification",
        freeze_policy="immutable-envelope",
        description="Public checkpoint Tier 1 certificate under docs/certifications/",
    ),
    SchemaRegistryEntry(
        schema_version=MTP_SCHEMA_VERSION,
        model=PublicMtpAccelerationCertification,
        compatibility_class="public-certification",
        freeze_policy="immutable-envelope",
        description="Public MTP acceleration Tier 2 certificate under docs/certifications/",
    ),
)

_OPERATIONAL_ENTRIES: Final[tuple[SchemaRegistryEntry, ...]] = (
    SchemaRegistryEntry(
        schema_version="axquant.expert-stream.v1",
        model=ExpertStreamManifest,
        compatibility_class="operational",
        freeze_policy="additive-ok",
        description="Layer-stack SSD expert streaming contract for AX Engine.",
    ),
)

_REGISTERED_ENTRIES: Final[tuple[SchemaRegistryEntry, ...]] = (
    *_PUBLIC_CERT_ENTRIES,
    *_OPERATIONAL_ENTRIES,
)


def schema_registry() -> tuple[SchemaRegistryEntry, ...]:
    """Return explicit schema policies (public-certification entries first)."""

    return _REGISTERED_ENTRIES


def schema_entry(schema_version: str) -> SchemaRegistryEntry:
    for entry in _REGISTERED_ENTRIES:
        if entry.schema_version == schema_version:
            return entry
    raise KeyError(f"unregistered schema_version: {schema_version!r}")


def public_certification_schema_versions() -> frozenset[str]:
    return frozenset(
        entry.schema_version
        for entry in _PUBLIC_CERT_ENTRIES
        if entry.compatibility_class == "public-certification"
    )
