"""Provenance records for adapted (non-parent-only) MTP sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axquant.schema._base import utc_now
from axquant.serde import file_sha256, write_data

ADAPTED_GRAFT_KIND = "holo3-adapted-mtp-v1"
PARENT_GRAFT_KIND = "parent-qwen35-moe-mtp"


def write_adapted_graft_record(
    output_dir: str | Path,
    *,
    trunk_model_id: str,
    trunk_revision: str,
    donor_model_id: str,
    donor_revision: str,
    init_mtp_sha256: str,
    output_mtp_sha256: str,
    train_summary: dict[str, Any],
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "axquant_mtp_graft.json"
    write_data(
        path,
        {
            "schema_version": "axquant.mtp-graft.v1",
            "graft_kind": ADAPTED_GRAFT_KIND,
            "trunk_model": {"model_id": trunk_model_id, "revision": trunk_revision},
            "donor_model": {"model_id": donor_model_id, "revision": donor_revision},
            "init_mtp_sha256": init_mtp_sha256,
            "output_mtp_sha256": output_mtp_sha256,
            "train": train_summary,
            "notes": [
                "MTP head adapted on Holo3 trunk labels; not full co-training of the trunk.",
                "Acceleration claims still require Tier 2 exactness/speedup evidence.",
            ],
            "created_at": utc_now().isoformat(),
        },
    )
    return path


def sidecar_sha256(path: str | Path) -> str:
    return file_sha256(Path(path).expanduser().resolve())
