"""Minimal synthetic Tier 1 / Tier 2 certificate pairs for loader tests.

The public catalog was withdrawn pending re-certification, so tests that
need a certificate pair build one here instead of reading live records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_STAMP = "2026-10-03T00:00:00+00:00"
_HOST = "synthetic-test-host"
_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def tier1_payload(
    *,
    hub_repo_id: str = "AutomatosX/AX-Demo-MLX-AXQ-6bit",
    status: str = "certified",
    mtp_status: str = "certified-see-tier2-record",
    display_name: str = "Demo MLX AXQ 6-bit",
    sort_order: int = 1,
    listed: bool = True,
    tier2_certificate: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "axquant.public-checkpoint-certification.v1",
        "status": status,
        "host_id": _HOST,
        "certified_at": _STAMP,
        "artifact": {
            "hub_repo_id": hub_repo_id,
            "hub_commit": _COMMIT,
            "product_class": "6bit",
            "candidate_manifest_sha256": "b" * 64,
        },
        "plan": {},
        "size": {},
        "quality": {},
        "thresholds": {},
        "mtp_acceleration": {
            "status": mtp_status,
            **({"tier2_certificate": tier2_certificate} if tier2_certificate is not None else {}),
        },
        "toolchain": {},
        "public_index": {
            "display_name": display_name,
            "sort_order": sort_order,
            "edition_label": "main",
            "listed": listed,
        },
    }


def tier2_payload(
    *,
    hub_repo_id: str = "AutomatosX/AX-Demo-MLX-AXQ-6bit",
    status: str = "certified",
) -> dict[str, Any]:
    return {
        "schema_version": "axquant.public-mtp-acceleration-certification.v1",
        "status": status,
        "host_id": _HOST,
        "certified_at": _STAMP,
        "artifact": {
            "hub_repo_id": hub_repo_id,
            "hub_commit": _COMMIT,
            "product_class": "6bit",
            "candidate_manifest_sha256": "b" * 64,
        },
        "thresholds": {
            "prompt_median_speedup_min": 1.1,
            "token_weighted_decode_speedup_min": 1.2,
        },
        "mtp_acceleration": {
            "status": "certified",
            "ax_engine_version": "9.9.9-test",
            "profiles": {
                "default": {
                    "exactness_pass": True,
                    "prompt_median_speedup": 1.5,
                    "token_weighted_decode_speedup": 1.6,
                }
            },
        },
        "toolchain": {},
    }


def write_cert_pair(
    directory: Path,
    stem: str,
    *,
    hub_repo_id: str = "AutomatosX/AX-Demo-MLX-AXQ-6bit",
    tier1_status: str = "certified",
    tier2_status: str = "certified",
    tier1_mtp_status: str = "certified-see-tier2-record",
    with_tier2: bool = True,
    listed: bool = True,
) -> tuple[Path, Path | None]:
    """Write a consistent Tier 1 / Tier 2 pair plus markdown companions."""

    tier1_path = directory / f"{stem}-tier1.json"
    tier1_path.write_text(
        json.dumps(
            tier1_payload(
                hub_repo_id=hub_repo_id,
                status=tier1_status,
                mtp_status=tier1_mtp_status,
                display_name=f"Demo {stem}",
                listed=listed,
                tier2_certificate=f"{stem}-tier2.json" if with_tier2 else None,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / f"{stem}-tier1.md").write_text("# tier 1\n", encoding="utf-8")
    if not with_tier2:
        return tier1_path, None
    tier2_path = directory / f"{stem}-tier2.json"
    tier2_path.write_text(
        json.dumps(
            tier2_payload(status=tier2_status, hub_repo_id=hub_repo_id),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / f"{stem}-tier2.md").write_text("# tier 2\n", encoding="utf-8")
    return tier1_path, tier2_path
