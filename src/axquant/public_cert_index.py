"""Public certification index: certificate JSON is the single source of truth.

Machine-readable records under ``docs/certifications/*.json`` drive the
README matrix, the certification index table, the release certification
matrix, and the Hub model-card certification section. Hand-edited table
cells that disagree with those records are a documentation bug.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from axquant.schema.certification import CheckpointCertificationClaim

_CHECKPOINT_SCHEMA = "axquant.public-checkpoint-certification.v1"
_MTP_SCHEMA = "axquant.public-mtp-acceleration-certification.v1"

BEGIN_MARKER = "<!-- BEGIN:AXQUANT_CERTIFICATION_MATRIX -->"
END_MARKER = "<!-- END:AXQUANT_CERTIFICATION_MATRIX -->"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CERT_DIR = _REPO_ROOT / "docs" / "certifications"
_DEFAULT_README = _REPO_ROOT / "README.md"
_DEFAULT_INDEX = _DEFAULT_CERT_DIR / "README.md"
_DEFAULT_RELEASE_MATRIX = _REPO_ROOT / "docs" / "releases" / "certification-matrix.md"

_GITHUB_CERT_BASE = "https://github.com/defai-digital/axquant/blob/main/docs/certifications"

Tier1Label = Literal["Certified", "Not Certified"]
Tier2Label = Literal["Certified", "Not Certified", "N/A (no MTP)"]


@dataclass(frozen=True, slots=True)
class PublicCertRow:
    """One public index row derived from a checkpoint Tier 1 certificate."""

    record_id: str
    """Stem shared by tier files, e.g. ``gemma4-12b-axq4``."""

    display_name: str
    sort_order: int
    listed: bool
    edition_label: str
    tier1_status: Literal["certified", "not_certified"]
    tier2_status: Literal["certified", "not_certified", "not_applicable"]
    hub_repo_id: str
    hub_commit: str
    host_id: str
    product_class: str
    candidate_manifest_sha256: str | None
    certified_at: str | None
    mtp_acceleration_status: str
    mtp_acceleration_reason: str | None
    tier1_path: Path
    tier2_path: Path | None

    @property
    def tier1_stem(self) -> str:
        return f"{self.record_id}-tier1"

    @property
    def tier2_stem(self) -> str | None:
        return f"{self.record_id}-tier2" if self.tier2_path is not None else None

    @property
    def tier1_label(self) -> Tier1Label:
        return "Certified" if self.tier1_status == "certified" else "Not Certified"

    @property
    def tier2_label(self) -> Tier2Label:
        if self.tier2_status == "not_applicable":
            return "N/A (no MTP)"
        if self.tier2_status == "certified":
            return "Certified"
        return "Not Certified"


def certifications_dir(root: Path | None = None) -> Path:
    return (root or _REPO_ROOT) / "docs" / "certifications"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: certification record must be a JSON object")
    return data


def _require_str(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: missing or empty string field {key!r}")
    return value


def _public_index_block(data: dict[str, Any], *, context: str) -> dict[str, Any]:
    block = data.get("public_index")
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"{context}: public_index must be an object")
    return block


def _edition_label(
    data: dict[str, Any],
    artifact: dict[str, Any],
    *,
    context: str,
) -> str:
    block = _public_index_block(data, context=context)
    explicit = block.get("edition_label")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    edition = artifact.get("artifact_edition")
    if isinstance(edition, str) and edition.strip():
        return edition.strip()
    commit = artifact.get("hub_commit")
    if isinstance(commit, str) and len(commit) >= 8:
        return f"main@`{commit[:8]}`"
    raise ValueError(f"{context}: cannot derive edition_label without hub_commit")


def _display_name(
    data: dict[str, Any],
    *,
    context: str,
    record_id: str,
) -> str:
    block = _public_index_block(data, context=context)
    explicit = block.get("display_name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    raise ValueError(
        f"{context}: public_index.display_name is required for listed certificate "
        f"{record_id!r} (certification metadata is the public index SSOT)"
    )


def _sort_order(data: dict[str, Any], *, context: str, default: int) -> int:
    block = _public_index_block(data, context=context)
    value = block.get("sort_order", default)
    if type(value) is not int:
        raise ValueError(f"{context}: public_index.sort_order must be an int")
    return value


def _listed(data: dict[str, Any], *, context: str) -> bool:
    block = _public_index_block(data, context=context)
    value = block.get("listed", True)
    if type(value) is not bool:
        raise ValueError(f"{context}: public_index.listed must be a bool")
    return value


def _tier2_status(
    *,
    tier1: dict[str, Any],
    tier2: dict[str, Any] | None,
    context: str,
) -> Literal["certified", "not_certified", "not_applicable"]:
    mtp = tier1.get("mtp_acceleration")
    if not isinstance(mtp, dict):
        raise ValueError(f"{context}: mtp_acceleration object is required")
    status = mtp.get("status")
    if status == "not-applicable":
        return "not_applicable"
    if tier2 is not None:
        t2_status = tier2.get("status")
        if t2_status == "certified":
            return "certified"
        if t2_status == "not_certified":
            return "not_certified"
    if status in {"not-certified", "not_certified"}:
        return "not_certified"
    if status in {"certified", "certified-scoped", "certified-see-tier2-record"}:
        # Claimed on the Tier 1 record but no certified Tier 2 file yet.
        return "not_certified"
    raise ValueError(f"{context}: unrecognized mtp_acceleration.status {status!r}")


def _claim_mtp_status(
    row: PublicCertRow,
) -> Literal["certified", "certified-scoped", "not-certified"]:
    status = row.mtp_acceleration_status
    if status in {"not-certified", "not_certified", "not-applicable"}:
        return "not-certified"
    if status == "certified-scoped":
        return "certified-scoped"
    if status == "certified-see-tier2-record":
        return "certified-scoped" if row.tier2_status == "certified" else "not-certified"
    if status == "certified":
        return "certified" if row.tier2_status == "certified" else "not-certified"
    return "not-certified"


def load_public_cert_rows(
    cert_dir: Path | None = None,
    *,
    listed_only: bool = True,
) -> list[PublicCertRow]:
    """Load checkpoint Tier 1 certificates and resolve Tier 2 companions."""

    directory = cert_dir or _DEFAULT_CERT_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"certifications directory not found: {directory}")

    rows: list[PublicCertRow] = []
    for path in sorted(directory.glob("*-tier1.json")):
        data = _load_json(path)
        schema = data.get("schema_version")
        if schema != _CHECKPOINT_SCHEMA:
            raise ValueError(f"{path.name}: expected schema {_CHECKPOINT_SCHEMA!r}, got {schema!r}")
        if data.get("certification_tier") != "checkpoint":
            raise ValueError(f"{path.name}: certification_tier must be 'checkpoint'")

        raw_status = data.get("status")
        if raw_status == "certified":
            status: Literal["certified", "not_certified"] = "certified"
        elif raw_status == "not_certified":
            status = "not_certified"
        else:
            raise ValueError(f"{path.name}: status must be certified or not_certified")

        artifact = data.get("artifact")
        if not isinstance(artifact, dict):
            raise ValueError(f"{path.name}: artifact object is required")
        hub_repo_id = _require_str(artifact, "hub_repo_id", context=path.name)
        hub_commit = _require_str(artifact, "hub_commit", context=path.name)
        product_class = _require_str(artifact, "product_class", context=path.name)
        host_id = _require_str(data, "host_id", context=path.name)

        manifest_sha = artifact.get("candidate_manifest_sha256")
        if manifest_sha is not None and (
            not isinstance(manifest_sha, str) or len(manifest_sha) != 64
        ):
            raise ValueError(f"{path.name}: candidate_manifest_sha256 must be 64 hex chars")

        mtp = data.get("mtp_acceleration")
        if not isinstance(mtp, dict):
            raise ValueError(f"{path.name}: mtp_acceleration object is required")
        mtp_status = _require_str(mtp, "status", context=path.name)
        mtp_reason = mtp.get("reason")
        if mtp_reason is not None and not isinstance(mtp_reason, str):
            raise ValueError(f"{path.name}: mtp_acceleration.reason must be a string")

        record_id = path.name.removesuffix("-tier1.json")
        tier2_path = directory / f"{record_id}-tier2.json"
        tier2_data: dict[str, Any] | None = None
        if tier2_path.is_file():
            tier2_data = _load_json(tier2_path)
            if tier2_data.get("schema_version") != _MTP_SCHEMA:
                raise ValueError(
                    f"{tier2_path.name}: expected schema {_MTP_SCHEMA!r}, "
                    f"got {tier2_data.get('schema_version')!r}"
                )
            md_path = directory / f"{record_id}-tier2.md"
            if not md_path.is_file():
                raise ValueError(f"{tier2_path.name}: missing companion markdown {md_path.name}")

        md_path = directory / f"{record_id}-tier1.md"
        if not md_path.is_file():
            raise ValueError(f"{path.name}: missing companion markdown {md_path.name}")

        listed = _listed(data, context=path.name)
        # Unlisted records may omit display_name; listed ones must declare it.
        if listed:
            display_name = _display_name(data, context=path.name, record_id=record_id)
        else:
            block = _public_index_block(data, context=path.name)
            raw = block.get("display_name")
            display_name = raw.strip() if isinstance(raw, str) and raw.strip() else record_id

        certified_at = data.get("certified_at") or data.get("evaluated_at")
        if certified_at is not None and not isinstance(certified_at, str):
            raise ValueError(f"{path.name}: certified_at/evaluated_at must be a string")

        rows.append(
            PublicCertRow(
                record_id=record_id,
                display_name=display_name,
                sort_order=_sort_order(data, context=path.name, default=10_000),
                listed=listed,
                edition_label=_edition_label(data, artifact, context=path.name),
                tier1_status=status,
                tier2_status=_tier2_status(tier1=data, tier2=tier2_data, context=path.name),
                hub_repo_id=hub_repo_id,
                hub_commit=hub_commit,
                host_id=host_id,
                product_class=product_class,
                candidate_manifest_sha256=manifest_sha if isinstance(manifest_sha, str) else None,
                certified_at=certified_at if isinstance(certified_at, str) else None,
                mtp_acceleration_status=mtp_status,
                mtp_acceleration_reason=mtp_reason if isinstance(mtp_reason, str) else None,
                tier1_path=path,
                tier2_path=tier2_path if tier2_path.is_file() else None,
            )
        )

    rows.sort(key=lambda row: (row.sort_order, row.record_id))
    if listed_only:
        return [row for row in rows if row.listed]
    return rows


def _tier1_cell(row: PublicCertRow, *, link_prefix: str) -> str:
    return f"[{row.tier1_label}]({link_prefix}{row.tier1_stem}.md)"


def _tier2_cell(row: PublicCertRow, *, link_prefix: str) -> str:
    if row.tier2_status == "not_applicable":
        return "N/A (no MTP)"
    if row.tier2_status == "certified":
        stem = row.tier2_stem
        assert stem is not None
        return f"[Certified]({link_prefix}{stem}.md)"
    return f"[Not Certified]({link_prefix}{row.tier1_stem}.md#tier-2-status)"


def render_readme_matrix(rows: list[PublicCertRow] | None = None) -> str:
    """README top-of-page certification matrix (no edition column)."""

    catalog = rows if rows is not None else load_public_cert_rows()
    lines = [
        "| Pack family | Tier 1 (Quality) | Tier 2 (MTP -- Scoped) |",
        "| --- | --- | --- |",
    ]
    prefix = "docs/certifications/"
    for row in catalog:
        lines.append(
            f"| {row.display_name} | {_tier1_cell(row, link_prefix=prefix)} | "
            f"{_tier2_cell(row, link_prefix=prefix)} |"
        )
    return "\n".join(lines) + "\n"


def render_index_matrix(rows: list[PublicCertRow] | None = None) -> str:
    """``docs/certifications/README.md`` index table with edition pin."""

    catalog = rows if rows is not None else load_public_cert_rows()
    lines = [
        "| Checkpoint | Edition | Tier 1 (checkpoint) | Tier 2 (MTP acceleration) |",
        "| --- | --- | --- | --- |",
    ]
    for row in catalog:
        name_link = f"[{row.display_name}]({row.tier1_stem}.md)"
        lines.append(
            f"| {name_link} | {row.edition_label} | "
            f"{_tier1_cell(row, link_prefix='')} | "
            f"{_tier2_cell(row, link_prefix='')} |"
        )
    return "\n".join(lines) + "\n"


def render_release_matrix(rows: list[PublicCertRow] | None = None) -> str:
    """Standalone release certification matrix (fully generated file body)."""

    catalog = rows if rows is not None else load_public_cert_rows()
    lines = [
        "<!-- Generated by axquant.public_cert_index — do not edit by hand. -->",
        "",
        "# Public certification matrix",
        "",
        "Source of truth: machine-readable certificates under",
        "[`docs/certifications/`](../certifications/).",
        "Regenerate with `python scripts/render_certification_docs.py --write`.",
        "",
        "| Pack family | Hub repository | Tier 1 | Tier 2 | Host |",
        "| --- | --- | --- | --- | --- |",
    ]
    prefix = "../certifications/"
    for row in catalog:
        hub = f"[`{row.hub_repo_id}`](https://huggingface.co/{row.hub_repo_id})"
        lines.append(
            f"| {row.display_name} | {hub} | "
            f"{_tier1_cell(row, link_prefix=prefix)} | "
            f"{_tier2_cell(row, link_prefix=prefix)} | "
            f"`{row.host_id}` |"
        )
    lines.append("")
    return "\n".join(lines)


def render_model_card_certification_section(row: PublicCertRow) -> str:
    """Certification section suitable for embedding in a Hub model card."""

    day = (row.certified_at or "")[:10] or "unknown"
    cert_url = f"{_GITHUB_CERT_BASE}/{row.tier1_stem}.md"
    if row.tier1_status != "certified":
        return (
            f"> **Not certified** for AXQuant checkpoint Tier 1 on `{row.host_id}`.\n"
            f"> This Hub pack is development evidence only. See the evaluation record:\n"
            f"> [{row.display_name}]({cert_url}).\n"
        )

    mtp = _claim_mtp_status(row)
    mtp_text = {
        "certified": "certified on the certification host; see the certificate for its exact scope",
        "certified-scoped": (
            "certified for the certificate's authorizing profiles only; outside that scope "
            "there is no speedup claim"
        ),
        "not-certified": "**not certified**; no MTP speedup claim for this checkpoint",
    }[mtp]
    if row.mtp_acceleration_reason and mtp == "not-certified":
        mtp_text = f"{mtp_text} ({row.mtp_acceleration_reason})"

    return (
        f"> **Checkpoint Tier 1 certified** on `{row.host_id}` ({day}) for this exact\n"
        f"> revision — measured size against a matched uniform baseline, quality retention, and\n"
        f"> conversion integrity. Tier 1 is a checkpoint claim, **not** a speed claim: MTP\n"
        f"> acceleration is {mtp_text}.\n"
        f"> See the [checkpoint Tier 1 certificate]({cert_url}) for the bound evidence and "
        f"thresholds.\n"
    )


def claim_from_public_row(row: PublicCertRow) -> CheckpointCertificationClaim | None:
    """Build a model-card claim from a public certificate row.

    Returns ``None`` when the pack is not checkpoint Tier 1 certified or the
    record lacks a bound candidate manifest digest.
    """

    if row.tier1_status != "certified":
        return None
    if not row.candidate_manifest_sha256 or not row.certified_at:
        return None
    from datetime import datetime

    return CheckpointCertificationClaim(
        hub_repo_id=row.hub_repo_id,
        hub_commit=row.hub_commit,
        candidate_manifest_sha256=row.candidate_manifest_sha256,
        host_id=row.host_id,
        certified_at=datetime.fromisoformat(row.certified_at),
        certificate_url=f"{_GITHUB_CERT_BASE}/{row.tier1_stem}.md",
        mtp_acceleration_status=_claim_mtp_status(row),
        mtp_acceleration_note=row.mtp_acceleration_reason,
    )


def public_row_for_repo(
    hub_repo_id: str,
    *,
    cert_dir: Path | None = None,
    listed_only: bool = False,
) -> PublicCertRow | None:
    for row in load_public_cert_rows(cert_dir, listed_only=listed_only):
        if row.hub_repo_id == hub_repo_id:
            return row
    return None


def replace_marked_section(text: str, body: str) -> str:
    """Replace the exclusive region between BEGIN/END markers with ``body``."""

    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"document is missing markers {BEGIN_MARKER!r} / {END_MARKER!r}")
    replacement = f"{BEGIN_MARKER}\n{body.rstrip()}\n{END_MARKER}"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError("expected exactly one certification matrix marker pair")
    return updated


def expected_documents(
    *,
    root: Path | None = None,
    rows: list[PublicCertRow] | None = None,
) -> dict[Path, str]:
    """Map of document path → full file content after regeneration."""

    repo = root or _REPO_ROOT
    catalog = rows if rows is not None else load_public_cert_rows(certifications_dir(repo))
    readme_path = repo / "README.md"
    index_path = certifications_dir(repo) / "README.md"
    release_path = repo / "docs" / "releases" / "certification-matrix.md"

    readme = replace_marked_section(
        readme_path.read_text(encoding="utf-8"),
        render_readme_matrix(catalog),
    )
    index = replace_marked_section(
        index_path.read_text(encoding="utf-8"),
        render_index_matrix(catalog),
    )
    release = render_release_matrix(catalog)
    return {
        readme_path: readme,
        index_path: index,
        release_path: release,
    }


def check_documents(*, root: Path | None = None) -> list[str]:
    """Return human-readable drift messages; empty means docs match the SSOT."""

    messages: list[str] = []
    for path, expected in expected_documents(root=root).items():
        if not path.is_file():
            messages.append(f"missing generated document: {path}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            rel = path.relative_to(root or _REPO_ROOT) if path.is_absolute() else path
            messages.append(
                f"{rel}: certification matrix is out of date with docs/certifications/*.json "
                "(run: python scripts/render_certification_docs.py --write)"
            )
    return messages


def write_documents(*, root: Path | None = None) -> list[Path]:
    """Regenerate marked sections / release matrix; return paths written."""

    written: list[Path] = []
    for path, content in expected_documents(root=root).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.read_text(encoding="utf-8") if path.is_file() else None
        if previous != content:
            path.write_text(content, encoding="utf-8")
            written.append(path)
    return written
