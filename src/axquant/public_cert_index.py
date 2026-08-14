"""Public certification index: certificate JSON is the single source of truth.

Machine-readable records under ``docs/certifications/*.json`` drive the
README matrix, the certification index table, the release certification
matrix, and the Hub model-card certification section. Hand-edited table
cells that disagree with those records are a documentation bug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from axquant.schema.certification import CheckpointCertificationClaim
from axquant.schema.public_certification import (
    CHECKPOINT_SCHEMA_VERSION,
    MTP_SCHEMA_VERSION,
    PublicCheckpointCertification,
    PublicMtpAccelerationCertification,
    load_public_checkpoint_certification,
    load_public_mtp_acceleration_certification,
)

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


def _tier2_status(
    *,
    tier1: PublicCheckpointCertification,
    tier2: PublicMtpAccelerationCertification | None,
) -> Literal["certified", "not_certified", "not_applicable"]:
    status = tier1.mtp_acceleration.status
    if status == "not-applicable":
        return "not_applicable"
    if tier2 is not None:
        if tier2.status == "certified":
            return "certified"
        if tier2.status == "not_certified":
            return "not_certified"
    if status == "not-certified":
        return "not_certified"
    if status in {"certified", "certified-scoped", "certified-see-tier2-record"}:
        # Claimed on the Tier 1 record but no certified Tier 2 file yet.
        return "not_certified"
    raise ValueError(f"unrecognized mtp_acceleration.status {status!r}")


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
    """Load checkpoint Tier 1 certificates and resolve Tier 2 companions.

    Records are validated through the frozen public certification models
    (AXQ-042) before any documentation matrix is rendered.
    """

    directory = cert_dir or _DEFAULT_CERT_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"certifications directory not found: {directory}")

    rows: list[PublicCertRow] = []
    for path in sorted(directory.glob("*-tier1.json")):
        cert = load_public_checkpoint_certification(path)
        if cert.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"{path.name}: expected schema {CHECKPOINT_SCHEMA_VERSION!r}, "
                f"got {cert.schema_version!r}"
            )

        record_id = path.name.removesuffix("-tier1.json")
        tier2_path = directory / f"{record_id}-tier2.json"
        tier2: PublicMtpAccelerationCertification | None = None
        if tier2_path.is_file():
            tier2 = load_public_mtp_acceleration_certification(tier2_path)
            if tier2.schema_version != MTP_SCHEMA_VERSION:
                raise ValueError(
                    f"{tier2_path.name}: expected schema {MTP_SCHEMA_VERSION!r}, "
                    f"got {tier2.schema_version!r}"
                )
            identity_fields = (
                "hub_repo_id",
                "hub_commit",
                "product_class",
                "candidate_manifest_sha256",
                "source_model_id",
                "source_revision",
            )
            mismatched = [
                field
                for field in identity_fields
                if getattr(tier2.artifact, field) != getattr(cert.artifact, field)
            ]
            if mismatched:
                raise ValueError(
                    f"{tier2_path.name}: Tier 2 artifact does not match Tier 1 "
                    f"{path.name} on fields {mismatched}"
                )
            if tier2.status == "certified" and cert.status != "certified":
                raise ValueError(
                    f"{tier2_path.name}: certified Tier 2 requires a certified Tier 1 record"
                )
            md_path = directory / f"{record_id}-tier2.md"
            if not md_path.is_file():
                raise ValueError(f"{tier2_path.name}: missing companion markdown {md_path.name}")

        md_path = directory / f"{record_id}-tier1.md"
        if not md_path.is_file():
            raise ValueError(f"{path.name}: missing companion markdown {md_path.name}")

        stamp = cert.event_timestamp
        certified_at = stamp.isoformat()

        rows.append(
            PublicCertRow(
                record_id=record_id,
                display_name=cert.public_index.display_name,
                sort_order=cert.public_index.sort_order,
                listed=cert.public_index.listed,
                edition_label=cert.public_index.edition_label,
                tier1_status=cert.status,
                tier2_status=_tier2_status(tier1=cert, tier2=tier2),
                hub_repo_id=cert.artifact.hub_repo_id,
                hub_commit=cert.artifact.hub_commit,
                host_id=cert.host_id,
                product_class=cert.artifact.product_class,
                candidate_manifest_sha256=cert.artifact.candidate_manifest_sha256,
                certified_at=certified_at,
                mtp_acceleration_status=cert.mtp_acceleration.status,
                mtp_acceleration_reason=cert.mtp_acceleration.reason,
                tier1_path=path,
                tier2_path=tier2_path if tier2_path.is_file() else None,
            )
        )

    # Dual Tier 1+2 certified packs lead the public matrices. Tier-1-only
    # (including no-MTP / T2 N/A) and T1-with-uncertified-MTP follow, still
    # ordered by each record's public_index.sort_order within the group.
    rows.sort(key=lambda row: (_matrix_completeness_rank(row), row.sort_order, row.record_id))
    if listed_only:
        return [row for row in rows if row.listed]
    return rows


def _matrix_completeness_rank(row: PublicCertRow) -> int:
    """Sort rank for public matrices (lower sorts first).

    Only packs that pass **both** checkpoint Tier 1 and scoped MTP Tier 2 are
    treated as the fully certified front of the catalog. Everything else keeps
    a secondary rank so dual-certified flagship packs are not buried under
    no-MTP siblings or T1-only rows.
    """

    if row.tier1_status == "certified" and row.tier2_status == "certified":
        return 0
    if row.tier1_status == "certified" and row.tier2_status == "not_applicable":
        return 1
    if row.tier1_status == "certified":
        return 2  # Tier 1 certified; MTP present but Tier 2 not certified
    return 3


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
        "Headline (listed) packs only. For every AXQ certificate record including",
        "unlisted no-MTP siblings and evaluation archives, see",
        "[full certification list](../certifications/full-list.md).",
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


def render_full_cert_list(rows: list[PublicCertRow] | None = None) -> str:
    """Full list of every AXQ certificate record (listed and unlisted).

    Written to ``docs/certifications/full-list.md``. The README headline matrix
    only shows ``public_index.listed`` packs; this document is the complete
    inventory.
    """

    catalog = rows if rows is not None else load_public_cert_rows(listed_only=False)
    lines = [
        "<!-- Generated by axquant.public_cert_index — do not edit by hand. -->",
        "",
        "# Full AXQ certification list",
        "",
        "Every public AXQuant certificate record under",
        "[`docs/certifications/`](./), including packs omitted from the README",
        "headline matrix (`public_index.listed = false`).",
        "",
        "Source of truth: `*-tier1.json` / companion `*-tier2.json` files.",
        "Regenerate with `python scripts/render_certification_docs.py --write`.",
        "",
        "Sort order: dual Tier 1+2 certified first, then Tier 1 only (T2 N/A),",
        "then Tier 1 with Tier 2 not certified, then non-certified evaluation",
        "records. Within each group, `public_index.sort_order` applies.",
        "",
        "| Pack family | Hub repository | Edition | Tier 1 | Tier 2 | Host | In headline matrix |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in catalog:
        hub = f"[`{row.hub_repo_id}`](https://huggingface.co/{row.hub_repo_id})"
        name = f"[{row.display_name}]({row.tier1_stem}.md)"
        in_matrix = "yes" if row.listed else "no"
        lines.append(
            f"| {name} | {hub} | {row.edition_label} | "
            f"{_tier1_cell(row, link_prefix='')} | "
            f"{_tier2_cell(row, link_prefix='')} | "
            f"`{row.host_id}` | {in_matrix} |"
        )
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    listed_n = sum(1 for row in catalog if row.listed)
    dual_n = sum(
        1 for row in catalog if row.tier1_status == "certified" and row.tier2_status == "certified"
    )
    t1_only_n = sum(
        1 for row in catalog if row.tier1_status == "certified" and row.tier2_status != "certified"
    )
    not_cert_n = sum(1 for row in catalog if row.tier1_status != "certified")
    lines.append(f"- Total certificate records: **{len(catalog)}**")
    lines.append(f"- In README headline matrix (`listed`): **{listed_n}**")
    lines.append(f"- Dual Tier 1 + scoped Tier 2 certified: **{dual_n}**")
    lines.append(f"- Tier 1 certified without Tier 2 certified: **{t1_only_n}**")
    lines.append(f"- Not checkpoint-certified (evaluation only): **{not_cert_n}**")
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
    full_catalog = load_public_cert_rows(certifications_dir(repo), listed_only=False)
    readme_path = repo / "README.md"
    index_path = certifications_dir(repo) / "README.md"
    release_path = repo / "docs" / "releases" / "certification-matrix.md"
    full_list_path = certifications_dir(repo) / "full-list.md"

    readme = replace_marked_section(
        readme_path.read_text(encoding="utf-8"),
        render_readme_matrix(catalog),
    )
    index = replace_marked_section(
        index_path.read_text(encoding="utf-8"),
        render_index_matrix(catalog),
    )
    release = render_release_matrix(catalog)
    full_list = render_full_cert_list(full_catalog)
    return {
        readme_path: readme,
        index_path: index,
        release_path: release,
        full_list_path: full_list,
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
