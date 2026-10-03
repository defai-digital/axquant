from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from _cert_fixtures import write_cert_pair

from axquant.cli._parser import _build_parser
from axquant.public_cert_index import (
    BEGIN_MARKER,
    END_MARKER,
    check_documents,
    claim_from_public_row,
    load_public_cert_rows,
    public_row_for_repo,
    render_full_cert_list,
    render_index_matrix,
    render_model_card_certification_section,
    render_release_matrix,
)
from axquant.schema_contracts import check_schema_contracts, render_schema_catalog

_ROOT = Path(__file__).resolve().parents[1]
_PUBLIC_DOCS = ("README.md", "CONTRIBUTING.md", "THIRD_PARTY_NOTICES.md")
# Published markdown outside AGENTS.md (agent-local conventions may mention
# .internal/tmp for throwaway work, but must never ship that tree).
_PUBLIC_MARKDOWN_GLOBS = (
    "README.md",
    "CONTRIBUTING.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/**/*.md",
    "examples/**/*.md",
    "examples/**/*.yaml",
    "examples/**/*.yml",
)


def _read(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def _public_text_paths() -> list[Path]:
    paths: list[Path] = []
    for pattern in _PUBLIC_MARKDOWN_GLOBS:
        paths.extend(sorted(_ROOT.glob(pattern)))
    return [path for path in paths if path.is_file()]


def test_public_docs_do_not_reference_local_only_material() -> None:
    forbidden = (".internal/", "/Users/", "/Volumes/", "192.168.", "devop@")
    offenders: list[str] = []
    for path in _public_text_paths():
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                offenders.append(f"{path.relative_to(_ROOT)}: {marker}")
    assert not offenders


def test_readme_points_to_hub_org_instead_of_mirroring_pack_catalog() -> None:
    """The README directs readers to the AutomatosX org page for the live catalog.

    The 2026-09-19 Hub catalog cleanup removed every AXQ 4/6/8-bit and experimental
    2-bit repo (public catalog is MXFP4 / embedding / OCR packs only). A README-mirrored
    pack table goes stale and accumulates dead repo links, so the README must reference
    the org page and must not link individual pack repos.
    """
    readme = _read("README.md")
    assert "https://huggingface.co/AutomatosX" in readme
    pack_links = re.findall(r"https://huggingface\.co/AutomatosX/AX-", readme)
    assert not pack_links, f"README links deleted pack repos: {sorted(set(pack_links))}"
    assert BEGIN_MARKER not in readme
    assert END_MARKER not in readme
    assert "support-matrix" in readme
    assert "docs/certifications/" in readme


def test_internal_tree_is_not_tracked() -> None:
    """``.internal/`` is local-only; force-adds must not re-enter the public tree."""
    git_dir = _ROOT / ".git"
    if not git_dir.exists():
        return
    listed = subprocess.check_output(
        ["git", "-C", str(_ROOT), "ls-files", "--", ".internal"],
        text=True,
    )
    tracked = [line for line in listed.splitlines() if line.strip()]
    assert not tracked, f"tracked .internal paths (remove from index): {tracked}"


def test_readme_product_path_is_install_then_convert() -> None:
    """The public front door is PyPI + quantize, not a git clone."""
    readme = _read("README.md")
    install = readme.index("## Install\n")
    convert = readme.index("## Convert\n")
    pip = readme.index("python -m pip install 'axquant[mlx]==1.8.1'")
    quantize = readme.index("axquant quantize /path/to/model-bf16")
    clone = readme.find("git clone https://github.com/defai-digital/axquant.git")

    assert install < convert
    assert install < pip
    assert convert < quantize
    assert clone == -1 or quantize < clone
    assert "You do not need to clone this repository." in readme


def test_cli_reference_table_covers_every_command() -> None:
    parser = _build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    commands = set(subparsers.choices)

    reference = _read("docs/cli-reference.md")
    documented = set(re.findall(r"^\| `([^`]+)` \|", reference, flags=re.MULTILINE))
    assert documented == commands
    # The slim README keeps a short entry-point section that links here.
    readme = _read("README.md")
    assert "## CLI workflow" in readme
    assert "docs/cli-reference.md" in readme


def test_live_docs_do_not_name_factory_machines() -> None:
    """Live docs describe the factory host by spec, never by machine identity.

    Machine hostnames and internal volume names are a privacy leak in live
    prose. Historical evidence (reports, eval dumps, evidence bundles, tagged
    release notes, the frozen certification spec) keeps its recorded names;
    everything else must use the hardware spec.
    """
    forbidden = (
        "df-macstudio-m2",
        "df-macbookpro-m5",
        "df-macbookpro-m3",
        "df-macmini",
        "tn-macstudio",
        "localadacStudio",
        "Ext16TR0",
        "Ext12T",
        "Ext4T",
    )
    live_paths = [
        "README.md",
        "CONTRIBUTING.md",
        "THIRD_PARTY_NOTICES.md",
        "docs/README.md",
        "docs/cli-reference.md",
        "docs/certifications/README.md",
        "docs/certifications/adr033-mapping.md",
        "docs/certifications/full-list.md",
        "docs/releases/certification-matrix.md",
        "docs/releases/1.9.0.md",
        "docs/contracts/axq-pack-interchange-v1.md",
        "docs/contracts/expert-ssd-stream-contract.md",
        *sorted(str(path.relative_to(_ROOT)) for path in (_ROOT / "docs" / "guides").glob("*.md")),
        *sorted(
            str(path.relative_to(_ROOT)) for path in (_ROOT / "docs" / "runbooks").glob("*.md")
        ),
        *sorted(
            str(path.relative_to(_ROOT)) for path in (_ROOT / "docs" / "hub-cards").glob("*.md")
        ),
        *sorted(
            str(path.relative_to(_ROOT)) for path in (_ROOT / "docs" / "migrations").glob("*.md")
        ),
    ]
    offenders: list[str] = []
    for relative in live_paths:
        text = _read(relative)
        for marker in forbidden:
            if marker in text:
                offenders.append(f"{relative}: {marker}")
    assert not offenders


def test_public_markdown_local_links_resolve() -> None:
    link_pattern = re.compile(r"\[[^]]*]\(([^)]+)\)")
    missing: list[str] = []
    for relative in _PUBLIC_DOCS:
        source = _ROOT / relative
        for target in link_pattern.findall(source.read_text(encoding="utf-8")):
            target = target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (source.parent / target).is_file():
                missing.append(f"{relative}: {target}")
    assert not missing


def _extract_marked_matrix(text: str) -> str:
    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r"\n(.*?)\n" + re.escape(END_MARKER),
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    assert match is not None, "missing certification matrix markers"
    return match.group(1).strip() + "\n"


def test_public_certification_json_is_loadable_ssot(tmp_path: Path) -> None:
    """The loader accepts the empty catalog and synthetic pairs alike."""

    assert load_public_cert_rows(listed_only=False) == []
    assert load_public_cert_rows(listed_only=True) == []
    write_cert_pair(tmp_path, "demo-dual")
    write_cert_pair(
        tmp_path,
        "demo-eval",
        tier1_status="not_certified",
        tier1_mtp_status="not-certified",
        with_tier2=False,
        listed=False,
    )
    rows = load_public_cert_rows(tmp_path, listed_only=False)
    assert {row.record_id for row in rows} == {"demo-dual", "demo-eval"}
    assert [row.record_id for row in load_public_cert_rows(tmp_path)] == ["demo-dual"]


def test_public_certification_rows_are_flagship_first_and_deterministic(
    tmp_path: Path,
) -> None:
    """Dual Tier 1+2 certified packs lead; remaining groups keep sort_order."""

    write_cert_pair(
        tmp_path,
        "demo-eval",
        tier1_status="not_certified",
        tier1_mtp_status="not-certified",
        with_tier2=False,
        listed=False,
    )
    write_cert_pair(
        tmp_path,
        "demo-nomtp",
        tier1_mtp_status="not-applicable",
        with_tier2=False,
    )
    write_cert_pair(tmp_path, "demo-dual")
    rows = load_public_cert_rows(tmp_path, listed_only=False)
    assert [row.record_id for row in rows] == ["demo-dual", "demo-nomtp", "demo-eval"]
    assert [row.record_id for row in load_public_cert_rows(tmp_path)] == [
        "demo-dual",
        "demo-nomtp",
    ]


def test_certification_docs_match_certificate_json_exactly() -> None:
    """Cert index and release matrix must equal the generated SSOT output."""

    messages = check_documents(root=_ROOT)
    assert not messages, "\n".join(messages)

    rows = load_public_cert_rows()
    all_rows = load_public_cert_rows(listed_only=False)
    assert rows == []
    assert all_rows == []
    index_body = _extract_marked_matrix(_read("docs/certifications/README.md"))
    assert index_body == render_index_matrix(rows)
    assert _read("docs/releases/certification-matrix.md") == render_release_matrix(rows)
    assert _read("docs/certifications/full-list.md") == render_full_cert_list(all_rows)
    assert "full-list.md" in _read("README.md")
    full = _read("docs/certifications/full-list.md")
    assert "Total certificate records: **0**" in full
    assert "In headline matrices" in full
    assert "Tier 1 (quality)" in full
    assert "Tier 2 (MTP -- Scoped)" in full
    assert "checkpoint **quality**" in full


def test_model_card_certification_section_matches_public_records(tmp_path: Path) -> None:
    """Hub card certification prose is derived from the same certificate rows."""

    write_cert_pair(tmp_path, "demo-dual")
    write_cert_pair(
        tmp_path,
        "demo-eval",
        tier1_status="not_certified",
        tier1_mtp_status="not-certified",
        with_tier2=False,
        listed=False,
    )
    assert public_row_for_repo("AutomatosX/AX-Demo-MLX-AXQ-6bit", listed_only=False) is None
    certified = public_row_for_repo(
        "AutomatosX/AX-Demo-MLX-AXQ-6bit",
        cert_dir=tmp_path,
        listed_only=False,
    )
    assert certified is not None
    section = render_model_card_certification_section(certified)
    assert "Checkpoint Tier 1 certified" in section
    assert certified.host_id in section
    claim = claim_from_public_row(certified)
    assert claim is not None
    assert claim.hub_repo_id == certified.hub_repo_id
    assert claim.hub_commit == certified.hub_commit
    assert claim.candidate_manifest_sha256 == certified.candidate_manifest_sha256
    assert claim.mtp_acceleration_status == "certified-scoped"

    write_cert_pair(
        tmp_path,
        "demo-open",
        hub_repo_id="AutomatosX/AX-Demo-Open-MLX-AXQ-6bit",
        tier2_status="not_certified",
    )
    open_row = public_row_for_repo(
        "AutomatosX/AX-Demo-Open-MLX-AXQ-6bit",
        cert_dir=tmp_path,
        listed_only=False,
    )
    assert open_row is not None
    assert "not certified" in render_model_card_certification_section(open_row).lower()

    failed = load_public_cert_rows(tmp_path, listed_only=False)[-1]
    assert failed.listed is False
    failed_section = render_model_card_certification_section(failed)
    assert "Not certified" in failed_section
    assert claim_from_public_row(failed) is None


def test_tier2_cells_disclose_engine_binding_and_scope() -> None:
    """A certified Tier 2 cell names its AX Engine binding and the matrix states its scope.

    ADR-033 splits AX Engine's "MTP Tier 2" into MTP-S / MTP-P / MTP-D. An AXQuant
    Tier 2 certificate is scoped acceleration evidence bound to one engine build; the
    rendered matrix must say so, and must not read as an MTP-S or MTP-D claim.
    """

    rows = load_public_cert_rows()
    assert rows == []
    release = render_release_matrix(rows)

    certified = [row for row in rows if row.tier2_status == "certified"]
    assert certified == []
    assert "No certified Tier 2 row is present" in release
    assert "Tier 2 (MTP -- Scoped)" in release
    assert "MTP-S" in release
    assert "MTP-D" in release
    assert "adr033-mapping.md" in release

    # The link target of the disclosure must exist.
    assert (_ROOT / "docs" / "certifications" / "adr033-mapping.md").is_file()


def test_schema_catalog_matches_registry_generator() -> None:
    """Human schema catalog must stay byte-identical to the freeze generator."""

    assert not check_schema_contracts(root=_ROOT)
    assert _read("docs/guides/schema-catalog.md") == render_schema_catalog()


def test_every_listed_certificate_has_public_index_metadata() -> None:
    cert_dir = _ROOT / "docs" / "certifications"
    for path in sorted(cert_dir.glob("*-tier1.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        block = data.get("public_index")
        assert isinstance(block, dict), f"{path.name}: missing public_index"
        assert isinstance(block.get("display_name"), str) and block["display_name"].strip()
        assert type(block.get("sort_order")) is int
        assert type(block.get("listed")) is bool
        if block["listed"]:
            assert isinstance(block.get("edition_label"), str) and block["edition_label"].strip()
