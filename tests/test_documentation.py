from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

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
    render_readme_matrix,
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
    "docs/*.md",
    "docs/certifications/*.md",
    "docs/releases/*.md",
    "docs/roadmap/**/*.md",
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


def test_public_stable_catalog_preserves_migration_and_lists_multimodal_additions() -> None:
    readme_catalog = (
        _read("README.md")
        .split(
            "### AutomatosX Hub catalog (AXQ)",
            1,
        )[1]
        .split("**Development naming:**", 1)[0]
    )
    completion = _read("docs/model-fleet-v2.md").split("## Completed migration", 1)[1]
    repository_link = re.compile(
        r"https://huggingface\.co/AutomatosX/"
        r"(AX-[A-Za-z0-9._-]+-MLX-AXQ-(?:4bit|6bit|8bit)(?:-MTP)?)"
    )
    readme_repositories = repository_link.findall(readme_catalog)
    completion_repositories = repository_link.findall(completion)
    multimodal_additions = {
        "AX-Qwen3-ASR-1.7B-MLX-AXQ-4bit",
        "AX-Qwen3-ASR-1.7B-MLX-AXQ-6bit",
        "AX-Qwen3-VL-8B-Instruct-MLX-AXQ-4bit",
        "AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit",
        "AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit",
        "AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit",
    }
    # Post-v2 fleet growth: Gemma-4 26B-A4B / 31B Tier 1 packs published after the
    # historical completed-migration table (which still covers the original 12b pair).
    gemma_tier1_additions = {
        "AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP",
        "AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP",
        "AX-gemma-4-31b-MLX-AXQ-4bit-MTP",
        "AX-gemma-4-31b-MLX-AXQ-6bit-MTP",
    }
    post_migration_additions = multimodal_additions | gemma_tier1_additions
    # Protection floors collapsed these AXQ-4bit siblings onto their 6bit packs; the
    # 4bit Hub repos were deleted so the public catalog must not list them.
    floor_collapsed_4bit = {
        "AX-Qwen3.5-9B-MLX-AXQ-4bit-MTP",
        "AX-MiniCPM5-1B-MLX-AXQ-4bit",
        "AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit",
    }
    floor_collapsed_6bit = {
        "AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP",
        "AX-MiniCPM5-1B-MLX-AXQ-6bit",
        "AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit",
    }

    gpt_oss_additions = {
        "AX-gpt-oss-20b-MLX-AXQ-4bit",
        "AX-gpt-oss-20b-MLX-AXQ-6bit",
        "AX-gpt-oss-120b-MLX-AXQ-6bit",
    }
    # 120B 4-bit failed agent-coding Tier 1; Hub pack deleted (evaluation record only).
    gpt_oss_removed_uncertified = {
        "AX-gpt-oss-120b-MLX-AXQ-4bit",
    }
    # Secondary development/cert packs published after the historical migration table.
    secondary_family_additions = {
        "AX-Ornith-1.0-35B-MLX-AXQ-4bit",
        "AX-Ornith-1.0-35B-MLX-AXQ-6bit",
        "AX-DeepSeek-OCR-2-MLX-AXQ-4bit",
        "AX-DeepSeek-OCR-2-MLX-AXQ-6bit",
        "AX-Muse-Glimmer-30B-MLX-AXQ-4bit",
        "AX-Muse-Glimmer-30B-MLX-AXQ-6bit",
        "AX-Holo3-35B-A3B-MLX-AXQ-4bit",
        "AX-Holo3-35B-A3B-MLX-AXQ-6bit",
    }
    # Qwen 3.6 no-MTP siblings + Qwen3.8-27B four-pack (Tier 1 / Tier 2 as certified).
    qwen_family_additions = {
        "AX-Qwen3.6-27B-MLX-AXQ-4bit",
        "AX-Qwen3.6-27B-MLX-AXQ-6bit",
        "AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit",
        "AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit",
        "AX-Qwen3.8-27B-MLX-AXQ-4bit",
        "AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
        "AX-Qwen3.8-27B-MLX-AXQ-6bit",
        "AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
        "AX-Qwen3.8-27B-MLX-AXQ-8bit",
    }
    post_migration_additions = (
        post_migration_additions
        | gpt_oss_additions
        | secondary_family_additions
        | qwen_family_additions
    )
    assert len(readme_repositories) == 55
    assert len(set(readme_repositories)) == 55
    # Historical completion table keeps non-link rows for deleted 4bit IDs; live Hub
    # links cover the original 28 minus those three 4bit packs (unique = 25).
    assert len(set(completion_repositories)) == 25
    assert set(completion_repositories) < set(readme_repositories)
    assert set(readme_repositories) - set(completion_repositories) == post_migration_additions
    assert floor_collapsed_4bit.isdisjoint(set(readme_repositories))
    assert gpt_oss_removed_uncertified.isdisjoint(set(readme_repositories))
    assert floor_collapsed_6bit < set(readme_repositories)
    catalog_lower = readme_catalog.lower()
    assert "no distinct AXQ-4bit pack" in catalog_lower or "no 4bit sibling" in readme_catalog
    assert "legacy-pre-v2" in readme_catalog
    assert "tagged `v2`" in readme_catalog
    assert not re.search(r"https://huggingface\.co/AutomatosX/[^)\s]+-v2(?:-MTP)?", readme_catalog)
    assert "(factory)" not in readme_catalog
    assert "regeneration required" not in readme_catalog


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


def test_readme_cli_table_covers_every_command() -> None:
    parser = _build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    commands = set(subparsers.choices)

    readme = _read("README.md")
    section = readme.split("## CLI workflow", 1)[1].split("## Measured planning and validation", 1)[
        0
    ]
    documented = set(re.findall(r"^\| `([^`]+)` \|", section, flags=re.MULTILINE))
    assert documented == commands


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


def test_public_certification_json_is_loadable_ssot() -> None:
    """Every checkpoint Tier 1 JSON must load with a companion markdown file."""

    rows = load_public_cert_rows(listed_only=False)
    assert rows, "expected at least one public certification record"
    listed = [row for row in rows if row.listed]
    assert listed, "expected listed public certification rows"
    # Gemma 4 4-bit and 6-bit packs are both Tier 1 certified (not 6-bit-only).
    gemma = [row for row in listed if row.record_id.startswith("gemma4-")]
    assert {row.record_id for row in gemma} == {
        "gemma4-12b-axq4",
        "gemma4-12b-axq6",
        "gemma4-26b-a4b-axq4",
        "gemma4-26b-a4b-axq6",
        "gemma4-31b-axq4",
        "gemma4-31b-axq6",
    }
    assert all(row.tier1_status == "certified" for row in gemma)
    assert all(row.tier2_status == "not_certified" for row in gemma)
    # Unlisted evaluation records remain loadable without entering the public matrix.
    unlisted = [row for row in rows if not row.listed]
    unlisted_ids = {row.record_id for row in unlisted}
    assert "gpt-oss-120b-axq4" in unlisted_ids
    assert "gpt-oss-20b-axq4" not in unlisted_ids  # certified + listed
    assert "holo3-35b-axq4" not in unlisted_ids  # certified + listed
    assert "holo3-35b-axq6" not in unlisted_ids  # certified + listed
    # No-MTP Qwen / Coder-Next siblings stay certified on disk but off the matrix.
    for rid in (
        "qwen38-27b-axq4",
        "qwen38-27b-axq6",
        "qwen36-27b-axq4-nomtp",
        "qwen36-27b-axq6-nomtp",
        "qwen36-35b-axq4-nomtp",
        "qwen36-35b-axq6-nomtp",
        "qwen3-coder-next-axq4",
        "qwen3-coder-next-axq6",
    ):
        assert rid in unlisted_ids


def test_public_certification_rows_are_flagship_first_and_deterministic() -> None:
    """Dual Tier 1+2 certified packs lead; remaining groups keep sort_order."""

    rows = load_public_cert_rows(listed_only=False)
    assert len({row.sort_order for row in rows}) == len(rows)
    # Completeness: both tiers certified → T1-only (T2 N/A) → T1 with T2 not certified.
    assert [row.record_id for row in rows] == [
        # Dual certified (Tier 1 + scoped Tier 2)
        "qwen38-27b-axq4-mtp",
        "qwen38-27b-axq6-mtp",
        "qwen36-27b-axq4",
        "qwen36-27b-axq6",
        "qwen36-35b-axq4",
        "qwen36-35b-axq6",
        # Tier 1 only, no MTP (T2 N/A)
        "qwen38-27b-axq-mxfp4",
        "qwen38-27b-axq4",
        "qwen38-27b-axq6",
        "qwen38-27b-axq8",
        "qwen36-27b-axq4-nomtp",
        "qwen36-27b-axq6-nomtp",
        "qwen36-35b-axq4-nomtp",
        "qwen36-35b-axq6-nomtp",
        "qwen3-coder-next-axq4",
        "qwen3-coder-next-axq6",
        "qwen3-vl-30b-axq4",
        "qwen3-vl-30b-axq6",
        "holo3-35b-axq4",
        "holo3-35b-axq6",
        "gpt-oss-20b-axq4",
        "gpt-oss-20b-axq6",
        "gpt-oss-120b-axq6",
        # Tier 1 certified; MTP present but Tier 2 not certified
        "qwen38-27b-axq-mxfp4-mtp",
        "qwen38-27b-axq8-mtp",
        "deepseek-v4-flash-axq2",
        "deepseek-v4-flash-axq3",
        "gemma4-12b-axq4",
        "gemma4-12b-axq6",
        "gemma4-26b-a4b-axq4",
        "gemma4-26b-a4b-axq6",
        "gemma4-31b-axq4",
        "gemma4-31b-axq6",
        # Not checkpoint-certified (unlisted evaluation record)
        "gpt-oss-120b-axq4",
    ]
    dual = [
        row for row in rows if row.tier1_status == "certified" and row.tier2_status == "certified"
    ]
    assert dual
    assert all(
        rows.index(dual[0]) < rows.index(row) for row in rows if row.tier2_status != "certified"
    )
    unlisted_ids = {row.record_id for row in rows if not row.listed}
    assert "holo3-35b-axq4" not in unlisted_ids
    assert "holo3-35b-axq6" not in unlisted_ids


def test_certification_docs_match_certificate_json_exactly() -> None:
    """README, cert index, and release matrix must equal the generated SSOT output."""

    messages = check_documents(root=_ROOT)
    assert not messages, "\n".join(messages)

    rows = load_public_cert_rows()
    all_rows = load_public_cert_rows(listed_only=False)
    readme_body = _extract_marked_matrix(_read("README.md"))
    index_body = _extract_marked_matrix(_read("docs/certifications/README.md"))
    assert readme_body == render_readme_matrix(rows)
    assert index_body == render_index_matrix(rows)
    assert _read("docs/releases/certification-matrix.md") == render_release_matrix(rows)
    assert _read("docs/certifications/full-list.md") == render_full_cert_list(all_rows)
    assert "full-list.md" in _read("README.md")
    # Full list includes unlisted no-MTP / Coder-Next records omitted from headline.
    full = _read("docs/certifications/full-list.md")
    assert "Qwen3.8-27B MLX AXQ 4-bit]" in full or "Qwen3.8-27B MLX AXQ 4-bit |" in full
    assert "qwen38-27b-axq4-tier1.md" in full
    assert "qwen3-coder-next-axq4-tier1.md" in full
    assert "In headline matrix" in full

    # Display names and Tier 1 verdicts agree across every generated surface.
    def _data_rows(matrix: str) -> list[str]:
        names: list[str] = []
        for line in matrix.splitlines():
            if not line.startswith("| "):
                continue
            if (
                line.startswith("| ---")
                or line.startswith("| Pack")
                or line.startswith("| Checkpoint")
            ):
                continue
            cell = line.split("|", 2)[1].strip()
            names.append(re.sub(r"^\[([^\]]+)\]\([^)]+\)$", r"\1", cell))
        return names

    assert _data_rows(readme_body) == [row.display_name for row in rows]
    listed_ids = [row.record_id for row in rows]
    assert listed_ids.index("qwen38-27b-axq-mxfp4") < listed_ids.index("qwen38-27b-axq4-mtp")
    assert listed_ids.index("qwen38-27b-axq-mxfp4-mtp") < listed_ids.index("qwen38-27b-axq4-mtp")
    assert listed_ids.index("qwen38-27b-axq-mxfp4") < listed_ids.index("qwen38-27b-axq-mxfp4-mtp")
    assert listed_ids.index("qwen38-27b-axq8") < listed_ids.index("qwen38-27b-axq8-mtp")
    assert _data_rows(index_body) == [row.display_name for row in rows]
    for row in rows:
        assert f"| {row.display_name} |" in readme_body
        assert f"[{row.tier1_label}](docs/certifications/{row.tier1_stem}.md)" in readme_body


def test_model_card_certification_section_matches_public_records() -> None:
    """Hub card certification prose is derived from the same certificate rows."""

    certified = public_row_for_repo(
        "AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP",
        listed_only=False,
    )
    assert certified is not None
    section = render_model_card_certification_section(certified)
    assert "Checkpoint Tier 1 certified" in section
    assert certified.host_id in section
    assert "not certified" in section.lower()  # Tier 2 / MTP still open
    claim = claim_from_public_row(certified)
    assert claim is not None
    assert claim.hub_repo_id == certified.hub_repo_id
    assert claim.hub_commit == certified.hub_commit
    assert claim.candidate_manifest_sha256 == certified.candidate_manifest_sha256
    assert claim.mtp_acceleration_status == "not-certified"

    failed = public_row_for_repo(
        "AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit",
        listed_only=False,
    )
    assert failed is not None
    assert failed.listed is False
    failed_section = render_model_card_certification_section(failed)
    assert "Not certified" in failed_section
    assert claim_from_public_row(failed) is None


def test_schema_catalog_matches_registry_generator() -> None:
    """Human schema catalog must stay byte-identical to the freeze generator."""

    assert not check_schema_contracts(root=_ROOT)
    assert _read("docs/schema-catalog.md") == render_schema_catalog()


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
