from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from axquant.cli._parser import _build_parser

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

    assert len(readme_repositories) == 33
    assert len(set(readme_repositories)) == 33
    # Historical completion table keeps non-link rows for deleted 4bit IDs; live Hub
    # links cover the original 28 minus those three 4bit packs (unique = 25).
    assert len(set(completion_repositories)) == 25
    assert set(completion_repositories) < set(readme_repositories)
    assert set(readme_repositories) - set(completion_repositories) == post_migration_additions
    assert floor_collapsed_4bit.isdisjoint(set(readme_repositories))
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
