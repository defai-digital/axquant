from __future__ import annotations

import argparse
import re
from pathlib import Path

from axquant.cli._parser import _build_parser

_ROOT = Path(__file__).resolve().parents[1]
_PUBLIC_DOCS = ("README.md", "THIRD_PARTY_NOTICES.md")


def _read(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def test_public_docs_do_not_reference_local_only_material() -> None:
    text = "\n".join(_read(relative) for relative in _PUBLIC_DOCS)
    forbidden = (".internal/", "/Users/", "/Volumes/", "192.168.", "devop@")
    assert not [marker for marker in forbidden if marker in text]


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
