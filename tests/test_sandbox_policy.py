"""Sandbox profile rendering contract (macOS Seatbelt coding sandbox)."""

from __future__ import annotations

from pathlib import Path

from axquant.sandbox_policy import render_sandbox_profile


def _process_exec_rules(profile: str) -> list[str]:
    return [rule for rule in profile.split("(allow ") if rule.startswith("process-exec")]


def test_process_exec_rule_per_entrypoint_literal(tmp_path: Path) -> None:
    """Regression: SBPL ANDs filters in one rule, so each entrypoint needs its own.

    An entrypoint whose lexical path differs from its resolved target (a
    version-manager/Homebrew shim) produces two distinct paths. Joining both
    literals into a single ``(allow process-exec ...)`` rule requires the exec
    target to equal both at once, which never matches, and the deny-default policy
    then blocks the entrypoint itself.
    """
    real = tmp_path / "real" / "node"
    real.parent.mkdir(parents=True)
    real.write_text("#!/bin/sh\n")
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "node"
    shim.symlink_to(real)

    profile = render_sandbox_profile(
        input_dir=tmp_path,
        output_dir=tmp_path,
        toolchain_paths=[tmp_path],
        entrypoint=shim,
        allow_subprocesses=False,
    )

    rules = _process_exec_rules(profile)
    assert len(rules) == 2, rules
    assert all(rule.count("(literal ") == 1 for rule in rules), rules
    literals = {rule.split("(literal ", 1)[1].rstrip(") ") for rule in rules}
    assert len(literals) == 2, literals


def test_single_entrypoint_emits_one_rule(tmp_path: Path) -> None:
    entrypoint = tmp_path / "runner"
    entrypoint.write_text("#!/bin/sh\n")
    profile = render_sandbox_profile(
        input_dir=tmp_path,
        output_dir=tmp_path,
        toolchain_paths=[tmp_path],
        entrypoint=entrypoint,
        allow_subprocesses=False,
    )
    assert len(_process_exec_rules(profile)) == 1
