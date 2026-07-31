from __future__ import annotations

import json
from pathlib import Path

from axquant.cli import main
from axquant.schema import BenchmarkSuiteManifest
from axquant.serde import file_sha256, load_model
from axquant.suites import build_benchmark_suites


def _task_ids(path: Path) -> set[str]:
    return {
        value["task_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if (value := json.loads(line))
    }


def _check_kinds(path: Path) -> set[str]:
    kinds: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        task = json.loads(line)
        for check in task.get("checks", []):
            kinds.add(check["kind"])
    return kinds


def test_benchmark_suite_has_disjoint_profile_runtime_prompts(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "suite"
    manifest = build_benchmark_suites(directory)
    agent_path = directory / manifest.files["agent_coding_ax_engine_prompts"]
    general_path = directory / manifest.files["general_ax_engine_prompts"]

    assert agent_path.is_file()
    assert general_path.is_file()
    assert file_sha256(agent_path) == manifest.sha256["agent_coding_ax_engine_prompts"]
    assert file_sha256(general_path) == manifest.sha256["general_ax_engine_prompts"]
    assert file_sha256(agent_path) != file_sha256(general_path)
    assert _task_ids(agent_path).isdisjoint(_task_ids(general_path))
    assert manifest.samples["agent_coding_ax_engine_prompts"] == 18
    # 10 prose + 3 JSON + 3 syntax tasks; prompts include the full general quality set.
    assert manifest.samples["general_ax_engine_prompts"] == 16
    assert manifest.samples["general_quality"] == 16
    general_kinds = _check_kinds(directory / manifest.files["general_quality"])
    agent_kinds = _check_kinds(directory / manifest.files["agent_coding_quality"])
    # Dual-profile release validation requires both governed structured rates.
    for kinds in (general_kinds, agent_kinds):
        assert "json-valid" in kinds
        assert "python-syntax" in kinds


def test_prepare_suite_cli_emits_dual_profile_structured_checks(
    tmp_path: Path,
) -> None:
    """Drive the real prepare-suite CLI entry for dual-profile metric completeness."""
    output = tmp_path / "cli-suite"
    assert main(["prepare-suite", "--output-dir", str(output)]) == 0
    manifest = load_model(output / "suite-manifest.json", BenchmarkSuiteManifest)
    general_kinds = _check_kinds(output / manifest.files["general_quality"])
    agent_kinds = _check_kinds(output / manifest.files["agent_coding_quality"])
    assert "json-valid" in general_kinds and "python-syntax" in general_kinds
    assert "json-valid" in agent_kinds and "python-syntax" in agent_kinds


def test_prepare_suite_cli_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert main(["prepare-suite", "--output-dir", str(first)]) == 0
    assert main(["prepare-suite", "--output-dir", str(second)]) == 0

    first_manifest = load_model(first / "suite-manifest.json", BenchmarkSuiteManifest)
    second_manifest = load_model(second / "suite-manifest.json", BenchmarkSuiteManifest)
    assert first_manifest.model_dump(exclude={"created_at"}) == second_manifest.model_dump(
        exclude={"created_at"}
    )
    for relative_path in first_manifest.files.values():
        assert (first / relative_path).read_bytes() == (second / relative_path).read_bytes()
