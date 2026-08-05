from __future__ import annotations

from pathlib import Path

import pytest

from axquant.cli import main
from axquant.dataset_overlap import build_campaign_overlap_report
from axquant.errors import ArtifactError, ValidationGateError
from axquant.schema import CampaignOverlapReport
from axquant.serde import load_model


def _jsonl(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(f'{{"id":"{record_id}","text":"{text}"}}\n' for record_id, text in records),
        encoding="utf-8",
    )


def test_campaign_overlap_reports_normalized_exact_match_without_content(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    compared = tmp_path / "compared.jsonl"
    _jsonl(dataset, [("private-a", "Alpha, beta gamma delta epsilon zeta.")])
    _jsonl(compared, [("private-b", "ALPHA beta gamma delta epsilon zeta")])

    report = build_campaign_overlap_report(
        dataset_path=dataset,
        compared_paths=[compared],
    )

    assert not report.passed
    assert report.exact_match_count == 1
    assert report.near_duplicate_count == 0
    assert report.comparison_pair_count == 1
    payload = report.model_dump_json()
    assert "private-a" not in payload
    assert "private-b" not in payload
    assert "Alpha" not in payload


def test_campaign_overlap_passes_disjoint_sets_and_cli_writes_report(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    compared_a = tmp_path / "compared-a.jsonl"
    compared_b = tmp_path / "compared-b.jsonl"
    output = tmp_path / "overlap.json"
    _jsonl(dataset, [("a", "compiler register allocation graph coloring")])
    _jsonl(compared_a, [("b", "marine biology coral reef ecology")])
    _jsonl(compared_b, [("c", "medieval trade routes and ceramic archaeology")])

    exit_code = main(
        [
            "campaign-overlap",
            "--dataset",
            str(dataset),
            "--compare",
            str(compared_a),
            "--compare",
            str(compared_b),
            "--output",
            str(output),
        ]
    )

    report = load_model(output, CampaignOverlapReport)
    assert exit_code == 0
    assert report.passed
    assert report.dataset_record_count == 1
    assert report.comparison_pair_count == 2
    assert len(report.compared_record_count_by_sha256) == 2


def test_campaign_overlap_limits_quadratic_work(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    compared = tmp_path / "compared.jsonl"
    _jsonl(dataset, [("a", "one two three four five"), ("b", "six seven eight nine ten")])
    _jsonl(compared, [("c", "alpha beta gamma delta epsilon")])

    with pytest.raises(ValidationGateError, match="above the configured maximum"):
        build_campaign_overlap_report(
            dataset_path=dataset,
            compared_paths=[compared],
            max_comparison_pairs=1,
        )


def test_campaign_overlap_rejects_symlink_input(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    compared = tmp_path / "compared.jsonl"
    linked = tmp_path / "linked.jsonl"
    _jsonl(dataset, [("a", "one two three four five")])
    _jsonl(compared, [("b", "alpha beta gamma delta epsilon")])
    linked.symlink_to(dataset)

    with pytest.raises(ArtifactError, match="must not be a symlink"):
        build_campaign_overlap_report(
            dataset_path=linked,
            compared_paths=[compared],
        )
