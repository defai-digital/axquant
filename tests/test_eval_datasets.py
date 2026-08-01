from __future__ import annotations

import json
from pathlib import Path

import pytest

from axquant.schema import QualityTask

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
DATASETS = sorted(EVAL_DIR.glob("*.jsonl"))


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda p: p.stem)
def test_eval_dataset_validates_against_schema(dataset: Path) -> None:
    lines = dataset.read_text().splitlines()
    assert len(lines) >= 10
    for line in lines:
        task = QualityTask.model_validate(json.loads(line))
        assert task.task_id
        assert task.category
        assert len(task.checks) >= 1


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda p: p.stem)
def test_eval_dataset_has_unique_task_ids(dataset: Path) -> None:
    ids = [
        json.loads(line)["task_id"]
        for line in dataset.read_text().splitlines()
        if line.strip()
    ]
    assert len(ids) == len(set(ids))


def test_eval_datasets_cover_expected_categories() -> None:
    categories = set()
    for dataset in DATASETS:
        for line in dataset.read_text().splitlines():
            if line.strip():
                categories.add(json.loads(line)["category"])
    assert {"coding", "reasoning", "json-tool", "instruction"}.issubset(categories)
