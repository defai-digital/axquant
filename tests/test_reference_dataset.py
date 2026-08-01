from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from axquant.calibration_dataset import validate_calibration_dataset


def _dataset_path() -> Path:
    return Path(str(files("axquant.data") / "reference_calibration.jsonl"))


def _load_samples() -> list[dict]:
    text = _dataset_path().read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_reference_dataset_passes_validation() -> None:
    issues = validate_calibration_dataset(_dataset_path())
    assert issues == [], f"Validation issues: {issues}"


def test_reference_dataset_has_required_domains() -> None:
    domains = {s["domain"] for s in _load_samples()}
    required = {"coding", "json", "tool", "multilingual", "long-context"}
    assert required.issubset(domains)


def test_reference_dataset_exceeds_minimum_samples() -> None:
    assert len(_load_samples()) >= 128


def test_reference_dataset_has_unique_ids() -> None:
    ids = [s["id"] for s in _load_samples()]
    assert len(ids) == len(set(ids))


def test_reference_dataset_long_context_samples_are_long() -> None:
    long_ctx = [s for s in _load_samples() if s["domain"] == "long-context"]
    assert len(long_ctx) >= 10
    for sample in long_ctx:
        assert len(sample["text"]) >= 2000
