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


def _minimal_valid_samples(*, extra: list[dict] | None = None) -> list[dict]:
    """Build enough samples to clear count/token floors for unit tests."""
    domains = ["coding", "json", "tool", "multilingual", "long-context"]
    samples: list[dict] = []
    for i in range(128):
        domain = domains[i % len(domains)]
        text = ("x" * 2000) if domain == "long-context" else ("hello world " * 20)
        samples.append({"id": f"s-{i}", "domain": domain, "text": text})
    if extra:
        samples.extend(extra)
    return samples


def _write_jsonl(path: Path, samples: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )
    return path


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


def test_validate_duplicate_ids_with_mixed_int_and_str_types(tmp_path: Path) -> None:
    """Mixed int/str IDs must not TypeError when reporting duplicates."""
    samples = _minimal_valid_samples()
    # Two independent duplicate pairs of different types: sorted({1, "dup"})
    # used to raise TypeError: '<' not supported between instances of 'str' and 'int'.
    samples[0]["id"] = 1
    samples[1]["id"] = 1
    samples[2]["id"] = "dup"
    samples[3]["id"] = "dup"
    path = _write_jsonl(tmp_path / "mixed.jsonl", samples)

    issues = validate_calibration_dataset(path)

    assert any(issue.startswith("duplicate IDs:") for issue in issues)
    # Stable, comparable string form
    dup_issue = next(issue for issue in issues if issue.startswith("duplicate IDs:"))
    assert "1" in dup_issue
    assert "dup" in dup_issue


def test_validate_int_and_str_same_value_are_duplicate(tmp_path: Path) -> None:
    samples = _minimal_valid_samples()
    samples[0]["id"] = 42
    samples[1]["id"] = "42"
    path = _write_jsonl(tmp_path / "coerced.jsonl", samples)

    issues = validate_calibration_dataset(path)

    assert any("duplicate IDs:" in issue and "42" in issue for issue in issues)
