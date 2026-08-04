from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from axquant.errors import BenchmarkError
from axquant.quality import MlxQualityBackend, compare_quality, evaluate_quality
from axquant.schema import ModelIdentity


class _FakeQualityBackend:
    def load_model(self, model: str, revision: str | None) -> None:
        assert model == "test-model"
        assert revision == "pinned"

    def perplexity_loss(self, text: str, max_length: int) -> tuple[float, int]:
        tokens = min(max_length, max(2, len(text.split())))
        return tokens * 0.1, tokens

    def generate(self, prompt: str, max_tokens: int, random_seed: int) -> str:
        del max_tokens, random_seed
        if "code" in prompt:
            return "```python\ndef add(a, b):\n    return a + b\n```"
        if "tool" in prompt:
            return '{"name":"search","arguments":{"query":"mlx"}}'
        if "Japanese" in prompt:
            return "これは日本語です"
        return "needle"

    def generation_metadata(self) -> tuple[str, str | None]:
        return "raw", None


def _quality_dataset(path: Path) -> None:
    tasks = [
        {
            "task_id": "coding-1",
            "category": "coding",
            "prompt": "write code",
            "reference": "def add(a, b): return a + b",
            "checks": [{"kind": "python-syntax"}, {"kind": "contains", "value": "return"}],
        },
        {
            "task_id": "tool-1",
            "category": "tool",
            "prompt": "make a tool call",
            "checks": [
                {"kind": "json-valid"},
                {"kind": "json-keys", "value": ["name", "arguments"]},
            ],
        },
        {
            "task_id": "json-1",
            "category": "json",
            "prompt": "make a tool JSON response",
            "checks": [{"kind": "json-valid"}],
        },
        {
            "task_id": "multilingual-1",
            "category": "multilingual",
            "prompt": "answer in Japanese",
            "checks": [{"kind": "contains", "value": "日本語"}],
        },
        {
            "task_id": "long-1",
            "category": "long_context",
            "prompt": "retrieve the marker",
            "reference": "needle",
            "checks": [{"kind": "exact"}],
        },
    ]
    path.write_text("\n".join(json.dumps(task) for task in tasks), encoding="utf-8")


def test_quality_evaluation_scores_tasks_and_perplexity(tmp_path: Path) -> None:
    dataset = tmp_path / "quality.jsonl"
    _quality_dataset(dataset)

    result = evaluate_quality(
        model=ModelIdentity(model_id="test-model", revision="pinned"),
        dataset_path=dataset,
        max_sequence_length=64,
        max_generation_tokens=32,
        random_seed=9,
        backend=_FakeQualityBackend(),
    )

    assert result.samples == 5
    assert result.evaluated_tokens > 0
    assert math.isclose(result.metrics.perplexity or 0.0, math.exp(0.1))
    assert set(result.metrics.task_scores) == {
        "coding",
        "tool",
        "json",
        "multilingual",
        "long_context",
    }
    assert all(score == 1.0 for score in result.metrics.task_scores.values())
    assert result.metrics.json_valid_rate == 1.0
    assert result.metrics.syntax_valid_rate == 1.0


def test_quality_dataset_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "quality.jsonl"
    task = {
        "task_id": "duplicate",
        "category": "coding",
        "prompt": "write code",
        "checks": [{"kind": "python-syntax"}],
    }
    dataset.write_text(f"{json.dumps(task)}\n{json.dumps(task)}", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="unique"):
        evaluate_quality(
            model=ModelIdentity(model_id="test-model", revision="pinned"),
            dataset_path=dataset,
            backend=_FakeQualityBackend(),
        )


def test_quality_comparison_preserves_per_task_visibility(tmp_path: Path) -> None:
    dataset = tmp_path / "quality.jsonl"
    _quality_dataset(dataset)
    reference = evaluate_quality(
        model=ModelIdentity(model_id="test-model", revision="pinned"),
        dataset_path=dataset,
        random_seed=9,
        backend=_FakeQualityBackend(),
    )
    candidate = reference.model_copy(
        update={
            "model": ModelIdentity(model_id="candidate", revision="pinned-candidate"),
            "metrics": reference.metrics.model_copy(
                update={"task_scores": {**reference.metrics.task_scores, "coding": 0.5}}
            ),
            "task_results": [
                task.model_copy(update={"score": 0.5}) if task.task_id == "coding-1" else task
                for task in reference.task_results
            ],
        }
    )
    comparison = compare_quality(reference, candidate)
    assert comparison.aggregate.retention == 0.9
    assert comparison.categories["coding"].retention == 0.5
    assert comparison.tasks[0].task_id == "coding-1"
    assert comparison.tasks[0].delta == -0.5


def test_generation_failure_counts_as_failed_structured_output_check(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "quality.jsonl"
    tasks = [
        {
            "task_id": "json-ok",
            "category": "json",
            "prompt": "return JSON",
            "checks": [{"kind": "json-valid"}],
        },
        {
            "task_id": "json-error",
            "category": "json",
            "prompt": "generation fails",
            "checks": [{"kind": "json-valid"}],
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(task) for task in tasks),
        encoding="utf-8",
    )

    class FailingBackend(_FakeQualityBackend):
        def generate(self, prompt: str, max_tokens: int, random_seed: int) -> str:
            if "fails" in prompt:
                raise RuntimeError("backend generation failed")
            return "{}"

    result = evaluate_quality(
        model=ModelIdentity(model_id="test-model", revision="pinned"),
        dataset_path=dataset,
        backend=FailingBackend(),
    )

    failed = next(task for task in result.task_results if task.task_id == "json-error")
    assert failed.score == 0.0
    assert failed.check_scores == {"json-valid:0": 0.0}
    assert failed.error == "backend generation failed"
    assert result.metrics.json_valid_rate == 0.5


def test_scoring_failure_zeros_every_declared_check(tmp_path: Path) -> None:
    dataset = tmp_path / "quality.jsonl"
    task = {
        "task_id": "invalid-check",
        "category": "json",
        "prompt": "return JSON",
        "checks": [
            {"kind": "json-valid"},
            {"kind": "regex", "value": "["},
        ],
    }
    dataset.write_text(json.dumps(task), encoding="utf-8")

    result = evaluate_quality(
        model=ModelIdentity(model_id="test-model", revision="pinned"),
        dataset_path=dataset,
        backend=_FakeQualityBackend(),
    )

    failed = result.task_results[0]
    assert failed.score == 0.0
    assert failed.check_scores == {"json-valid:0": 0.0, "regex:1": 0.0}
    assert failed.error
    assert result.metrics.json_valid_rate == 0.0


def test_mlx_backend_resets_chat_template_state_between_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def parameters(self) -> list[object]:
            return []

    first_tokenizer = SimpleNamespace(chat_template="{{ messages }}")
    second_tokenizer = SimpleNamespace(chat_template=None)
    tokenizers = iter((first_tokenizer, second_tokenizer))
    fake_mlx_lm = SimpleNamespace(load=lambda *_args, **_kwargs: (FakeModel(), next(tokenizers)))
    fake_mx = SimpleNamespace(eval=lambda *_args: None)

    import importlib

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_mx if name == "mlx.core" else fake_mlx_lm,
    )
    backend = MlxQualityBackend()

    backend.load_model("first", None)
    assert backend.generation_metadata()[0] == "chat-template"
    assert backend.generation_metadata()[1] is not None

    backend.load_model("second", None)
    assert backend.generation_metadata() == ("raw", None)
