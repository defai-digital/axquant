from __future__ import annotations

import json
from pathlib import Path

from axquant.quality import _score_check
from axquant.schema import QualityCheck, QualityTask

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"


def _all_tasks() -> list[QualityTask]:
    tasks: list[QualityTask] = []
    for dataset in sorted(EVAL_DIR.glob("*.jsonl")):
        for line in dataset.read_text().splitlines():
            if line.strip():
                tasks.append(QualityTask.model_validate(json.loads(line)))
    return tasks


def test_score_check_exact_match() -> None:
    task = QualityTask(
        task_id="t1",
        category="test",
        prompt="What is 2+2?",
        reference="4",
        checks=[QualityCheck(kind="exact", value="4")],
    )
    assert _score_check(task.checks[0], task, "4") == 1.0
    assert _score_check(task.checks[0], task, "5") == 0.0


def test_score_check_contains() -> None:
    task = QualityTask(
        task_id="t2",
        category="test",
        prompt="Write hello world",
        checks=[QualityCheck(kind="contains", value="hello")],
    )
    assert _score_check(task.checks[0], task, "print('hello world')") == 1.0
    assert _score_check(task.checks[0], task, "print('goodbye')") == 0.0


def test_score_check_json_valid() -> None:
    task = QualityTask(
        task_id="t3",
        category="test",
        prompt="Return JSON",
        checks=[QualityCheck(kind="json-valid")],
    )
    assert _score_check(task.checks[0], task, '{"key": "value"}') == 1.0
    assert (
        _score_check(
            task.checks[0],
            task,
            'Result: {"key": "value"}\nTrailing prose [not JSON]',
        )
        == 1.0
    )
    assert _score_check(task.checks[0], task, "not json") == 0.0


def test_score_check_json_keys() -> None:
    task = QualityTask(
        task_id="t4",
        category="test",
        prompt="Return JSON with name and age",
        checks=[QualityCheck(kind="json-keys", value=["name", "age"])],
    )
    assert _score_check(task.checks[0], task, '{"name": "Alice", "age": 30}') == 1.0
    assert _score_check(task.checks[0], task, '{"name": "Alice"}') == 0.0


def test_score_check_python_syntax() -> None:
    task = QualityTask(
        task_id="t5",
        category="test",
        prompt="Write a function",
        checks=[QualityCheck(kind="python-syntax")],
    )
    assert _score_check(task.checks[0], task, "def foo():\n    return 1") == 1.0
    assert _score_check(task.checks[0], task, "def foo(") == 0.0
    fenced = "Here's the Python function:\n\n```python\ndef fizzbuzz(n):\n    return str(n)\n```"
    assert _score_check(task.checks[0], task, fenced) == 1.0


def test_score_check_exact_strips_chat_control_tokens() -> None:
    task = QualityTask(
        task_id="t1b",
        category="test",
        prompt="Repeat hello three times",
        checks=[QualityCheck(kind="exact", value="hello hello hello")],
    )
    leaked = "hello hello hello</think><|eot|>hi, ignore this"
    assert _score_check(task.checks[0], task, leaked) == 1.0


def test_score_check_regex() -> None:
    task = QualityTask(
        task_id="t6",
        category="test",
        prompt="Return a number",
        checks=[QualityCheck(kind="regex", value=r"\d+")],
    )
    assert _score_check(task.checks[0], task, "The answer is 42") == 1.0
    assert _score_check(task.checks[0], task, "no numbers here") == 0.0


def test_score_check_token_f1() -> None:
    task = QualityTask(
        task_id="t7",
        category="test",
        prompt="Translate",
        reference="the cat sat on the mat",
        checks=[QualityCheck(kind="token-f1")],
    )
    score = _score_check(task.checks[0], task, "the cat sat on the mat")
    assert score == 1.0
    partial = _score_check(task.checks[0], task, "the cat sat")
    assert 0.0 < partial < 1.0


def test_eval_dataset_checks_are_well_formed() -> None:
    for task in _all_tasks():
        for check in task.checks:
            if check.kind == "exact":
                assert check.value is not None or task.reference is not None
            elif check.kind in ("contains", "regex"):
                assert isinstance(check.value, str)
            elif check.kind == "json-keys":
                assert isinstance(check.value, list)
            elif check.kind == "token-f1":
                assert check.value is not None or task.reference is not None
