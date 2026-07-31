from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import structlog

from axquant.errors import BackendUnavailableError, BenchmarkError
from axquant.schema import (
    ModelIdentity,
    QualityCheck,
    QualityComparisonReport,
    QualityEvaluationResult,
    QualityGenerationConfig,
    QualityMetrics,
    QualityScoreComparison,
    QualityTask,
    QualityTaskComparison,
    QualityTaskResult,
)
from axquant.serde import file_sha256
from axquant.versioning import collect_versions

log = structlog.get_logger()


class QualityBackend(Protocol):
    def load_model(self, model: str, revision: str | None) -> None: ...

    def perplexity_loss(self, text: str, max_length: int) -> tuple[float, int]: ...

    def generate(self, prompt: str, max_tokens: int, random_seed: int) -> str: ...

    def generation_metadata(self) -> tuple[str, str | None]: ...


class MlxQualityBackend:
    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._mx: Any = None
        self._mlx_lm: Any = None
        self._prompt_format = "raw"
        self._chat_template_sha256: str | None = None

    def load_model(self, model: str, revision: str | None) -> None:
        try:
            import importlib

            self._mx = importlib.import_module("mlx.core")
            self._mlx_lm = importlib.import_module("mlx_lm")
        except ImportError as exc:
            raise BackendUnavailableError(
                f"quality evaluation requires mlx and mlx-lm: {exc}"
            ) from exc
        loaded = self._mlx_lm.load(model, revision=revision, lazy=False)
        self._model, self._tokenizer = loaded[:2]
        self._mx.eval(self._model.parameters())
        template = getattr(self._tokenizer, "chat_template", None)
        if isinstance(template, str) and template:
            self._prompt_format = "chat-template"
            self._chat_template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    def generation_metadata(self) -> tuple[str, str | None]:
        return self._prompt_format, self._chat_template_sha256

    def perplexity_loss(self, text: str, max_length: int) -> tuple[float, int]:
        if self._model is None or self._tokenizer is None:
            raise BenchmarkError("quality model is not loaded")
        try:
            import mlx.nn as nn
        except ImportError:
            raise BackendUnavailableError("quality evaluation requires mlx") from None
        token_ids = [
            int(token)
            for token in self._tokenizer.encode(text, add_special_tokens=True)[:max_length]
        ]
        if len(token_ids) < 2:
            return 0.0, 0
        inputs = self._mx.array([token_ids[:-1]])
        targets = self._mx.array([token_ids[1:]])
        logits = self._model(inputs)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        total = self._mx.sum(losses)
        self._mx.eval(total)
        return float(total.item()), len(token_ids) - 1

    def generate(self, prompt: str, max_tokens: int, random_seed: int) -> str:
        if self._model is None or self._tokenizer is None:
            raise BenchmarkError("quality model is not loaded")
        try:
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            raise BackendUnavailableError("quality evaluation requires mlx-lm") from None
        rendered_prompt = prompt
        if self._prompt_format == "chat-template":
            rendered_prompt = str(
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        self._mx.random.seed(random_seed)
        return str(
            self._mlx_lm.generate(
                self._model,
                self._tokenizer,
                rendered_prompt,
                verbose=False,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=0.0),
            )
        )


def _load_tasks(dataset: Path) -> list[QualityTask]:
    tasks: list[QualityTask] = []
    try:
        with dataset.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    tasks.append(QualityTask.model_validate_json(line))
                except ValueError as exc:
                    raise BenchmarkError(
                        f"invalid quality task at {dataset}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise BenchmarkError(f"cannot read quality dataset {dataset}: {exc}") from exc
    if not tasks:
        raise BenchmarkError("quality dataset contains no tasks")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise BenchmarkError("quality task IDs must be unique")
    return tasks


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _unfenced(value: str) -> str:
    stripped = value.strip()
    match = re.fullmatch(r"```(?:json|python)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1) if match else stripped


def _json_value(output: str) -> Any:
    text = _unfenced(output)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((index for index in (text.find("{"), text.find("[")) if index >= 0), default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _token_f1(reference: str, candidate: str) -> float:
    reference_tokens = _normalized_text(reference).split()
    candidate_tokens = _normalized_text(candidate).split()
    if not reference_tokens or not candidate_tokens:
        return float(reference_tokens == candidate_tokens)
    overlap = sum((Counter(reference_tokens) & Counter(candidate_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _score_check(check: QualityCheck, task: QualityTask, output: str) -> float:
    if check.kind == "exact":
        expected = check.value if isinstance(check.value, str) else task.reference
        if expected is None:
            raise BenchmarkError(f"{task.task_id}: exact check requires a value or reference")
        return float(_normalized_text(output) == _normalized_text(expected))
    if check.kind == "contains":
        if not isinstance(check.value, str):
            raise BenchmarkError(f"{task.task_id}: contains check requires a string value")
        return float(_normalized_text(check.value) in _normalized_text(output))
    if check.kind == "regex":
        if not isinstance(check.value, str):
            raise BenchmarkError(f"{task.task_id}: regex check requires a string value")
        return float(re.search(check.value, output, flags=re.IGNORECASE | re.DOTALL) is not None)
    if check.kind == "json-valid":
        try:
            _json_value(output)
            return 1.0
        except (json.JSONDecodeError, ValueError):
            return 0.0
    if check.kind == "json-keys":
        if not isinstance(check.value, list) or not all(
            isinstance(key, str) for key in check.value
        ):
            raise BenchmarkError(f"{task.task_id}: json-keys check requires a string list")
        try:
            value = _json_value(output)
        except (json.JSONDecodeError, ValueError):
            return 0.0
        return float(isinstance(value, dict) and all(key in value for key in check.value))
    if check.kind == "python-syntax":
        try:
            ast.parse(_unfenced(output))
            return 1.0
        except SyntaxError:
            return 0.0
    if check.kind == "token-f1":
        expected = check.value if isinstance(check.value, str) else task.reference
        if expected is None:
            raise BenchmarkError(f"{task.task_id}: token-f1 requires a value or reference")
        return _token_f1(expected, output)
    raise AssertionError(f"unhandled quality check {check.kind}")


def evaluate_quality(
    *,
    model: ModelIdentity,
    dataset_path: str | Path,
    max_sequence_length: int = 2048,
    max_generation_tokens: int = 256,
    random_seed: int = 0,
    max_samples: int | None = None,
    backend: QualityBackend | None = None,
) -> QualityEvaluationResult:
    dataset = Path(dataset_path).expanduser().resolve()
    if not dataset.is_file():
        raise BenchmarkError(f"quality dataset does not exist: {dataset}")
    tasks = _load_tasks(dataset)
    if max_samples is not None:
        if max_samples < 1:
            raise BenchmarkError("max_samples must be at least one")
        tasks = tasks[:max_samples]
    active_backend = backend or MlxQualityBackend()
    active_backend.load_model(model.local_path or model.model_id, model.revision)
    prompt_format, chat_template_sha256 = active_backend.generation_metadata()

    task_results: list[QualityTaskResult] = []
    category_scores: dict[str, list[float]] = {}
    json_scores: list[float] = []
    syntax_scores: list[float] = []
    total_loss = 0.0
    evaluated_tokens = 0
    for index, task in enumerate(tasks):
        loss_text = task.perplexity_text or (
            f"{task.prompt}\n{task.reference}" if task.reference else task.prompt
        )
        loss, token_count = active_backend.perplexity_loss(loss_text, max_sequence_length)
        total_loss += loss
        evaluated_tokens += token_count
        try:
            output = active_backend.generate(
                task.prompt,
                max_generation_tokens,
                random_seed + index,
            )
            check_scores = {
                f"{check.kind}:{check_index}": _score_check(check, task, output)
                for check_index, check in enumerate(task.checks)
            }
            score = sum(check_scores.values()) / len(check_scores)
            error = None
        except (BenchmarkError, RuntimeError, ValueError) as exc:
            output = ""
            check_scores = {}
            score = 0.0
            error = str(exc)
        category_scores.setdefault(task.category, []).append(score)
        json_scores.extend(
            value
            for key, value in check_scores.items()
            if key.startswith(("json-valid:", "json-keys:"))
        )
        syntax_scores.extend(
            value for key, value in check_scores.items() if key.startswith("python-syntax:")
        )
        task_results.append(
            QualityTaskResult(
                task_id=task.task_id,
                category=task.category,
                output=output,
                score=score,
                check_scores=check_scores,
                error=error,
            )
        )
        log.info(
            "quality_task_completed",
            task_id=task.task_id,
            category=task.category,
            task=index + 1,
            tasks=len(tasks),
            score=score,
            error=error,
        )

    if evaluated_tokens == 0:
        raise BenchmarkError("quality evaluation produced no perplexity tokens")
    metrics = QualityMetrics(
        perplexity=math.exp(total_loss / evaluated_tokens),
        task_scores={
            category: sum(scores) / len(scores)
            for category, scores in sorted(category_scores.items())
        },
        json_valid_rate=(sum(json_scores) / len(json_scores) if json_scores else None),
        syntax_valid_rate=(sum(syntax_scores) / len(syntax_scores) if syntax_scores else None),
    )
    log.info(
        "quality_evaluation_completed",
        model=model.model_id,
        samples=len(tasks),
        tokens=evaluated_tokens,
        perplexity=metrics.perplexity,
    )
    return QualityEvaluationResult(
        model=model,
        dataset_sha256=file_sha256(dataset),
        generation=QualityGenerationConfig(
            prompt_format=prompt_format,
            chat_template_sha256=chat_template_sha256,
            thinking_enabled=False,
            max_sequence_length=max_sequence_length,
            max_generation_tokens=max_generation_tokens,
        ),
        metrics=metrics,
        task_results=task_results,
        samples=len(tasks),
        evaluated_tokens=evaluated_tokens,
        random_seed=random_seed,
        software_versions=collect_versions(),
    )


def _comparison(reference: float, candidate: float) -> QualityScoreComparison:
    return QualityScoreComparison(
        reference=reference,
        candidate=candidate,
        delta=candidate - reference,
        retention=candidate / reference if reference > 0.0 else None,
    )


def compare_quality(
    reference: QualityEvaluationResult,
    candidate: QualityEvaluationResult,
) -> QualityComparisonReport:
    if reference.dataset_sha256 != candidate.dataset_sha256:
        raise BenchmarkError("quality evaluations use different datasets")
    if reference.random_seed != candidate.random_seed:
        raise BenchmarkError("quality evaluations use different random seeds")
    if reference.generation != candidate.generation:
        raise BenchmarkError("quality evaluations use different generation settings")
    reference_tasks = {task.task_id: task for task in reference.task_results}
    candidate_tasks = {task.task_id: task for task in candidate.task_results}
    if reference_tasks.keys() != candidate_tasks.keys():
        raise BenchmarkError("quality evaluations contain different task IDs")
    task_comparisons: list[QualityTaskComparison] = []
    for task_id, reference_task in reference_tasks.items():
        candidate_task = candidate_tasks[task_id]
        if reference_task.category != candidate_task.category:
            raise BenchmarkError(f"quality task category differs for {task_id}")
        comparison = _comparison(reference_task.score, candidate_task.score)
        task_comparisons.append(
            QualityTaskComparison(
                task_id=task_id,
                category=reference_task.category,
                **comparison.model_dump(),
            )
        )
    categories = set(reference.metrics.task_scores) | set(candidate.metrics.task_scores)
    if set(reference.metrics.task_scores) != set(candidate.metrics.task_scores):
        raise BenchmarkError("quality evaluations contain different categories")
    reference_aggregate = sum(task.score for task in reference.task_results) / len(
        reference.task_results
    )
    candidate_aggregate = sum(task.score for task in candidate.task_results) / len(
        candidate.task_results
    )
    reference_perplexity = reference.metrics.perplexity
    candidate_perplexity = candidate.metrics.perplexity
    perplexity_ratio = (
        candidate_perplexity / reference_perplexity
        if candidate_perplexity is not None and reference_perplexity is not None
        else None
    )

    def optional_comparison(
        reference_value: float | None,
        candidate_value: float | None,
    ) -> QualityScoreComparison | None:
        if reference_value is None and candidate_value is None:
            return None
        if reference_value is None or candidate_value is None:
            raise BenchmarkError("quality evaluations have mismatched optional metrics")
        return _comparison(reference_value, candidate_value)

    return QualityComparisonReport(
        reference_model=reference.model,
        candidate_model=candidate.model,
        dataset_sha256=reference.dataset_sha256,
        random_seed=reference.random_seed,
        aggregate=_comparison(reference_aggregate, candidate_aggregate),
        categories={
            category: _comparison(
                reference.metrics.task_scores[category],
                candidate.metrics.task_scores[category],
            )
            for category in sorted(categories)
        },
        perplexity_ratio=perplexity_ratio,
        json_validity=optional_comparison(
            reference.metrics.json_valid_rate,
            candidate.metrics.json_valid_rate,
        ),
        syntax_validity=optional_comparison(
            reference.metrics.syntax_valid_rate,
            candidate.metrics.syntax_valid_rate,
        ),
        tasks=task_comparisons,
        reference_errors=sum(task.error is not None for task in reference.task_results),
        candidate_errors=sum(task.error is not None for task in candidate.task_results),
    )
