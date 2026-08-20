# ruff: noqa: RUF001
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

    def count_tokens(self, text: str) -> int: ...


class MlxQualityBackend:
    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._mx: Any = None
        self._mlx_lm: Any = None
        self._prompt_format = "raw"
        self._chat_template_sha256: str | None = None

    def load_model(self, model: str, revision: str | None) -> None:
        # Backends may be reused for a reference/candidate pair. Reset prompt
        # rendering state before every load so a tokenizer without a template
        # cannot inherit the previous model's chat-template metadata.
        self._prompt_format = "raw"
        self._chat_template_sha256 = None
        try:
            import importlib
            import os

            self._mx = importlib.import_module("mlx.core")
            self._mlx_lm = importlib.import_module("mlx_lm")
        except ImportError as exc:
            raise BackendUnavailableError(
                f"quality evaluation requires mlx and mlx-lm: {exc}"
            ) from exc
        # Large hybrid BF16/AXQ evals can trip Metal command-buffer faults on
        # some hosts; honor the same CPU force flag used by conversion.
        force_cpu = os.environ.get("AXQUANT_FORCE_CPU", "").strip().lower() in {
            "1",
            "true",
            "yes",
        } or os.environ.get("MLX_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
        if force_cpu:
            self._mx.set_default_device(self._mx.cpu)
        loaded = self._mlx_lm.load(model, revision=revision, lazy=False)
        self._model, self._tokenizer = loaded[:2]
        self._mx.eval(self._model.parameters())
        template = getattr(self._tokenizer, "chat_template", None)
        if isinstance(template, str) and template:
            self._prompt_format = "chat-template"
            self._chat_template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    def generation_metadata(self) -> tuple[str, str | None]:
        return self._prompt_format, self._chat_template_sha256

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise BenchmarkError("quality tokenizer is not loaded")
        return len(self._tokenizer.encode(text, add_special_tokens=False))

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


_MLX_VLM_QUALITY_MODEL_TYPES = frozenset({"muse_glimmer", "qwen3_vl"})


def _config_model_type(model: str) -> str | None:
    path = Path(model)
    config_path = path / "config.json" if path.is_dir() else None
    if config_path is None or not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("model_type")
    return raw if isinstance(raw, str) else None


def select_quality_backend(model: ModelIdentity | str) -> QualityBackend:
    """Pick mlx-lm or mlx-vlm from the checkpoint ``model_type``."""

    if isinstance(model, ModelIdentity):
        local = model.local_path or model.model_id
        identity = f"{model.model_id} {model.local_path or ''}"
    else:
        local = model
        identity = model
    model_type = _config_model_type(local)
    lowered = identity.casefold()
    if (
        model_type in _MLX_VLM_QUALITY_MODEL_TYPES
        or "muse-glimmer" in lowered
        or ("qwen3-vl" in lowered and "embedding" not in lowered)
    ):
        return MlxVlmQualityBackend()
    return MlxQualityBackend()


class MlxVlmQualityBackend:
    """Text quality via public MLX-VLM generate/forward (Muse Glimmer)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._mx: Any = None
        self._mlx_vlm: Any = None
        self._prompt_format = "raw"
        self._chat_template_sha256: str | None = None

    def load_model(self, model: str, revision: str | None) -> None:
        self._prompt_format = "raw"
        self._chat_template_sha256 = None
        try:
            import importlib
            import os

            self._mx = importlib.import_module("mlx.core")
            self._mlx_vlm = importlib.import_module("mlx_vlm")
        except ImportError as exc:
            raise BackendUnavailableError(
                f"muse_glimmer quality evaluation requires mlx and mlx-vlm: {exc}"
            ) from exc
        force_cpu = os.environ.get("AXQUANT_FORCE_CPU", "").strip().lower() in {
            "1",
            "true",
            "yes",
        } or os.environ.get("MLX_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
        if force_cpu:
            self._mx.set_default_device(self._mx.cpu)
        loaded = self._mlx_vlm.load(model, revision=revision, lazy=False)
        self._model, self._processor = loaded[:2]
        self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
        self._mx.eval(self._model.parameters())
        template = getattr(self._tokenizer, "chat_template", None)
        if isinstance(template, str) and template:
            self._prompt_format = "chat-template"
            self._chat_template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    def generation_metadata(self) -> tuple[str, str | None]:
        return self._prompt_format, self._chat_template_sha256

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            raise BenchmarkError("quality tokenizer is not loaded")
        return len(self._tokenizer.encode(text, add_special_tokens=False))

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
        output = self._model(inputs)
        logits = output.logits if hasattr(output, "logits") else output
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        total = self._mx.sum(losses)
        self._mx.eval(total)
        return float(total.item()), len(token_ids) - 1

    def generate(self, prompt: str, max_tokens: int, random_seed: int) -> str:
        if self._model is None or self._processor is None:
            raise BenchmarkError("quality model is not loaded")
        rendered_prompt = prompt
        if self._prompt_format == "chat-template" and hasattr(
            self._tokenizer, "apply_chat_template"
        ):
            rendered_prompt = str(
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        self._mx.random.seed(random_seed)
        result = self._mlx_vlm.generate(
            self._model,
            self._processor,
            rendered_prompt,
            image=None,
            verbose=False,
            max_tokens=max_tokens,
            temp=0.0,
        )
        text = getattr(result, "text", result)
        return str(text)


def load_quality_tasks(dataset_path: str | Path) -> list[QualityTask]:
    dataset = Path(dataset_path).expanduser().resolve()
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


# Chat-control markers that leak into completions when EOS is late or the
# template prefixes non-thinking with ``</think>``. Truncate at the first hit
# so exact/syntax scoring is not poisoned by the next turn.
_CONTROL_TOKEN_MARKERS = (
    "</think>",
    "<|eot|>",
    "<|endoftext|>",
    "<|im_end|>",
    "<｜User｜>",
    "<｜end▁of▁sentence｜>",
)
_FENCE_PYTHON = re.compile(r"```(?:python|py)\b[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FENCE_ANY = re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL)


def _strip_control_tokens(value: str) -> str:
    cut = len(value)
    for marker in _CONTROL_TOKEN_MARKERS:
        index = value.find(marker)
        if 0 <= index < cut:
            cut = index
    return value[:cut].strip()


def _unfenced(value: str) -> str:
    stripped = value.strip()
    match = re.fullmatch(r"```(?:json|python)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1)
    embedded = re.search(
        r"```(?:json|python|py)?\s*(.*?)```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return embedded.group(1).strip() if embedded else stripped


def _python_source(value: str) -> str:
    """Body to ``ast.parse`` for python-syntax: first fenced block, else bare text."""

    text = _strip_control_tokens(value)
    python_fence = _FENCE_PYTHON.search(text)
    if python_fence:
        return python_fence.group(1).strip()
    any_fence = _FENCE_ANY.search(text)
    if any_fence:
        return any_fence.group(1).strip()
    return text.strip()


def _json_value(output: str) -> Any:
    text = _unfenced(output)
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        for start, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            return value
        raise original_error


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
    scored = _strip_control_tokens(output)
    if check.kind == "exact":
        expected = check.value if isinstance(check.value, str) else task.reference
        if expected is None:
            raise BenchmarkError(f"{task.task_id}: exact check requires a value or reference")
        return float(_normalized_text(scored) == _normalized_text(expected))
    if check.kind == "contains":
        if not isinstance(check.value, str):
            raise BenchmarkError(f"{task.task_id}: contains check requires a string value")
        return float(_normalized_text(check.value) in _normalized_text(scored))
    if check.kind == "regex":
        if not isinstance(check.value, str):
            raise BenchmarkError(f"{task.task_id}: regex check requires a string value")
        return float(re.search(check.value, scored, flags=re.IGNORECASE | re.DOTALL) is not None)
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
            ast.parse(_python_source(output))
            return 1.0
        except SyntaxError:
            return 0.0
    if check.kind == "token-f1":
        expected = check.value if isinstance(check.value, str) else task.reference
        if expected is None:
            raise BenchmarkError(f"{task.task_id}: token-f1 requires a value or reference")
        return _token_f1(expected, scored)
    raise AssertionError(f"unhandled quality check {check.kind}")


def score_quality_task_output(
    task: QualityTask,
    output: str,
) -> tuple[float, dict[str, float]]:
    check_scores = {
        f"{check.kind}:{check_index}": _score_check(check, task, output)
        for check_index, check in enumerate(task.checks)
    }
    return sum(check_scores.values()) / len(check_scores), check_scores


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
    tasks = load_quality_tasks(dataset)
    if max_samples is not None:
        if max_samples < 1:
            raise BenchmarkError("max_samples must be at least one")
        tasks = tasks[:max_samples]
    active_backend = backend or select_quality_backend(model)
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
            score, check_scores = score_quality_task_output(task, output)
            error = None
        except (BenchmarkError, RuntimeError, ValueError, re.error) as exc:
            output = ""
            # A failed generation/scoring attempt is a failed sample, not a
            # missing observation. Preserve every declared check at zero so
            # structured-output validity denominators cannot silently shrink.
            check_scores = {
                f"{check.kind}:{check_index}": 0.0 for check_index, check in enumerate(task.checks)
            }
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
    # Generation protocol must match (format, lengths, thinking). Chat-template
    # digests may differ between an AXQ pack and a third-party uniform reference
    # even when both use the same suite controls — that packaging difference is
    # recorded, not a failed evaluation protocol.
    ref_gen = reference.generation
    cand_gen = candidate.generation
    if (
        ref_gen.prompt_format != cand_gen.prompt_format
        or ref_gen.thinking_enabled != cand_gen.thinking_enabled
        or ref_gen.max_sequence_length != cand_gen.max_sequence_length
        or ref_gen.max_generation_tokens != cand_gen.max_generation_tokens
    ):
        raise BenchmarkError("quality evaluations use different generation settings")
    if ref_gen.chat_template_sha256 != cand_gen.chat_template_sha256:
        log.warning(
            "quality_chat_template_differs",
            reference=ref_gen.chat_template_sha256,
            candidate=cand_gen.chat_template_sha256,
        )
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
