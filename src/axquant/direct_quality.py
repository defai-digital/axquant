from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from axquant.errors import BenchmarkError
from axquant.identity import same_model_identity
from axquant.quality import (
    MlxQualityBackend,
    QualityBackend,
    load_quality_tasks,
    score_quality_task_output,
)
from axquant.schema import (
    DirectGeneralQualityModelOutput,
    DirectGeneralQualityState,
    DirectQualityEvaluation,
    DirectQualityTaskOutcome,
    ModelIdentity,
    ProfileName,
    QualityGenerationConfig,
)
from axquant.serde import file_sha256, load_model, write_data, write_text
from axquant.versioning import collect_versions


def evaluate_general_quality(
    *,
    model: ModelIdentity,
    model_artifact_sha256: str,
    dataset_path: str | Path,
    tokenizer_sha256: str,
    output_path: str | Path,
    state_path: str | Path,
    raw_log_dir: str | Path,
    max_sequence_length: int = 4096,
    max_generation_tokens: int = 256,
    random_seed: int = 20260803,
    backend: QualityBackend | None = None,
) -> DirectQualityEvaluation:
    """Evaluate the direct track's general holdout and retain every raw model output."""

    dataset = Path(dataset_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    raw_logs = Path(raw_log_dir).expanduser().resolve()
    try:
        raw_logs.relative_to(output_file.parent)
    except ValueError as exc:
        raise BenchmarkError(
            "raw general-quality logs must be stored beneath the quality evaluation directory"
        ) from exc

    tasks = load_quality_tasks(dataset)
    active_backend = backend or MlxQualityBackend()
    active_backend.load_model(model.local_path or model.model_id, model.revision)
    prompt_format, chat_template_sha256 = active_backend.generation_metadata()
    generation = QualityGenerationConfig(
        prompt_format=prompt_format,
        chat_template_sha256=chat_template_sha256,
        thinking_enabled=False,
        max_sequence_length=max_sequence_length,
        max_generation_tokens=max_generation_tokens,
    )
    expected_state = DirectGeneralQualityState(
        dataset_sha256=file_sha256(dataset),
        model=model,
        model_artifact_sha256=model_artifact_sha256,
        tokenizer_sha256=tokenizer_sha256,
        generation=generation,
        random_seed=random_seed,
    )
    state_file = Path(state_path).expanduser().resolve()
    if state_file.exists():
        state = load_model(state_file, DirectGeneralQualityState)
        if (
            state.dataset_sha256 != expected_state.dataset_sha256
            or not same_model_identity(state.model, expected_state.model)
            or state.model_artifact_sha256 != expected_state.model_artifact_sha256
            or state.tokenizer_sha256 != expected_state.tokenizer_sha256
            or state.generation != expected_state.generation
            or state.random_seed != expected_state.random_seed
        ):
            raise BenchmarkError(
                "general quality evaluation state does not match the requested run"
            )
    else:
        state = expected_state
    completed = {output.task_id: output for output in state.outputs}
    task_ids = {task.task_id for task in tasks}
    unexpected = set(completed) - task_ids
    if unexpected:
        raise BenchmarkError(
            f"general quality evaluation state contains unexpected task IDs: {sorted(unexpected)}"
        )

    for index, task in enumerate(tasks):
        if task.task_id in completed:
            continue
        loss_text = task.perplexity_text or (
            f"{task.prompt}\n{task.reference}" if task.reference else task.prompt
        )
        loss, perplexity_tokens = active_backend.perplexity_loss(
            loss_text,
            max_sequence_length,
        )
        try:
            output = active_backend.generate(
                task.prompt,
                max_generation_tokens,
                random_seed + index,
            )
            generated_tokens = active_backend.count_tokens(output)
            score, check_scores = score_quality_task_output(task, output)
            model_error = None
        except (BenchmarkError, RuntimeError, ValueError, re.error) as exc:
            output = ""
            generated_tokens = 0
            score = 0.0
            # Match quality.py: a failed sample keeps every declared check at
            # zero so validity denominators cannot silently shrink.
            check_scores = {
                f"{check.kind}:{check_index}": 0.0 for check_index, check in enumerate(task.checks)
            }
            model_error = str(exc)
        completed[task.task_id] = DirectGeneralQualityModelOutput(
            task_id=task.task_id,
            output=output,
            score=score,
            check_scores=check_scores,
            generated_tokens=generated_tokens,
            perplexity_loss=loss,
            perplexity_tokens=perplexity_tokens,
            model_error=model_error,
        )
        state = state.model_copy(
            update={
                "outputs": [completed[item.task_id] for item in tasks if item.task_id in completed]
            }
        )
        write_data(state_file, state)

    ordered_outputs = [completed[task.task_id] for task in tasks]
    total_loss = sum(output.perplexity_loss for output in ordered_outputs)
    total_perplexity_tokens = sum(output.perplexity_tokens for output in ordered_outputs)
    if total_perplexity_tokens == 0:
        raise BenchmarkError("general quality evaluation produced no perplexity tokens")

    raw_logs.mkdir(parents=True, exist_ok=True)
    outcomes: list[DirectQualityTaskOutcome] = []
    for index, output_record in enumerate(ordered_outputs):
        task_digest = hashlib.sha256(output_record.task_id.encode("utf-8")).hexdigest()[:16]
        output_name = f"{index:03d}-{task_digest}.model-output.txt"
        output_path_for_task = raw_logs / output_name
        write_text(output_path_for_task, output_record.output)
        relative_output = output_path_for_task.relative_to(output_file.parent).as_posix()
        outcomes.append(
            DirectQualityTaskOutcome(
                task_id=output_record.task_id,
                score=output_record.score,
                scored_tokens=output_record.generated_tokens,
                model_error=output_record.model_error is not None,
                output_file=relative_output,
                output_sha256=file_sha256(output_path_for_task),
            )
        )

    evaluation = DirectQualityEvaluation(
        profile=ProfileName.GENERAL,
        model=model,
        model_artifact_sha256=model_artifact_sha256,
        evaluation_manifest_sha256=file_sha256(dataset),
        dataset_sha256=file_sha256(dataset),
        tokenizer_sha256=tokenizer_sha256,
        generation=generation,
        random_seed=random_seed,
        evaluated_tokens=total_perplexity_tokens,
        software_versions=collect_versions(),
        perplexity=math.exp(total_loss / total_perplexity_tokens),
        outcomes=outcomes,
    )
    write_data(output_file, evaluation)
    write_data(state_file, state.model_copy(update={"completed": True}))
    return evaluation
