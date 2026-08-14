"""Offline teacher-forced MTP top-1 vs Holo3 trunk greedy next-token."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError
from axquant.mtp_align.qwen_mtp_head import QwenMtpHead


@dataclass(frozen=True, slots=True)
class TeacherForceReport:
    positions: int
    correct: int
    top1: float
    prompts_used: int
    model_dir: str
    mtp_path: str
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "axquant.mtp-align-teacher-force.v1",
            "positions": self.positions,
            "correct": self.correct,
            "top1": self.top1,
            "prompts_used": self.prompts_used,
            "model_dir": self.model_dir,
            "mtp_path": self.mtp_path,
            "notes": list(self.notes),
        }


def _load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            prompts.append(line)
            continue
        if isinstance(record, str):
            prompts.append(record)
        elif isinstance(record, dict):
            text = record.get("prompt") or record.get("text") or record.get("content")
            if isinstance(text, str) and text:
                prompts.append(text)
    if not prompts:
        raise ArtifactError(f"no prompts in {path}")
    return prompts


def _iter_token_windows(
    tokenizer: Any,
    prompts: list[str],
    *,
    max_positions: int,
    max_prompt_tokens: int,
) -> Iterator[tuple[list[int], int]]:
    """Yield (token_ids_prefix_including_position, position_index) for labels."""
    count = 0
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) < 2:
            continue
        ids = ids[:max_prompt_tokens]
        # Teacher-force positions 0..len-2 predict token at i+1 from prefix ids[:i+1]
        for i in range(len(ids) - 1):
            yield ids[: i + 1], ids[i + 1]
            count += 1
            if count >= max_positions:
                return


def _trunk_hidden_and_logits(model: Any, token_ids: list[int]) -> tuple[Any, Any]:
    """Return (final_hidden [H], logits [V]) for last position."""
    import mlx.core as mx

    ids = mx.array([token_ids], dtype=mx.int32)
    # Prefer language_model path used by qwen3_5_moe.
    lm = getattr(model, "language_model", model)
    core = lm["model"] if hasattr(lm, "__getitem__") and "model" in lm else lm.model
    lm_head = lm["lm_head"] if hasattr(lm, "__getitem__") and "lm_head" in lm else lm.lm_head

    h = core.embed_tokens(ids)
    for layer in core.layers:
        h = layer(h)
    h = core.norm(h)
    last = h[:, -1, :]
    logits = last @ mx.swapaxes(lm_head.weight, -1, -2)
    return last[0], logits[0]


def run_teacher_force(
    model_dir: str | Path,
    mtp_path: str | Path,
    prompts_path: str | Path,
    *,
    max_positions: int = 64,
    max_prompt_tokens: int = 128,
    max_prompts: int | None = None,
) -> TeacherForceReport:
    """Score depth-1 draft top-1 against trunk greedy labels."""
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mlx_lm is required for teacher-force probe") from exc

    model_dir = Path(model_dir).expanduser().resolve()
    mtp_path = Path(mtp_path).expanduser().resolve()
    prompts_path = Path(prompts_path).expanduser().resolve()
    if not model_dir.is_dir():
        raise ArtifactError(f"model dir missing: {model_dir}")
    if not mtp_path.is_file():
        # allow mtp inside model dir
        candidate = model_dir / "mtp.safetensors"
        if candidate.is_file():
            mtp_path = candidate
        else:
            raise ArtifactError(f"mtp sidecar missing: {mtp_path}")

    prompts = _load_prompts(prompts_path)
    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    loaded = load(str(model_dir))
    model: Any = loaded[0]
    tokenizer: Any = loaded[1]
    head = QwenMtpHead.from_safetensors(mtp_path)
    lm = getattr(model, "language_model", model)
    core = lm["model"] if hasattr(lm, "__getitem__") and "model" in lm else lm.model
    lm_head = lm["lm_head"] if hasattr(lm, "__getitem__") and "lm_head" in lm else lm.lm_head
    embed = core.embed_tokens

    correct = 0
    total = 0
    notes: list[str] = []
    for prefix, _label in _iter_token_windows(
        tokenizer,
        prompts,
        max_positions=max_positions,
        max_prompt_tokens=max_prompt_tokens,
    ):
        hidden, trunk_logits = _trunk_hidden_and_logits(model, prefix)
        teacher = int(mx.argmax(trunk_logits).item())
        # Prefer greedy trunk next-token as teacher (may differ from dataset token).
        label_token = teacher
        prev = prefix[-1]
        prev_embed = embed(mx.array([prev], dtype=mx.int32))[0]
        draft_logits = head.draft_logits(
            main_hidden=hidden,
            prev_token_embed=prev_embed,
            lm_head_weight=lm_head.weight,
        )[0]
        pred = int(mx.argmax(draft_logits).item())
        if pred == label_token:
            correct += 1
        total += 1

    if total == 0:
        raise ArtifactError("no positions evaluated — check prompts length")

    top1 = correct / total
    if top1 < 0.05:
        notes.append(
            "top-1 near zero matches grafted parent head on Holo3 trunk "
            "(distributional mismatch expected)"
        )
    return TeacherForceReport(
        positions=total,
        correct=correct,
        top1=top1,
        prompts_used=min(len(prompts), total),
        model_dir=str(model_dir),
        mtp_path=str(mtp_path),
        notes=tuple(notes),
    )
