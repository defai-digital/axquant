"""Self-distill dataset builder for Holo3 MTP head adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError


SAMPLE_SCHEMA = "axquant.mtp-align-sample.v1"


def load_prompt_strings(path: str | Path) -> list[str]:
    path = Path(path).expanduser().resolve()
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            prompts.append(stripped)
            continue
        if isinstance(record, str) and record:
            prompts.append(record)
        elif isinstance(record, dict):
            text = record.get("prompt") or record.get("text") or record.get("content")
            if isinstance(text, str) and text:
                prompts.append(text)
    if not prompts:
        raise ArtifactError(f"no prompts in {path}")
    return prompts


def write_samples(path: str | Path, samples: list[dict[str, Any]]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return path


def read_samples(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    samples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict):
            samples.append(record)
    return samples


def prepare_self_distill_dataset(
    model_dir: str | Path,
    prompts_path: str | Path,
    output_path: str | Path,
    *,
    max_prompts: int = 32,
    max_new_tokens: int = 64,
    max_samples: int = 512,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Generate token windows labeled by trunk greedy next-token.

    Uses mlx_lm generate for continuations, then teacher-labels with trunk
    greedy argmax along the resulting sequences (self-distill style).
    """
    try:
        import mlx.core as mx
        from mlx_lm import generate, load
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mlx_lm is required for prepare-data") from exc

    model_dir = Path(model_dir).expanduser().resolve()
    prompts = load_prompt_strings(prompts_path)[:max_prompts]
    model, tokenizer = load(str(model_dir))
    samples: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompts):
        if len(samples) >= max_samples:
            break
        # Greedy continuation text
        try:
            text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False,
            )
        except TypeError:
            text = generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens)
        # Tokenize full sequence (prompt + continuation when possible)
        full = prompt + (text or "")
        ids = tokenizer.encode(full)
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) < 3:
            continue
        # Label each position with trunk greedy next (recompute short prefixes)
        lm = getattr(model, "language_model", model)
        core = lm["model"] if hasattr(lm, "__getitem__") and "model" in lm else lm.model
        lm_head = lm["lm_head"] if hasattr(lm, "__getitem__") and "lm_head" in lm else lm.lm_head
        # Cap prefix length for memory
        max_len = min(len(ids), 256)
        for i in range(1, max_len - 1):
            if len(samples) >= max_samples:
                break
            prefix = ids[: i + 1]
            arr = mx.array([prefix], dtype=mx.int32)
            h = core.embed_tokens(arr)
            for layer in core.layers:
                h = layer(h)
            h = core.norm(h)
            last = h[:, -1, :]
            logits = last @ mx.swapaxes(lm_head.weight, -1, -2)
            label = int(mx.argmax(logits[0]).item())
            samples.append(
                {
                    "schema_version": SAMPLE_SCHEMA,
                    "prompt_index": prompt_index,
                    "position": i,
                    "input_ids": prefix,
                    "prev_token": prefix[-1],
                    "label_token": label,
                    "dataset_token": ids[i + 1] if i + 1 < len(ids) else label,
                    "label_source": "trunk_greedy",
                    "seed": seed,
                }
            )

    out = write_samples(output_path, samples)
    return {
        "schema_version": "axquant.mtp-align-dataset.v1",
        "output": str(out),
        "samples": len(samples),
        "prompts": len(prompts),
        "model_dir": str(model_dir),
        "max_new_tokens": max_new_tokens,
    }
