"""Self-distill dataset builder for Holo3 MTP head adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError

SAMPLE_SCHEMA = "axquant.mtp-align-sample.v1"
FEATURE_SCHEMA = "axquant.mtp-align-features.v1"


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


def _trunk_modules(model: Any) -> tuple[Any, Any, Any]:
    lm = getattr(model, "language_model", model)
    core = lm["model"] if hasattr(lm, "__getitem__") and "model" in lm else lm.model
    lm_head = lm["lm_head"] if hasattr(lm, "__getitem__") and "lm_head" in lm else lm.lm_head
    return core, core.embed_tokens, lm_head


def prepare_self_distill_dataset(
    model_dir: str | Path,
    prompts_path: str | Path,
    output_path: str | Path,
    *,
    max_prompts: int = 32,
    max_new_tokens: int = 64,
    max_samples: int = 512,
    max_seq_len: int = 128,
    seed: int = 20260728,
    write_features: bool = True,
) -> dict[str, Any]:
    """Generate token windows labeled by trunk greedy next-token.

    Single forward per sequence (teacher-forced): all positions labeled from
    one trunk pass. Optionally dumps stacked features for fast stage-1 adapt
    without reloading the 35B trunk each step.
    """
    try:
        import mlx.core as mx
        from mlx_lm import generate, load
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mlx_lm is required for prepare-data") from exc

    model_dir = Path(model_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    prompts = load_prompt_strings(prompts_path)[:max_prompts]
    model, tokenizer = load(str(model_dir))
    core, embed, lm_head = _trunk_modules(model)

    samples: list[dict[str, Any]] = []
    hidden_rows: list[Any] = []
    prev_embed_rows: list[Any] = []
    labels: list[int] = []

    for prompt_index, prompt in enumerate(prompts):
        if len(samples) >= max_samples:
            break
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
        full = prompt + (text or "")
        ids = tokenizer.encode(full)
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) < 3:
            continue
        max_len = min(len(ids), max_seq_len)
        ids = ids[:max_len]
        arr = mx.array([ids], dtype=mx.int32)
        h = embed(arr)
        for layer in core.layers:
            h = layer(h)
        h = core.norm(h)  # [1, T, H]
        logits = h @ mx.swapaxes(lm_head.weight, -1, -2)  # [1, T, V]
        embeds = embed(arr)  # [1, T, H]
        # Position i (0-based in ids): use hidden at i to predict next; prev token = ids[i]
        for i in range(0, max_len - 1):
            if len(samples) >= max_samples:
                break
            label = int(mx.argmax(logits[0, i]).item())
            samples.append(
                {
                    "schema_version": SAMPLE_SCHEMA,
                    "prompt_index": prompt_index,
                    "position": i,
                    "input_ids": ids[: i + 1],
                    "prev_token": ids[i],
                    "label_token": label,
                    "dataset_token": ids[i + 1],
                    "label_source": "trunk_greedy",
                    "seed": seed,
                    "feature_index": len(labels),
                }
            )
            if write_features:
                hidden_rows.append(h[0, i])
                prev_embed_rows.append(embeds[0, i])
                labels.append(label)

    out = write_samples(output_path, samples)
    feature_path = None
    if write_features and labels:
        feature_path = output_path.with_suffix(".features.safetensors")
        payload = {
            "hidden": mx.stack(hidden_rows, axis=0),
            "prev_embed": mx.stack(prev_embed_rows, axis=0),
            "label_token": mx.array(labels, dtype=mx.int32),
            # Allow adapt without reloading the full trunk for CE.
            "lm_head_weight": lm_head.weight,
        }
        mx.save_safetensors(str(feature_path), payload)
        meta = {
            "schema_version": FEATURE_SCHEMA,
            "features": str(feature_path),
            "count": len(labels),
            "hidden_shape": list(hidden_rows[0].shape),
            "has_lm_head_weight": True,
        }
        feature_path.with_suffix(".features.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

    return {
        "schema_version": "axquant.mtp-align-dataset.v1",
        "output": str(out),
        "features": str(feature_path) if feature_path is not None else None,
        "samples": len(samples),
        "prompts": len(prompts),
        "model_dir": str(model_dir),
        "max_new_tokens": max_new_tokens,
        "max_seq_len": max_seq_len,
    }


def load_feature_bundle(feature_path: str | Path) -> tuple[list[dict[str, Any]], Any | None]:
    """Load stacked features (+ optional lm_head) from prepare-data."""
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mlx is required to load feature rows") from exc
    path = Path(feature_path).expanduser().resolve()
    if not path.is_file():
        raise ArtifactError(f"feature file missing: {path}")
    data = mx.load(str(path))
    n = int(data["label_token"].shape[0])
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "hidden": data["hidden"][i],
                "prev_embed": data["prev_embed"][i],
                "label_token": int(data["label_token"][i].item()),
            }
        )
    lm_head = data.get("lm_head_weight")
    return rows, lm_head


def load_feature_rows(feature_path: str | Path) -> list[dict[str, Any]]:
    """Load stacked features written by :func:`prepare_self_distill_dataset`."""
    rows, _lm = load_feature_bundle(feature_path)
    return rows
