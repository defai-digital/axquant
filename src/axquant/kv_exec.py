"""Execute a planned per-layer KV-cache precision table through MLX-LM.

The compatibility runtime's public API accepts one cache object per layer
(``prompt_cache``), and its attention helper dispatches per cache object —
quantized SDPA for layers carrying a ``QuantizedKVCache`` and standard SDPA
otherwise. Building that list from the artifact's planned per-layer table
therefore executes the plan's exact per-layer KV precisions at runtime with
no private interfaces.

Runs as a subprocess (``python -m axquant.kv_exec``) so the MLX runtime stays
isolated from the toolkit process, mirroring the other runtime checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_layer_table(model_dir: Path) -> tuple[list[int], list[int]]:
    runtime_path = model_dir / "axquant_runtime.json"
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    kv = payload.get("kv_cache")
    if not isinstance(kv, dict):
        raise ValueError("artifact has no planned KV-cache table")
    bits = kv.get("layer_bits")
    groups = kv.get("layer_group_sizes")
    if (
        not isinstance(bits, list)
        or not isinstance(groups, list)
        or len(bits) != len(groups)
        or not bits
    ):
        raise ValueError("artifact KV-cache table is malformed")
    return [int(b) for b in bits], [int(g) for g in groups]


def _build_layer_caches(model: Any, bits: list[int], groups: list[int]) -> list[Any]:
    from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache

    base = make_prompt_cache(model)
    if len(base) != len(bits):
        raise ValueError(
            f"planned KV table covers {len(bits)} layers but the model builds {len(base)} caches"
        )
    caches: list[Any] = []
    replaced = 0
    for index, (cache, layer_bits, layer_group) in enumerate(zip(base, bits, groups, strict=True)):
        if layer_bits < 16 and type(cache) is KVCache:
            caches.append(QuantizedKVCache(group_size=layer_group, bits=layer_bits))
            replaced += 1
        else:
            # BF16-planned layers and non-KV caches (recurrent state in
            # hybrid families) keep the model's own cache object.
            caches.append(cache)
        del index
    if replaced == 0:
        raise ValueError(
            "no layer accepted a quantized KV cache; the family's cache layout "
            "does not expose standard KV layers for the planned precisions"
        )
    return caches


def _execution_summary(
    bits: list[int],
    executed: list[int],
    quantized_active: int,
    *,
    ok: bool,
    output_characters: int,
) -> dict[str, Any]:
    """Build the runtime-fidelity summary from a completed generation run.

    ``per_layer_execution`` must be an exact match against the planned table:
    a layer that silently reverted to BF16 (e.g. a plan/model layer-index
    mismatch, or a cache type the family doesn't expose as quantizable) is a
    real runtime-fidelity failure and must not be masked by any other layer
    having quantized successfully.
    """
    return {
        "ok": ok,
        "output_characters": output_characters,
        "planned_layer_bits": bits,
        "executed_layer_bits": executed,
        "quantized_layers_active": quantized_active,
        "per_layer_execution": executed == [b if b < 16 else 16 for b in bits],
    }


def run(model_dir: str, max_tokens: int) -> dict[str, Any]:
    from mlx_lm import generate, load

    directory = Path(model_dir).expanduser().resolve()
    bits, groups = _load_layer_table(directory)
    loaded = load(str(directory))
    model, tokenizer = loaded[0], loaded[1]
    caches = _build_layer_caches(model, bits, groups)
    text = generate(
        model,
        tokenizer,
        prompt="Reply with OK.",
        max_tokens=max_tokens,
        prompt_cache=caches,
        verbose=False,
    )
    executed = [int(getattr(cache, "bits", 16)) for cache in caches]
    quantized_active = sum(
        1 for cache in caches if hasattr(cache, "bits") and int(getattr(cache, "offset", 0)) > 0
    )
    return _execution_summary(
        bits,
        executed,
        quantized_active,
        ok=bool(text.strip()),
        output_characters=len(text.strip()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="axquant.kv_exec")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        report = run(args.model, args.max_tokens)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
