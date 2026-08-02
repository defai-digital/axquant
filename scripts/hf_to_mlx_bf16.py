#!/usr/bin/env python3
"""Download (if needed) and convert a Hub model to MLX BF16 for AXQuant.

Sentence-Transformers / embedding exports often store weights without the
``model.`` prefix. This helper remaps keys (including multi-shard checkpoints)
before ``mlx_lm convert``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

_CONFIG_NAMES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def _weight_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("model*.safetensors"))
    if not files:
        files = sorted(p for p in directory.glob("*.safetensors") if p.is_file())
    return files


def _needs_model_prefix(path: Path) -> bool:
    weights = mx.load(str(path))
    keys = list(weights)
    if not keys:
        return False
    has_model = sum(1 for key in keys if key.startswith("model."))
    has_flat = sum(
        1 for key in keys if key.startswith("layers.") or key.startswith("embed_tokens")
    )
    return has_model == 0 and has_flat > 0


def _remap_key(key: str) -> str:
    if key.startswith("model.") or key.startswith("lm_head"):
        return key
    return f"model.{key}"


def _prepare_with_prefix(snapshot: Path, prepared: Path) -> Path:
    if prepared.exists():
        shutil.rmtree(prepared)
    prepared.mkdir(parents=True)
    for name in _CONFIG_NAMES:
        src = snapshot / name
        if src.exists():
            shutil.copy(src, prepared / name)

    weight_files = _weight_files(snapshot)
    index_path = snapshot / "model.safetensors.index.json"
    weight_map: dict[str, str] = {}

    for weight_file in weight_files:
        raw = mx.load(str(weight_file))
        remapped = {_remap_key(key): value for key, value in raw.items()}
        out_name = weight_file.name
        mx.save_safetensors(str(prepared / out_name), remapped)
        for key in remapped:
            weight_map[key] = out_name
        print(f"remapped {weight_file.name} -> {len(remapped)} tensors")

    # Embedding exports often omit lm_head while config has tie_word_embeddings=false.
    # Materialize lm_head from embed_tokens so mlx_lm can load a causal Qwen3 layout.
    has_lm_head = any(key == "lm_head.weight" or key.endswith(".lm_head.weight") for key in weight_map)
    embed_key = next(
        (
            key
            for key in weight_map
            if key.endswith("embed_tokens.weight") or key == "model.embed_tokens.weight"
        ),
        None,
    )
    if not has_lm_head and embed_key is not None:
        embed_file = prepared / weight_map[embed_key]
        tensors = dict(mx.load(str(embed_file)))
        lm_key = "lm_head.weight"
        tensors[lm_key] = tensors[embed_key]
        mx.save_safetensors(str(embed_file), tensors)
        weight_map[lm_key] = weight_map[embed_key]
        config_path = prepared / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["tie_word_embeddings"] = True
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"synthesized {lm_key} from {embed_key} and set tie_word_embeddings=true")

    if len(weight_files) > 1 or index_path.exists():
        total = sum((prepared / name).stat().st_size for name in sorted(set(weight_map.values())))
        index = {
            "metadata": {"total_size": total},
            "weight_map": weight_map,
        }
        if index_path.exists():
            try:
                original = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(original.get("metadata"), dict):
                    index["metadata"] = {**original["metadata"], **index["metadata"]}
            except json.JSONDecodeError:
                pass
        (prepared / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"prepared remapped checkpoint at {prepared}")
    return prepared


def prepare_hf_dir(hf_id: str, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    snap = Path(
        snapshot_download(
            hf_id,
            local_dir=str(work / "snapshot"),
        )
    )
    weight_files = _weight_files(snap)
    if not weight_files:
        raise SystemExit(f"no safetensors in {snap}")
    if any(_needs_model_prefix(path) for path in weight_files):
        return _prepare_with_prefix(snap, work / "prepared")
    print(f"using snapshot as-is {snap}")
    return snap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-id", required=True)
    parser.add_argument("--mlx-path", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    if (args.mlx_path / "config.json").exists() and list(args.mlx_path.glob("*.safetensors")):
        print(f"skip existing {args.mlx_path}")
        return
    hf_dir = prepare_hf_dir(args.hf_id, args.work)
    args.mlx_path.parent.mkdir(parents=True, exist_ok=True)
    if args.mlx_path.exists():
        shutil.rmtree(args.mlx_path)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        str(hf_dir),
        "--mlx-path",
        str(args.mlx_path),
        "--dtype",
        "bfloat16",
    ]
    print("run", " ".join(cmd))
    subprocess.check_call(cmd)
    print("ok", args.mlx_path)


if __name__ == "__main__":
    main()
