#!/usr/bin/env python3
"""Download (if needed) and convert a Hub model to MLX BF16 for AXQuant.

Sentence-Transformers / embedding exports often store weights without the
``model.`` prefix. This helper remaps keys (including multi-shard checkpoints)
before ``mlx_lm convert``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

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
_IMMUTABLE_HUB_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_SOURCE_PROVENANCE_NAME = "axquant_source.json"


def _mlx_core() -> Any:
    try:
        return import_module("mlx.core")
    except ModuleNotFoundError as exc:
        raise SystemExit("this helper requires MLX on Apple Silicon") from exc


def _weight_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("model*.safetensors"))
    if not files:
        files = sorted(p for p in directory.glob("*.safetensors") if p.is_file())
    return files


def _needs_model_prefix(path: Path) -> bool:
    mx = _mlx_core()
    weights = mx.load(str(path))
    keys = list(weights)
    if not keys:
        return False
    has_model = sum(1 for key in keys if key.startswith("model."))
    has_flat = sum(1 for key in keys if key.startswith("layers.") or key.startswith("embed_tokens"))
    return has_model == 0 and has_flat > 0


def _remap_key(key: str) -> str:
    if key.startswith("model.") or key.startswith("lm_head"):
        return key
    return f"model.{key}"


def _prepare_with_prefix(snapshot: Path, prepared: Path) -> Path:
    mx = _mlx_core()
    if prepared.exists():
        raise SystemExit(f"prepared staging path already exists: {prepared}")
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
    has_lm_head = any(
        key == "lm_head.weight" or key.endswith(".lm_head.weight") for key in weight_map
    )
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


def prepare_hf_dir(hf_id: str, revision: str, work: Path, prepared: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    snap = Path(
        snapshot_download(
            hf_id,
            revision=revision,
            local_dir=str(work / f"snapshot-{revision}"),
        )
    )
    weight_files = _weight_files(snap)
    if not weight_files:
        raise SystemExit(f"no safetensors in {snap}")
    if any(_needs_model_prefix(path) for path in weight_files):
        return _prepare_with_prefix(snap, prepared)
    print(f"using snapshot as-is {snap}")
    return snap


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-id", required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="Immutable 40-character Hugging Face commit SHA",
    )
    parser.add_argument("--mlx-path", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args(argv)
    if not _IMMUTABLE_HUB_REVISION.fullmatch(args.revision):
        raise SystemExit("--revision must be an immutable 40-character Hub commit SHA")
    if args.mlx_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.mlx_path}")
    args.mlx_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{args.mlx_path.name}.", dir=args.mlx_path.parent)
    )
    staging = temporary_root / "artifact"
    prepared = temporary_root / "prepared-source"
    try:
        hf_dir = prepare_hf_dir(args.hf_id, args.revision, args.work, prepared)
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            str(hf_dir),
            "--mlx-path",
            str(staging),
            "--dtype",
            "bfloat16",
        ]
        print("run", " ".join(cmd))
        subprocess.check_call(cmd)
        if not (staging / "config.json").is_file() or not list(staging.glob("*.safetensors")):
            raise SystemExit("MLX conversion completed without a usable checkpoint")
        (staging / _SOURCE_PROVENANCE_NAME).write_text(
            json.dumps(
                {
                    "schema_version": "axquant.source-conversion.v1",
                    "source_model": args.hf_id,
                    "source_revision": args.revision.lower(),
                    "dtype": "bfloat16",
                    "key_remap_applied": hf_dir == prepared,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if args.mlx_path.exists():
            raise SystemExit(f"output appeared during conversion: {args.mlx_path}")
        staging.rename(args.mlx_path)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    print("ok", args.mlx_path)


if __name__ == "__main__":
    main()
