"""Convert-time source preparation for MLX-LM architecture gaps.

Some public checkpoints (notably Gemma-4 ``gemma4_unified``) are fully
classifiable by AXQuant but cannot be loaded by the pinned MLX-LM until the
config type is remapped and multimodal tensors that the text convert path
rejects are filtered. Preparation is deterministic and writes beside the
staging root so system temp disks are not required for multi-GB models.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import structlog

from axquant.errors import ArtifactError
from axquant.serde import file_sha256

log = structlog.get_logger()

# Substrings of tensor names dropped from the MLX text convert view. The
# original checkpoint keeps them for post-convert protected vision extraction.
_GEMMA4_MULTIMODAL_DROP = (
    "vision_tower",
    "vision_embedder",
    "multi_modal_projector",
    "audio_tower",
    "embed_audio",
    "embed_vision",
)


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise ArtifactError(f"missing config.json in {model_dir}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read config.json: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError("config.json must contain a JSON object")
    return value


def needs_gemma4_unified_prep(config: dict[str, Any]) -> bool:
    return str(config.get("model_type", "")) == "gemma4_unified"


def needs_tekken_tokenizer_prep(source_dir: str | Path) -> bool:
    """Mistral/Devstral tekken-only exports lack transformers tokenizer.json."""
    source = Path(source_dir).expanduser().resolve()
    return (source / "tekken.json").is_file() and not (source / "tokenizer.json").is_file()


# Known HF packs that already ship a transformers-compatible tokenizer for
# tekken-based Mistral/Devstral releases (used only when the base export has
# tekken.json but no tokenizer.json). Each entry pins a Hub commit so prep is
# revision-reproducible (fail closed: never download from floating main).
# (match needle, repo_id, revision sha)
_TEKKEN_TOKENIZER_PACKS: tuple[tuple[str, str, str], ...] = (
    (
        "devstral-small-2505",
        "mlx-community/Devstral-Small-2505-bf16",
        "32aa4f5a17b8f7d302677d5d7e5f8b50351de159",
    ),
    (
        "devstral-small-2507",
        "lmstudio-community/Devstral-Small-2507-MLX-bf16",
        "7add794afc502cde50f3a536d91a12b987d15f8d",
    ),
    (
        "devstral-small-2-24b",
        "mlx-community/Devstral-Small-2-24B-Instruct-2512-bf16",
        "5cc5ba993ad5220ec4cbac1ab7126cd80189094a",
    ),
    (
        "mistral-small-3.1",
        "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-bf16",
        "93f1ae32cb76d99ec94c12ab00e759b2465f2cf6",
    ),
    (
        "mistral-small-3",
        "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-bf16",
        "93f1ae32cb76d99ec94c12ab00e759b2465f2cf6",
    ),
)


def needs_ministral3_model_prefix_prep(model_dir: str | Path, config: dict[str, Any]) -> bool:
    """True when Ministral3/Nemotron-Embed weights omit the MLX ``model.`` prefix.

    Public ``nvidia/Nemotron-3-Embed-*`` BF16 packs store language tensors as
    ``embed_tokens.*`` / ``layers.*`` while ``mlx_lm.models.ministral3`` expects
    ``model.embed_tokens.*`` / ``model.layers.*``. Convert-time prep rewrites
    the keys into a prepared view; the source snapshot is left unchanged.
    """
    if str(config.get("model_type", "")) != "ministral3":
        return False
    directory = Path(model_dir).expanduser().resolve()
    sample_keys = _sample_safetensor_keys(directory)
    if not sample_keys:
        return False
    has_unprefixed = any(
        key == "embed_tokens.weight"
        or key.startswith("layers.")
        or key.startswith("embed_tokens.")
        for key in sample_keys
    )
    has_prefixed = any(key.startswith("model.") for key in sample_keys)
    return has_unprefixed and not has_prefixed


def needs_conversion_prep(model_dir: str | Path) -> bool:
    """True when the checkpoint requires a prepared MLX text-path view."""
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir():
        return False
    try:
        config = _read_config(directory)
    except ArtifactError:
        return False
    return needs_gemma4_unified_prep(config) or needs_ministral3_model_prefix_prep(
        directory, config
    )


def _prepared_directory(source: Path, work_dir: str | Path, name: str) -> Path:
    """Create an empty preparation directory without ever overlapping the source."""
    root = Path(work_dir).expanduser().resolve()
    prepared = root / name
    resolved_prepared = prepared.resolve()
    if (
        source == resolved_prepared
        or source in resolved_prepared.parents
        or resolved_prepared in source.parents
    ):
        raise ArtifactError(
            f"preparation output must not overlap the source checkpoint: {resolved_prepared}"
        )
    root.mkdir(parents=True, exist_ok=True)
    if prepared.is_symlink():
        raise ArtifactError(f"preparation output must not be a symlink: {prepared}")
    if prepared.exists():
        if not prepared.is_dir():
            raise ArtifactError(f"preparation output is not a directory: {prepared}")
        shutil.rmtree(prepared)
    prepared.mkdir(parents=True)
    return prepared


def prepare_gemma4_unified_source(
    source_dir: str | Path,
    *,
    work_dir: str | Path,
) -> Path:
    """Build a ``gemma4`` text-path view of a ``gemma4_unified`` checkpoint.

    - Remaps ``model_type`` to ``gemma4`` (the MLX-LM module name).
    - Filters multimodal tensors that MLX-LM's gemma4 loader rejects.
    - Symlinks tokenizer / generation configs from the source.

    The prepared directory is suitable for ``mlx_lm.convert`` / load. Protected
    multimodal tensors remain on the original source for sidecar extraction.
    """
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise ArtifactError(f"gemma4 preparation requires a local directory: {source}")
    config = _read_config(source)
    if not needs_gemma4_unified_prep(config):
        raise ArtifactError(
            "gemma4 preparation expected model_type=gemma4_unified, got "
            f"{config.get('model_type')!r}"
        )

    prepared = _prepared_directory(source, work_dir, "gemma4-text-path")

    prepared_config = dict(config)
    prepared_config["model_type"] = "gemma4"
    (prepared / "config.json").write_text(
        json.dumps(prepared_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        candidate = source / name
        if candidate.is_file():
            (prepared / name).symlink_to(candidate)

    weight_path = source / "model.safetensors"
    index_path = source / "model.safetensors.index.json"
    if weight_path.is_file():
        _filter_single_shard(weight_path, prepared / "model.safetensors")
    elif index_path.is_file():
        _filter_sharded(source, prepared)
    else:
        raise ArtifactError(f"no Safetensors weights found under {source}")

    log.info(
        "gemma4_unified_source_prepared",
        source=str(source),
        prepared=str(prepared),
        model_type="gemma4",
    )
    return prepared


def _resolve_tekken_tokenizer_pack(
    source: Path,
    *,
    model_id: str | None,
    config: dict[str, Any],
) -> tuple[str, str]:
    """Return ``(repo_id, revision)`` for a pinned tokenizer pack."""
    blob = " ".join(
        [
            model_id or "",
            str(config.get("_name_or_path", "")),
            source.name,
        ]
    ).lower()
    for needle, repo, revision in _TEKKEN_TOKENIZER_PACKS:
        if needle in blob:
            return repo, revision
    raise ArtifactError(
        "tekken.json source has no tokenizer.json and no revision-pinned tokenizer pack; "
        "pass a model id that matches Devstral/Mistral packs or place tokenizer.json "
        "next to the weights (floating Hub main is not allowed for provenance)"
    )


# Back-compat alias used by unit tests.
def _resolve_tekken_tokenizer_repo(
    source: Path,
    *,
    model_id: str | None,
    config: dict[str, Any],
) -> str:
    repo, _revision = _resolve_tekken_tokenizer_pack(source, model_id=model_id, config=config)
    return repo


def prepare_tekken_tokenizer_source(
    source_dir: str | Path,
    *,
    work_dir: str | Path,
    model_id: str | None = None,
) -> Path:
    """Build a convert view that adds transformers tokenizer files for tekken exports.

    Weights/config are hard-linked or symlinked from the source; only tokenizer
    sidecars are fetched from a known-good pack so MLX-LM can load.
    """
    source = Path(source_dir).expanduser().resolve()
    if not needs_tekken_tokenizer_prep(source):
        raise ArtifactError("tekken tokenizer prep requires tekken.json without tokenizer.json")
    config = _read_config(source)
    repo, revision = _resolve_tekken_tokenizer_pack(source, model_id=model_id, config=config)

    prepared = _prepared_directory(source, work_dir, "tekken-tokenizer-path")

    # Link all source files (weights, config, tekken) into the prepared tree.
    for item in source.iterdir():
        if item.name.startswith("."):
            continue
        target = prepared / item.name
        try:
            target.symlink_to(item)
        except OSError:
            if item.is_file():
                shutil.copy2(item, target)
            else:
                shutil.copytree(item, target)

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ArtifactError(
            "tekken tokenizer prep requires huggingface_hub to fetch tokenizer.json"
        ) from exc

    fetched_sha256: dict[str, str] = {}
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        try:
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
            )
        except Exception:
            if filename == "tokenizer.json":
                raise ArtifactError(
                    f"cannot download tokenizer.json from {repo}@{revision} "
                    f"for tekken source {source}"
                ) from None
            continue
        downloaded_path = Path(downloaded)
        if not downloaded_path.is_file():
            raise ArtifactError(
                f"tokenizer pack returned a non-file for {filename}: {downloaded_path}"
            )
        expected_sha256 = file_sha256(downloaded_path)
        dest = prepared / filename
        if dest.is_symlink() or dest.exists():
            if dest.exists() and not dest.is_file() and not dest.is_symlink():
                raise ArtifactError(f"tokenizer destination is not a file: {dest}")
            dest.unlink()
        shutil.copy2(downloaded_path, dest)
        if file_sha256(dest) != expected_sha256:
            raise ArtifactError(f"tokenizer checksum changed while copying {filename}")
        fetched_sha256[filename] = expected_sha256

    if not (prepared / "tokenizer.json").is_file():
        raise ArtifactError(f"tokenizer.json missing after tekken prep from {repo}@{revision}")

    provenance = {
        "schema_version": "axquant.tekken-tokenizer-prep.v1",
        "source_dir": str(source),
        "tokenizer_repo": repo,
        "tokenizer_revision": revision,
        "fetched_files": sorted(fetched_sha256),
        "fetched_sha256": dict(sorted(fetched_sha256.items())),
    }
    provenance_path = prepared / "axquant_tekken_tokenizer_provenance.json"
    if provenance_path.is_symlink():
        provenance_path.unlink()
    elif provenance_path.exists():
        if not provenance_path.is_file():
            raise ArtifactError(
                f"tokenizer provenance destination is not a file: {provenance_path}"
            )
        provenance_path.unlink()
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    log.info(
        "tekken_tokenizer_source_prepared",
        source=str(source),
        prepared=str(prepared),
        tokenizer_repo=repo,
        tokenizer_revision=revision,
    )
    return prepared


def _sample_safetensor_keys(source: Path, *, limit: int = 32) -> list[str]:
    """Read a small set of tensor names without loading full weight payloads."""
    single = source / "model.safetensors"
    index = source / "model.safetensors.index.json"
    if index.is_file():
        try:
            payload = json.loads(
                index.read_text(encoding="utf-8"),
                object_pairs_hook=_json_object_without_duplicate_keys,
            )
        except (OSError, json.JSONDecodeError):
            return []
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict):
            return []
        return list(weight_map.keys())[:limit]
    if single.is_file():
        try:
            from safetensors import safe_open
        except ModuleNotFoundError:
            # Fallback via MLX when safetensors is unavailable.
            try:
                mx = _mlx_core()
                return list(_load_mlx_weights(mx, single).keys())[:limit]
            except ArtifactError:
                return []
        try:
            with safe_open(str(single), framework="np") as handle:
                return list(handle.keys())[:limit]
        except OSError:
            return []
    return []


def _prefix_ministral3_key(key: str) -> str:
    if key.startswith("model."):
        return key
    if key.startswith(("layers.", "embed_tokens.", "norm.")) or key in {
        "embed_tokens.weight",
        "norm.weight",
    }:
        return f"model.{key}"
    # lm_head and other top-level module keys stay unprefixed.
    return key


def prepare_ministral3_model_prefix_source(
    source_dir: str | Path,
    *,
    work_dir: str | Path,
) -> Path:
    """Rewrite unprefixed Ministral3 language keys to ``model.*`` for MLX-LM."""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise ArtifactError(f"ministral3 preparation requires a local directory: {source}")
    config = _read_config(source)
    if not needs_ministral3_model_prefix_prep(source, config):
        raise ArtifactError(
            "ministral3 model-prefix preparation expected unprefixed layers./embed_tokens. "
            f"weights under model_type=ministral3 (got model_type={config.get('model_type')!r})"
        )

    prepared = _prepared_directory(source, work_dir, "ministral3-model-prefix")
    sample_keys = _sample_safetensor_keys(source, limit=10_000)
    has_lm_head = any(key == "lm_head.weight" or key.endswith(".lm_head.weight") for key in sample_keys)
    prepared_config = dict(config)
    # Nemotron-3-Embed-8B ships tie_word_embeddings=false without an lm_head
    # tensor (feature-extraction export). Force tie so mlx_lm.ministral3 does
    # not expect a missing head during convert preflight.
    if not has_lm_head and not bool(prepared_config.get("tie_word_embeddings", True)):
        prepared_config["tie_word_embeddings"] = True
        log.info(
            "ministral3_forced_tie_word_embeddings",
            source=str(source),
            reason="lm_head.weight missing from checkpoint",
        )
    (prepared / "config.json").write_text(
        json.dumps(prepared_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "sentence_bert_config.json",
        "config_sentence_transformers.json",
        "modules.json",
    ):
        candidate = source / name
        if candidate.is_file():
            (prepared / name).symlink_to(candidate)
    pooling = source / "1_Pooling"
    if pooling.is_dir():
        (prepared / "1_Pooling").symlink_to(pooling)

    weight_path = source / "model.safetensors"
    index_path = source / "model.safetensors.index.json"
    if weight_path.is_file():
        _rewrite_ministral3_single_shard(weight_path, prepared / "model.safetensors")
    elif index_path.is_file():
        _rewrite_ministral3_sharded(source, prepared)
    else:
        raise ArtifactError(f"no Safetensors weights found under {source}")

    log.info(
        "ministral3_model_prefix_source_prepared",
        source=str(source),
        prepared=str(prepared),
        model_type="ministral3",
        tie_word_embeddings=bool(prepared_config.get("tie_word_embeddings")),
    )
    return prepared


def prepare_conversion_source(
    source_dir: str | Path,
    *,
    work_dir: str | Path,
    model_id: str | None = None,
) -> Path | None:
    """Return a prepared directory when required, else ``None`` (use source as-is)."""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        return None
    config = _read_config(source)
    if needs_gemma4_unified_prep(config):
        return prepare_gemma4_unified_source(source, work_dir=work_dir)
    if needs_ministral3_model_prefix_prep(source, config):
        return prepare_ministral3_model_prefix_source(source, work_dir=work_dir)
    if needs_tekken_tokenizer_prep(source):
        return prepare_tekken_tokenizer_source(source, work_dir=work_dir, model_id=model_id)
    return None


def _rewrite_ministral3_single_shard(source_file: Path, destination: Path) -> None:
    mx = _mlx_core()
    weights = _load_mlx_weights(mx, source_file)
    rewritten = {_prefix_ministral3_key(str(key)): value for key, value in weights.items()}
    if len(rewritten) != len(weights):
        raise ArtifactError(
            "ministral3 model-prefix rewrite collapsed tensor names; refusing convert prep"
        )
    mx.save_safetensors(str(destination), rewritten)
    _verify_saved_tensor_names(mx, destination, set(rewritten))


def _rewrite_ministral3_sharded(source: Path, prepared: Path) -> None:
    """Rewrite each shard and the index so keys use the MLX ``model.`` prefix."""
    source = source.resolve()
    index_path = source / "model.safetensors.index.json"
    try:
        index = json.loads(
            index_path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid {index_path}: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ArtifactError("model.safetensors.index.json has no weight_map")

    mx = _mlx_core()
    new_weight_map: dict[str, str] = {}
    shards_by_name: dict[str, dict[str, Any]] = {}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ArtifactError("model.safetensors.index.json contains an empty tensor name")
        if not isinstance(shard_name, str) or not shard_name:
            raise ArtifactError(
                "model.safetensors.index.json contains a non-string shard reference"
            )
        relative = Path(shard_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactError(f"index contains an unsafe shard path: {shard_name}")
        shard_path = source / relative
        if not shard_path.is_file():
            raise ArtifactError(f"missing shard referenced by index: {shard_name}")
        if shard_name not in shards_by_name:
            weights = _load_mlx_weights(mx, shard_path)
            shards_by_name[shard_name] = {
                _prefix_ministral3_key(str(key)): value for key, value in weights.items()
            }
        new_key = _prefix_ministral3_key(tensor_name)
        new_weight_map[new_key] = shard_name

    for shard_name, weights in shards_by_name.items():
        destination = prepared / shard_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(destination), weights)
        _verify_saved_tensor_names(mx, destination, set(weights))

    new_index = dict(index)
    new_index["weight_map"] = new_weight_map
    (prepared / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mlx_core() -> Any:
    try:
        return __import__("mlx.core", fromlist=["*"])
    except ModuleNotFoundError as exc:
        raise ArtifactError(
            "gemma4 source preparation requires the MLX backend (axquant[mlx])"
        ) from exc


def _filter_single_shard(source_file: Path, destination: Path) -> None:
    mx = _mlx_core()
    weights = _load_mlx_weights(mx, source_file)
    filtered = {
        key: value
        for key, value in weights.items()
        if not any(token in key.lower() for token in _GEMMA4_MULTIMODAL_DROP)
    }
    if not filtered:
        raise ArtifactError("gemma4 preparation dropped every tensor; refusing empty checkpoint")
    dropped = len(weights) - len(filtered)
    mx.save_safetensors(str(destination), filtered)
    _verify_saved_tensor_names(mx, destination, set(filtered))
    log.info(
        "gemma4_weights_filtered",
        kept=len(filtered),
        dropped=dropped,
        destination=str(destination),
    )


def _load_mlx_weights(mx: Any, path: Path) -> Any:
    try:
        weights = mx.load(str(path))
    except Exception as exc:
        raise ArtifactError(f"cannot load Safetensors weights from {path}: {exc}") from exc
    if not hasattr(weights, "items"):
        raise ArtifactError(f"MLX returned a non-mapping weight payload for {path}")
    keys = list(weights)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ArtifactError(f"MLX returned an invalid tensor name for {path}")
    return weights


def _verify_saved_tensor_names(mx: Any, path: Path, expected: set[str]) -> None:
    saved = _load_mlx_weights(mx, path)
    actual = set(saved)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ArtifactError(
            f"prepared Safetensors output coverage mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def _filter_sharded(source: Path, prepared: Path) -> None:
    """Filter a sharded checkpoint; write one consolidated safetensors file."""
    source = source.resolve()
    index_path = source / "model.safetensors.index.json"
    try:
        index = json.loads(
            index_path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"invalid {index_path}: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ArtifactError("model.safetensors.index.json has no weight_map")
    indexed_names: dict[Path, set[str]] = {}
    shard_labels: dict[Path, str] = {}
    resolved_shards: dict[Path, str] = {}
    for tensor_name, shard_name in weight_map.items():
        if not tensor_name:
            raise ArtifactError("model.safetensors.index.json contains an empty tensor name")
        if not isinstance(shard_name, str) or not shard_name:
            raise ArtifactError(
                "model.safetensors.index.json contains a non-string shard reference"
            )
        relative = Path(shard_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactError(f"index contains an unsafe shard path: {shard_name}")
        shard_path = source / relative
        if shard_path.suffix != ".safetensors":
            raise ArtifactError(f"index references a non-Safetensors shard: {shard_name}")
        if not shard_path.is_file():
            raise ArtifactError(f"missing shard referenced by index: {shard_name}")
        resolved_shard = shard_path.resolve()
        previous_label = resolved_shards.get(resolved_shard)
        if previous_label is not None and previous_label != shard_name:
            raise ArtifactError(
                "index references the same physical shard under multiple paths: "
                f"{previous_label!r}, {shard_name!r}"
            )
        resolved_shards[resolved_shard] = shard_name
        indexed_names.setdefault(shard_path, set()).add(tensor_name)
        shard_labels[shard_path] = shard_name

    # Index path safety is pure validation and must surface without MLX so
    # non-MLX installs (and CI) reject unsafe shard maps before backend work.
    mx = _mlx_core()
    filtered: dict[str, Any] = {}
    for shard_path in sorted(indexed_names):
        shard_name = shard_labels[shard_path]
        weights = _load_mlx_weights(mx, shard_path)
        expected = indexed_names[shard_path]
        actual = set(weights)
        if actual != expected:
            missing = sorted(expected - actual)
            unindexed = sorted(actual - expected)
            raise ArtifactError(
                f"{shard_name} does not match model.safetensors.index.json: "
                f"missing={missing[:10]}, unindexed={unindexed[:10]}"
            )
        for key in sorted(expected):
            value = weights[key]
            if any(token in key.lower() for token in _GEMMA4_MULTIMODAL_DROP):
                continue
            filtered[key] = value
    if not filtered:
        raise ArtifactError("gemma4 preparation dropped every tensor; refusing empty checkpoint")
    destination = prepared / "model.safetensors"
    mx.save_safetensors(str(destination), filtered)
    _verify_saved_tensor_names(mx, destination, set(filtered))
    log.info("gemma4_sharded_weights_filtered", kept=len(filtered), shards=len(indexed_names))
