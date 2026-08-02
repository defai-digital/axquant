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


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise ArtifactError(f"missing config.json in {model_dir}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
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


def needs_conversion_prep(model_dir: str | Path) -> bool:
    """True when the checkpoint requires a prepared MLX text-path view."""
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir():
        return False
    try:
        config = _read_config(directory)
    except ArtifactError:
        return False
    return needs_gemma4_unified_prep(config)


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

    root = Path(work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prepared = root / "gemma4-text-path"
    if prepared.exists():
        shutil.rmtree(prepared)
    prepared.mkdir(parents=True)

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

    root = Path(work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prepared = root / "tekken-tokenizer-path"
    if prepared.exists():
        shutil.rmtree(prepared)
    prepared.mkdir(parents=True)

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

    fetched: dict[str, str] = {}
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
        dest = prepared / filename
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        shutil.copy2(downloaded, dest)
        fetched[filename] = revision

    if not (prepared / "tokenizer.json").is_file():
        raise ArtifactError(f"tokenizer.json missing after tekken prep from {repo}@{revision}")

    provenance = {
        "schema_version": "axquant.tekken-tokenizer-prep.v1",
        "source_dir": str(source),
        "tokenizer_repo": repo,
        "tokenizer_revision": revision,
        "fetched_files": sorted(fetched),
    }
    (prepared / "axquant_tekken_tokenizer_provenance.json").write_text(
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
    if needs_tekken_tokenizer_prep(source):
        return prepare_tekken_tokenizer_source(source, work_dir=work_dir, model_id=model_id)
    return None


def _mlx_core() -> Any:
    try:
        return __import__("mlx.core", fromlist=["*"])
    except ModuleNotFoundError as exc:
        raise ArtifactError(
            "gemma4 source preparation requires the MLX backend (axquant[mlx])"
        ) from exc


def _filter_single_shard(source_file: Path, destination: Path) -> None:
    mx = _mlx_core()
    weights = mx.load(str(source_file))
    filtered = {
        key: value
        for key, value in weights.items()
        if not any(token in key.lower() for token in _GEMMA4_MULTIMODAL_DROP)
    }
    if not filtered:
        raise ArtifactError("gemma4 preparation dropped every tensor; refusing empty checkpoint")
    dropped = len(weights) - len(filtered)
    mx.save_safetensors(str(destination), filtered)
    log.info(
        "gemma4_weights_filtered",
        kept=len(filtered),
        dropped=dropped,
        destination=str(destination),
    )


def _filter_sharded(source: Path, prepared: Path) -> None:
    """Filter a sharded checkpoint; write one consolidated safetensors file."""
    mx = _mlx_core()
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ArtifactError("model.safetensors.index.json has no weight_map")
    shards = sorted({str(value) for value in weight_map.values() if isinstance(value, str)})
    filtered: dict[str, Any] = {}
    for shard_name in shards:
        relative = Path(shard_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactError(f"index contains an unsafe shard path: {shard_name}")
        shard_path = source / relative
        if shard_path.suffix != ".safetensors":
            raise ArtifactError(f"index references a non-Safetensors shard: {shard_name}")
        if not shard_path.is_file():
            raise ArtifactError(f"missing shard referenced by index: {shard_name}")
        weights = mx.load(str(shard_path))
        for key, value in weights.items():
            if any(token in key.lower() for token in _GEMMA4_MULTIMODAL_DROP):
                continue
            filtered[key] = value
    if not filtered:
        raise ArtifactError("gemma4 preparation dropped every tensor; refusing empty checkpoint")
    mx.save_safetensors(str(prepared / "model.safetensors"), filtered)
    log.info("gemma4_sharded_weights_filtered", kept=len(filtered), shards=len(shards))
