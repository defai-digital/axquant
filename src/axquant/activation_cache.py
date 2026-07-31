"""Tokenized calibration and activation cache.

Manages the calibration cache directory structure with content-addressed
shards, checksum verification, and atomic completion markers.  Changed
inputs always create a new cache directory rather than overwriting.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import structlog

from axquant.errors import BackendUnavailableError, CacheError
from axquant.schema import (
    ModelIdentity,
    ProfileName,
    TokenizedCacheManifest,
)
from axquant.serde import file_sha256, stable_sha256, write_data
from axquant.versioning import collect_versions

log = structlog.get_logger()

_SHARD_PREFIX = "shard-"
_SHARD_SUFFIX = ".npz"
_COMPLETION_MARKER = "completion.json"
_MANIFEST_NAME = "tokenized_cache_manifest.json"
_BACKEND_VERSION = "axquant-tokenizer-v1"
_SAMPLES_PER_SHARD = 100


def compute_cache_key(
    *,
    model: ModelIdentity,
    dataset_sha256: str,
    profile: ProfileName,
    sequence_length: int,
    random_seed: int,
    tokenizer_revision: str | None = None,
    config_digest: str | None = None,
    backend_version: str = _BACKEND_VERSION,
    mlx_version: str | None = None,
    mlx_lm_version: str | None = None,
    capture_points: tuple[str, ...] = ("output", "hidden"),
    separation_attested: bool = False,
    domains: tuple[str, ...] = (),
) -> str:
    """Compute a deterministic cache key from all identity-defining fields.

    The cache key includes: source revision, config digest, tokenizer
    revision, dataset digest, profile, sequence length, seed, backend
    version, MLX/MLX-LM versions, and capture-point definition.
    """
    identity = {
        "model_id": model.model_id,
        "revision": model.revision,
        "config_digest": config_digest,
        "tokenizer_revision": tokenizer_revision,
        "dataset_sha256": dataset_sha256,
        "profile": profile.value,
        "sequence_length": sequence_length,
        "random_seed": random_seed,
        "backend_version": backend_version,
        "mlx_version": mlx_version,
        "mlx_lm_version": mlx_lm_version,
        "capture_points": list(capture_points),
        "calibration_evaluation_separation_attested": separation_attested,
        "domains": list(domains),
    }
    return stable_sha256(identity)


def _shard_path(directory: Path, index: int) -> Path:
    return directory / f"{_SHARD_PREFIX}{index:04d}{_SHARD_SUFFIX}"


def verify_shard(shard: Path) -> bool:
    """Verify a shard file exists and is non-empty."""
    return shard.is_file() and shard.stat().st_size > 0


def verify_cache_integrity(cache_dir: Path, manifest: TokenizedCacheManifest) -> list[str]:
    """Verify all shards in the cache directory against the manifest.

    Returns a list of issues (empty means all checks passed).
    """
    issues: list[str] = []
    tokenized_dir = cache_dir / "tokenized"
    if not tokenized_dir.is_dir():
        issues.append("tokenized directory missing")
        return issues

    for i in range(manifest.shard_count):
        shard = _shard_path(tokenized_dir, i)
        if not shard.is_file():
            issues.append(f"missing shard: {shard.name}")
        elif shard.stat().st_size == 0:
            issues.append(f"empty shard: {shard.name}")
        else:
            expected_sha256 = manifest.shard_sha256.get(shard.name)
            if expected_sha256 is not None and file_sha256(shard) != expected_sha256:
                issues.append(f"checksum mismatch: {shard.name}")
                continue
            try:
                import numpy as np

                with np.load(shard, allow_pickle=False) as data:
                    required = {"input_ids", "attention_mask", "sample_indices"}
                    missing = required - set(data.files)
                    if missing:
                        issues.append(
                            f"invalid shard {shard.name}: missing arrays {sorted(missing)}"
                        )
                        continue
                    input_ids = data["input_ids"]
                    attention_mask = data["attention_mask"]
                    sample_indices = data["sample_indices"]
                    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
                        issues.append(f"invalid shard {shard.name}: token array shape mismatch")
                    if sample_indices.ndim != 1 or len(sample_indices) != len(input_ids):
                        issues.append(f"invalid shard {shard.name}: sample index shape mismatch")
            except (OSError, ValueError) as exc:
                issues.append(f"invalid shard {shard.name}: {exc}")

    # Check for extra shards
    existing_shards = sorted(tokenized_dir.glob(f"{_SHARD_PREFIX}*{_SHARD_SUFFIX}"))
    if len(existing_shards) > manifest.shard_count:
        issues.append(
            f"found {len(existing_shards)} shards but manifest declares {manifest.shard_count}"
        )

    return issues


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CacheError(f"{path}:{line_number} must contain a JSON object")
                samples.append(value)
    except json.JSONDecodeError as exc:
        raise CacheError(f"invalid JSONL in {path}:{exc.lineno}: {exc.msg}") from exc
    except OSError as exc:
        raise CacheError(f"cannot read calibration dataset: {exc}") from exc
    if not samples:
        raise CacheError("calibration dataset contains no samples")
    return samples


def _sample_text(sample: dict[str, Any], tokenizer: Any) -> str:
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            rendered = apply_template(messages, tokenize=False, add_generation_prompt=False)
            if isinstance(rendered, str) and rendered:
                return rendered
    for key in ("text", "prompt", "content", "instruction"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            response = sample.get("response") or sample.get("output")
            return f"{value}\n{response}" if isinstance(response, str) and response else value
    raise CacheError(
        "calibration samples require non-empty text, prompt, content, instruction, or messages"
    )


def _load_tokenizer(model: ModelIdentity, tokenizer_revision: str | None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise BackendUnavailableError(
            "tokenization requires transformers; install axquant[mlx]"
        ) from None
    source = model.local_path or model.model_id
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if model.local_path is not None:
        kwargs["local_files_only"] = True
    elif tokenizer_revision or model.revision:
        kwargs["revision"] = tokenizer_revision or model.revision
    try:
        return AutoTokenizer.from_pretrained(source, **kwargs)
    except (OSError, ValueError) as exc:
        raise CacheError(f"cannot load tokenizer for {source}: {exc}") from exc


def _tokenizer_sha256(tokenizer: Any) -> str:
    try:
        vocabulary = tokenizer.get_vocab()
    except (AttributeError, TypeError):
        vocabulary = {}
    return stable_sha256(
        {
            "class": type(tokenizer).__name__,
            "vocabulary": vocabulary,
            "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
        }
    )


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    import numpy as np

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            np.savez_compressed(destination, **arrays)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def is_cache_complete(cache_dir: Path) -> bool:
    """Check whether the atomic completion marker exists."""
    return (cache_dir / _COMPLETION_MARKER).is_file()


def load_cache_manifest(cache_dir: Path) -> TokenizedCacheManifest | None:
    """Load the tokenized cache manifest if it exists."""
    manifest_path = cache_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return TokenizedCacheManifest.model_validate(data)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def tokenize_calibration(
    *,
    model: ModelIdentity,
    dataset_path: str | Path,
    output_dir: str | Path,
    profile: ProfileName,
    sequence_length: int,
    random_seed: int,
    tokenizer_revision: str | None = None,
    config_digest: str | None = None,
    tokenizer: Any | None = None,
    calibration_manifest_sha256: str | None = None,
    separation_attested: bool = False,
    domains: list[str] | None = None,
) -> TokenizedCacheManifest:
    """Tokenize a calibration dataset and write the cache structure.

    This function requires MLX-LM for tokenization.  If MLX-LM is not
    available, it raises BackendUnavailableError.

    Cache structure:
        calibration-cache/
          tokenized_cache_manifest.json
          tokenized/
            shard-0000.npz ... shard-NNNN.npz
          completion.json
    """
    dataset = Path(dataset_path).expanduser().resolve()
    if not dataset.is_file():
        raise CacheError(f"calibration dataset does not exist: {dataset}")

    dataset_sha = file_sha256(dataset)
    samples_data = _read_dataset(dataset)
    recorded_domains = [
        domain.strip()
        for sample in samples_data
        if isinstance((domain := sample.get("domain")), str) and domain.strip()
    ]
    observed_domains = sorted(set(recorded_domains))
    requested_domains = sorted(set(domains or observed_domains))
    if observed_domains and not set(requested_domains).issubset(observed_domains):
        missing = sorted(set(requested_domains) - set(observed_domains))
        raise CacheError(f"declared calibration domains have no matching samples: {missing}")
    domain_provenance = (
        "sample-records"
        if len(recorded_domains) == len(samples_data) and observed_domains
        else "declared"
    )
    versions = collect_versions()
    cache_dir = Path(output_dir).expanduser().resolve()
    existing_manifest = load_cache_manifest(cache_dir)
    if existing_manifest is not None and (
        existing_manifest.dataset_sha256 != dataset_sha
        or existing_manifest.model != model
        or existing_manifest.profile != profile
        or existing_manifest.sequence_length != sequence_length
    ):
        raise CacheError(
            f"calibration cache already exists with different inputs at {cache_dir}; "
            "use a new output directory"
        )
    tokenizer_instance = tokenizer or _load_tokenizer(model, tokenizer_revision)
    tokenizer_sha = _tokenizer_sha256(tokenizer_instance)

    cache_key = compute_cache_key(
        model=model,
        dataset_sha256=dataset_sha,
        profile=profile,
        sequence_length=sequence_length,
        random_seed=random_seed,
        tokenizer_revision=tokenizer_revision or model.revision,
        config_digest=config_digest or tokenizer_sha,
        mlx_version=versions.mlx,
        mlx_lm_version=versions.mlx_lm,
        separation_attested=separation_attested,
        domains=tuple(requested_domains),
    )

    # Check for existing cache with same key
    if existing_manifest is not None:
        if existing_manifest.cache_key_sha256 == cache_key:
            if is_cache_complete(cache_dir):
                issues = verify_cache_integrity(cache_dir, existing_manifest)
                if issues:
                    raise CacheError(f"completed calibration cache failed verification: {issues}")
                log.info("calibration_cache_reused", path=str(cache_dir))
                return existing_manifest
            # Incomplete cache with same key - verify and resume
            issues = verify_cache_integrity(cache_dir, existing_manifest)
            if not issues:
                final_manifest = existing_manifest.model_copy(update={"complete": True})
                write_data(cache_dir / _MANIFEST_NAME, final_manifest)
                _write_completion_marker(cache_dir, final_manifest)
                return final_manifest
            raise CacheError(f"existing cache is incomplete and cannot be resumed: {issues}")
        # Different inputs - refuse to overwrite
        raise CacheError(
            f"calibration cache already exists with different inputs at {cache_dir}; "
            "use a new output directory"
        )

    # Create cache structure
    tokenized_dir = cache_dir / "tokenized"
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
    except ImportError:
        raise BackendUnavailableError(
            "tokenization requires numpy; install with: pip install numpy"
        ) from None

    order = list(range(len(samples_data)))
    random.Random(random_seed).shuffle(order)
    encoded: list[tuple[int, list[int]]] = []
    for sample_index in order:
        text = _sample_text(samples_data[sample_index], tokenizer_instance)
        token_ids = tokenizer_instance.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=sequence_length,
        )
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)
        normalized_ids = [int(token) for token in token_ids[:sequence_length]]
        if not normalized_ids:
            fallback_token = getattr(tokenizer_instance, "eos_token_id", None)
            normalized_ids = [int(fallback_token) if fallback_token is not None else 0]
        encoded.append((sample_index, normalized_ids))

    shard_count = max(1, (len(encoded) + _SAMPLES_PER_SHARD - 1) // _SAMPLES_PER_SHARD)
    total_tokens = sum(len(tokens) for _, tokens in encoded)
    shard_sha256: dict[str, str] = {}
    pad_token = getattr(tokenizer_instance, "pad_token_id", None)
    if pad_token is None:
        pad_token = getattr(tokenizer_instance, "eos_token_id", None)
    pad_token_id = int(pad_token) if pad_token is not None else 0

    for shard_index in range(shard_count):
        shard_samples = encoded[
            shard_index * _SAMPLES_PER_SHARD : (shard_index + 1) * _SAMPLES_PER_SHARD
        ]
        width = max(len(tokens) for _, tokens in shard_samples)
        input_ids = np.full((len(shard_samples), width), pad_token_id, dtype=np.int32)
        attention_mask = np.zeros((len(shard_samples), width), dtype=np.uint8)
        sample_indices = np.empty((len(shard_samples),), dtype=np.int64)
        for row, (sample_index, token_ids) in enumerate(shard_samples):
            input_ids[row, : len(token_ids)] = token_ids
            attention_mask[row, : len(token_ids)] = 1
            sample_indices[row] = sample_index
        shard_file = _shard_path(tokenized_dir, shard_index)
        _write_npz_atomic(
            shard_file,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_indices=sample_indices,
        )
        shard_sha256[shard_file.name] = file_sha256(shard_file)

    manifest = TokenizedCacheManifest(
        cache_key_sha256=cache_key,
        model=model,
        dataset_sha256=dataset_sha,
        profile=profile,
        domains=requested_domains,
        domain_provenance=domain_provenance,
        sequence_length=sequence_length,
        samples=len(samples_data),
        shard_count=shard_count,
        total_tokens=total_tokens,
        tokenizer_revision=tokenizer_revision or model.revision,
        tokenizer_sha256=tokenizer_sha,
        sample_order_sha256=stable_sha256(order),
        calibration_manifest_sha256=calibration_manifest_sha256,
        calibration_evaluation_separation_attested=separation_attested,
        backend_version=_BACKEND_VERSION,
        shard_sha256=shard_sha256,
        software_versions=versions,
        complete=False,
    )

    # Write manifest
    write_data(cache_dir / _MANIFEST_NAME, manifest)

    # Verify all shards before writing completion marker
    issues = verify_cache_integrity(cache_dir, manifest)
    if issues:
        raise CacheError(f"cache verification failed after writing: {issues}")

    # Write atomic completion marker
    final_manifest = manifest.model_copy(update={"complete": True})
    write_data(cache_dir / _MANIFEST_NAME, final_manifest)
    _write_completion_marker(cache_dir, final_manifest)

    log.info(
        "calibration_cache_created",
        path=str(cache_dir),
        shards=shard_count,
        samples=len(samples_data),
        total_tokens=total_tokens,
    )
    return final_manifest


def _write_completion_marker(cache_dir: Path, manifest: TokenizedCacheManifest) -> None:
    """Write the atomic completion marker."""
    marker_data = {
        "complete": True,
        "cache_key_sha256": manifest.cache_key_sha256,
        "shard_count": manifest.shard_count,
        "total_tokens": manifest.total_tokens,
    }
    marker_path = cache_dir / _COMPLETION_MARKER
    marker_path.write_text(
        json.dumps(marker_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
