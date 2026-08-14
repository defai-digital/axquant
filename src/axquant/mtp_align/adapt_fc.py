"""Stage-1/2 MTP head adaptation (freeze Holo3 trunk)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError
from axquant.grafted_mtp import QWEN35_MOE_PACKED_MTP_SHAPES, compose_grafted_mtp_onto_pack
from axquant.mtp_align.dataset import read_samples
from axquant.mtp_align.provenance import sidecar_sha256, write_adapted_graft_record
from axquant.mtp_align.qwen_mtp_head import QwenMtpHead
from axquant.serde import write_data

# Stage-1: projection + norms only.
FC_NORM_KEYS = (
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
)
# Back-compat alias.
TRAIN_KEYS = FC_NORM_KEYS

# Stage-2: entire packed MTP sidecar.
FULL_LAYER_KEYS = tuple(sorted(QWEN35_MOE_PACKED_MTP_SHAPES))


def adapt_head_from_features(
    head: QwenMtpHead,
    features: list[dict[str, Any]],
    lm_head_weight: Any,
    *,
    train_keys: Sequence[str] = FC_NORM_KEYS,
    steps: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 1,
) -> tuple[QwenMtpHead, list[float]]:
    """CE-train selected MTP tensors on precomputed feature rows."""
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mlx is required for MTP adapt") from exc

    if not features:
        raise ArtifactError("no training features")
    keys = tuple(train_keys)
    missing = [k for k in keys if k not in head.weights]
    if missing:
        raise ArtifactError(f"train keys missing from head: {missing}")
    params = {k: head.weights[k] for k in keys}

    def loss_fn(p: dict[str, Any], batch: list[dict[str, Any]]) -> Any:
        weights = dict(head.weights)
        weights.update(p)
        total = mx.array(0.0)
        for sample in batch:
            tmp = QwenMtpHead(weights=weights, config=head.config)
            logits = tmp.draft_logits(
                main_hidden=sample["hidden"],
                prev_token_embed=sample["prev_embed"],
                lm_head_weight=lm_head_weight,
            )[0]
            label = int(sample["label_token"])
            logz = logits.astype(mx.float32)
            logz = logz - mx.logsumexp(logz)
            total = total + (-logz[label])
        return total / max(len(batch), 1)

    loss_and_grad = mx.value_and_grad(loss_fn)
    history: list[float] = []
    step = 0
    idx = 0
    while step < steps:
        batch = features[idx : idx + batch_size]
        if not batch:
            idx = 0
            batch = features[:batch_size]
        idx = (idx + batch_size) % max(len(features), 1)
        loss, grads = loss_and_grad(params, batch)
        params = {k: params[k] - learning_rate * grads[k] for k in params}
        history.append(float(loss.item()))
        step += 1

    head.weights.update(params)
    return head, history


# Back-compat name used by tests/CLI stage-1.
def adapt_fc_norms_from_features(
    head: QwenMtpHead,
    features: list[dict[str, Any]],
    lm_head_weight: Any,
    *,
    steps: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 1,
) -> tuple[QwenMtpHead, list[float]]:
    return adapt_head_from_features(
        head,
        features,
        lm_head_weight,
        train_keys=FC_NORM_KEYS,
        steps=steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )


def write_adapted_mtp_bundle(
    head: QwenMtpHead,
    output_dir: str | Path,
    *,
    init_mtp_sha256: str,
    train_summary: dict[str, Any],
    trunk_model_id: str,
    trunk_revision: str,
    donor_model_id: str,
    donor_revision: str,
    graft_kind: str | None = None,
) -> dict[str, Any]:
    """Persist adapted sidecar + manifest + graft provenance under ``output_dir``."""
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ArtifactError(f"output already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mtp_out = head.save_safetensors(output_dir / "mtp.safetensors")
    out_sha = sidecar_sha256(mtp_out)
    write_data(
        output_dir / "axquant_mtp_sidecar_manifest.json",
        {
            "schema_version": "axquant.protected-tensor-sidecar.v1",
            "source_model": {"model_id": trunk_model_id, "revision": trunk_revision},
            "role": "mtp",
            "tensor_count": len(QWEN35_MOE_PACKED_MTP_SHAPES),
            "parameters": 1,
            "dtypes": ["BF16"],
            "tensor_names_sha256": "0" * 64,
            "source_files": [],
            "output": {
                "path": "mtp.safetensors",
                "size_bytes": mtp_out.stat().st_size,
                "sha256": out_sha,
            },
        },
    )
    graft = write_adapted_graft_record(
        output_dir,
        trunk_model_id=trunk_model_id,
        trunk_revision=trunk_revision,
        donor_model_id=donor_model_id,
        donor_revision=donor_revision,
        init_mtp_sha256=init_mtp_sha256,
        output_mtp_sha256=out_sha,
        train_summary=train_summary,
        graft_kind=graft_kind,
    )
    history = train_summary.get("loss_history")
    hist_name = (
        "adapt_full_history.json"
        if str(train_summary.get("stage", "")).startswith("full")
        else "adapt_fc_history.json"
    )
    if isinstance(history, list):
        (output_dir / hist_name).write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "mtp": str(mtp_out),
        "graft_record": str(graft),
        "train": train_summary,
        "output_mtp_sha256": out_sha,
    }


def _load_features_and_lm_head(
    model_dir: Path,
    data_path: Path,
    *,
    features_path: str | Path | None,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], Any, int, str | None]:
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mlx_lm is required for MTP adapt") from exc

    if features_path is None:
        candidate = data_path.with_suffix(".features.safetensors")
        if candidate.is_file():
            features_path = candidate

    if features_path is not None:
        from axquant.mtp_align.dataset import load_feature_bundle

        features, lm_head_weight = load_feature_bundle(features_path)
        if max_samples is not None:
            features = features[:max_samples]
        if lm_head_weight is None:
            loaded = load(str(model_dir))
            feature_model: Any = loaded[0]
            lm = getattr(feature_model, "language_model", feature_model)
            lm_head = (
                lm["lm_head"] if hasattr(lm, "__getitem__") and "lm_head" in lm else lm.lm_head
            )
            lm_head_weight = lm_head.weight
        return features, lm_head_weight, len(features), str(features_path)

    samples = read_samples(data_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ArtifactError("no training samples")
    loaded = load(str(model_dir))
    trunk_model: Any = loaded[0]
    lm = getattr(trunk_model, "language_model", trunk_model)
    core = lm["model"] if hasattr(lm, "__getitem__") and "model" in lm else lm.model
    lm_head = lm["lm_head"] if hasattr(lm, "__getitem__") and "lm_head" in lm else lm.lm_head
    embed = core.embed_tokens
    features = []
    for sample in samples:
        ids = sample["input_ids"]
        arr = mx.array([ids], dtype=mx.int32)
        h = embed(arr)
        for layer in core.layers:
            h = layer(h)
        h = core.norm(h)
        features.append(
            {
                "hidden": h[0, -1, :],
                "prev_embed": embed(mx.array([sample["prev_token"]], dtype=mx.int32))[0],
                "label_token": int(sample["label_token"]),
            }
        )
    return features, lm_head.weight, len(samples), None


def adapt_mtp_head(
    model_dir: str | Path,
    data_path: str | Path,
    init_mtp: str | Path,
    output_dir: str | Path,
    *,
    stage: str = "fc_norms",
    steps: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    max_samples: int | None = 256,
    features_path: str | Path | None = None,
    trunk_model_id: str = "Hcompany/Holo3-35B-A3B",
    trunk_revision: str = "unknown",
    donor_model_id: str = "Qwen/Qwen3.5-35B-A3B",
    donor_revision: str = "unknown",
) -> dict[str, Any]:
    """Adapt MTP head. ``stage`` is ``fc_norms`` (stage-1) or ``full_layer`` (stage-2)."""
    if stage not in {"fc_norms", "full_layer"}:
        raise ArtifactError(f"unknown adapt stage: {stage}")
    train_keys = FC_NORM_KEYS if stage == "fc_norms" else FULL_LAYER_KEYS
    graft_kind = "holo3-adapted-mtp-v1" if stage == "fc_norms" else "holo3-adapted-mtp-full-v1"

    model_dir = Path(model_dir).expanduser().resolve()
    init_mtp = Path(init_mtp).expanduser().resolve()
    data_path = Path(data_path).expanduser().resolve()

    head = QwenMtpHead.from_safetensors(init_mtp)
    init_sha = sidecar_sha256(init_mtp)
    features, lm_head_weight, sample_count, resolved_features = _load_features_and_lm_head(
        model_dir,
        data_path,
        features_path=features_path,
        max_samples=max_samples,
    )
    if not features:
        raise ArtifactError("no training features")

    head, history = adapt_head_from_features(
        head,
        features,
        lm_head_weight,
        train_keys=train_keys,
        steps=steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    train_summary = {
        "stage": stage,
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "samples": sample_count,
        "features_path": resolved_features,
        "loss_start": history[0] if history else None,
        "loss_end": history[-1] if history else None,
        "loss_history": history,
        "trainable": list(train_keys),
        "trainable_count": len(train_keys),
    }
    return write_adapted_mtp_bundle(
        head,
        output_dir,
        init_mtp_sha256=init_sha,
        train_summary=train_summary,
        trunk_model_id=trunk_model_id,
        trunk_revision=trunk_revision,
        donor_model_id=donor_model_id,
        donor_revision=donor_revision,
        graft_kind=graft_kind,
    )


def adapt_fc_norms(
    model_dir: str | Path,
    data_path: str | Path,
    init_mtp: str | Path,
    output_dir: str | Path,
    *,
    steps: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    max_samples: int | None = 256,
    features_path: str | Path | None = None,
    trunk_model_id: str = "Hcompany/Holo3-35B-A3B",
    trunk_revision: str = "unknown",
    donor_model_id: str = "Qwen/Qwen3.5-35B-A3B",
    donor_revision: str = "unknown",
) -> dict[str, Any]:
    """Stage-1: freeze MTP transformer; train fc + pre_fc norms + mtp.norm."""
    return adapt_mtp_head(
        model_dir,
        data_path,
        init_mtp,
        output_dir,
        stage="fc_norms",
        steps=steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_samples=max_samples,
        features_path=features_path,
        trunk_model_id=trunk_model_id,
        trunk_revision=trunk_revision,
        donor_model_id=donor_model_id,
        donor_revision=donor_revision,
    )


def adapt_full_layer(
    model_dir: str | Path,
    data_path: str | Path,
    init_mtp: str | Path,
    output_dir: str | Path,
    *,
    steps: int = 200,
    learning_rate: float = 1e-4,
    batch_size: int = 2,
    max_samples: int | None = 512,
    features_path: str | Path | None = None,
    trunk_model_id: str = "Hcompany/Holo3-35B-A3B",
    trunk_revision: str = "unknown",
    donor_model_id: str = "Qwen/Qwen3.5-35B-A3B",
    donor_revision: str = "unknown",
) -> dict[str, Any]:
    """Stage-2: unfreeze all packed MTP tensors; continue from stage-1 init."""
    return adapt_mtp_head(
        model_dir,
        data_path,
        init_mtp,
        output_dir,
        stage="full_layer",
        steps=steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_samples=max_samples,
        features_path=features_path,
        trunk_model_id=trunk_model_id,
        trunk_revision=trunk_revision,
        donor_model_id=donor_model_id,
        donor_revision=donor_revision,
    )


def compose_adapted_onto_pack(
    pack_dir: str | Path,
    mtp_bundle: str | Path,
    *,
    output_dir: str | Path,
) -> Path:
    """Compose adapted MTP bundle onto a certified trunk pack."""
    return compose_grafted_mtp_onto_pack(pack_dir, mtp_bundle, output_dir=output_dir)
