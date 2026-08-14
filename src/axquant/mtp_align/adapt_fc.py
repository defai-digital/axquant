"""Stage-1 adaptation: train mtp.fc + pre_fc norms + mtp.norm only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError
from axquant.grafted_mtp import QWEN35_MOE_PACKED_MTP_SHAPES, compose_grafted_mtp_onto_pack
from axquant.mtp_align.dataset import read_samples
from axquant.mtp_align.provenance import sidecar_sha256, write_adapted_graft_record
from axquant.mtp_align.qwen_mtp_head import QwenMtpHead, linear, rms_norm
from axquant.serde import write_data

TRAIN_KEYS = (
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
)


def adapt_fc_norms_from_features(
    head: QwenMtpHead,
    features: list[dict[str, Any]],
    lm_head_weight: Any,
    *,
    steps: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 1,
) -> tuple[QwenMtpHead, list[float]]:
    """CE-train fc/norms on precomputed ``{hidden, prev_embed, label_token}`` rows.

    This is the shipped stage-1 training loop. Callers that have a full trunk
    build features via :func:`adapt_fc_norms`; unit tests inject tiny tensors.
    """
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mlx is required for adapt-fc") from exc

    if not features:
        raise ArtifactError("no training features")
    params = {k: head.weights[k] for k in TRAIN_KEYS}

    def loss_fn(p: dict[str, Any], batch: list[dict[str, Any]]) -> Any:
        weights = dict(head.weights)
        weights.update(p)
        total = mx.array(0.0)
        for sample in batch:
            hidden = sample["hidden"]
            prev_embed = sample["prev_embed"]
            e_n = rms_norm(prev_embed, p["mtp.pre_fc_norm_embedding.weight"])
            h_n = rms_norm(hidden, p["mtp.pre_fc_norm_hidden.weight"])
            cat = mx.concatenate([e_n[None, :], h_n[None, :]], axis=-1)
            x = linear(cat, p["mtp.fc.weight"])
            tmp = QwenMtpHead(weights=weights, config=head.config)
            x = tmp._decoder_layer(x)
            x = rms_norm(x, p["mtp.norm.weight"])
            logits = linear(x, lm_head_weight)[0]
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
    )
    history = train_summary.get("loss_history")
    if isinstance(history, list):
        (output_dir / "adapt_fc_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    return {
        "output_dir": str(output_dir),
        "mtp": str(mtp_out),
        "graft_record": str(graft),
        "train": train_summary,
        "output_mtp_sha256": out_sha,
    }


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
    """Freeze MTP transformer; CE-train fc/norms on teacher labels.

    Prefer ``features_path`` from prepare-data (fast). Without features, rebuilds
    hiddens via mlx_lm trunk (slow on 35B).
    """
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mlx_lm is required for adapt-fc") from exc

    model_dir = Path(model_dir).expanduser().resolve()
    init_mtp = Path(init_mtp).expanduser().resolve()
    data_path = Path(data_path).expanduser().resolve()

    # Prefer sibling .features.safetensors when not explicit.
    if features_path is None:
        candidate = data_path.with_suffix(".features.safetensors")
        if candidate.is_file():
            features_path = candidate

    head = QwenMtpHead.from_safetensors(init_mtp)
    init_sha = sidecar_sha256(init_mtp)

    if features_path is not None:
        from axquant.mtp_align.dataset import load_feature_bundle

        features, lm_head_weight = load_feature_bundle(features_path)
        if max_samples is not None:
            features = features[:max_samples]
        if lm_head_weight is None:
            model, _tokenizer = load(str(model_dir))
            lm = getattr(model, "language_model", model)
            lm_head = (
                lm["lm_head"] if hasattr(lm, "__getitem__") and "lm_head" in lm else lm.lm_head
            )
            lm_head_weight = lm_head.weight
        sample_count = len(features)
    else:
        samples = read_samples(data_path)
        if max_samples is not None:
            samples = samples[:max_samples]
        if not samples:
            raise ArtifactError("no training samples")
        model, _tokenizer = load(str(model_dir))
        lm = getattr(model, "language_model", model)
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
            hidden = h[0, -1, :]
            prev_embed = embed(mx.array([sample["prev_token"]], dtype=mx.int32))[0]
            features.append(
                {
                    "hidden": hidden,
                    "prev_embed": prev_embed,
                    "label_token": int(sample["label_token"]),
                }
            )
        lm_head_weight = lm_head.weight
        sample_count = len(samples)

    if not features:
        raise ArtifactError("no training features")

    head, history = adapt_fc_norms_from_features(
        head,
        features,
        lm_head_weight,
        steps=steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    train_summary = {
        "stage": "fc_norms",
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "samples": sample_count,
        "features_path": str(features_path) if features_path is not None else None,
        "loss_start": history[0] if history else None,
        "loss_end": history[-1] if history else None,
        "loss_history": history,
        "trainable": list(TRAIN_KEYS),
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
    )


def compose_adapted_onto_pack(
    pack_dir: str | Path,
    mtp_bundle: str | Path,
    *,
    output_dir: str | Path,
) -> Path:
    """Compose adapted MTP bundle onto a certified trunk pack."""
    return compose_grafted_mtp_onto_pack(pack_dir, mtp_bundle, output_dir=output_dir)
