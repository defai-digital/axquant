"""Pure-MLX Qwen3.5/3.6-style packed MTP head (depth-1 draft).

Matches AX Engine documentation contract::

    fc(cat([rms_norm(embed(prev), pre_fc_norm_embedding),
            rms_norm(main_hidden, pre_fc_norm_hidden)], dim=-1))
    → one transformer MoE layer
    → rms_norm(h, mtp_norm) @ lm_head

Offline align/training only. Online product MTP remains AX Engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError
from axquant.grafted_mtp import QWEN35_MOE_PACKED_MTP_SHAPES


def _mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mlx is required for MTP head forward/adapt") from exc
    return mx


def rms_norm(x: Any, weight: Any, *, eps: float = 1e-6) -> Any:
    mx = _mlx()
    x32 = x.astype(mx.float32)
    variance = mx.mean(mx.square(x32), axis=-1, keepdims=True)
    y = x32 * mx.rsqrt(variance + eps)
    return (y * weight.astype(mx.float32)).astype(x.dtype)


def linear(x: Any, weight: Any) -> Any:
    """``weight`` is (out_features, in_features)."""
    mx = _mlx()
    return x @ mx.swapaxes(weight, -1, -2)


def silu(x: Any) -> Any:
    mx = _mlx()
    return x * mx.sigmoid(x.astype(mx.float32)).astype(x.dtype)


@dataclass
class QwenMtpHeadConfig:
    hidden_size: int = 2048
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320


@dataclass
class QwenMtpHead:
    """MTP head parameters (mlx arrays) for offline draft logits."""

    weights: dict[str, Any]
    config: QwenMtpHeadConfig

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        *,
        config: QwenMtpHeadConfig | None = None,
    ) -> QwenMtpHead:
        mx = _mlx()
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ArtifactError(f"mtp sidecar not found: {path}")
        loaded = mx.load(str(path))
        missing = sorted(set(QWEN35_MOE_PACKED_MTP_SHAPES) - set(loaded))
        if missing:
            raise ArtifactError(f"mtp sidecar missing tensors: {missing}")
        base = config or QwenMtpHeadConfig()
        q = loaded["mtp.layers.0.self_attn.q_proj.weight"]
        head_dim = int(loaded["mtp.layers.0.self_attn.q_norm.weight"].shape[0])
        # q_proj is interleaved [query||gate] per head → out = n_heads * head_dim * 2
        n_q = int(q.shape[0] // (2 * head_dim))
        k = loaded["mtp.layers.0.self_attn.k_proj.weight"]
        n_kv = int(k.shape[0] // head_dim)
        gate = loaded["mtp.layers.0.mlp.gate.weight"]
        down = loaded["mtp.layers.0.mlp.experts.down_proj"]
        cfg = QwenMtpHeadConfig(
            hidden_size=int(loaded["mtp.fc.weight"].shape[0]),
            num_experts=int(gate.shape[0]),
            num_experts_per_tok=base.num_experts_per_tok,
            moe_intermediate_size=int(down.shape[-1]),
            num_attention_heads=n_q,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
            rms_norm_eps=base.rms_norm_eps,
            vocab_size=base.vocab_size,
        )
        return cls(weights={name: loaded[name] for name in QWEN35_MOE_PACKED_MTP_SHAPES}, config=cfg)

    @classmethod
    def from_weight_dict(
        cls,
        weights: dict[str, Any],
        *,
        config: QwenMtpHeadConfig,
    ) -> QwenMtpHead:
        missing = sorted(set(QWEN35_MOE_PACKED_MTP_SHAPES) - set(weights))
        if missing:
            raise ArtifactError(f"weight dict missing tensors: {missing}")
        return cls(weights=dict(weights), config=config)

    def trainable_fc_norm_params(self) -> dict[str, Any]:
        keys = (
            "mtp.fc.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.norm.weight",
        )
        return {k: self.weights[k] for k in keys}

    def save_safetensors(self, path: str | Path) -> Path:
        mx = _mlx()
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(path), self.weights)
        return path

    def draft_logits(
        self,
        *,
        main_hidden: Any,
        prev_token_embed: Any,
        lm_head_weight: Any,
    ) -> Any:
        """Depth-1 draft logits. ``main_hidden``/``prev_token_embed``: [B,H] or [H]."""
        mx = _mlx()
        w = self.weights
        cfg = self.config
        h = main_hidden if main_hidden.ndim == 2 else main_hidden[None, :]
        e = prev_token_embed if prev_token_embed.ndim == 2 else prev_token_embed[None, :]
        e_n = rms_norm(e, w["mtp.pre_fc_norm_embedding.weight"], eps=cfg.rms_norm_eps)
        h_n = rms_norm(h, w["mtp.pre_fc_norm_hidden.weight"], eps=cfg.rms_norm_eps)
        cat = mx.concatenate([e_n, h_n], axis=-1)
        x = linear(cat, w["mtp.fc.weight"])
        x = self._decoder_layer(x)
        x = rms_norm(x, w["mtp.norm.weight"], eps=cfg.rms_norm_eps)
        return linear(x, lm_head_weight)

    def _decoder_layer(self, x: Any) -> Any:
        w = self.weights
        cfg = self.config
        residual = x
        x = rms_norm(x, w["mtp.layers.0.input_layernorm.weight"], eps=cfg.rms_norm_eps)
        x = residual + self._attention(x)
        residual = x
        x = rms_norm(x, w["mtp.layers.0.post_attention_layernorm.weight"], eps=cfg.rms_norm_eps)
        return residual + self._moe_mlp(x)

    def _attention(self, x: Any) -> Any:
        mx = _mlx()
        w = self.weights
        cfg = self.config
        batch = int(x.shape[0])
        hd, nq, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads

        # q_proj: interleaved per-head [query(hd), gate(hd)] → (B, nq, 2*hd)
        qg = linear(x, w["mtp.layers.0.self_attn.q_proj.weight"]).reshape(batch, nq, 2 * hd)
        q = qg[:, :, :hd]
        gate = qg[:, :, hd:]
        k = linear(x, w["mtp.layers.0.self_attn.k_proj.weight"]).reshape(batch, nkv, hd)
        v = linear(x, w["mtp.layers.0.self_attn.v_proj.weight"]).reshape(batch, nkv, hd)
        q = rms_norm(q, w["mtp.layers.0.self_attn.q_norm.weight"], eps=cfg.rms_norm_eps)
        k = rms_norm(k, w["mtp.layers.0.self_attn.k_norm.weight"], eps=cfg.rms_norm_eps)
        if nq != nkv:
            repeats = nq // max(nkv, 1)
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)
        scale = hd**-0.5
        # seq length 1: attention weight is 1 after softmax over a singleton.
        scores = mx.sum(q * k, axis=-1, keepdims=True) * scale
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        ctx = (weights * v).reshape(batch, nq * hd)
        gate_flat = gate.reshape(batch, nq * hd)
        gated = ctx * mx.sigmoid(gate_flat.astype(mx.float32)).astype(x.dtype)
        return linear(gated, w["mtp.layers.0.self_attn.o_proj.weight"])

    def _moe_mlp(self, x: Any) -> Any:
        mx = _mlx()
        w = self.weights
        cfg = self.config
        router = linear(x, w["mtp.layers.0.mlp.gate.weight"])
        k = min(cfg.num_experts_per_tok, cfg.num_experts)
        # argpartition not always available; use top-k via sort for modest E in tests.
        # For E=256 production, sort is OK for offline micro-batches.
        order = mx.argsort(-router, axis=-1)
        top_idx = order[:, :k]
        top_logits = mx.take_along_axis(router, top_idx, axis=-1)
        top_w = mx.softmax(top_logits.astype(mx.float32), axis=-1).astype(x.dtype)

        gate_up = w["mtp.layers.0.mlp.experts.gate_up_proj"]
        down = w["mtp.layers.0.mlp.experts.down_proj"]
        batch = int(x.shape[0])
        rows: list[Any] = []
        for bi in range(batch):
            acc = mx.zeros((cfg.hidden_size,), dtype=x.dtype)
            for j in range(k):
                expert = int(top_idx[bi, j].item())
                weight = top_w[bi, j]
                gu = gate_up[expert]
                d = down[expert]
                gu_y = linear(x[bi][None, :], gu)[0]
                inter = int(gu_y.shape[0]) // 2
                act = silu(gu_y[:inter]) * gu_y[inter:]
                acc = acc + weight * linear(act[None, :], d)[0]
            rows.append(acc)
        out = mx.stack(rows, axis=0)

        g = linear(x, w["mtp.layers.0.mlp.shared_expert.gate_proj.weight"])
        u = linear(x, w["mtp.layers.0.mlp.shared_expert.up_proj.weight"])
        shared = silu(g) * u
        shared = linear(shared, w["mtp.layers.0.mlp.shared_expert.down_proj.weight"])
        sgate = w["mtp.layers.0.mlp.shared_expert_gate.weight"]
        alpha = mx.sigmoid(linear(x, sgate).astype(mx.float32)).astype(x.dtype)
        return out + alpha * shared
