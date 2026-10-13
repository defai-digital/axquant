"""MLX model for Tencent Hy-MT2 (``model_type`` ``hy_v3``).

mlx_lm does not ship this architecture. :func:`register_hy_v3_model` inserts
this module into ``sys.modules`` under ``mlx_lm.models.hy_v3`` so
``mlx_lm.utils._get_classes`` resolves it. Structure mirrors
``mlx_lm.models.qwen3_moe`` (RMSNorm, RoPE, SwitchGLU experts, sanitize that
stacks unfused experts); routing is Hy-MT2 style: sigmoid scores,
expert-bias top-k selection, renormalize, router scaling, plus a shared
expert. Layer 0 is dense; layers 1..N-1 are MoE.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    moe_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    tie_word_embeddings: bool
    router_scaling_factor: float = 2.826
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 262144
    rope_scaling: dict[str, Any] | None = None


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            head_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Any | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.gate(x)


class HyV3SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.router_scaling_factor = args.router_scaling_factor

        self.router = Router(dim, num_experts)
        self.expert_bias = mx.zeros((num_experts,))
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)
        self.shared_mlp = MLP(dim, intermediate_size)

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        logits = self.router(x)
        scores = mx.sigmoid(logits)

        k = self.top_k
        biased = scores + self.expert_bias
        inds = mx.argpartition(biased, kth=-k, axis=-1)[..., -k:]
        top_scores = mx.take_along_axis(scores, inds, axis=-1)
        top_scores = top_scores / mx.sum(top_scores, axis=-1, keepdims=True)
        top_scores = top_scores * self.router_scaling_factor

        y = self.switch_mlp(x, inds)
        y = (y * top_scores[..., None]).sum(axis=-2)

        return y + self.shared_mlp(x)


class HyV3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args, layer_idx)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

        if layer_idx < args.first_k_dense_replace:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = HyV3SparseMoeBlock(args)

    def __call__(
        self,
        x: mx.array,
        mask: Any | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class HyV3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            HyV3DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Any | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HyV3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Any | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for layer in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            if f"{prefix}.mlp.experts.0.up_proj.weight" not in weights:
                continue
            for name in ["up_proj", "down_proj", "gate_proj"]:
                to_join = [
                    weights.pop(f"{prefix}.mlp.experts.{e}.{name}.weight")
                    for e in range(self.args.num_experts)
                ]
                weights[f"{prefix}.mlp.switch_mlp.{name}.weight"] = mx.stack(to_join)
        return weights

    @property
    def quant_predicate(self) -> Callable[[str, Any], Any]:
        def predicate(path: str, _module: Any) -> Any:
            if path.endswith("mlp.router.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self) -> list[HyV3DecoderLayer]:
        return self.model.layers


def register_hy_v3_model() -> None:
    """Register this module as ``mlx_lm.models.hy_v3`` for mlx_lm loading."""

    sys.modules["mlx_lm.models.hy_v3"] = sys.modules[__name__]
