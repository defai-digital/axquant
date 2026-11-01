"""MLX port of Cloudflare Clef / Clef-Flash (Qwen3.5 backbone + joint schema head).

Vendored for AXQuant packs. Apache-2.0, following Cloudflare/clef.
The joint-head math follows the upstream joint_schema_model.py. This file is
the public MLX loader, kept so the decision head runs on Apple Silicon without
PyTorch. AXQuant does not quantize the joint head.

Torch-free. The backbone is loaded with mlx-vlm (text + images/video) or mlx-lm
(text only), bf16 or quantized. The joint head is loaded from the original
``joint_head.safetensors`` / ``joint_head_config.json`` with no conversion.

    import clef_mlx
    model = clef_mlx.load("mlx-community/clef-flash-4bit")
    model.predict(record)        # {question_id: {option_id: probability}}
    model.systemone(request)     # Jev/SystemOne /v1/systemone response body

Command line (from a downloaded repo, --model defaults to that repo):

    python clef_mlx.py predict --state "..." --questions '{"q": {"type": "noul"}}'
    python clef_mlx.py serve --port 8000     # POST /v1/systemone on 127.0.0.1
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import socket
import ssl
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}

# This loader uses mlx-vlm internals (get_input_embeddings, inputs_embeds/position_ids on the
# text model), which can change between minor releases.
# Tested: mlx 0.32.3, mlx-lm 0.32.0, mlx-vlm 0.7.4.
TESTED_VERSIONS = {"mlx-vlm": (0, 7), "mlx-lm": (0, 32)}


class ContextTooLong(ValueError):
    pass


def _check_versions(packages: tuple[str, ...]) -> None:
    import importlib.metadata
    import warnings

    for name in packages:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        major_minor = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
        major, minor = TESTED_VERSIONS[name]
        if major_minor != (major, minor):
            warnings.warn(
                f"clef_mlx was tested with {name} {major}.{minor}.x but {version} is installed; "
                "if loading fails or results look wrong, run: pip install "
                f'"{name}>={major}.{minor},<{major}.{minor + 1}"',
                stacklevel=3,
            )


# --------------------------------------------------------------------------- encoding
# Mirrors joint_schema_model.encode_record (text-only path) exactly.


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    question_type = str(question["type"])
    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    media: dict[str, Any] | None = dataclass_field(default=None, compare=False, repr=False)


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def sample_frames(video, fps=2.0, source_fps=24.0, min_frames=4, max_frames=768):
    """Uniform frame sampling matching transformers' Qwen3VLVideoProcessor.

    Frame arrays carry no metadata, so (like transformers) assume a 24 fps source.
    Returns (frames, source frame indices).
    """
    import numpy as np

    video = np.asarray(video)
    total = len(video)
    n = int(total / source_fps * fps)
    n = min(max(n, min_frames), max_frames, total)
    indices = np.linspace(0, total - 1, n).round().astype(int)
    return video[indices], indices.tolist()


def _timestamps(indices: list[int], source_fps: float, temporal_patch: int) -> list[float]:
    indices = list(indices)
    if len(indices) % temporal_patch:
        indices.extend(indices[-1] for _ in range(temporal_patch - len(indices) % temporal_patch))
    seconds = [i / source_fps for i in indices]
    return [
        (seconds[i] + seconds[i + temporal_patch - 1]) / 2
        for i in range(0, len(seconds), temporal_patch)
    ]


def _encode_media(
    processor: Any, record: dict[str, Any]
) -> tuple[list[int], dict[str, Any] | None]:
    """Pixel preprocessing via the (torch-free) mlx-vlm processor; the token layout is
    built here to match the transformers Qwen3-VL processor the model was trained with."""
    import numpy as np

    images = list(record.get("images") or [])
    videos = list(record.get("videos") or [])
    if not images and not videos:
        return [], None
    if processor is None:
        raise ValueError("records with images or videos require a processor")
    kwargs = dict(record.get("media_kwargs") or {})
    sample = {
        k: kwargs.pop(k) for k in ("fps", "source_fps", "min_frames", "max_frames") if k in kwargs
    }
    source_fps = sample.get("source_fps", 24.0)
    do_sample = kwargs.pop("do_sample_frames", True)

    media: dict[str, Any] = {}
    text = ""
    if images:
        out = processor.image_processor(images=images, **kwargs)
        media["pixel_values"] = mx.array(np.asarray(out["pixel_values"]))
        grids = np.asarray(out["image_grid_thw"])
        media["image_grid_thw"] = mx.array(grids)
        merge = processor.image_processor.merge_size**2
        for grid in grids:
            text += (
                "<|vision_start|>"
                + "<|image_pad|>" * (int(np.prod(grid)) // merge)
                + "<|vision_end|>"
            )
    if videos:
        vp = processor.video_processor
        frames, frame_indices = [], []
        for v in videos:
            f, idx = (
                sample_frames(v, **sample) if do_sample else (np.asarray(v), list(range(len(v))))
            )
            frames.append(f)
            frame_indices.append(idx)
        out = vp(videos=frames, **kwargs)
        media["pixel_values_videos"] = mx.array(np.asarray(out["pixel_values_videos"]))
        grids = np.asarray(out["video_grid_thw"])
        media["video_grid_thw"] = mx.array(grids)
        for grid, idx in zip(grids, frame_indices, strict=False):
            per_frame = int(grid[1] * grid[2]) // vp.merge_size**2
            stamps = _timestamps(idx, source_fps, vp.temporal_patch_size)
            text += "<|vision_start|>"
            for t in range(int(grid[0])):
                text += (
                    f"<{stamps[t]:.1f} seconds><|vision_start|>"
                    + "<|video_pad|>" * per_frame
                    + "<|vision_end|>"
                )
            text += "<|vision_end|>"
    text += "\n"
    return _tokens(processor.tokenizer, text), media


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = 16384,
    max_state_tokens: int | None = None,
    processor: Any | None = None,
    truncate: bool = True,
) -> EncodedRecord:
    """Encode a record. Like the reference, the state is truncated to fit ``max_length``
    unless ``truncate=False``, in which case ``ContextTooLong`` is raised instead."""
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for qi, (question_id, question) in enumerate(record["questions"].items()):
        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {qi + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
            )
        )
        q_start = len(schema_ids)
        instructions = question.get("instructions") or str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        q_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))
        option_spans, option_ids = [], []
        for oi, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {oi + 1}: "))
            o_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((o_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            EncodedQuestion(
                str(question_id),
                QUESTION_TYPES[str(question["type"])],
                (q_start, q_end),
                tuple(option_spans),
                tuple(option_ids),
            )
        )

    prefix_ids = _tokens(
        tokenizer, f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    media_ids, media = _encode_media(processor, record)
    prefix_ids = prefix_ids + media_ids
    state_ids = _tokens(tokenizer, render(record["state"]))
    if max_state_tokens is not None:
        state_ids = state_ids[:max_state_tokens]
    fixed = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed > max_length:
        raise ContextTooLong(
            f"schema requires {fixed} tokens before state; maximum is {max_length}"
        )
    if not truncate and fixed + len(state_ids) > max_length:
        raise ContextTooLong(
            f"request needs {fixed + len(state_ids)} tokens; maximum is {max_length}"
        )
    state_ids = state_ids[: max_length - fixed]
    off = len(prefix_ids) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(
            q.question_id,
            q.question_type,
            (q.question_span[0] + off, q.question_span[1] + off),
            tuple((s + off, e + off) for s, e in q.option_spans),
            q.option_ids,
        )
        for q in questions
    )
    return EncodedRecord(tuple(prefix_ids + state_ids + schema_ids + suffix_ids), shifted, media)


# --------------------------------------------------------------------------- head


class MultiheadAttention(nn.Module):
    """torch.nn.MultiheadAttention (batch_first, packed in_proj) in MLX."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * width, width))
        self.in_proj_bias = mx.zeros((3 * width,))
        self.out_proj = nn.Linear(width, width)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        w = self.in_proj_weight
        b = self.in_proj_bias
        d = w.shape[1]
        q = q @ w[:d].T + b[:d]
        k = k @ w[d : 2 * d].T + b[d : 2 * d]
        v = v @ w[2 * d :].T + b[2 * d :]
        B, Lq, _ = q.shape
        Lk = k.shape[1]
        hd = d // self.heads
        q = q.reshape(B, Lq, self.heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, Lk, self.heads, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, Lk, self.heads, hd).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, Lq, d))


class FeedForward(nn.Module):
    def __init__(self, width: int, feedforward: int):
        super().__init__()
        self.fc1 = nn.Linear(width, feedforward)
        self.fc2 = nn.Linear(feedforward, width)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = MultiheadAttention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = FeedForward(width, feedforward)

    def __call__(self, queries, memory):
        m = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), m, m)
        return queries + self.feedforward(self.feedforward_norm(queries))


class TransformerDecoderLayer(nn.Module):
    """torch.nn.TransformerDecoderLayer(norm_first=True, activation='gelu')."""

    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.self_attn = MultiheadAttention(width, heads)
        self.multihead_attn = MultiheadAttention(width, heads)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, x, memory):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h)
        x = x + self.multihead_attn(self.norm2(x), memory, memory)
        return x + self.linear2(nn.gelu(self.linear1(self.norm3(x))))


def _l2norm(x, eps=1e-12):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


class JointSchemaHead(nn.Module):
    def __init__(self, hidden_size, width, routing_layers, layers, heads, feedforward):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = [
            EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)
        ]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [TransformerDecoderLayer(width, heads, feedforward) for _ in range(layers)]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.scorer1 = nn.Linear(width * 4, width)
        self.scorer2 = nn.Linear(width, 1)
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        out = {}
        for k, v in weights.items():
            k = k.replace(".feedforward.0.", ".feedforward.fc1.")
            k = k.replace(".feedforward.3.", ".feedforward.fc2.")
            k = k.replace("residual_scorer.0.", "scorer1.").replace(
                "residual_scorer.3.", "scorer2."
            )
            out[k] = v
        return out

    def __call__(self, hidden, input_ids, record: EncodedRecord, lexical_lookup):
        """hidden: (L, H) final backbone states for one record. Returns list of (n_opts,)."""
        h = self.hidden_norm(hidden)
        memory = self.memory_projection(h)[None]
        global_vector = h[-1]
        qs = record.questions
        question_vectors = mx.stack([h[s:e].mean(0) for s, e in (q.question_span for q in qs)])
        type_ids = mx.array([q.question_type for q in qs])

        option_contexts, lexical_options, counts = [], [], []
        for q in qs:
            option_contexts.append(mx.stack([h[s:e].mean(0) for s, e in q.option_spans]))
            lexical_options.append(
                mx.stack([lexical_lookup(input_ids[s:e]).mean(0) for s, e in q.option_spans])
            )
            counts.append(len(q.option_spans))

        option_queries = [
            self.option_context_projection(c)
            + self.option_lexical_projection(lx)
            + self.option_question_projection(question_vectors[i])[None]
            for i, (c, lx) in enumerate(zip(option_contexts, lexical_options, strict=False))
        ]
        routed = mx.concatenate(option_queries, axis=0)[None]
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        routed = routed[0]
        splits = [int(x) for x in mx.cumsum(mx.array(counts))[:-1].tolist()]
        split_options = mx.split(routed, splits, axis=0) if splits else [routed]

        base_fields = self.question_projection(question_vectors)
        summaries = []
        for field, options in zip(base_fields, split_options, strict=False):
            w = mx.softmax((options @ field) / math.sqrt(options.shape[-1]), axis=0)
            summaries.append((w[:, None] * options).sum(0))
        fields = (
            base_fields
            + self.option_summary_norm(mx.stack(summaries))
            + self.global_projection(global_vector)[None]
            + self.type_embedding(type_ids)
        )[None]
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])

        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, math.log(100.0)))
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, math.log(100.0)))
        gate = mx.sigmoid(self.residual_gate)
        logits = []
        for i, (field, lexical, routed_opts) in enumerate(
            zip(fields, lexical_options, split_options, strict=False)
        ):
            anchor = _l2norm(question_vectors[i] + global_vector)
            prior = prior_scale * (_l2norm(lexical) @ anchor)
            options = self.option_norm(routed_opts)
            rf = mx.broadcast_to(field[None], options.shape)
            cosine = (rf * options).sum(-1) / mx.maximum(
                mx.linalg.norm(rf, axis=-1) * mx.linalg.norm(options, axis=-1), 1e-8
            )
            feats = mx.concatenate([rf, options, rf * options, mx.abs(rf - options)], axis=-1)
            residual = self.scorer2(nn.gelu(self.scorer1(feats)))[:, 0]
            logits.append(prior + gate * (joint_scale * cosine + residual))
        return logits


# --------------------------------------------------------------------------- model


class ClefMLX:
    def __init__(self, backbone, tokenizer, head: JointSchemaHead, processor=None):
        self.backbone = backbone
        if type(tokenizer).__name__ == "TokenizerWrapper":  # mlx-lm wrapper
            tokenizer = tokenizer._tokenizer
        self.tokenizer = tokenizer
        self.processor = processor
        self.head = head
        self.vision = hasattr(backbone, "vision_tower")
        lm = backbone.language_model
        self._text_model = lm.model
        self._out = lm.lm_head if hasattr(lm, "lm_head") else lm.model.embed_tokens

    def _lexical(self, ids: mx.array) -> mx.array:
        """Rows of the output-embedding matrix, dequantized if needed."""
        m = self._out
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            return mx.dequantize(
                m.weight[ids],
                m.scales[ids],
                m.get("biases")[ids] if "biases" in m else None,
                group_size=m.group_size,
                bits=m.bits,
                mode=getattr(m, "mode", "affine"),
            )
        return m.weight[ids]

    def _hidden(self, ids: mx.array, media: dict[str, Any] | None) -> mx.array:
        if not self.vision:
            if media:
                raise ValueError("this checkpoint has no vision tower; convert with mlx_vlm")
            return self._text_model(ids[None])[0]
        media = media or {}
        if "pixel_values" in media and "pixel_values_videos" in media:
            raise NotImplementedError("mixing images and videos in one record is not supported")
        pixel_values = media.get("pixel_values", media.get("pixel_values_videos"))
        feats = self.backbone.get_input_embeddings(
            ids[None],
            pixel_values=pixel_values,
            image_grid_thw=media.get("image_grid_thw"),
            video_grid_thw=media.get("video_grid_thw"),
        )
        return self._text_model(
            ids[None], inputs_embeds=feats.inputs_embeds, position_ids=feats.position_ids
        )[0]

    def logits(
        self, record: dict[str, Any], **encode_kwargs
    ) -> tuple[EncodedRecord, list[mx.array]]:
        enc = encode_record(self.tokenizer, record, processor=self.processor, **encode_kwargs)
        ids = mx.array(enc.input_ids)
        hidden = self._hidden(ids, enc.media)
        out = self.head(hidden, ids, enc, self._lexical)
        mx.eval(out)
        return enc, out

    def predict(self, record: dict[str, Any], **kw) -> dict[str, dict[str, float]]:
        enc, logits = self.logits(record, **kw)
        return {
            q.question_id: dict(
                zip(q.option_ids, mx.softmax(lg.astype(mx.float32)).tolist(), strict=False)
            )
            for q, lg in zip(enc.questions, logits, strict=False)
        }

    def systemone(
        self, request: dict[str, Any], max_length: int = 16384, truncate: bool = True
    ) -> dict[str, Any]:
        """Answer a Jev/SystemOne ``POST /v1/systemone`` request body (same response body)."""
        questions = request.get("questions")
        if not isinstance(request.get("model"), str) or "state" not in request:
            raise ValueError("model and state are required")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("at least one question is required")
        for qid, q in questions.items():
            if q.get("type") not in QUESTION_TYPES:
                raise ValueError(f"{qid}: type must be noul, choice, or score")
            if q["type"] != "noul" and not q.get("criteria"):
                raise ValueError(f"{qid}: criteria must not be empty")
        enc, logits = self.logits(request, max_length=max_length, truncate=truncate)
        answers = {
            q.question_id: systemone_answer(
                questions[q.question_id],
                dict(zip(q.option_ids, mx.softmax(lg.astype(mx.float32)).tolist(), strict=False)),
            )
            for q, lg in zip(enc.questions, logits, strict=False)
        }
        return {
            "model": request["model"],
            "answers": answers,
            "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0},
        }


def systemone_answer(question: dict[str, Any], probabilities: dict[str, float]) -> dict[str, Any]:
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(o) for o in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {o: round(probabilities[o], 4) for o in options},
        }
    levels = [str(i) for i in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(sum(i * probabilities[lv] for i, lv in enumerate(levels)), 4),
        "confidence": round(max(probabilities[lv] for lv in levels), 4),
        "legend": dict(zip(levels, question["criteria"], strict=False)),
        "probabilities": {lv: round(probabilities[lv], 4) for lv in levels},
    }


def _has_vision_weights(path: Path) -> bool:
    index = path / "model.safetensors.index.json"
    if index.exists():
        keys = json.loads(index.read_text())["weight_map"]
    else:
        keys = mx.load(str(next(path.glob("*.safetensors"))))
    return any(k.startswith(("vision_tower", "model.visual")) for k in keys)


def load(path: str | Path, head_dtype=mx.bfloat16, backend: str = "auto") -> ClefMLX:
    """Load an MLX Clef checkpoint (local dir or HF repo id).

    backend: "vlm" (mlx-vlm, text + images/video), "lm" (mlx-lm, text only), or "auto".
    """
    path = Path(path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(str(path)))
    if backend == "auto":
        backend = "vlm" if _has_vision_weights(path) else "lm"
    processor = None
    _check_versions(("mlx-vlm",) if backend == "vlm" else ("mlx-lm",))
    if backend == "vlm":
        from mlx_vlm import load as vlm_load

        backbone, processor = vlm_load(str(path))
        tokenizer = processor.tokenizer
    else:
        from mlx_lm import load as lm_load

        backbone, tokenizer = lm_load(str(path))
    cfg = json.loads((path / "joint_head_config.json").read_text())
    head = JointSchemaHead(**cfg)
    weights = JointSchemaHead.sanitize(mx.load(str(path / "joint_head.safetensors")))
    head.load_weights(list(weights.items()), strict=True)
    head.set_dtype(head_dtype)
    mx.eval(head.parameters())
    return ClefMLX(backbone, tokenizer, head, processor)


# --------------------------------------------------------------------------- CLI and server


def _default_model() -> str | None:
    """When run from inside a downloaded repo, default to that repo."""
    # absolute(), not resolve(): Hub cache snapshots are symlinks into a blob store.
    here = Path(__file__).absolute().parent
    return str(here) if (here / "joint_head.safetensors").exists() else None


def _model_name(model: str) -> str:
    """Short response name for a repo id, local directory, or Hub cache snapshot."""
    path = Path(model)
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        return path.parent.parent.name.split("--")[-1]
    return path.name


def _json_arg(value: str) -> Any:
    """A JSON string, a path to a JSON file, or '-' for stdin."""
    if value == "-":
        return json.load(__import__("sys").stdin)
    try:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return json.loads(candidate.read_text())
    except OSError:  # e.g. inline JSON too long to be a file name
        pass
    return json.loads(value)


def _state_arg(value: str) -> Any:
    """State text; JSON objects/arrays are parsed so they are rendered like the reference does."""
    stripped = value.strip()
    if stripped[:1] in "{[":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return value


_REMOTE_IMAGE_MAX_BYTES = 32 * 1024 * 1024
_REMOTE_IMAGE_MAX_REDIRECTS = 3
_REMOTE_IMAGE_TIMEOUT_S = 30.0
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)


def _is_public_address(address: str) -> bool:
    """True for a globally routable address; IPv4 embedded in IPv6 is judged as IPv4."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip in ipaddress.ip_network("64:ff9b::/96"):
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            return _is_public_address(str(embedded))
    return ip.is_global and not ip.is_multicast


def _resolve_public_host(host: str, port: int) -> str:
    """Resolve a host and return an address to connect to; any non-public answer is refused."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise ValueError("could not fetch image URL") from None
    addresses = [str(info[4][0]) for info in infos]
    if not addresses or not all(_is_public_address(address) for address in addresses):
        raise ValueError("image URL must point to a public host")
    return addresses[0]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Connects to a pre-validated address so a later DNS answer cannot change the target."""

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Pinned TLS connection; the certificate is still verified against the URL host."""

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        self._tls_context = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=self._tls_context)
        self._address = address

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._tls_context.wrap_socket(sock, server_hostname=self.host)


def _fetch_remote_image(url: str) -> bytes:
    """Fetch an image URL for the HTTP server without reaching private networks.

    The request body is untrusted, so an unrestricted fetch would let a client probe loopback
    services, link-local metadata endpoints and the private network (SSRF). The first URL and
    every redirect hop must be http(s) without credentials and resolve only to public
    addresses; the connection is pinned to the validated address. Size and time are bounded.
    """
    import time
    import urllib.parse

    deadline = time.monotonic() + _REMOTE_IMAGE_TIMEOUT_S
    for _ in range(_REMOTE_IMAGE_MAX_REDIRECTS + 1):
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
        if parts.scheme not in ("http", "https") or not host or parts.username or parts.password:
            raise ValueError("image URL must be http(s) without credentials")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        address = _resolve_public_host(host, port)
        connection_class = (
            _PinnedHTTPSConnection if parts.scheme == "https" else _PinnedHTTPConnection
        )
        connection = connection_class(host, port, address, _REMOTE_IMAGE_TIMEOUT_S)
        try:
            target = urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, ""))
            connection.request("GET", target, headers={"Accept": "image/*"})
            response = connection.getresponse()
            if response.status in _REDIRECT_STATUSES:
                location = response.getheader("Location")
                if not location:
                    raise ValueError("image URL redirected without a location")
                url = urllib.parse.urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f"image URL returned HTTP {response.status}")
            declared = response.getheader("Content-Length") or ""
            if declared.isdigit() and int(declared) > _REMOTE_IMAGE_MAX_BYTES:
                raise ValueError("image URL response is too large")
            chunks: list[bytes] = []
            total = 0
            while chunk := response.read(65536):
                total += len(chunk)
                if total > _REMOTE_IMAGE_MAX_BYTES:
                    raise ValueError("image URL response is too large")
                if time.monotonic() > deadline:
                    raise ValueError("image URL fetch timed out")
                chunks.append(chunk)
            return b"".join(chunks)
        except (OSError, http.client.HTTPException):
            raise ValueError("could not fetch image URL") from None
        finally:
            connection.close()
    raise ValueError("image URL redirected too many times")


def _decode_image(item: Any):
    """An image from a data URL, raw base64 or an http(s) URL (no local paths over HTTP).

    Remote URLs must point to public hosts; see ``_fetch_remote_image``.
    """
    import base64
    import io

    from PIL import Image

    if not isinstance(item, str) or not item:
        raise ValueError("images must be strings: data URLs, base64, or http(s) URLs")
    if item.startswith(("http://", "https://")):
        data = _fetch_remote_image(item)
    else:
        if item.startswith("data:"):
            item = item.split(",", 1)[-1]
        try:
            data = base64.b64decode(item, validate=True)
        except ValueError:
            raise ValueError(
                "could not decode image: expected a data URL, base64, or http(s) URL"
            ) from None
    return Image.open(io.BytesIO(data)).convert("RGB")


def _cmd_predict(args) -> None:
    import sys

    if args.request:
        request = _json_arg(args.request)
    else:
        if args.state is None or args.questions is None:
            sys.exit("predict needs --state and --questions, or --request")
        request = {"state": _state_arg(args.state), "questions": _json_arg(args.questions)}
    request.setdefault("model", args.name or _model_name(args.model))
    if args.image:
        from PIL import Image

        request["images"] = [Image.open(Path(p).expanduser()).convert("RGB") for p in args.image]
    model = load(args.model)
    try:
        response = model.systemone(
            request, max_length=args.max_length, truncate=not args.no_truncate
        )
    except ValueError as error:
        sys.exit(f"error: {error}")
    print(json.dumps(response, indent=2, ensure_ascii=False))


def _cmd_serve(args) -> None:
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    model = load(args.model)
    served_name = args.name or _model_name(args.model)
    model.systemone(
        {
            "model": served_name,
            "state": "warmup",
            "questions": {"w": {"type": "noul", "instructions": "Is this a warmup?"}},
        }
    )
    lock = threading.Lock()  # one GPU: run requests one at a time
    max_body = args.max_body_mb * 1024 * 1024

    class Handler(BaseHTTPRequestHandler):
        server_version = "clef-mlx"

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str) -> None:
            self._send(status, {"error": {"message": message}})

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": served_name})
            elif self.path == "/v1/models":
                self._send(
                    200, {"object": "list", "data": [{"id": served_name, "object": "model"}]}
                )
            else:
                self._error(404, f"not found: {self.path}")

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._error(404, f"not found: {self.path}")
            length = int(self.headers.get("Content-Length") or 0)
            if length > max_body:
                return self._error(413, f"request body larger than {args.max_body_mb} MB")
            try:
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request body must be a JSON object")
                if request.get("videos"):
                    raise ValueError(
                        "videos are not supported over HTTP; use clef_mlx.py from Python"
                    )
                request.setdefault("model", served_name)
                if request.get("images"):
                    request["images"] = [_decode_image(i) for i in request["images"]]
                truncate = request.pop("truncate", not args.no_truncate)
            except (json.JSONDecodeError, ValueError, OSError) as error:
                return self._error(400, str(error))
            started = time.perf_counter()
            try:
                with lock:
                    response = model.systemone(
                        request, max_length=args.max_length, truncate=bool(truncate)
                    )
            except ContextTooLong as error:
                return self._error(413, f"maximum context length exceeded: {error}")
            except (ValueError, KeyError, TypeError) as error:
                return self._error(400, str(error))
            except Exception as error:
                return self._error(500, f"{type(error).__name__}: {error}")
            response["usage"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self._send(200, response)

        def log_message(self, fmt, *fmt_args):
            if not args.quiet:
                super().log_message(fmt, *fmt_args)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Serving {served_name} on http://{args.host}:{args.port}/v1/systemone "
        f"(truncate={'off' if args.no_truncate else 'on'}, max_length={args.max_length})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    import argparse

    default_model = _default_model()
    parser = argparse.ArgumentParser(prog="clef_mlx.py", description="Run Clef locally with MLX.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument(
            "--model",
            default=default_model,
            required=default_model is None,
            help="Hub repo id or local dir (default: the repo this file is in)",
        )
        p.add_argument(
            "--name", help="model name echoed in responses (default: the model dir/repo name)"
        )
        p.add_argument(
            "--max-length", type=int, default=16384, help="prompt token limit (default 16384)"
        )
        p.add_argument(
            "--no-truncate",
            action="store_true",
            help="refuse over-length inputs instead of truncating the state",
        )

    p = sub.add_parser("predict", help="answer one request and print the SystemOne response")
    common(p)
    p.add_argument("--state", help="state text (JSON objects/arrays are parsed)")
    p.add_argument("--questions", help="questions as JSON, a .json file, or '-' for stdin")
    p.add_argument(
        "--request", help="a full /v1/systemone request body as JSON, a .json file, or '-'"
    )
    p.add_argument("--image", action="append", help="image file to attach (repeatable)")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("serve", help="serve POST /v1/systemone over HTTP")
    common(p)
    p.add_argument(
        "--host", default="127.0.0.1", help="bind address (default 127.0.0.1, local only)"
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-body-mb", type=int, default=64)
    p.add_argument("--quiet", action="store_true", help="no per-request access log")
    p.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
