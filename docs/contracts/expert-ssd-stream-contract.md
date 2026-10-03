# AX Expert SSD Stream — shared contract (v1)

**Status:** implemented and merged (2026-08-23). This document is the normative v1
contract; the original cross-repo work plan is superseded.
**Goal:** Run Super-class MoE (Qwen3.8-2.4T-A95B) on Macs whose unified memory is smaller than the full 2-bit expert table (~0.8 TB). A 512 GB Mac cannot resident-load that pack.
**Non-goal:** Copy, translate, or vendor mlx-optiq. Clean-room implementation only.
**v1 scope:** *layer-stack paging* — page one MoE layer's fused expert stack from SSD, run the existing `gather_qmm`, then evict. Not per-expert unfused GEMM (v2).

## Why this shape

AX Engine already executes fused MoE via `gather_qmm` on packed tensors
`[num_experts, out, in]`. Per-expert streaming would require a new kernel path
and an unfused layout (ADR-0005 deferred that). Layer-stack paging reuses the
current kernel: peak RAM ≈ resident trunk + one layer's experts + KV.

For a 2.4T / ~80-layer model at 2-bit, one layer is on the order of several GB,
not 0.8 TB. That is what makes 192–512 GB Macs viable.

## On-disk contract

AXQuant writes `ax_expert_stream.json` next to `config.json` / weight shards.

```json
{
  "schema_version": "axquant.expert-stream.v1",
  "generated_by": "axquant",
  "required": true,
  "mode": "layer-stack",
  "num_experts": 256,
  "experts_per_tok": 8,
  "estimated_resident_bytes": 40000000000,
  "estimated_full_resident_bytes": 800000000000,
  "estimated_max_layer_expert_bytes": 10000000000,
  "resident_roles": ["embedding", "attention", "router", "shared_expert", "norm", "lm_head", "mtp"],
  "streamed_roles": ["expert"],
  "tensors": [
    {
      "name": "model.layers.0.mlp.switch_mlp.gate_proj.weight",
      "file": "model-00001-of-00080.safetensors",
      "layer": 0,
      "proj": "gate_up",
      "expert_axis": 0,
      "num_experts": 256,
      "bits": 2,
      "group_size": 64
    }
  ]
}
```

Rules:

- `required=true` means AX Engine **must** stream. Loading the pack without
  `--stream-experts` / `AX_STREAM_EXPERTS=1` is a hard error. Do not silently
  attempt a full resident load (that OOMs or swap-thrashes).
- `mode` v1 is only `"layer-stack"`. Unknown modes fail closed.
- `tensors[].name` is the **runtime / sanitized MLX module path** AX Engine
  already uses (`switch_mlp.*` / packed expert roles), not the raw HF name.
- `tensors[].file` is repo-relative. One safetensors file may hold many layers;
  the engine must load **only the named tensors** from that file, never
  `eval` the whole file.
- `proj` is one of: `gate_up`, `gate`, `up`, `down`.
- `expert_axis` is 0 for packed `[E, out, in]` (and packed quantized views).
- Resident roles stay in unified memory for the process lifetime.
- Streamed roles are **absent** from the initial `load_weights` map.

Also record a pointer on the existing artifact manifest:

- `RuntimeMetadata.memory_policy["expert_stream"] = "required" | "optional" | "off"`
- `RuntimeMetadata.memory_policy["expert_stream_manifest"] = "ax_expert_stream.json"`

Do **not** bump `axquant.artifact.v2` in a breaking way. Additive memory_policy
keys are enough for v1.

## Runtime behavior (AX Engine)

- Parse `ax_expert_stream.json` from the model directory. Unknown
  `schema_version` / `mode` is a hard error.
- Streaming is enabled by `--stream-experts` (serve/load) or
  `AX_STREAM_EXPERTS=1`. If the manifest says `required=true` and streaming
  is off, fail closed (`ExpertStreamRequired`, reporting
  `estimated_full_resident_bytes`); never fall through to full
  `load_weights`. If streaming is on and no manifest is present, fail
  closed; do not guess.
- Initial load builds the weight map from **resident tensors only**,
  skipping names listed in the stream manifest without `eval`. An
  expert-only safetensors file is not opened at init.
- The pager caches one layer stack at a time (key `(layer_idx, proj)`,
  budget `max(1 layer, AX_STREAM_EXPERT_LAYERS)`), constructs the same
  `QuantizedWeight` the resident path would have built, reuses the existing
  `gather_qmm` path unchanged, and evicts LRU stacks.
- Serve/doctor surfaces stream mode as
  `expert_stream: {enabled, required, resident_bytes, max_layer_bytes, cached_layers}`.
- `qwen3_5_moe_text` is accepted as the Qwen 3.8 text MoE family; it must
  never be silently treated as Qwen 3.6 35B-A3B.

## Do not

- Copy, translate, or vendor mlx-optiq (clean-room implementation only).
- Change dense / Qwen 3.6 default load: streaming stays off unless a stream
  manifest is present.
- Use `AX_MMAP_WEIGHTS` as a substitute (it still materializes every
  tensor).
- Reshard weights into one-file-per-expert in v1.
