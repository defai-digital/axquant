# AX Expert SSD Stream v1

AXQuant can describe fused MoE expert stacks that AX Engine should page from SSD one layer at a
time. The native path is intended for Super-class checkpoints whose quantized expert table is
larger than the target Mac's unified memory. Qwen3.8-2.4T-A95B is the first thin-track adapter.

This document expands the shared v1 contract. AXQuant emits the artifact metadata; successful
loading of a real Qwen 3.8 pack remains gated on the corresponding AX Engine implementation and
runtime validation. Outputs are development evidence and carry no quality or certification claim.

## Convert controls

Both conversion entry points accept:

```text
--expert-stream {off,auto,required}
```

The default is `auto`.

- `auto` emits `ax_expert_stream.json` when conversion produces packed expert stacks. It marks
  the manifest `required=true` when estimated full residency exceeds 256 GiB.
- `required` requires a packed expert inventory, emits the manifest, and always marks it required.
- `off` suppresses the manifest for ordinary packs. It is rejected for
  Qwen3.8-2.4T-A95B because that Super-class pack cannot be run safely as a resident load, even on
  a 512 GB Mac.

Examples:

```bash
axquant convert \
  --model /models/Qwen3.8-2.4T-A95B \
  --plan qwen38-plan.json \
  --expert-stream required \
  --output ./AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit

axquant quantize /models/packed-moe-bf16 \
  --expert-stream auto \
  --output ./packed-moe-axq
```

Conversion remains fail-closed when the plan does not cover every module. The stream option does
not relax tensor classification, quantization, or converted-weight verification.

## Emitted files

`ax_expert_stream.json` sits beside `config.json` and the Safetensors shards. The artifact's
`axquant_runtime.json` and embedded runtime metadata record:

```json
{
  "expert_stream": "required",
  "expert_stream_manifest": "ax_expert_stream.json"
}
```

The policy value is `required`, `optional`, or `off`. The pointer is present only when a stream
manifest exists.

The manifest envelope is `axquant.expert-stream.v1` with compatibility class `operational` and
freeze policy `additive-ok`. Its v1 mode is only `layer-stack`; consumers must reject unknown
versions or modes.

Each tensor record contains:

- its sanitized runtime tensor name and existing repo-relative shard filename;
- layer number and projection (`gate_up`, `gate`, `up`, or `down`);
- expert axis `0`, expert count, bits, and group size.

Weights and their quantization sidecars are listed separately so the runtime can exclude every
streamed array from its initial resident map. A shard may contain several layers or resident and
streamed tensors. Consumers must open only the named tensor slices.

## v1 runtime shape

The v1 unit of paging is a layer's fused expert stack `[experts, out, in]` plus its packed
quantization views. AX Engine pages that stack, executes its existing gathered MoE matrix path,
then evicts it according to the runtime cache budget. Embeddings, attention, routing state,
shared experts, norms, output head, and MTP state remain resident.

When `required=true`, AX Engine must reject a load unless expert streaming is explicitly enabled.
It must never fall through to a full resident load. Optional manifests allow an operator to opt
into the same layer-stack path on smaller packed MoE artifacts.

## Storage invariants

- Conversion preserves the normal Safetensors sharding layout. v1 does not create one file per
  expert and does not reshard an artifact around routing granularity.
- Tensor names are runtime/sanitized module paths, not raw upstream names.
- Streamed expert arrays are absent from the runtime's initial resident weight map.
- The existing disk prefix cache remains a KV-cache feature and is unrelated to expert paging.
- The implementation is independent; no external streaming implementation is imported, vendored,
  translated, or copied.

## Current Qwen 3.8 status

The adapter `qwen38-moe-v1` recognizes only the Qwen3.8-2.4T-A95B text MoE identity under
`model_type=qwen3_5_moe_text` or `qwen3_5_moe`. It is a thin, development-only conversion track
with no certification track. Qwen 3.6 35B-A3B remains owned by `qwen36-v1`.

The currently published experimental Qwen 3.8 serving artifacts use a separate OptiQ path. They
are not AXQ artifacts and must not be loaded by AX Engine. See
[Qwen 3.8 experimental OptiQ packs](qwen38-optiq-experimental.md).
