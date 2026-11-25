# DFlash2 target identity

AX Engine runs DFlash2 block drafting for a small set of admitted base
checkpoints. It selects the drafter by the pack's recorded
`source_model.model_id`, and the drafter reuses the pack's token embedding and
LM head. A pack built from the right weights under a local path or the wrong
ID silently loses that route, so AXQuant checks the source identity when it
converts.

Admitted targets mirrored by this toolkit (AX Engine `929e1f135`, observed
2026-11-24):

| Source model | Drafter |
| --- | --- |
| `Qwen/Qwen3.8-27B` | `z-lab/Qwen3.8-27B-DFlash2` |
| `Qwen/Qwen3.6-35B-A3B` | `incoai/Qwen3.6-35B-A3B-DFlash2` |

The mirror ships as package data (`axquant/data/dflash2_admitted_targets.json`,
schema `axquant.dflash2_admitted_targets.v1`). AX Engine owns admission; the
mirror is a dated copy, not a runtime claim.

## What `convert` and `quantize` do

A checkpoint whose config matches an admitted architecture (model type,
hidden size, layer count, vocabulary and expert count) needs an identity:

| Source | Result |
| --- | --- |
| Local path, no `--model-id` | Fails before conversion |
| `--model-id Qwen/Qwen3.8-27B --revision <40-hex>` | DFlash2 target |
| Admitted ID without an immutable revision | Fails before conversion |
| Another Hub ID, for example a fine-tune | Non-target; AX Engine selects no drafter |
| `--not-dflash2-target "reason"` | Declared non-target; the reason is recorded |

A DFlash2 target must also use a format lane AX Engine loads for it. For
`Qwen/Qwen3.8-27B` these are MXFP4, MXFP8 and affine 6-bit at group size 64.

Example for a local copy of the admitted 27B checkpoint:

```bash
axquant quantize /path/to/Qwen3.8-27B \
  --model-id Qwen/Qwen3.8-27B --revision <immutable-commit> --format mxfp4
```

## The declaration file

Every pack built from a matching checkpoint carries
`axquant_dflash2_target.json` (schema `axquant.dflash2_target.v1`), listed in
`axquant_manifest.json`. It records the status, the mirrored drafter pin, the
trunk format lane, and SHA-256 digests of the stored token embedding and LM
head tensors. Local paths are never written into it.

`publish-prepare` and `publish` refuse a matching MLX pack whose declaration is
missing, unbound, inconsistent with the manifest, pinned to an outdated
drafter, or whose embedding or LM head bytes changed after conversion. CUDA
packs are not checked; AX Engine does not load them.

The declaration is development metadata. It is not evidence that DFlash2
loads, accepts drafts or runs faster with the pack, and it is not a quality
certificate.
