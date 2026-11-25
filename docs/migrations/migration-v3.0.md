# Migrating to AXQuant 3.0

AXQuant 3.0 writes only **MLX MXFP4, MXFP8, BF16** and **CUDA NVFP4, BF16**.
Existing packs, certificates and artifact schemas stay readable; only what the
toolkit produces changed.

## Recipes and plans

| 2.x recipe or plan content | 3.0 |
| --- | --- |
| 4-bit `method: affine` rule (converted with `--q-mode mxfp4`) | Write `method: mxfp4`, group 32. `plan-manual` refuses 4-bit affine rules. Existing plan files with 4-bit affine allocations still convert, stored as MXFP4 |
| 8-bit `method: affine`, group 64 | Use group 32; it is stored as MXFP8 |
| 8-bit `method: affine`, group 32 | Unchanged; stored as MXFP8 (no `--allow-unmeasured` needed for MXFP8) |
| 2-, 3- or 6-bit rules | Not supported. Use MXFP4 with measured role raises to MXFP8, or MXFP8 |
| `awq`, `gptq`, `gptq-act`, `dwq` | Not supported |
| No `hardware:` block | Add one that lists `mxfp4` and group 32 when a rule uses `method: mxfp4` |

A manual recipe hardware block for MLX packs:

```yaml
hardware:
  name: ax-engine-apple-silicon-mxfp
  runtime: ax-engine
  supported_bits: [4, 8, 16]
  supported_methods: [affine, mxfp4, bf16]
  supported_group_sizes: [32]
```

## Command line

| 2.x | 3.0 |
| --- | --- |
| `--allow-legacy-4bit` | Removed |
| `quantize --format 2bit|3bit|6bit` | Removed; use `mxfp4` or `mxfp8` |
| `convert --q-mode affine` | Rejected; `mxfp4` / `mxfp8` are accepted and ignored |
| `convert --q-mode mxfp8` for 8-bit floors | Not needed; 8-bit is always MXFP8 |
| `convert --calibration-activations` | Removed (AWQ/GPTQ) |
| `--ladder refine-awq-dwq` | Removed |
| `plan-experimental-mix` | Removed |
| `quantize-mtp-sidecar` | Removed |
| `quantize-cuda --q-mode mix6` | Removed; CUDA is NVFP4 only |
| `axquant name --target-class 4bit` | Use `MXFP4`, `MXFP4-attn8` or `MXFP8` |

## Output names

Default output directories and `axquant name` now name the storage format:
`AX-<model>-MLX-AXQ-MXFP4`, `AX-<model>-MLX-AXQ-MXFP4-attn8`,
`AX-<model>-MLX-AXQ-MXFP8`. `MXFP4-attn8` is an MXFP4 trunk whose attention
projections are raised to MXFP8 after a measured study.

## Pack metadata

The top-level `quantization` default in `config.json` is now the pack's
dominant quantized lane (for example `bits: 4, group_size: 32, mode: mxfp4`).
Every quantized module still carries its own override. 2.x packs built with
`--q-mode mxfp8` carry an invalid `bits: 4, mode: mxfp8` default; rebuild them
from the BF16 source rather than editing metadata.

## Unchanged

Inspection, `analyze` probes, `validate`, `release-audit` and `verify-cert`
read affine packs as before. Hub repositories keep their names; affected
packs are rebuilt and uploaded as new revisions.
