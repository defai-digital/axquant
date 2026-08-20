# DeepSeek V4 Flash-0731 — large-memory Tier 1 on `tn-macstudio-m3`

**Host:** `tn-macstudio-m3` (observed hostname `localadacStudio`, Apple M3 Ultra, 512 GB).  
**Engine:** AX Engine **7.1.x**. Latest published release is **v7.1.5**.  
**OS gate:** AX Engine 7.1.x requires **macOS 26 (Tahoe)+**. This Studio is currently **15.5** — preflight fails closed until the OS is upgraded.

Do not use this host for flagship Qwen convert. Factory convert/cert for packs that fit in 192 GB remains `df-macstudio-m2`. This host is the **scoped recert** machine for Flash-0731 SKUs that OOMed or could not generate on 192 GB.

## Packs

| `--pack` | Hub leaf | Recipe | Notes |
| --- | --- | --- | --- |
| `axq2` | `…-2bit-MTP` | `examples/deepseek-v4-experimental-2bit-v0.1.yaml` | Hub pack exists; factory 15+15 viability 0.633 on m2 (AX Engine 7.1.5 native recert, still below 0.90) |
| `axq4` | `…-4bit-MTP` | `examples/deepseek-v4-experimental-4bit-g128-v0.1.yaml` | Local g128 pack on Ext12T; Hub still a stub |
| `mxfp4` | `…-MXFP4` | `examples/deepseek-v4-experimental-mxfp4-v0.1.yaml` | Not converted; g32 class ~179 GB |
| `axq6` | `…-6bit` | `examples/deepseek-v4-experimental-6bit-g128-v0.1.yaml` | Not converted; estimated 200 GB+ |

3-bit remains withdrawn.

## Bootstrap

```bash
# on tn-macstudio-m3
export HF_HOME=$HOME/.cache/huggingface
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CACHE=$HF_HOME/xet
unset HF_HUB_ENABLE_HF_TRANSFER

cd ~/code/axquant
PYTHONPATH=src .venv/bin/python scripts/run_deepseek_v4_0731_tier1.py preflight
# expected today: exit non-zero, macos 15.5 < 26

# after macOS 26+:
PYTHONPATH=src .venv/bin/python scripts/run_deepseek_v4_0731_tier1.py install-engine
PYTHONPATH=src .venv/bin/python scripts/run_deepseek_v4_0731_tier1.py --pack axq2 all
```

Override engine: `AX_ENGINE_RELEASE=v7.1.5` (default).

If the Hub 4-bit repo is still a stub, copy the factory g128 pack onto this
host into `~/models/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-4bit-MTP/` before
`--pack axq4`.

Copy factory datasets (76+44 generation-viability jsonl) to `~/axquant-certification/datasets/`.

## Gates

Experimental Flash track: **generation viability ≥ 0.90** on agent-coding and general (not BF16 retention), mlx-lm smoke, AX Engine doctor/generate-manifest on 7.1.x. MTP acceleration is still unclaimed until a separate Tier 2 A/B.
