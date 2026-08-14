# Holo3-35B-A3B AXQ 6-bit-MTP — MTP acceleration Tier 2

**Verdict: not certified.**

This record covers
[`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP)
commit
[`f474549461817cafb73909847af43af2431d4a0d`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP/tree/f474549461817cafb73909847af43af2431d4a0d).

## Why not certified

Soft `axquant mtp-diagnose` on **`df-macstudio-m2`** / AX Engine **6.15.0**
(short probe set, 2 measured trials, draft depth 1):

| Profile | Exactness | Prompt-median speedup | Token-weighted decode speedup | Release ready |
| --- | --- | ---: | ---: | --- |
| baseline | pass | ~0.43× | ~0.45× | no |
| disable-post-input-metal | pass | ~0.43× | ~0.45× | no |
| disable-la-decode-metal | pass | ~0.44× | ~0.45× | no |

Gates require exactness 100%, token-weighted decode speedup ≥ 1.20×, and
prompt-median speedup ≥ 1.10×. Speedup failed (MTP slower than direct on this
host/probe). MTP is a **grafted parent Qwen3.5 head**, not co-trained on Holo3.

Product default remains **direct decode**.

## Related

- Checkpoint Tier 1: [holo3-35b-axq6-mtp-tier1.md](holo3-35b-axq6-mtp-tier1.md)
- Machine-readable: [holo3-35b-axq6-mtp-tier2.json](holo3-35b-axq6-mtp-tier2.json)
