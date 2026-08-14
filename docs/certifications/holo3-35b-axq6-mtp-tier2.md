# Holo3-35B-A3B AXQ 6-bit-MTP — MTP acceleration Tier 2

**Verdict: not certified.**

This record covers
[`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP)
commit
[`f474549461817cafb73909847af43af2431d4a0d`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit-MTP/tree/f474549461817cafb73909847af43af2431d4a0d).

## Why not certified

Two independent probes on **`df-macstudio-m2`** / AX Engine **6.15.0**:

### 1) Qwen MoE exact-profile probe (decisive)

Same env family as certified Qwen 3.6 35B-A3B Tier 2
(`--qwen36-moe-exact-profile`: certification candidate, async draft, verify-submit
interval 8, pipeline layer, `LINEAR_EXACT_REPLAY=0`), agent-coding subset
(8 prompts, max 128 tokens, 2 measured trials, draft depth 1):

| Gate | Requirement | Result |
| --- | ---: | ---: |
| Greedy exactness | 100% | **Pass** (0 divergent) |
| Draft accept rate | (diagnostic) | **0%** (0 / 128 accepted) |
| Token-weighted decode speedup | ≥ 1.20× | **0.502×** fail |
| Prompt-median speedup | ≥ 1.10× | **0.502×** fail |
| Release ready | true | **false** |

Exactness passes because verify **rejects all drafts** and falls back to direct
tokens. That is safe, but every step pays draft+verify cost → **slower** than
direct. Accept rate 0% is the smoking gun for a **grafted parent head** that
does not match the Holo3 fine-tune trunk.

### 2) Soft `mtp-diagnose` kill-switch matrix

| Profile | Exactness | Speedup (approx.) | Accept |
| --- | --- | ---: | ---: |
| baseline | pass | ~0.45× | 0 / 14 |
| disable-post-input-metal | pass | ~0.45× | 0 / 14 |
| disable-la-decode-metal | pass | ~0.46× | 0 / 14 |

Kill-switch variants did not recover accepts or speedup.

## Decision (improve or stop?)

| Path | Chance of Tier 2 |
| --- | --- |
| More env / host tuning only | **Low** — 0% accept under formal MoE exact env already |
| Full M5 authorizing scoreboard without new weights | **Low** — same grafted head |
| Train / adapt MTP on Holo3 (or co-trained head) | **Real chance**, high cost |
| Ship `-MTP` as sidecar asset only | **Current product stance** |

**Recommendation:** stop optimizing this graft for Tier 2; keep **non-MTP** as
primary SKU; keep `-MTP` optional with acceleration **not claimed**.

## Related

- Checkpoint Tier 1: [holo3-35b-axq6-mtp-tier1.md](holo3-35b-axq6-mtp-tier1.md)
- Evidence: [evidence/holo3-35b-axq6-mtp-tier2/](evidence/holo3-35b-axq6-mtp-tier2/)
- Probe script: [`scripts/run_holo3_35b_mtp_tier2_probe.py`](../../scripts/run_holo3_35b_mtp_tier2_probe.py)
- Machine-readable: [holo3-35b-axq6-mtp-tier2.json](holo3-35b-axq6-mtp-tier2.json)
