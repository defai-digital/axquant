# DeepSeek V4 Flash-0731 — complete-assignment 2-bit v0.5

| Field | Value |
| --- | --- |
| Status | Development recipe + factory runbook; **not** a certification |
| Target host | `df-macstudio-m2` (Apple M2 Ultra, 192 GB) + Ext12T |
| Date | 2026-08-20 |
| Protocol | Factory `v-extract` (same as AXQ T1): seed `20260728`, coding 384 / general 64, suite system prompts, DSV4 non-thinking, greedy |

Product question: the market 2-bit pack clears the `v-extract` 0.90 floor at
combined **0.967** while AXQ uniform v0.1 scores **0.887**. Isolated one-axis
AXQ raises (attention 6-bit DWQ or affine, shared experts 4-bit) all scored
**worse** than uniform. The remaining untested hypothesis is the **complete
assignment** — attention 6 / shared 8 / LM-head 8 / routed 2 — re-derived as
an AXQ-owned manual recipe.

Recipe: [`examples/deepseek-v4-experimental-2bit-complete-v0.5.yaml`](../examples/deepseek-v4-experimental-2bit-complete-v0.5.yaml)

| Role | Bits | Method | Notes |
| --- | --- | --- | --- |
| Routed trunk (experts + remaining MLP) | 2 | affine g32 | experimental 2-bit trunk |
| Attention | 6 | affine g32 | raised from the 4-bit floor |
| Shared experts | 8 | affine g32 | fire every token |
| LM head | 8 | affine g32 | **AXQ-026 governed opt-in** (`lm_head_min_bits: 8`); default stays BF16 |
| Embeddings, routers | 8 | affine g32 | protection floors unchanged |
| Norms | 16 | bf16 | protection floor unchanged |
| MTP | 16 | byte-preserved sidecar | AXQ convention; acceleration not claimed |

## Clean-room basis

The role-level bit classes above are public facts from the market pack's model
card. The recipe, script, and any measurements are AXQ-owned; no competitor
code, calibration data, or metadata is used. This follows the boundary set in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the
[v-extract record](deepseek-v4-flash-0731-optiq2-v-extract.md).

## Factory job

Orchestrator (runs only on `df-macstudio-m2`):

```text
scripts/run_deepseek_v4_0731_axq2_complete.sh
```

Steps: `plan-manual` from the existing Flash-0731 inventory → role/bit sanity
print (including `constraints.lm_head_min_bits`) → `convert
--allow-unmeasured --ax-engine-manifest skip` → T1 `v-extract` through
[`scripts/run_deepseek_v4_0731_axq2_axengine.py`](../scripts/run_deepseek_v4_0731_axq2_axengine.py)
with `DSV4_QA_PROTOCOL=v-extract`. The pack is development evidence; it is not
uploaded or certified from this run.

## Acceptance gate and next step

| Outcome (v-extract combined) | Decision |
| --- | --- |
| ≥ 0.90 | Complete assignment was the lever. Defer plan-search productization; treat measured sensitivity and fused-expert refinement as ceiling-raisers, not blockers. Record the result in a comparison doc and consider the 2-bit SKU for the certification track. |
| < 0.90 | Assignment alone is insufficient. Next AXQ-owned levers, in order: (1) measured fused-module sensitivity ranks (`STREAMING_PARTIAL` probes, `MEASURED_DEVELOPMENT` only) feeding `plan-experimental-mix`; (2) fused-trunk DWQ clip at 2-bit; (3) per-channel AWQ scales for `SwitchLinear`. GPTQ on fused stacks stays out of scope (O(in³) on fused down-proj). |

Either way this run is a measured protocol record, not a certificate, and it
does not certify the OptiQ pack or adopt any of its artifacts.

Related: [uniform v0.1 vs OptiQ on v-extract](deepseek-v4-flash-0731-optiq2-v-extract.md),
[mixed v0.1](deepseek-v4-flash-0731-optiq2-vs-axq2-mixed.md),
[experimental trunk mix](experimental-trunk-mix.md),
[known issues](known-issues.md).
