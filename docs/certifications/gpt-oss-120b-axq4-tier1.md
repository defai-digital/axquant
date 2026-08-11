# GPT-OSS 120B AXQ 4-bit — evaluation record (not certified)

**Verdict:** **not certified** for AXQuant checkpoint Tier 1 (evaluated 2026-08-10).
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).
**Public Hub status:** the development pack
`AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit` remains on Hugging Face as **development evidence only** after
this evaluation — AXQuant will not publish or certify this class until agent-coding
quality clears the gate.

This record is kept only as an internal/public explanation of *why* the 4-bit
120B product class is not available. It is **not** a certificate and must not be
cited as a supported download.

## Bound artifact (historical evaluation)

| Property | Value |
| --- | --- |
| Architecture | `GptOssForCausalLM` (MoE, no MTP) |
| Product class | `4bit` |
| Source (convert input) | `mlx-community/gpt-oss-120b-MXFP4-Q4@bce781bef0f2fc85ed4e575af74054f5aad73ddd` |
| Upstream lineage | OpenAI `gpt-oss-120b` |
| Former Hub commit (deleted) | `7e0f77ed63c0fb83d0fcc57d84b3018f269ec8f3` |
| Measured total BPW | `4.800009864578611` |
| Evaluation host | `df-macbookpro-m5` |

## Evaluation results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `4.8` | `4.800010` | Pass |
| Weight-size ratio vs MXFP4-Q4 | ≤ `1.20` | `1.124620` | Pass |
| General quality retention | ≥ `0.98` | `1.002438` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `0.952000` | **Fail** |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `70,097,638,062` vs MXFP4-Q4 reference `62,330,057,589` → **1.125×**.

### Quality suites

| Profile | Retention | Gate |
| --- | ---: | --- |
| Agent-coding | `0.952` | Fail (&lt; 0.98) |
| General | `1.002` | Pass |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.

## Remarks — why this pack cannot be certified

1. **Checkpoint Tier 1 requires agent-coding retention ≥ 0.98** against the matched
   `mlx-community` MXFP4-Q4 reference. The evaluated 4-bit re-pack scored **0.952**
   on the agent-coding suite while general quality still passed.
2. **Size and load are not enough.** The pack met BPW/size and MLX-LM smoke gates.
   AXQuant still refuses certification when any required quality profile fails.
3. **Plan path was `architecture_prior` MXFP4 re-pack**, not the later manual
   agent-coding recipe used for the certified **120B 6-bit** pack (experts 6-bit,
   attention 8-bit, no 4-bit trunk). A future 4-bit recovery would need a new plan,
   re-measurement, and a clean certificate — not republication of this revision.
4. **Hub removal policy.** Because certification is blocked and the pack was only
   development evidence, the public repository was deleted so users are not offered
   an uncertified 120B 4-bit download. Prefer the certified
   [120B 6-bit](gpt-oss-120b-axq6-tier1.md) product, or rebuild privately from source
   for research without public claims.

## Tier 2 status

**Not applicable.** No MTP.

## Related

- Certified sibling: [gpt-oss-120b-axq6-tier1.md](gpt-oss-120b-axq6-tier1.md)
- 20B 4-bit (certified recovery recipe): [gpt-oss-20b-axq4-tier1.md](gpt-oss-20b-axq4-tier1.md)
- Certification index: [README.md](README.md)

Machine-readable: [gpt-oss-120b-axq4-tier1.json](gpt-oss-120b-axq4-tier1.json).

## Recert attempts (2026-08-11)

Further manual recovery recipes (attention 8-bit + expert 4-bit; early-layer 8-bit
boosts) were run on `df-macbookpro-m5` and did **not** clear agent-coding retention
≥ 0.98 (best prior ~0.952). Operator decision: **skip 120B 4-bit certification**.

## Recert decision (2026-08-11)

Additional recovery recipes on `df-macbookpro-m5` did not clear agent-coding retention
≥ 0.98. **Further 120B 4-bit certification is skipped.**
