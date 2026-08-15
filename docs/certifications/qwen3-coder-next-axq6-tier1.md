# Qwen3-Coder-Next AXQ 6-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-10.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This certificate covers
[`AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit)
commit
[`c6f3ae556f95ce13b7d319486ad2d4d753726216`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit/tree/c6f3ae556f95ce13b7d319486ad2d4d753726216).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3NextForCausalLM` (hybrid MoE, no MTP) |
| Product class | `6bit` |
| Source | `Qwen/Qwen3-Coder-Next@a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` |
| Candidate manifest SHA-256 | `590757bf1f3825f48a5ec931a07e8ecfbbe2e5da46d466e39873947789fce35a` |
| Plan evidence | `architecture_prior` |
| Measured total BPW | `5.998995986154411` |
| Certification host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `6.0` | `5.998995986154411` | Pass |
| Weight-size ratio vs uniform | ≤ `1.1` | `0.922716` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.000000` | Pass |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `59,745,794,198` vs uniform reference
`64,749,929,465` (`AutomatosX/AX-Qwen3-Coder-Next-MLX-6bit`) → **0.9227×**.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Agent-coding (76) | `0.901316` | `0.901316` | `1.000000` | `0.997531` |
| General (44) | `1.000000` | `1.000000` | `1.000000` | `1.000736` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.
Quality is measured against the matched uniform quantized reference (not BF16).

## Tier 2 status

**Not applicable.** Qwen3-Coder-Next has no declared MTP weights; this certificate
is non-MTP direct-decode checkpoint Tier 1 only. No speculative-decode speedup
claim is authorized.

## Related

- Sibling 4-bit: [qwen3-coder-next-axq4-tier1.md](qwen3-coder-next-axq4-tier1.md)
- Certification index: [README.md](README.md)

Machine-readable: [qwen3-coder-next-axq6-tier1.json](qwen3-coder-next-axq6-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `not-applicable` | `false` | vision not supported (no tower config and no sidecar weights) |
| Audio | `not-applicable` | `false` | audio not supported (no tower config and no sidecar weights) |
