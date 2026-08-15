# Qwen3-Coder-Next AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-10.
**MTP acceleration Tier 2 is not applicable** (source declares no MTP).

This certificate covers
[`AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-4bit)
commit
[`53dce509aa115e7fae583516b494a5dafebf31a9`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-4bit/tree/53dce509aa115e7fae583516b494a5dafebf31a9).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3NextForCausalLM` (hybrid MoE, no MTP) |
| Product class | `4bit` |
| Source | `Qwen/Qwen3-Coder-Next@a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` |
| Candidate manifest SHA-256 | `0cabf5e744e2bd1a24c5a525dd4b71bb7193a26883c182c102b2852962e944b5` |
| Plan evidence | `architecture_prior` |
| Measured total BPW | `4.7977524983638125` |
| Certification host | `df-macbookpro-m5` |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | ≈ target `4.8` | `4.7977524983638125` | Pass |
| Weight-size ratio vs uniform | ≤ `1.15` | `1.065515` | Pass |
| General quality retention | ≥ `0.98` | `1.000000` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.022727` | Pass |
| MLX-LM runtime | load + smoke | Pass | Pass |

Candidate weight bytes `47,782,251,237` vs uniform reference
`44,844,286,500` (`mlx-community/Qwen3-Coder-Next-4bit`) → **1.0655×**.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Agent-coding (76) | `0.868421` | `0.888158` | `1.022727` | `0.978543` |
| General (44) | `1.000000` | `1.000000` | `1.000000` | `1.016376` |

Seed `20260728`, max gen 64, host `df-macbookpro-m5`, AXQuant `1.6.1`.
Quality is measured against the matched uniform quantized reference (not BF16).

## Tier 2 status

**Not applicable.** Qwen3-Coder-Next has no declared MTP weights; this certificate
is non-MTP direct-decode checkpoint Tier 1 only. No speculative-decode speedup
claim is authorized.

## Related

- Sibling 6-bit: [qwen3-coder-next-axq6-tier1.md](qwen3-coder-next-axq6-tier1.md)
- Certification index: [README.md](README.md)

Machine-readable: [qwen3-coder-next-axq4-tier1.json](qwen3-coder-next-axq4-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `not-applicable` | `false` | vision not supported (no tower config and no sidecar weights) |
| Audio | `not-applicable` | `false` | audio not supported (no tower config and no sidecar weights) |
