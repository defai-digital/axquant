# Holo3-35B-A3B AXQ 4-bit — checkpoint Tier 1 certification

**Verdict:** certified for AXQuant checkpoint Tier 1 on 2026-08-14.
**MTP acceleration Tier 2 is not applicable** (no MTP weights).

This certificate covers
[`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit)
commit
[`7b2256130cd55ea6b7489817a9a00c46e9874403`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit/tree/7b2256130cd55ea6b7489817a9a00c46e9874403).

## Bound artifact

| Property | Value |
| --- | --- |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (35B-A3B MoE) |
| Product class | `4bit` (attention **6-bit** / experts **4-bit** recovery layout) |
| Source | `Hcompany/Holo3-35B-A3B@208d5ae3a03f99d561f32ab5e606f73397a390ea` |
| Candidate manifest SHA-256 | `34e1fffbd7f27caa22276d950db85e49c8628930369e8bdf5d30f84a05dc5852` |
| Plan | `plan-manual` + [`examples/holo3-35b-axq4-agent-v0.1.yaml`](../../examples/holo3-35b-axq4-agent-v0.1.yaml) |
| Measured total BPW | `5.665439451180904` (includes BF16 vision) |
| Certification host | `df-macstudio-m2` |
| Adapter | `qwen35-moe-v1` (not Qwen 3.6 cert track) |

## Certification results

| Gate | Requirement | Result | Verdict |
| --- | ---: | ---: | --- |
| Measured total BPW | product class `4bit` | `5.665439` | Pass |
| Weight-size ratio vs uniform-4 | ≤ `1.15` | `1.146910` | Pass |
| General quality retention | ≥ `0.98` | `1.048780` | Pass |
| Agent-coding quality retention | ≥ `0.98` | `1.006897` | Pass |
| AX Engine / MLX-LM runtime | load + eval | Pass | Pass |

Candidate weight bytes `24,862,201,695` vs uniform-4 reference `21,677,556,069` → **1.1469×**.

### Quality suites

| Profile | Reference | Candidate | Retention | Perplexity ratio |
| --- | ---: | ---: | ---: | ---: |
| Agent-coding (76) | `0.953947` | `0.960526` | `1.006897` | `0.922347` |
| General (44) | `0.931818` | `0.977273` | `1.048780` | `0.951576` |

Seed `20260728`, max gen 64, host `df-macstudio-m2`, AXQuant `1.6.2`.

### Recovery note

The first `architecture_prior` 4-bit pack failed **agent-coding** retention
(`0.9793`, long_context `0.875`). Certified pack raises **attention to 6-bit**
while keeping **experts at 4-bit**. An attention-8 trial cleared quality but
exceeded the size gate (`1.162` &gt; `1.15`).

## Tier 2 status

**Not applicable.** No MTP weights.

## Related

- Sibling 6-bit: [holo3-35b-axq6-tier1.md](holo3-35b-axq6-tier1.md)
- Runbook: [../holo3-35b-axq-dev-runbook.md](../holo3-35b-axq-dev-runbook.md)

Machine-readable: [holo3-35b-axq4-tier1.json](holo3-35b-axq4-tier1.json).

## Modalities (capability-gated)

Text checkpoint Tier 1 does **not** imply vision or audio quality. `Vision present=true` on a pack is not a quality pass.

| Modality | Claim | Supported | Reason |
| --- | --- | --- | --- |
| Vision | `present-not-certified` | `true` | vision present sidecar=['vision.safetensors']; mlx-vlm smoke failed on df-macstudio-m2 (mlx-vlm expects vision_tower.*; sidecar/layout mismatch). Text Tier 1 unchanged. Evidence: docs/certifications/evidence/modality-recert-capability-gated/results/AX-Holo3-35B-A3B-MLX-AXQ-4bit.json |
| Audio | `not-applicable` | `false` | audio not supported (no tower config and no sidecar weights) |
