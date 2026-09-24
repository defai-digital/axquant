# Tiel Coder 35B-A3B MLX AXQ MXFP4 MTP: certification status

**Checkpoint Tier 1: Certified. MTP Tier 2: Not Certified.**

2026-09-23 re-conversion edition, measured on `df-macstudio-m2` (Apple M2 Ultra,
192 GiB) with AXQuant 1.9.0 at `723eaeb` and AX Engine 7.5.4. This record
certifies checkpoint quality for the redo edition only; it does not certify MTP
acceleration, MTP-S safety, or defaults.

Pack: [`AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP`](https://huggingface.co/AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP/tree/607a7ba019a0b7478f49505d62d69fa1f4164ff5)
at `607a7ba019a0b7478f49505d62d69fa1f4164ff5`. This is the same-day metadata
correction revision: `target_class` in `axquant_plan.json` / `axquant_manifest.json`
was corrected from the allocator-inherited `8bit` to the actual product class
`MXFP4` (weights byte-identical to the first redo revision; see the README
notice on the model page).
Machine-readable record: [tiel-coder-35b-axq-mxfp4-mtp-redo-tier1.json](tiel-coder-35b-axq-mxfp4-mtp-redo-tier1.json).

## Quality evidence

Dual 15-sample development suite through the MLX-LM text route, pack against the
pinned already-quantized oQ6e source (no BF16 reference exists for this repack;
the source loads through the layout-gated MTP-isolated prepared view). Seed
20260728, 64 max tokens per sample.

| Suite | Candidate | Reference (oQ6e source) | Retention | Required | Perplexity ratio |
| --- | --- | --- | --- | --- | --- |
| agent-coding | 0.8667 | 0.8667 | 1.0000 | >= 0.98 | 1.3549 |
| general | 1.0000 | 1.0000 | 1.0000 | >= 0.98 | 1.7308 |

Evidence:
[ref agent-coding](evidence/tiel-redo-20260923/tiel/quality/ref-agent-coding.json),
[cand agent-coding](evidence/tiel-redo-20260923/tiel/quality/cand-agent-coding.json),
[compare agent-coding](evidence/tiel-redo-20260923/tiel/quality/compare-agent-coding.json),
[ref general](evidence/tiel-redo-20260923/tiel/quality/ref-general.json),
[cand general](evidence/tiel-redo-20260923/tiel/quality/cand-general.json),
[compare general](evidence/tiel-redo-20260923/tiel/quality/compare-general.json).

## Size and integrity

`measured_main_bpw` 4.6342 and `measured_total_bpw` 4.9013 (manifest-bound);
candidate weight bytes 22,026,200,211. No matched uniform/BF16 size reference
exists for this already-quantized source class. All 20 manifest-listed files
passed size and SHA-256 verification:
[integrity receipt](evidence/tiel-redo-20260923/tiel/size/artifact-integrity.json).

## Runtime evidence

| Route | Result |
| --- | --- |
| MLX-LM text smoke (pack main index) | pass — greedy 2+2 returned `4`; the pack weight index does not reference the MTP sidecar, so text loads are not norm-shifted |
| AX Engine doctor 7.5.4 | `ready` |
| AX Engine bench, MTP requested | finished, 32 output tokens, qwen-linear MTP route active, zero direct fallback |
| AX Engine server, `--mlx-mtp-policy required` | finished, 32 output tokens, MTP counters positive, zero direct fallback |

Evidence:
[mlx-lm smoke](evidence/tiel-redo-20260923/tiel/runtime/mlx-lm-text-smoke.json),
[doctor](evidence/tiel-redo-20260923/tiel/runtime/ax-engine-doctor.json),
[bench](evidence/tiel-redo-20260923/tiel/runtime/ax-engine-bench.json),
[server response](evidence/tiel-redo-20260923/tiel/runtime/ax-engine-server-response.json).

## Tier 2 status

Native drafting and verification executed with no recorded direct fallback. This
does not establish the Tier 2 weighted 1.20x / median 1.10x acceleration
thresholds: no paired direct baseline or authorizing workloads were measured.
MTP-S, MTP-P and MTP-D remain unassessed; `auto` is not promoted.

Vision weights are present and hash-verified (byte-preserved sidecar), not
modality-certified. Audio is not supported. This pack is a lossy requantization
of an oQ6e source; upstream performance claims do not transfer.
