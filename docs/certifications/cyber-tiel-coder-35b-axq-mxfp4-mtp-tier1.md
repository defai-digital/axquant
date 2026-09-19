# Cyber Tiel Coder 35B-A3B MLX AXQ MXFP4 MTP: certification status

**Checkpoint Tier 1: Not Certified. MTP Tier 2: Not Certified.**

Native MTP execution was observed on MacBook Pro M5 Max 128 GiB using
AX Engine 7.4.0 development binaries whose source content is committed as
`51137c71a6794964d52c32c95e275c0923ed8d0b`. This record updates runtime compatibility evidence;
it does not certify quality retention, MTP acceleration, MTP-S safety, or defaults.

Pack: [`AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP`](https://huggingface.co/AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP/tree/fe05e871ec69ad9ae8eac01fd285555514ac7daf)
at `fe05e871ec69ad9ae8eac01fd285555514ac7daf`.
Machine-readable record: [cyber-tiel-coder-35b-axq-mxfp4-mtp-tier1.json](cyber-tiel-coder-35b-axq-mxfp4-mtp-tier1.json).

## Runtime evidence

The fixed input was token IDs 1 through 16 with a 32-token output cap.
The bench explicitly requested MTP; the server used `--mlx-mtp-policy required`
and `--mlx-mtp-disable-ngram-stacking`. The sidecar has one MTP layer;
the current throughput policy iterates it to a reported draft depth of 3.

| Route | Output tokens | Draft tokens | Verify tokens | Accepted tokens | Direct fallback steps |
| --- | --- | --- | --- | --- | --- |
| Native bench | 32 | 45 | 64 | 13 | 0 |
| HTTP server | 32 | 45 | 64 | 13 | 0 |

All 20 artifact-manifest files for this pack passed size and SHA-256 verification.
See [integrity receipt](evidence/tiel-mxfp4-mtp-m5-20260919/artifact-integrity.json),
[bench response](evidence/tiel-mxfp4-mtp-m5-20260919/cyber-bench.json), and
[server response](evidence/tiel-mxfp4-mtp-m5-20260919/cyber-server-response.json).
The certificate binds these files and the runtime binaries by SHA-256.
The original runs use generic response model IDs; those labels do not identify
the pack. No rendered chat, coding-quality, long-context, or image test is claimed.

## Tier 2 status

Drafting, verification, and accepted tokens prove execution for this probe.
They do not establish the Tier 2 weighted 1.20x / median 1.10x acceleration
thresholds. No paired direct baseline or authorizing workloads were measured.
MTP-S, MTP-P and MTP-D remain unassessed; `auto` is not promoted.

Checkpoint quality retention and uniform-baseline size gates also remain
unassessed. Vision weights are present and hash-verified, not modality-certified.
