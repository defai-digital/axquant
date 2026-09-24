# Live Hub compatibility verdict: AXQ packs vs MTPLX 2.5.2 / oMLX (2026-09-24)

Status: static-audit + manifest evidence only. No live runtime probe was run
for this write-up (MTPLX and oMLX are not installed on the audit machine);
every runtime claim below stays "unverified pending probe" by design. The
verdict matrix was cross-reviewed by two independent external models, both of
which rejected an earlier draft for static-to-runtime overclaims; the
corrections are incorporated. This is an evidence write-up, not a
certificate, and it implies no MTP acceleration claim.

## Evidence base

1. Live fleet audit (schema `axquant.mtp-hub-fleet-audit.v1`, anonymous
   read-only Hub access, 2026-09-24). The AutomatosX org held 27 public
   repositories; name-based discovery selected the 4 MTP packs and all 4 were
   audited:
   - PASS `AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP@d172f31e` (qwen-resident)
   - FAIL `AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP@0b0bf6c1`: the index
     carries 384 sharded n-gram keys without `ngram-table.safetensors`;
     MTPLX treats them as extraneous parameters (matches the black-box
     diagnostic; the relayout tool is the fix path)
   - PASS `AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP@607a7ba0`
     (qwen-expert-stream)
   - PASS `AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP@90b785d8`
     (qwen-expert-stream)
2. Full 27-repository manifest enumeration (`list_repo_files` per repo, not
   name search): 4 repositories carry an MTP sidecar (the audited set), 1
   carries `ax_expert_stream.json` without a sidecar
   (MiniMax-M3-MXFP4 - invisible to name-based discovery), and 22 are plain
   checkpoints (14 embedding, 4 OCR including one CUDA-AWQ non-MLX artifact,
   and 4 dense/VLM MXFP4 packs with no packaged MTP).

## Compatibility matrix (static evidence only)

| Live pack | MTPLX 2.5.2 | oMLX | Basis and limits |
| --- | --- | --- | --- |
| Qwen3.8-27B MXFP4-MTP | metadata-compatible; sidecar import UNVERIFIED | same; pin the exact oMLX build at probe time | audit PASS + canonical `qwen3-next-mtp`; a header-only audit cannot prove a live import |
| Flash-Next MXFP4-MTP | LM load FAILS today; `relayout-ngram-table` fixes the LM load only - MTP import stays UNCLAIMED (expert-stream, not resident import) | not claimed | audit FAIL + known n-gram root cause |
| Tiel / Cyber-Tiel 35B MXFP4-MTP | UNTESTED (packaging PASS says nothing about runtime load; MTPLX expert-stream support itself unresolved) | UNTESTED | audit PASS covers packaging only |
| MiniMax-M3 MXFP4 (stream, no sidecar) | UNTESTED plain load | UNTESTED | manifest enumeration; no packaged MTP mode |
| 22 plain packs | out of MTP scope; plain MLX load untested here; the CUDA-AWQ OCR pack is not an MLX artifact | same | manifest enumeration |

Layout compatibility is not exactness: the MTPLX Forge baseline stays
unverified for every pack, and no MTP acceleration claim may be published
from this evidence.

## Assurance change shipped with this write-up

The fleet audit's discovery previously matched repository names
(`search=MTP` + `-mlx-axq-` prefix + `-mtp` suffix), which is how the
stream-only MiniMax pack stayed outside the audit. Discovery is now unioned
with a manifest-based pass
(`discover_axq_mtp_artifact_repositories`): every repository under the
author that packages an MTP sidecar artifact (`mtp.safetensors` or
`mtp_head.safetensors`) is audited regardless of its name, so a pack with an
MTP mode cannot silently escape the audit. Stream-only packs have no
packaged MTP mode and correctly stay outside it.

## Outstanding

1. Factory-host runtime probes: MTPLX 2.5.2 sidecar import for the 27B pack;
   plain LM load for Flash-Next (post-relayout), Tiel, Cyber-Tiel, and
   MiniMax-M3; include Flash-Next pre-relayout as the negative control that
   proves the probe detects failure.
2. Decide and document whether MTPLX supports the expert-stream contract at
   all (`unsupported` vs `untested` for the Tiel/Cyber-Tiel/MiniMax rows).
3. Publish relaid-out Flash-Next variants only with a recorded MTPLX load
   receipt; the exactness baseline remains a separate, unresolved workstream.
