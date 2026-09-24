# Tiel / Cyber-Tiel redo checkpoint Tier 1 evidence (2026-09-23)

Measured on `df-macstudio-m2` (Apple M2 Ultra, 192 GiB, Ext16TR0) during the
2026-09-23 re-conversion campaign. AXQuant 1.9.0 at `723eaeb1fb5b24c4243fd243c1959555def70542`
(isolated campaign copy), AX Engine 7.5.4 (`/opt/homebrew/bin`), MLX 0.32.1,
MLX-LM 0.31.3. Driver: factory campaign `tools/tiel_redo_cert.py`; chain log:
[cert-chain.log](cert-chain.log).

Bound packs (same Hub model pages as the 2026-09-19 editions, new commits;
a same-day follow-up metadata revision corrected `target_class` to `MXFP4` —
weights byte-identical, README notice extended):

- `AutomatosX/AX-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` @ `607a7ba019a0b7478f49505d62d69fa1f4164ff5`
- `AutomatosX/AX-Cyber-Tiel-Coder-35B-A3B-MLX-AXQ-MXFP4-MTP` @ `90b785d81d68d022945289a2bb16febe500fd314`

[classfix-report.json](classfix-report.json) records the corrected plan/manifest
digests; [classfix-publication.json](classfix-publication.json) records the
uploaded files verified by SHA-256 at those revisions. Pre-edit files are
backed up on the factory campaign root under `classfix-backup/`.

## Layout

Per model (`tiel/`, `cyber-tiel/`):

- `views/views.json` — the source is evaluated through the layout-gated
  `qwen-quantized-text` prepared view (MTP shard isolated; public MLX-LM would
  otherwise double-shift trunk norms). The pack main weight index does not
  reference the MTP sidecar, so the pack evaluates as-is.
- `quality/` — `evaluate-quality` (seed 20260728, 64 max tokens, 15 samples)
  for reference (pinned oQ6e source) and candidate (pack) on both suites, plus
  `compare-quality` retention output. Campaign datasets under
  `datasets/development-{agent-coding,general}` match `data/eval/{coding,instruction}.jsonl`
  SHA-256.
- `runtime/` — `mlx-lm-text-smoke.json` (greedy arithmetic on the pack),
  `ax-engine-doctor.json`, `ax-engine-bench.json`, `ax-engine-server-response.json`
  (native MTP route, `--mlx-mtp-policy required`), and a server summary.
- `size/` — `artifact-integrity.json` (size + SHA-256 of every manifest-listed
  file; all matched) and `ratios.json`.

## Verdicts

- Tiel: both suites at retention 1.0 (>= 0.98) — **certified**.
- Cyber-Tiel: agent-coding retention 0.9615 (< 0.98), general 1.0 —
  **not certified** (measured and recorded).

MTP acceleration is not certified for either pack: the runs prove native
execution only, with no paired speed workloads. Vision remains
present-not-certified; audio is not supported.
