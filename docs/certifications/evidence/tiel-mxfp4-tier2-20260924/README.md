# Tiel / Cyber-Tiel MXFP4 MTP Tier 2 evidence — 2026-09-24

Two hosts, one question: can the 2026-09-23 redo MXFP4 packs carry a scoped MTP
acceleration (Tier 2) certificate? Answer: no — greedy exactness fails on both
packs on both hosts. This directory holds the measured evidence.

## Layout

- `factory/` — **authoritative measurements** on `df-macstudio-m2` (Mac Studio
  M2 Ultra, 192 GB) with the release AX Engine 7.5.4 benchmark CLI
  (`ax-engine-bench`, sha256 `eeb40413…`). Per pack: the A/B technical summary,
  both profile comparison artifacts, the exact dataset copies fed to the
  harness, and the run log. These are the numbers recorded in the public Tier 2
  records (`*-redo-tier2.json`).
- `comparison-df-macbookpro-m5/` — **development comparison evidence** on
  `df-macbookpro-m5` (MacBook Pro M5 Max), requested by the operator. Not
  certification authority; the certification contract restricts new Tier 2
  records to the factory host.
  - `engine-754/` — same packs, datasets, seed, runtime profile, and the same
    7.5.4 release `ax-engine-bench` binary (hash-identical, staged under
    `tiel-tier2/engine-754/` on that host) via
    `scripts/run_tiel_35b_mxfp4_tier2_mtp_ab.py` with `TIEL_TIER2_HOST` set.
  - `engine-dev-build/` — raw MTP-on logs from a first m5 round using the
    in-tree dev build `v7.5.4-40-g875da6b5`. Every trial shows
    `mtp.requested=true, mtp.available=false`
    (`qwen_linear_mtp_exact_eligible=0`, `mtp_model_gate_default_present=0`):
    that engine build never activates MTP for these packs. Recorded as an
    engine-side regression candidate for the >=7.5.5 development line.
  - `host-comparison.json` — per-pack, per-profile gate results for both hosts
    plus the written findings.

## Key findings

1. **Exactness is the blocker, and it is pack-inherent.** On the factory host,
   MTP-vs-direct greedy outputs diverge (tiel: 1/2 and 2/2 divergent measured
   trials; cyber-tiel general-long: 2/2). On m5 with the identical engine binary
   the divergent trials and first-divergence token positions differ, which is
   the signature of numeric (reduction-order) divergence in the MXFP4
   multi-token verify path, not a deterministic logic bug and not a host
   defect. The grafted oQ6e-source MTP head on a requantized MXFP4 trunk does
   not meet the 1.0 greedy-exactness contract.
2. **Speed floors are secondary.** Only cyber-tiel agent-coding clears 1.20x
   weighted / 1.10x median (1.2088x/1.2088x factory; 1.3778x/1.3778x m5).
   General-long stays under 1.20x on both hosts (1.01x–1.20x).
3. **Transient factory OOM.** One tiel agent-coding MTP trial on the factory
   host died with `[METAL] … Insufficient Memory` despite ~160 GB free unified
   memory; recorded in the comparison artifact (`failed_trial_count: 1`). It
   does not change the verdict (exactness already fails).
4. **Harness semantics** (shared with all published Tier 2 records): the
   benchmark harness runs `warmup_trials + measured_trials` generations,
   rotating through suite prompts — 1+2 trials exercises the first three
   prompts of each 15-prompt suite (2 measured prompts per profile).

## Reproduce

Factory authority run:

```bash
export PATH="/opt/homebrew/bin:$PATH"   # AX Engine 7.5.4 bench CLI
.venv/bin/python scripts/run_tiel_35b_mxfp4_tier2_mtp_ab.py --pack both
```

Comparison host (development evidence only):

```bash
export PATH="$HOME/tiel-tier2/engine-754/bin:$PATH"
export TIEL_TIER2_HOST=df-macbookpro-m5
~/tiel-tier2/venv/bin/python scripts/run_tiel_35b_mxfp4_tier2_mtp_ab.py \
  --pack both --models-root ~/tiel-tier2/models \
  --datasets ~/tiel-tier2/datasets --work ~/tiel-tier2/work754
```
