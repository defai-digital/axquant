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
  - `control-ab/` — **control experiment**: the Tier-2-certified Qwen 3.6
    35B-A3B AXQ-6bit/4bit-MTP packs re-run through the identical harness on
    the same 7.5.4 binary. On 7.5.4 the certified packs also fail greedy
    exactness (AXQ6 general-long; AXQ4 both profiles), isolating an
    engine-version regression from the MXFP4 pack format.
  - `determinism-probe/` — `scripts/probe_mtp_determinism.py`: same prompt
    repeated 8x per arm for all four packs. All 256 generations are internally
    bit-deterministic; cross-arm divergence is fixed per prompt (e.g. always
    token 21 for cyber-tiel agent-coding prompt 1): 9 exact / 7
    verify-differs / 0 nondeterministic.
  - `engine-version-bisect/` — `ax-engine-bench` built from source worktrees
    at release tags (v7.1.1, v7.3.0, v7.4.0, v7.5.0; control pack also run on
    v7.3.0/v7.4.0/v7.5.0). Key results: engines <= v7.4.0 never load these
    packs' MTP head (`raw_hf_delta` metadata vs already-converted norms; only
    the 7.5.x hash-bound compatibility shim loads it); v7.5.0 loads it with
    strong speedups (tiel 1.30x/1.14x, cyber-tiel 1.31x/1.23x) but this pack
    is inexact on both profiles, while the certified AXQ control pack stays
    exact on agent-coding there. The general-long regression is engine-wide
    across 7.3.0-7.5.4.
  - `host-comparison.json` — per-pack, per-profile gate results for both hosts
    plus the written findings, revised root-cause statement, and the full
    engine-version bisect matrix.

## Key findings

1. **The primary root cause is an engine regression, not the host and not
   purely the pack.** On the same 7.5.4 binary: the certified Qwen 3.6
   35B-A3B AXQ control packs fail greedy exactness (AXQ6 general-long; AXQ4
   both profiles) — those packs passed full Tier 2 on engine 6.14.1. The MXFP4
   Tiel/Cyber-Tiel packs fail more profiles, so pack/format modulation exists
   on top, but the verify-path row-exactness defect is engine-wide on 7.5.4.
2. **The failure is deterministic, not run-to-run noise.** The determinism
   probe repeats each prompt 8x per arm on all four packs: every generation is
   internally bit-identical, and cross-arm divergence occurs at a fixed token
   position for a given (pack, prompt) — the verify path computes
   deterministically different logits than the direct path on specific
   prompts (near-tie flips). Different hosts/chips shift which prompts flip.
3. **Speed floors are secondary.** Only cyber-tiel agent-coding clears 1.20x
   weighted / 1.10x median on the factory host (1.2088x/1.2088x; 1.3778x on
   m5). General-long stays under 1.20x on both hosts (1.01x–1.20x).
4. **Transient factory OOM.** One tiel agent-coding MTP trial on the factory
   host died with `[METAL] … Insufficient Memory` despite ~160 GB free unified
   memory; recorded in the comparison artifact (`failed_trial_count: 1`). It
   does not change the verdict (exactness already fails).
5. **Harness semantics** (shared with all published Tier 2 records): the
   benchmark harness runs `warmup_trials + measured_trials` generations,
   rotating through the seed-deterministically-shuffled suite prompts — 1+2
   trials exercises two measured prompts per profile.

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
