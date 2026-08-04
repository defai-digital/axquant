# Qwen 3.6 27B sensitivity recovery status

**Name:** Qwen 3.6 27B sensitivity recovery status (SR0–SR4)
**Program:** `.internal/product/sensitivity-recovery-prd.md`, decisions AXQ-030/AXQ-031
**Not:** "certified" or "release-ready" — this candidate does not pass formal dual-profile
validation. See Bottom line.

## Bottom line

SR1 (sensitivity regeneration) and the MTP-speed portion of SR2 are done cleanly, on real,
release-quality evidence, on the correct formal host (mbp-m5). Of the two independent quality-gate
gaps found during agent-coding profile validation, one (missing required eval-task-category
coverage) was closed this session by authoring a proper eval dataset; the other (perplexity
relative increase exceeding the 3% threshold) held up under a much larger, better-substantiated
re-measurement and converged toward the original AXQ-026 session's own historical number — it is a
real, consistent, credible finding, not a dataset artifact. `refine-select` correctly refuses to
select a non-passing candidate, which structurally blocks SR3 (hardware-registry,
compatibility-matrix, pareto) and SR4 (release-audit) — those commands consume `refine-select`'s
output and were never reached. This is the pipeline's fail-closed design working as intended, not
an incomplete run.

## SR1 — sensitivity regenerated (done)

- `axquant analyze` rerun for `Qwen/Qwen3.6-27B` @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`,
  profile `agent-coding`, methods `affine,dwq`, **`--token-budget 8192`** (the default 2048 is
  insufficient for release-quality evidence — see gotcha #14 below).
- Result: 1199/1199 tensors, `evidence_kind=measured` (release-quality).
- `stable_sha256`: `3b4b9ceef4e87daaa7556f4d4fc72e8c944165b934f9616c453512695c766385`
  (does **not** match the lost original's `analysis_sha256`
  `3248b9dd05d1ba2dc98e9f35516ab1aedb1fefcd12f1ebf424b585405e53651e` — expected and fine per
  AXQ-030; not required to match).
- Archived per AXQ-031 to `/Volumes/Ext4T/qwen36-v1-axq026-recovered-sr1/
  qwen36-27b-sensitivity-recovered-v2.json` on the primary development machine (M3 Max), dated
  2026-08-03/04.
- A first attempt at the default token budget produced `evidence_kind=measured_development`
  (not release-quality); a second attempt crashed on a real bug (see gotcha #13). Both discarded;
  this is the third, clean attempt.

## SR2 — measurement set (partially complete: MTP passes, quality validation does not)

### Candidate

- `cand-0000-000`: `refine --target-bpw 5.3 --lm-head-floor 8bit` (governed 8-bit LM-head floor
  added to `refine` this session — it previously only existed on `plan`). Selected over `refine`'s
  own proxy-quality pick (`cand-0000-002`, 6.65 BPW) because AXQ-026's actual target is the size
  gate, not proxy-optimal quality.
- Converted: `measured_total_bpw=5.2944124728985065`, `effective_bpw=5.294331719957949`.
- Size gate: candidate weight_bytes 18,385,792,333 vs uniform-4bit reference weight_bytes
  16,903,941,980 (`.internal/tmp/qwen36-v1-final-uniform4-size-evidence.json`, reused —
  `logical_parameters` matches exactly, 27,781,427,952) → ratio **≈108.8%**, under the 110% cap.

### MTP speed gate — PASS (this is the original blocker; it is resolved)

Measured on **mbp-m5** (M5 Max), the host all historical AXQ-026 MTP numbers came from — a first
measurement on the local M3 Max gave a false-low 1.05× (host/thermal artifact, not a real
regression; see gotcha #15) and was discarded in favor of the correct host.

| Arm | Speedup | Gate (1.20×) |
| --- | ---: | --- |
| Candidate (agent-coding workload) | 1.3623× | PASS |
| Candidate (general workload label, same data) | 1.3658× | PASS |
| Reference (agent-coding workload) | 1.7012× | PASS |
| Reference (general workload label, same data) | — | PASS |

`exactness_pass=true`, zero failed/divergent trials on every run. `ax_engine_version` resolved to
`6.12.1-5-g42bf581e` (local `~/code/ax-engine` HEAD, copied to mbp-m5 — see gotcha #16).

### Quality validation — FAIL (one real, well-substantiated reason; one gap closed)

`axquant validate` for `agent-coding` profile: `passed=false`.

**Missing required task-score categories — RESOLVED.** `thresholds_for(AGENT_CODING).
required_task_scores = ("coding", "tool", "json", "multilingual", "long_context")`. The original
60-task suite only tagged `coding` from that list; the properly domain-tagged dataset the
original AXQ-026 session used could not be located anywhere (traced by `dataset_sha256`, no
local `.jsonl` matches — same "referenced but gone" pattern AXQ-030/031 exist to prevent, for an
eval dataset instead of a sensitivity report). Fixed by clean-room-authoring four new 15-task
category files this session: `data/eval/{json,tool,multilingual,long_context}.jsonl` (committed,
see below), bringing the combined eval dataset to 120 tasks across 8 categories. All five required
categories now score 1.0 for both candidate and reference.

**Perplexity relative increase — real, remains failing, now well-substantiated.**
`max_perplexity_relative_increase=0.03` (3%). On the original 60-task dataset this measured 21.9%;
re-measured on the expanded 120-task dataset it dropped to **9.9%** — a large swing purely from
sample size, confirming the original number was significantly inflated by eval-set noise. Task
score retention stayed excellent throughout (1.0042 aggregate on the 120-task set — candidate
still slightly exceeds reference). The 9.9% figure is now much closer to, and consistent with, the
original AXQ-026 session's own comparable BF16-vs-final-candidate ratio (**8.9%**, also over the
3% threshold, from `.internal/tmp/qwen36-bf16-agent-coding-quality-current.json` /
`qwen36-v1-final-cand-000-agent-coding-quality.json` — a different eval dataset, so not an exact
match, but the same order of magnitude). Two independent measurements, on two different
candidates, two different eval methodologies, both land at 9-10% against a 3% threshold. This is
now a **real, consistent, credible finding**: a ~5.3 BPW governed-floor candidate for this model
does not clear a 3%-relative-perplexity budget against a near-BF16 reference. Per
`release-best-practices.md` §3/PRD §6.2, this is not a threshold to relax or a suite to keep
padding — it is the honest measured outcome.

### Pipeline stop point

`refine-measure` succeeds and honestly records `validation_passed=false` for the agent-coding
measurement, both on the original 60-task evidence (`sr2/refinement_measurements.json`,
`measurement_id=cand-0000-000-agent-coding`) and the expanded 120-task re-measurement
(`sr2/refinement_measurements-v2.json`, `measurement_id=cand-0000-000-agent-coding-v2`).
`refine-select` fails closed both times: `"no complete-model candidate passed validation"` — by
design, it will not select a candidate that didn't pass. SR3/SR4 were never reached; nothing
downstream of `refine-select` was skipped or faked, it structurally cannot run without a selected
candidate.

## SR3 / SR4 — not reached

Blocked on the above. No hardware-registry, compatibility-matrix, pareto, or release-audit
artifacts exist for this lineage.

## What would change the outcome

The eval-dataset gap is closed (see above); the perplexity gap is the sole remaining blocker, and
it is now measured consistently enough across two independent sessions/candidates/dataset sizes
that it reads as a real property of this precision class rather than an artifact:

- **Accept it and report non-certification.** This is the honest, currently-supported conclusion:
  a governed-8-bit-LM-head ~5.3 BPW Qwen3.6-27B candidate clears the MTP speed and size gates but
  does not clear a 3%-relative-perplexity quality budget against a near-BF16 reference, on two
  independent measurements (9.9% here, 8.9% historically).
- **`cand-0000-001` (5.571 BPW) was tried and is worse, not better — this path is closed.**
  Converted, evaluated, and validated 2026-08-04: weight-size ratio **114.45%** (fails the 110%
  gate — this candidate sits right on AXQ-026's original unmodified policy floor, ~5.577 BPW,
  which is exactly why the 8-bit LM-head governance existed in the first place), perplexity
  relative increase **12.36%** (worse than cand-0000-000's 9.9%), and a newly-seen failure not
  present on cand-0000-000: **MTP acceptance-rate drop 2.10%** exceeds the 2.00% threshold. MTP
  speedup itself still passes (1.34×). Three independent gate failures, none of them measurement
  noise (the size ratio in particular is an exact byte count, not subject to sampling variance).
  Evidence: `sr2/candidate-cand-0000-001/`, `sr2/eval/candidate001-*`,
  `sr2/eval/validate-cand001-agent-coding.json`. `cand-0000-002` (6.653 BPW) was not tried; given
  the trend (more BPW here made things worse, not better) it is not expected to help and was not
  pursued.
- Neither of these is a "fix the number until it passes" move — the first is honest reporting: per
  `release-best-practices.md` §3 / PRD §6.2 claim guardrails, a real gate failure is reported as
  exactly that, not padded away by re-running the eval suite indefinitely looking for a smaller
  ratio. The dataset expansion done this session was legitimate evidence-quality work (closing a
  suite that was provably incomplete against the profile's own declared requirements), not
  threshold-shopping — it was run once, the result recorded, and not repeated looking for a better
  number.

## Operational gotchas discovered this recovery (additive to tech-spec §8, items 1–12)

13. **`compute_cosine_distance` (`probe.py`) can return a small negative value for high-dimension
    near-identical vectors** (float rounding on `dot()/  (norm*norm)` fractionally above 1.0),
    which fails `MetricVector.cosine_distance`'s `ge=0.0` schema constraint. Hit at tensor 265/1199
    after ~4h of compute. Fixed: clamp to `max(0.0, 1.0 - similarity)`. Commit `a6463ff`.
14. **`analyze --token-budget` defaults to 2048, which caps `probe.py`'s release-evidence token
    accounting below `_MIN_RELEASE_CALIBRATION_TOKENS=8192`**, silently producing
    `evidence_kind=measured_development` instead of `measured` — `refine` then refuses it
    ("not release quality"). Always pass `--token-budget 8192` (or higher) for release-bound
    `analyze` runs, regardless of how large the underlying calibration cache actually is.
15. **MTP speedup measured on a non-formal host can be dramatically low, not just noisy.** A
    measurement on a local M3 Max gave 1.05×; the identical candidate on mbp-m5 (M5 Max, the
    project's formal host) gave 1.34–1.36×. Always measure MTP speed on mbp-m5, never as a
    convenience substitute on whatever machine is running the orchestration.
16. **A `target/release/ax-engine-bench` binary can be silently ABI-incompatible with the
    machine's *current* MLX runtime**, not just "possibly stale" — an old binary (compiled against
    MLX 0.31.2 headers) crashed every single trial against a machine now running MLX 0.32.0
    (`thread 'main' panicked ... MLX version mismatch`), rather than degrading gracefully. A
    binary built on a *different* machine but the *same* architecture (both Apple Silicon here)
    can work fine if it happens to target a compatible MLX version — worth trying before installing
    a full Rust toolchain to rebuild locally.
17. **`software_versions.ax_engine` only resolves from a binary path if the binary's parent
    directory name is *exactly* a version string** (regex `^v?\d+\.\d+\.\d+(?:[-+]...)?$`,
    `axquant/versioning.py`). `/tmp/ax-engine-6.12.1-5-g42bf581e/ax-engine-bench` does **not**
    resolve (extra `ax-engine-` prefix on the directory name); `/tmp/6.12.1-5-g42bf581e/
    ax-engine-bench` does. `doctor --json` did not populate an `install.version` field for this
    binary either, so the standalone-directory path was the only working resolution route.
18. **`--quality-evaluation` on `benchmark-ab` and `--candidate-size`/`quality`/`validation` inputs
    to `refine-measure` compare the *full* `ModelIdentity` for equality, including `local_path`.**
    Evidence generated against the same model on two different machines (different absolute paths)
    will not cross-match even with identical `model_id`/`revision`. Regenerate every identity-bound
    artifact on whichever single machine will consume it downstream, rather than mixing machines
    mid-lineage.
19. **`refine-measure`/`build_complete_candidate_measurement` requires
    `artifact.profile == plan.profile == validation.profile` exactly.** A single candidate/plan
    (built under one profile, e.g. `agent-coding`) cannot be fed a *different* profile's
    `ValidationReport` through `refine-measure` — "dual-profile" release evidence is not "one
    candidate measured twice under two profiles" at this layer; it is assembled later, at
    `validation-index`/`release-audit`, from independently-produced per-profile validate reports.
20. **`benchmark-ab` evaluation bundles require `hardware.power_mode`** (non-empty string) for
    `refine-measure` to accept them; omitting `--power-mode` produces a bundle that validates fine
    standalone but fails `refine-measure` with `"validation is missing hardware.power_mode"`. Use
    `pmset -g | grep powermode` on the host and pass e.g. `--power-mode AC-powermode-2`.
21. **`refine-select` fails closed with `"no complete-model candidate passed validation"`** when
    every measured candidate's `validation_passed=false` — by design, not a bug. This is the
    correct place to stop and report "not yet certified" rather than force a selection.

## Archive record (AXQ-031)

- Sensitivity report + `analyze.log`: `/Volumes/Ext4T/qwen36-v1-axq026-recovered-sr1/`
- Converted candidate, refine outputs, eval/benchmark/validate evidence:
  `/Volumes/Ext4T/qwen36-v1-axq026-recovered-sr1/sr2/`
- Archived 2026-08-04, on the same durable external volume used for the historical AXQ-026
  candidate artifacts (`/Volumes/Ext4T/qwen36-v1-axq026-*`).
