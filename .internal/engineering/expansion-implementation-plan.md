# AXQuant Expansion Implementation Plan

**Document status:** Active  
**Derived from:** `expansion-prd.md`, `expansion-technical-specification.md`, AXQ-017–AXQ-022  
**Last reviewed:** 2026-07-31

Phases are dependency-ordered. A phase is complete only when its exit criteria
pass and the full quality gates (`pytest`, `ruff check`, `ruff format --check`,
`mypy src` strict) are green.

Status summary (2026-07-31): E0–E4 are implemented, including the wave-2
toolkit items (release bundle packaging, remote `hf://` resolution per
AXQ-023, `support-matrix`), the wave-3 measured KV probing path (AXQ-024),
and the wave-4 measured-KV release chain plus head-to-head renderer
(AXQ-025, AXQ-022), all green under the full quality gates (`pytest` 351
passing, `ruff check`, `ruff format --check`, `mypy src` strict). Every
KV-planning contract decision is resolved; E5/E6 certification waves are
evidence programs and remain the only open work.

## Phase E0 — Governance (complete)

Deliverables: `expansion-prd.md`, ADRs AXQ-017–AXQ-022, the expansion technical
specification, and this plan.

## Phase E1 — Tiers and adapter framework (complete)

Work items:

1. `SupportTier` enum + `ArchitectureProfile.support_tier` (additive).
2. `architectures/dense_family.py`: `DenseFamilySpec`, `DenseFamilyAdapter`,
   shared fail-closed classification table factored from `Qwen36Adapter`.
3. Registry: ordered specs, all-matches collection, ambiguity error.
4. Family specs: `qwen35-dense-v1`, `gemma4-dense-v1`, `minicpm5-dense-v1`,
   `nemotron3-dense-v1` at `inspect-only`.
5. Tier enforcement: converter preflight refusal below `convertible`;
   publisher refusal below `certified`; inventory/report surfacing.
6. Tests per tech-spec §7.

Exit criteria: E1 rows of the capability truth table; Qwen 3.6 fixture
behavior byte-identical apart from the added tier field.

## Phase E2 — One-command conversion (complete)

Work items:

1. `quantize` subcommand orchestrating inspect → tier gate → plan (default
   recipe or priors) → convert → optional runtime smoke → summary.
2. Development-evidence labeling in summary, manifests, and `--json` output.
3. Docs: README "Quick start" becomes the quantize path; staged pipeline moves
   under "Release workflow".

Exit criteria: one command converts a Tier-convertible synthetic fixture end
to end in tests; summary carries the development-evidence sentence; no new
claim surface (publish still refuses quick-mode output).

## Phase E3 — Recipe bundles (complete, including wave 2)

Wave 2 (2026-07-31) additionally shipped: `publish-prepare` packages a release
recipe bundle with lineage digests; remote `hf://OWNER/REPO@REVISION[/PATH]`
resolution per AXQ-023; and the registry-derived `support-matrix` command.

Work items:

1. `axquant.recipe-bundle.v1` schema + digest/identity verification.
2. `quantize --recipe` local resolution; lineage recorded in the artifact
   manifest.
3. `publish-prepare` bundle export for certified releases.

Exit criteria: measured-evidence inheritance and tamper detection covered by
tests; a bundle exported from release evidence replays into an identical plan.

## Phase E4 — KV-cache groundwork (complete, including measured probing)

Wave 3 (2026-07-31) shipped AXQ-024: `analyze-kv` measured per-layer KV
sensitivity over verified calibration caches, `allocate_kv_cache_measured`
with digest-bound plans, and conversion acceptance of bound measured KV plans.
Wave 4 (2026-07-31) shipped AXQ-025: `convert --kv-sensitivity` packages the
bound report; publication re-verifies the digest and reproduces the exact
allocation from packaged evidence. It also shipped the AXQ-022 `head-to-head`
comparison-page renderer for E-T7.

Work items:

1. `KvCachePlan` / `KvLayerAllocation` additive schema.
2. Prior-based `allocate_kv_cache` with cover/floor invariants.
3. AX Engine runtime metadata emission + advisory MLX-LM fallback.
4. `plan`/`quantize` flags to opt in (`--kv-cache prior`), default off.

Exit criteria: plans without the section byte-identical to today; measured
basis rejected; metadata emission tested.

### E5 evidence log (2026-08-01, Apple M3 Max, 128 GB)

Real-evidence runs performed with the shipped tooling against the pinned BF16
Qwen3.6-27B source (`6a9e13bd…`), artifacts under `.internal/tmp/`:

- Three head-to-head pages rendered from checksum-verified bound benchmark
  evidence indexes (`head-to-head-{agent-coding,general,agent-cand002}-v1.md`).
- First real recipe bundle `qwen36-27b-measured-r1` exported from the
  measured selection plan (6.0293 BPW) and round-trip verified against the
  real source inventory.
- First real KV sensitivity report (`qwen36-kv-sensitivity-agent-v1.json`,
  agent-coding, 144 samples): exactly 16 of 64 layers are standard attention
  (indices 3, 7, …, 63); measured 4-bit output-KL median 0.139 (max 0.560 at
  layer 3), 6-bit median 0.022. Early attention layers are the most
  sensitive. Caveat: single-batch measurement; per-layer 6/8-bit ordering is
  noisy — raise the token budget before drawing per-layer conclusions.
- First digest-bound measured KV plan (budget 0.05): early attention layers
  keep BF16 KV, mid layers 6-bit, late layers 4-bit; the AXQ-025 publication
  gate reproduced the allocation from the packaged report.
- The real run also surfaced and fixed a hybrid-architecture defect the
  synthetic fixtures could not (commit 9c5379d): recurrent-cache layers are
  now fail-closed unsupported, and quantized-KV forwards use fake-quant
  KV caches because this mlx-lm version does not execute the packed
  QuantizedKVCache path for this family.

- Second real KV report (`qwen36-kv-sensitivity-general-v1.json`, general
  profile, 24 samples): same 16 attention layers; 4-bit KL median 0.0094 —
  ~15× lower than agent-coding, with the last layer most sensitive instead of
  the earliest. Workload-dependent KV sensitivity is measured fact.
- Full one-command real conversion (`quantize --recipe`): the r2 bundle
  (measured weights + measured KV) converted the BF16 source into a 19 GiB
  artifact at measured 6.0001 BPW (plan 6.0000), byte-identical MTP sidecar,
  packaged `kv_sensitivity.json`, per-layer KV table in the runtime metadata,
  and a passing real `mlx_lm.generate` smoke. The AXQ-025 gate reproduced the
  KV allocation from the packaged report on the real artifact.
- The failed first attempts were themselves evidence: the stale-tier refusal
  and the missing `quantize --kv-sensitivity` flag were found by real runs
  and fixed in commit 84a720e; the legacy raw MTP sidecar (old-format
  provenance) is presented without a provenance file, which the converter
  documents as the development-conversion path.

Remaining blockers are unchanged and external to the toolkit: the M2 MTP
speed floor (AX Engine runtime) and the named-approval size exception.

### AXQ-026 candidate evidence log (2026-08-01, Apple M3 Max, 128 GB)

The size direction was resolved the same day: the workspace owner approved the
governed 8-bit LM-head floor (AXQ-026), and the first size-gate-passing
candidate was produced end-to-end with the shipped tooling. All results below
are development evidence pending the formal supported-host suite.

- Targeted probe extension on the pinned BF16 source added the missing
  `lm_head.weight` 8-bit/group-64 measurement to the release lineage
  (`qwen36-sensitivity-release-lmhead8-v1.json`, base
  `qwen36-sensitivity-release-dwq-v1.json`, same 8,192-token protocol):
  measured output-KL **0.000097**, task-loss delta 0.003977 — far below
  ordinarily accepted trunk-tensor sensitivities, directly supporting the
  AXQ-026 quality thesis. Evidence kind stays `measured`.
- The real run surfaced two probe defects the synthetic fixtures could not,
  both fixed with regression tests: (1) the base-report architecture contract
  treated `support_tier` as evidence, so any base probed before a tier
  promotion was rejected (AXQ-017 violation — the tier is current policy);
  (2) pre-AXQ-017 reports hash an inventory serialization with no
  `support_tier` key, so the historical-hash reconstruction failed on every
  pre-expansion report; the loader now reproduces that exact legacy byte
  contract before failing closed.
- `plan --lm-head-floor 8bit --kv-cache measured` at target 5.30 produced the
  first release-lineage plan under the size gate: effective 5.2999933 BPW,
  `constraints.lm_head_min_bits=8`, LM head at 8-bit with the AXQ-026 reason,
  measured KV allocation identical to the E5 run (5×4-bit / 5×6-bit / 54×BF16).
- Atomic conversion to `/Volumes/Ext4T/qwen36-v1-axq026-candidate`: 22 files,
  measured **5.3000740** total BPW (plan drift +0.00008), byte-preserved MTP
  sidecar with the new `mtp_norm_layout: raw_hf_delta` declaration, packaged
  `kv_sensitivity.json`, per-layer KV table in the runtime metadata, and a
  generated AX Engine manifest.
- Real size evidence: **18,405,453,035** weight bytes against the audited
  uniform-4 reference 16,903,941,980 — ratio **1.088826**, the first candidate
  to pass the 110% size gate.
- AX Engine doctor and MLX-LM generation smoke both pass on the real artifact.
- Dual-profile development quality on the real artifact (same datasets, seeds,
  and generation contract as the BF16 references):
  - agent-coding (52 tasks): aggregate retention **1.0099** (candidate 0.9776
    vs BF16 0.9679), perplexity ratio 0.9991, `json_validity` 0.95 = reference,
    `syntax_validity` 0.90 vs 0.80 — every category at or above BF16.
  - general (new 16-task suite with the governed structured pairs, evaluated
    on both BF16 and the candidate): aggregate retention **1.0000** with
    identical per-task scores, perplexity ratio 0.9702, and `json_validity` =
    `syntax_validity` = 1.0 on both sides. This is the first measurement of
    the general structured-output coverage the 2026-07-31 formal audit
    flagged as missing (BF16 reference:
    `qwen36-bf16-general16-quality.json`).
- End-to-end MTP A/B smoke on the AX Engine HEAD build (post 1de301bd, the
  per-sidecar norm decision + `mtp_norm_layout` contract): depth 1, greedy,
  2 trials, exact profile. **Exactness passes with zero divergent trials and
  acceptance is 0.9381** (45.5 average accepted tokens, zero kernel
  fallbacks) on the byte-preserved raw sidecar that 6.11.1 loaded at 0/40
  acceptance — the declarative layout contract is proven on the real
  artifact, and the 8-bit LM head does not degrade the MTP verify path.
  The 0.816x smoke speed is a busy-interactive-host number with no formal
  standing; the 1.20x M2 gate is measured only on the idle formal host.
- A second conversion of the same plan with `--mtp-layout ax-engine-qwen36-v1`
  (`/Volumes/Ext4T/qwen36-v1-axq026-candidate-prepared`, identical
  5.3000740 BPW, provenance v3, `mtp_norm_layout: mlx_multiplier`) gives the
  formal host a path that works on the **existing notarized 6.12.1 runtime**
  without waiting for an AX Engine release: doctor passes and the depth-1
  greedy A/B smoke on that runtime passes exactness with the identical
  0.9381 acceptance and zero kernel fallbacks. Both formal-path options are
  therefore verified — prepared layout + notarized 6.12.1, or byte-preserved
  + a future release carrying the `mtp_norm_layout` loader.
- First M5 formal-host speed measurement attempt (2026-08-01, idle Apple M5
  Max, notarized 6.12.1, 18 agent prompts, 512 tokens, depth 1, only
  `AX_MLX_QWEN_LINEAR_MTP_EXACT=1`): exactness passed with acceptance 0.9615
  but the ratio was 0.8925, with rollback wall averaging ~22 ms/cycle. The
  follow-up investigation proved this measurement was **not protocol-valid**:
  same-artifact/same-binary/same-host discriminator runs showed the formal
  cand-002 tree also degrades to ~26 ms/cycle rollback without the formal
  runner's full environment, and recovers to ~0.9 ms/cycle with it. The
  formal exact-profile contract requires the full eight-variable set from
  `run_qwen36_v1_recovered_fast_formal.zsh` (`AX_MLX_MTP_BYPASS_MIN_SAMPLES`,
  `AX_MLX_MTP_DRAFT_MIN_CONFIDENCE`, `AX_MLX_MTP_LINEAR_EXACT_REPLAY`,
  `AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL`,
  `AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS`,
  `AX_MLX_SPECULATIVE_INVARIANT_PROJECTIONS=all`,
  `AX_MLX_SPECULATIVE_ROW_EXACT_POST_INPUT`,
  `AX_MLX_SPECULATIVE_SPLIT_FFN`), not the exact flag alone — without them
  the verifier falls off the invariant-projection graph and every cycle pays
  an expensive adoption path.
- With the full formal environment, per-cycle discriminator costs on M5 are
  **better for axq026 than for cand-002**: rollback 0.5 vs 0.9 ms, draft 4.2
  vs 6.2 ms, verify-eval 34.0 vs 37.0 ms.
- The authoritative formal-protocol A/B on M5 (full env contract, 5 agent
  prompts, 2 warmups + 5 trials, 512 tokens, seed 20260728, notarized
  6.12.1): **speedup 1.0969** (direct 32.68 tok/s, MTP 35.85), exactness
  passes with zero divergence, acceptance 0.8955, rollback 0.6 ms/cycle.
  Reading: axq026's absolute MTP throughput (35.85) exceeds cand-002's
  (35.22) — the verify pipeline is healthy — but the smaller model's faster
  direct decode (32.68 vs 30.31) lowers the ratio. Reaching 1.20x needs the
  MTP cycle to shed ~4.6 ms (verify-eval 35.0 ms/cycle is the dominant
  target), or the exact contract extended past depth 1; both are AX Engine
  runtime work, now precisely quantified with no configuration confusion.
- Toolkit hardening from this incident: `benchmark-ab --qwen36-exact-profile`
  injects the complete measurement contract (explicit `--runtime-env` wins),
  so a future runner cannot silently drop contract members again.
- Depth-2 exact experiment (AX Engine 08df0fdd, development evidence): the
  exact profile's eligibility gate was extended to serve drafts up to depth
  3 through the committed-prefix checkpoint (the invariant contract already
  covers a 1-4 token verifier). Depth 2 is **exact for the first time** —
  zero divergent trials on both the local M3 smoke and the M5
  formal-protocol run — at 2.59 emitted tokens/cycle, but measures 0.8684x
  because verify-eval scales ~linearly with verifier length (35 ms/cycle at
  2 tokens → 47 ms at 3; each extra verify token costs ~12 ms ≈ 0.39 direct
  steps on M5). Depth-2 M5 counters: acceptance 0.794/token, 301/423 full
  accepts, rollback 11.9 ms/cycle (partial-accept prefix recomputes).
  Conclusion: deeper drafts lose until the multi-token verify path
  approaches bandwidth-bound scaling; the M2 speed lever is now singular
  and precisely quantified — cut ~4.6 ms/cycle from the depth-1 verify
  (35.0 → ~30.4 ms) or make verify-token increments cheap enough for
  depth 2 to win.
- Final cycle accounting (M5, depth 1, full contract): the 45.0 ms cycle is
  draft 4.2 (near its ~3 ms bandwidth floor: 849 MB BF16 MTP head + 8-bit
  lm_head), verify-graph build 5.1 (pure CPU, GPU idle), verify-eval 35.0
  (≈ the ~32 ms weight-read floor for the 17.6 GB trunk at M5 bandwidth),
  rollback 0.6, accept/other ≈ 0.4; generation-loop overhead outside the
  step region measures only ~2.6 ms/token. The one clean lever left is the
  5.1 ms CPU build: AX Engine already carries per-layer compile
  infrastructure (`per_layer_compile.rs`, Track C2) that the verify forward
  does not use. Eliminating the build cost models to
  1.0969 × (45.0/39.9) ≈ **1.24x** — past the gate — and is scoped as the
  next AX Engine project (shapeless compile across the 64-layer hybrid
  stack with cache-mutation safety, reused for the 2-4 token verifier).
- Still open for this candidate: formal dual-profile validation bundles, MTP
  A/B speed (the 1.20x M2 gate on both workloads), and the formal
  supported-host suite.

## Phase E5 — Certification wave 1 (evidence work, not toolkit work)

Promote `qwen35-dense-v1` to `convertible` with a real-checkpoint smoke;
certify remaining official dense Qwen 3.6 sizes (AXQ-016 scope); publish the
first head-to-head evidence page (27B) per AXQ-022. Requires supported-host
hardware time; runs on the existing refine-run/release-audit tooling.

## Phase E6 — Certification wave 2

Promote and certify the first non-Qwen families (Gemma-4 first); schedule
measured KV probing (new ADR). Remote recipe resolution shipped early under
AXQ-023. Exit: E-T3.

## Sequencing and risk

```text
E0 → E1 → E2 → E3 → E4 → (E5 ∥ E6 evidence waves)
```

- E1 is the only phase that touches protection-critical classification; it
  therefore lands first and alone.
- E2–E4 are additive and independent of each other after E1; they land in
  order to keep review load linear.
- E5/E6 are evidence programs gated on hardware access, not code; the toolkit
  must never block on them (tiers make partial breadth truthful).
- Standing risk: any shortcut that lets development evidence look like release
  evidence. Mitigation: every phase's tests include at least one fail-closed
  case proving the label survives the new path.
