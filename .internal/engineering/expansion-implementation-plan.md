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
  5.1 ms CPU build: eliminating it models to 1.0969 × (45.0/39.9) ≈
  **1.24x** — past the gate.
- Two cheap variants of that lever were tested on 2026-08-01 and ruled out
  with measurements:
  1. Draft-confidence gating (dropping `AX_MLX_MTP_DRAFT_MIN_CONFIDENCE=0`)
     is not a legal knob on 6.12.1 — removing it also reverts the verifier
     to the expensive non-contract path (rollback 22.4 ms/cycle signature),
     and even modeled with the cheap path it reaches only ~1.19x.
  2. Extending the default-on dense-FFN decode compile
     (`AX_MLX_DENSE_FFN_COMPILE`) from seq==1 to the verifier's seq 2-4
     microbatch (seq-keyed closure cache) measured **no verify cost change**
     (verify-eval 53.0 vs 53.9 ms/cycle on M3, build unchanged) — the
     exact profile's invariant-projection custom Metal kernels do not
     benefit inside `mlx_compile`, and the build cost is dominated by the
     48 linear-attention layers, not the FFN. The change was reverted per
     the no-unproven-performance-changes discipline.
  The remaining implementation of the lever is therefore genuine engine
  engineering: either make the invariant/gated-delta custom kernels
  compile-hostable, or restructure the MTP cycle to overlap graph build
  with GPU execution (the direct path's double-buffer pattern applied to
  the verifier).
- The overlap restructure **landed the same day** (AX Engine b2d6afdd,
  `AX_MLX_MTP_ASYNC_DRAFT`, default off): the greedy zero-gate draft is
  scheduled with `async_eval` and the verifier chains directly on the lazy
  draft-token arrays through a new ids-taking forward variant, so the
  verify graph builds while the draft head's GPU forward runs and one eval
  batch materialises both. Exactness is preserved by construction and
  verified: byte-identical greedy output on M3 and M5 with unchanged
  acceptance (0.8955), draft wall 4.2 → 0.2 ms/cycle, M3 end-to-end −9.3%.
  The M5 formal-protocol depth-1 measurement improves from 1.0969x to
  **1.1912x** (direct 32.56 tok/s, MTP 38.79) — within ~0.75% (~0.2
  ms/cycle) of the 1.20x gate, i.e. inside single-run variance. The formal
  decision needs the formal suite on a notarized runtime carrying the
  flag; remaining headroom (the ~1.9 ms of verify build not covered by the
  draft overlap, deeper cross-cycle pipelining, compile-hostable kernels)
  stays scoped for the next engine round.
- Async-draft robustness matrix (local M3, 120-token greedy): all six
  configurations — flag off/on × {depth-1 prepared, depth-2 view, depth-1
  byte-preserved} — produce byte-identical output (one shared output hash)
  with identical accepted/drafted counts per configuration. The overlap
  holds exactness across both sidecar layouts and both admitted draft
  depths.
- Still open for this candidate: formal dual-profile validation bundles, MTP
  A/B speed (the 1.20x M2 gate on both workloads), and the formal
  supported-host suite.

## Phase E5 — Certification wave 1 (evidence work, not toolkit work)

Promote `qwen35-dense-v1` to `convertible` with a real-checkpoint smoke;
certify remaining official dense Qwen 3.6 sizes (AXQ-016 scope); publish the
first head-to-head evidence page (27B) per AXQ-022. Requires supported-host
hardware time; runs on the existing refine-run/release-audit tooling.

### qwen35-dense-v1 promotion evidence (2026-08-01, Apple M3 Max)

The AXQ-017 `convertible` contract (adapter unit tests + one real-checkpoint
conversion smoke with full coverage and integrity checks) is satisfied on
`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` (19.3 GB BF16 download):

- Real inspect: all 775 tensors classified (224 attention, 96 MLP, 105 norm,
  333 vision, 15 integrated MTP, embedding, LM head), dense, 32 text layers,
  MTP and vision detected — zero unclassified tensors.
- One-command `quantize` at target 7.0 BPW (policy minimum for this
  vision+MTP-heavy 9.65B model is 6.9675): measured **7.0001** total BPW,
  atomic output, AX Engine `model-manifest.json` generated, prior-based
  per-layer KV metadata, and a **passing MLX-LM generation smoke** plus a
  passing AX Engine doctor runtime check.
- The real run drove three genuine breadth fixes, each with regression
  tests: the pre-AXQ-017 hard-coded "restricted to Qwen 3.6" conversion
  gate is now tier-based (`assert_conversion_scope`); the external-sidecar
  requirement no longer fires for integrated-MTP sources; and the protected
  vision extraction generalized into a role-parameterized sidecar extractor
  that byte-copies **integrated MTP** (Qwen 3.5's 15-tensor, 243,290,624-
  parameter head, which the MLX-LM text mapping drops — caught by the
  fail-closed coverage check) into the canonical `mtp.safetensors` with a
  checksummed `axquant_mtp_sidecar_manifest.json` and the
  `mtp_norm_layout: raw_hf_delta` declaration, giving every artifact one
  MTP layout regardless of source packaging.
- Registry flip: `qwen35-dense-v1` → `SupportTier.CONVERTIBLE` (dense-only;
  MoE and missing-layer configs stay fail-closed inspect-only). All
  artifacts remain development evidence until the family is certified.

### Breadth wave 2 (2026-08-01, same session)

- **minicpm5-dense-v1 → `convertible`** on
  `openbmb/MiniCPM5-1B@4e9de7a0778dc1c362e983e6858f0e77542cbdca`: the public
  checkpoint is a plain Llama-arch export (`model_type: llama`, spec extended
  with the reference-scoped type), all 219 tensors classify, and the
  one-command conversion measures **7.4999 BPW** (policy minimum 7.3772 for
  a 1.08B model) with passing MLX-LM generation and AX Engine doctor smokes.
- **gemma4-dense-v1 stays `inspect-only`, now with full real inspection**:
  `google/gemma-4-12b@023679ed352d` declares `gemma4_unified` (spec
  extended); the real inventory drove two classifier fixes — MoE config keys
  present with null/false values no longer mark a checkpoint non-dense, and
  `layer_scalar` / `embed_audio` patterns classify the remaining 49 tensors
  (677/677 classified, dense=true). Promotion is blocked honestly: the
  pinned MLX-LM 0.31.3 cannot convert `gemma4_unified`, so the AXQ-017
  conversion smoke cannot run until an admitted MLX-LM version supports the
  family.
- **nemotron3-dense-v1 stays `inspect-only`**: the current public Nemotron 3
  catalog (Nano-30B-A3B, Super-120B-A12B, Ultra-550B-A55B, embed models) is
  MoE-only — there is no real dense checkpoint that can satisfy the
  promotion contract. MoE planning remains deferred scope.
- Sources are revision-pinned `snapshot_download` trees under
  `/Volumes/Ext4T/` (plain local_dir materialization, not the HF cache
  snapshot layout); dev-smoke artifacts sit alongside them.

### Low-bit range (2026-08-01, same session)

- Hardware profile `ax-engine-apple-silicon-affine-dwq-v2` extends the
  precision range to **2/3/4/6/8/BF16**. MLX affine kernels execute 2- and
  3-bit natively; AX Engine admits them behind documented experimental
  gates (3-bit pre-existing; 2-bit added in engine commit 93b9cbb2 with the
  same rejected-by-default contract).
- Real 3-bit evidence (MiniCPM5-1B, prior plan, 106 tensors at 3-bit):
  measured **6.8504** total BPW, passing MLX-LM generation smoke and AX
  Engine doctor under `AX_ENGINE_3BIT_EXPERIMENTAL=1`.
- Real 2-bit evidence (MiniCPM5-1B, prior plan, 64 tensors at 2-bit):
  measured **6.5495** total BPW, passing MLX-LM generation smoke and AX
  Engine doctor under `AX_ENGINE_2BIT_EXPERIMENTAL=1` with the new engine
  build.
- Both artifacts are development evidence; protection floors are untouched
  (low bits reach ordinary trunk tensors only). Release claims still require
  the ordinary measured quality/runtime gates.
- KV runtime execution now has a real compatibility-path implementation
  (below); AX Engine-native per-layer execution remains the scoped engine
  project for the primary runtime.

### KV-cache runtime execution, compatibility path (2026-08-01, same session)

- The MLX-LM generation smoke now executes a planned artifact's advisory KV
  quantization instead of ignoring it: `check_mlx_lm_generation` reads
  `advisory_mlx_lm_kv_bits`/`_group_size` from `axquant_runtime.json` and
  passes `--kv-bits/--kv-group-size` to `mlx_lm.generate`, which runs the
  public `QuantizedKVCache` path. The runtime-check report records the
  executed KV parameters; BF16 advisories keep the runtime default.
- Real evidence: the MiniCPM5-1B artifact rebuilt with `--kv-cache prior`
  (advisory 4-bit/group-64) passes generation **with quantized KV actually
  executing** — the recorded command carries `--kv-bits 4 --kv-group-size
  64` and the report binds the source as the artifact's advisory values.
- **Per-layer execution landed the same session**: `axquant.kv_exec` builds
  one cache object per layer from the plan's table (`QuantizedKVCache` for
  quantized layers, the model's own cache otherwise) and generates through
  MLX-LM's public `prompt_cache` API, whose attention helper dispatches per
  cache object — the plan's exact per-layer precisions execute at runtime
  with zero private interfaces. `runtime-check --runtime mlx-lm-kv` wraps
  it with a typed report. Real evidence on the MiniCPM5-1B KV artifact:
  all 24 quantized layer caches active, executed bits == planned bits
  (6 layers at 8-bit, 18 at 4-bit), generation passing.
- Honest scope notes: the hybrid Qwen 3.6 family fails closed in this path
  (MLX-LM's quantized-SDPA shape handling rejects the family's attention —
  reproduced: broadcast error at prefill; consistent with the E5 log), so
  its per-layer KV awaits AX Engine-native quantized KV (cache storage,
  append/read, MTP-clone interplay) — the one remaining scoped engine
  project. Standard-attention families (MiniCPM5, Qwen 3.5 dense) have
  full per-layer KV execution today.

### MoE conversion support (2026-08-01, same session)

The deferred-scope assumption fell to a real run: MLX-LM 0.31.3 natively
supports `qwen3_5_moe` through its public API (module
`mlx_lm.models.qwen3_5_moe`; its sanitize splits the packed
`experts.gate_up_proj` stack into fused `switch_mlp.gate_proj`/`up_proj`
modules) — no MLX-LM patching of the kind the competitive reference uses is
required, so AXQ-001/AXQ-004 stay intact. Implementation, all evidence on
the official catalog MoE size
`Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0` (67 GB
BF16):

- Classification: `mlp.gate` routers (distinct from `gate_proj`), packed
  3-D expert stacks, MoE MTP head, and vision — **1045/1045 tensors
  classified** after three real fixes (`qwen3_5_moe` model type accepted by
  the Qwen 3.6 adapter; `.mlp.gate.` router pattern; 3-D expert stacks are
  quantizable — MLX-LM quantizes them as fused switch modules).
- Planning: routers hold their 8-bit floor (40/40 at 8-bit), packed expert
  stacks take mixed 4/6/8-bit (78/144/18), policy minimum 5.1350 BPW, plan
  at 5.25 effective. Per-expert layouts additionally get fused-group
  uniformity (predicate fails closed on mixed precisions inside one switch
  group; planner normalizes groups budget-safely).
- Module identity: packed `experts.gate_up_proj` aliases both fused MLX
  switch halves from one allocation; `experts.down_proj` aliases its fused
  module — recorded in `module_paths.py` as the public-sanitize contract.
- Real conversion: **measured 5.2501 total BPW** (plan 5.2500), 20 files,
  22 GiB artifact, byte-preserved integrated-MTP extraction (19 tensors,
  MoE draft head included) and vision sidecar, passing **AX Engine doctor**
  and the MLX-LM generation smoke. Development evidence; certification
  follows the ordinary gates.

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
