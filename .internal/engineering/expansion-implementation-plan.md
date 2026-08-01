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
- Still open for this candidate: dual-profile quality/validation bundles, MTP
  A/B speed (the 1.20x M2 gate on both workloads), and the formal
  supported-host suite. Note the structural tension: a smaller candidate has
  faster direct decode, which raises the bar for the MTP speed ratio.

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
