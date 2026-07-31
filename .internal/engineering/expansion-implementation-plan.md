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
