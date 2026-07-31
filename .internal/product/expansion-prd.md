# AXQuant Expansion Product Requirements Document

**Document status:** Accepted for planning  
**Applies to:** AXQuant v1.1 through v2.0 (the "expansion program")  
**Supersedes:** Section 3.3 non-goal entries "Gemma or unrelated model families",
"arbitrary Qwen generations", and "runtime KV-cache quantization" in
`requirements.md` become expansion-program goals under the boundaries below. All
other v1 requirements, claim guardrails, and release gates remain in force.  
**Last reviewed:** 2026-07-31

## 1. Executive summary

AXQuant v1 proved a deep, evidence-gated conversion pipeline for one model
(Qwen3.6-27B). The market-leading alternative, mlx-optiq, wins adoption through
breadth (sixteen published models across five families) and a one-command user
experience, while offering no provenance, no reproducibility artifacts, no MTP
awareness, and no release audit.

The expansion program makes AXQuant competitive on breadth and ease of use
**without weakening** the evidence discipline that differentiates it:

1. **Family breadth** through a tiered support model: many families become
   *convertible* quickly; a smaller, growing set becomes *certified* through the
   existing M0–M8 gates.
2. **One-command conversion** (`axquant quantize`) whose ease matches mlx-optiq
   for end users, while every shortcut remains explicitly labeled development
   evidence.
3. **A published measured-recipe library** so users converting their own
   checkpoints reuse AutomatosX-measured sensitivity and plans instead of paying
   the measurement cost themselves.
4. **KV-cache mixed-precision planning**, closing the one functional gap that
   matters for long-context deployment cost on unified memory.
5. **A public model catalog** on Hugging Face: certified AXQuant models for the
   same bases mlx-optiq covers, each shipping its full evidence chain, plus
   head-to-head benchmark publications.

The strategic claim policy is unchanged: AXQuant competes by measured merit. It
does not claim to be a successor to mlx-optiq and never reuses its assets
(AXQ-001, AXQ-022).

## 2. Product vision (expansion)

AXQuant becomes the trustworthy way to quantize and deploy LLMs on Apple
Silicon:

- users download certified AXQuant models the way they download OptiQ models
  today, but each model carries checksums, provenance, a reproduction recipe,
  and a passed release audit;
- users convert their **own** checkpoints with one command, optionally binding a
  published measured recipe so the result inherits measured planning;
- deployment cost planning covers weights **and** KV cache;
- AX Engine remains the primary runtime; stock MLX-LM remains the compatibility
  runtime for every published artifact.

## 3. Product boundary

### 3.1 Expansion boundary (v1.1–v2.0)

In scope:

- a tiered family-support model (`certified` / `convertible` / `inspect-only`)
  recorded in every inventory and manifest;
- a declarative dense-transformer adapter framework so a new family adapter is
  primarily data plus tests, not a rewrite;
- adapters for, in priority order: all official dense Qwen 3.6 sizes,
  Qwen 3.5 dense, Gemma-4 dense text path, MiniCPM5 dense, Nemotron 3 dense;
- the `axquant quantize` one-command conversion path;
- recipe bundles: checksummed, versioned, publishable planning artifacts;
- per-layer KV-cache precision planning (schema, prior-based allocation, AX
  Engine metadata; measured KV probes follow in a later milestone);
- publication of certified models and head-to-head benchmark evidence.

### 3.2 Explicit non-goals for the expansion program

- MoE expert-level planning (unchanged; revisit after the dense catalog ships);
- full VLM quantization or VLM quality claims (vision tensors remain preserved);
- 2/3/5-bit production formats;
- GGUF, CUDA, Windows, or Linux output;
- LoRA rank allocation (tracked as a post-expansion candidate; the sensitivity
  artifacts are already the required input, so no expansion work may preclude
  it);
- claiming compatibility for a family whose adapter has not passed its tier's
  gate;
- any reuse of mlx-optiq code, data, metadata, sensitivity outputs, or
  allocation tables (AXQ-001 is unconditional).

## 4. Target users (expansion additions)

1. **Model downloaders** (new primary): people who today download
   `mlx-community/*-OptiQ-*` checkpoints. They never run AXQuant; they consume
   certified AXQuant models from Hugging Face. Ease requirement: zero — but
   catalog coverage and measured superiority are decisive.
2. **Self-converters** (new primary): developers with their own fine-tuned or
   niche checkpoints. Ease requirement: one command, defaults that work,
   readable output. They are the audience for `axquant quantize` and recipe
   bundles.
3. **Enterprise release engineers** (existing): unchanged; they use the full
   gated pipeline and are the audience for tiers, manifests, and audits.

## 5. User problems addressed

| Problem | Current state | Expansion answer |
| --- | --- | --- |
| "AXQuant only does one model" | One certified path (Qwen3.6-27B) | Tiered breadth: convertible quickly, certified deliberately |
| "Conversion takes a dozen commands" | Release pipeline UX is the only UX | `axquant quantize` single command for development conversion |
| "Measured planning is too expensive for me" | Every user would re-measure | Published recipe bundles bind AutomatosX measurements to user conversions |
| "Weights-only quantization ignores my real memory bill" | KV cache unplanned | Per-layer KV-cache precision planning and runtime metadata |
| "Why choose AXQuant over OptiQ?" | No public comparison | Certified catalog + published head-to-head evidence per base model |

## 6. Goals and claim policy

### 6.1 Primary goals

1. Ship certified AXQuant models for at least the base-model set covered by
   mlx-optiq's published catalog, prioritized by download traffic.
2. Make development conversion of a Tier-`convertible` checkpoint a single
   command with wall-clock and step-count parity with mlx-optiq.
3. Make every published AXQuant model verifiably better than, or measurably
   equivalent to, the corresponding uniform-4-bit baseline under the existing
   dual-profile gates — and publish the same metrics against the attributed
   OptiQ checkpoint as an external baseline.

### 6.2 Claim guardrails (additions)

- A family or size may be described as **supported** only at its recorded tier,
  and the tier name must appear in the claim ("convertible (development)" vs
  "certified").
- Quick-mode output is development evidence. The CLI, manifests, and README
  wording must make it impossible to mistake quick-mode output for a certified
  artifact.
- Recipe bundles inherit the evidence kind of their source: a bundle exported
  from measured release evidence is "measured, externally reproduced"; a bundle
  from priors is "architecture prior" and is labeled as such.
- Comparisons against OptiQ checkpoints follow AXQ-001: attributed, external,
  standard-load-contract only.

## 7. Success criteria

### 7.1 Required expansion targets

| ID | Target |
| --- | --- |
| E-T1 | ≥ 5 model families at tier `convertible` or better, each with adapter tests and a conversion smoke on a real checkpoint |
| E-T2 | Every official dense Qwen 3.6 size certified (extends AXQ-016) |
| E-T3 | ≥ 3 non-Qwen certified models published with full evidence chains |
| E-T4 | `axquant quantize` converts a Tier-convertible checkpoint end to end with one command and defaults |
| E-T5 | Recipe bundles: publish ≥ 1 measured bundle per certified model; `quantize --recipe` binds and verifies one by checksum |
| E-T6 | KV-cache plan schema, prior allocation, and AX Engine metadata shipped; measured KV probes scheduled in the roadmap |
| E-T7 | One public head-to-head benchmark page per certified base model (AXQuant vs uniform-4/6 vs attributed OptiQ external baseline) |

### 7.2 Failure conditions

- A tier claim without its gate evidence.
- Quick-mode output published to the official catalog.
- A family adapter that silently misclassifies protected tensors (fail-closed
  classification remains mandatory: unclassifiable tensors block conversion).
- KV-cache claims stated as measured before measured KV probes exist.

## 8. Functional requirements (expansion)

### 8.1 Tiered support

- `SupportTier` = `certified` | `convertible` | `inspect-only`, recorded in
  `ArchitectureProfile`, surfaced by `inspect`, and enforced by `convert`,
  `quantize`, and `publish`:
  - `inspect-only`: inventory and classification; conversion refused;
  - `convertible`: conversion permitted; output always development evidence
    until certified; publication to the official catalog refused;
  - `certified`: full release pipeline; publication permitted through existing
    gates.
- Tier promotion is evidence-bound: `convertible` requires adapter tests plus a
  real-checkpoint conversion smoke with coverage and integrity checks;
  `certified` requires the existing M0–M8 audit for at least one size of the
  family.

### 8.2 Adapter framework

- A declarative dense-family specification (identifier, product family, config
  `model_type` values, reference patterns, layer-count extraction, tensor-role
  classification rules, protection notes) drives a shared adapter
  implementation; families needing bespoke logic (MTP layouts, unusual
  sidecars) may still implement the adapter protocol directly.
- The registry resolves the most specific matching adapter; ambiguity is an
  error, not a silent pick.

### 8.3 Quick conversion

- `axquant quantize --model PATH --output DIR [--recipe BUNDLE] [--target-bpw N]`
  performs inspect → plan → convert → optional runtime smoke in one process,
  with progress reporting and a final summary of tier, evidence kind, measured
  BPW, and output location.
- Without `--recipe`, planning uses architecture priors or the family default
  recipe and the artifact is labeled development evidence.
- With a measured `--recipe`, the bundle's checksums and model identity must
  verify, and the resulting artifact records the bundle lineage.

### 8.4 Recipe bundles

- A recipe bundle is a versioned, checksummed artifact binding: model identity
  (id, revision), a plan or manual recipe, its evidence kind and lineage
  digests, and the AXQuant version that produced it.
- Bundles are publishable to Hugging Face alongside certified models and
  resolvable by `axquant quantize --recipe` from a local path (remote
  resolution follows in a later phase).

### 8.5 KV-cache planning

- The plan schema gains an optional per-layer KV-cache precision section
  (bits, group size) with policy floors, produced by a prior-based allocator in
  the first phase and measured KV sensitivity later.
- Converted artifacts emit KV-cache metadata for AX Engine and record that KV
  planning is prior-based until measured.
- Absence of the KV section preserves existing behavior exactly.

## 9. Competitive requirements

- Every certified model's Hugging Face page must state, with links to evidence:
  measured BPW, dual-profile quality deltas, MTP speedup where applicable, and
  the reproduction recipe.
- The published comparison set per base model: BF16, uniform-4, uniform-6, the
  AXQuant candidate, and the attributed external OptiQ checkpoint where one
  exists (external-baseline rules per AXQ-001).
- MTP-aware conversion remains an AXQuant-only capability and leads the
  positioning for families that ship MTP weights.

## 10. Non-functional requirements

- All existing reproducibility, transparency, and safety requirements apply to
  every tier, including quick mode (atomic output, fail-closed coverage,
  checksum manifests, no credentials in logs).
- Adding a family must not modify certified-family behavior: adapter resolution
  is deterministic and covered by regression tests.
- Quick mode on a ~30B checkpoint must add no more than one minute of overhead
  beyond the underlying MLX-LM conversion time on a supported host.

## 11. Roadmap

Phases are dependency-ordered; detailed work breakdown lives in
`.internal/engineering/expansion-implementation-plan.md`.

| Phase | Product result | Exit condition |
| ---: | --- | --- |
| E0 | Expansion governance | This PRD, ADRs AXQ-017–AXQ-022, expansion tech spec accepted |
| E1 | Tiered support + adapter framework | Tier recorded end to end; ≥ 2 families beyond Qwen 3.6 at `inspect-only`/`convertible` with tests |
| E2 | One-command conversion | `axquant quantize` ships; E-T4 met on a real checkpoint |
| E3 | Recipe bundles | Bundle schema + local resolution + verification ship; E-T5 partially met (local) |
| E4 | KV-cache groundwork | Plan schema + prior allocator + runtime metadata ship (E-T6, unmeasured stage) |
| E5 | Certification wave 1 | Remaining official dense Qwen 3.6 sizes certified (E-T2); head-to-head page for 27B (E-T7 start) |
| E6 | Certification wave 2 | First non-Qwen families certified (E-T3); remote recipe resolution; measured KV probes scheduled |

Release mapping:

```text
v1.1  E1 + E2 (tiers, adapters, quantize)
v1.2  E3 + E4 (recipe bundles, KV groundwork)
v1.3  E5 (Qwen 3.6 catalog certification, first head-to-head publication)
v2.0  E6 (multi-family certified catalog, measured KV planning scheduled or shipped)
```
