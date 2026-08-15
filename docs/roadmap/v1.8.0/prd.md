# PRD — AXQuant 1.8.0: certified Apple Silicon deployment standard

- **Status:** accepted for planning
- **Date:** 2026-08-14
- **Owner:** AXQuant maintainer
- **Release:** 1.8.0 (toolkit SemVer; not a pack edition)
- **Related:** ADR-0001–0006 (completion program), ADR-0007–0009 (this suite)

> 1.8.0 is the release that turns the already-built planner, provenance, and
> catalog into a **standard other people can quote and re-verify**. It is not
> a new quantizer, not a CUDA port, and not a rename of live Hub repos.

## 1. Decision and scope

AXQuant 1.8.0 is the certified deployment-precision standard for Apple
Silicon LLMs: the evidence system of record for how much precision a model
can safely lose on a Mac (weights, then KV).

Buyers are publishers who need a defensible MLX release, and teams running
coding/agent workloads on unified memory who need a quotable quality / size
/ speed claim. They do not buy “another convert.”

Dependency order (ADR-0007):

```text
B evidence protocol  →  D pack contract  →  C optimize UX  →  A catalog
```

Anyone can emit an MLX checkpoint. Almost nobody else will publish a
certificate they are willing to fail. GPT-OSS 120B 4-bit remaining
uncertified is the product, not a footnote.

## 2. Users and terminology

| Term | Meaning in 1.8.0 |
| --- | --- |
| Product class / SKU | Requested budget lane in the Hub name: `4bit`, `6bit`, … |
| Measured main BPW | Authoritative weight-only bits-per-weight from bound bytes |
| Edition | Immutable pack revision tagged `vN` inside a stable repo |
| Toolkit version | `axquant==1.8.0` — unrelated to pack editions |
| Tier 1 | Checkpoint quality, size vs matched baseline, conversion integrity |
| Tier 2 | Scoped MTP acceleration on a named host and engine. Never implied by Tier 1 or by `-MTP` |
| Deployment budget | `weight_bytes + kv_bytes + reserve <= requested` |

## 3. Naming decision (reviewed)

Two user camps were reviewed against the live catalog, `naming.py`,
`model_card.py`, `claims.py`, and published certificates. Reviewers:
Codex `gpt-5.6-sol` (max) and Qoder `Qwen3.8-Max`. Both landed on the
same verdict. Full normative rules are ADR-0008.

**Verdict:** keep `4bit` / `6bit` as stable Hub SKUs. Measured BPW is the
claim. Editions live in tags. Do not put BPW in the repository name. Do
not rename live certified repos. Do not add an unversioned alias repo.

| Criterion | Camp 1: one name, drop 4bit/6bit | Camp 2: keep class SKUs | Hybrid rejected extras (alias / MP repo) |
| --- | --- | --- | --- |
| Hub search (“qwen 4bit”) | Loses the query people type | Matches existing search | Alias competes with the real pack |
| Memory vs quality choice | Hidden behind tags or “current” | Visible sibling ladder | Same ladder plus a third identity |
| Certified URL stability | Breaks every `hub_repo_id` binding | Preserves every bound URL | Preserves URLs, adds sync cost |
| “4bit” pack at ~5.42 BPW | No label mismatch | Dishonest if shown alone; honest as “4bit class — 5.42 BPW measured” | Two names for one artifact |
| Floor-collapse twins | Fewer names, no general ladder | Already suppressed in cards | Alias must pick the survivor |
| MTP vs no-MTP | Still needs a suffix or tags | `-MTP` already distinct | Alias must pick one MTP state |
| Edition v3 | Confuses edition with product choice | Stable repo + `v3` tag | Alias needs its own update policy |
| Certificate binding | Titles and URLs diverge after rename | Repo + class + tag + commit + BPW | Title and repo disagree |
| vs GGUF Q4/Q6 | Avoids false equivalence | Familiar ladder; must say “AXQ class, not GGUF” | Same risk plus a second grammar |
| Small-team cost | High migration | Lowest: tighten existing behavior | Highest ongoing docs/redirect cost |

Camp 1’s real point is honesty. Cards already print measured main and total
BPW above the fold. 1.8.0 makes that the **heading and the certificate
title**, and stops `claims.py` from demanding an MP repository name.

## 4. Goals

1. **G-B.** Publish Certification Spec v1.0 and `axquant verify-cert` so a
   third party can reproduce a certificate verdict from published files.
2. **G-D.** Freeze the portable affine U32 interchange so the same unchanged
   weight hashes load in AX Engine and stock MLX-LM.
3. **G-C.** Ship `axquant optimize` that spends one memory budget on
   weights + KV and fails closed when the request is infeasible.
4. **G-A.** Catalog and cards follow ADR-0008. No fleet-wide rename.

## 5. Non-goals

- CUDA / ROCm / TensorRT / vLLM shipping work
- Method-agnostic backend federation (AutoRound-native, GGUF, NVFP4, FP8)
- Renaming live certified Hub repositories
- Fleet-wide recertification as a 1.8.0 gate
- Full 2.x joint optimizer (per-layer interaction search, activation quant)
- Vision-tower quality certification
- Training-time or weight-mutating recovery
- New algorithm labels (act-order, 2/3-bit hardening) as launch gates —
  those stay on the completion-program roadmap

## 6. Requirements

IDs are `R18-`. Priorities: P0 launch, P1 same window if capacity, P2 after.

### 6.1 Evidence protocol (B)

- **R18-01 (P0).** Public Certification Spec v1.0 document: Tier 1 vs
  scoped Tier 2, evidence kinds, floors, measured-BPW naming, host scope,
  fail-closed rules. Acceptance: the spec is versioned and linked from
  every new Spec v1.0 certificate.
- **R18-02 (P0).** New strict certificate schema(s) under a **new**
  `schema_version` (ADR-0001). Legacy v1 records remain readable and are
  not silently rewritten.
- **R18-03 (P0).** `axquant verify-cert` offline-checks certificate,
  manifest, plan, file hashes, repo/class agreement, tag/commit binding
  when present, and recomputed BPW. Exit 0 only when the bundle is
  consistent. Tamper of any bound digest, class, BPW, or tier exits
  nonzero with a machine-readable report.
- **R18-04 (P1).** Context-scoped certificates: a `certified` context
  length never implies a longer one.

### 6.2 Runtime contract (D)

- **R18-10 (P0).** Public pack interchange spec `axq-affine-u32-v1`:
  tensor naming, U32 packing, scales/biases, group metadata, sharding,
  protected tensors, runtime declarations.
- **R18-11 (P0).** Dual-runtime conformance: one golden pack’s weight
  hashes load and generate on AX Engine and MLX-LM with no intervening
  repack. Fail closed if a quantized tensor is not affine-packed.

### 6.3 Planner UX (C)

- **R18-20 (P0).** `axquant optimize` accepts at least `--model`,
  `--max-memory`, `--context`, `--profile`, `--runtime`,
  `--min-quality`, `--mode`. It orchestrates existing inspect / plan /
  KV paths plus joint accounting (ADR-0009).
- **R18-21 (P0).** Hard constraint:
  `weight_bytes + kv_bytes + reserve <= requested_budget`. Infeasible
  requests fail closed with the breakdown.
- **R18-22 (P0).** Output records the breakdown, evidence kind, product
  class, and measured or estimated BPW. Estimated results cannot be
  labeled measured or certified.
- **R18-23 (P1).** `--mode` changes objective weights for real
  (`balanced` / `quality` / `low-memory` / `speed`).
- **R18-24 (P1).** Kernel-latency table, when supplied, re-ranks inside
  the quality-feasible set (ADR-0003). Absent table: bit-identical to
  abstract-BPW planning.

### 6.4 Catalog and naming (A)

- **R18-30 (P0).** Implement ADR-0008 in `naming.py`, `claims.py`,
  `model_card.py`: class SKU repos, measured BPW in card H1 and
  certificate titles, MP string not a repository generator.
- **R18-31 (P0).** Mechanical floor-collapse rule (≥5% complete-weight
  savings or no new 4-bit sibling).
- **R18-32 (P1).** Generated catalog / collections have no hand-authored
  certification status cells (already true for the README matrix; keep it).

## 7. Success metrics

- A third party can run `verify-cert` on one published 4-bit and one
  published 6-bit Spec v1.0 bundle and get the same verdict as the
  certificate.
- The interchange suite passes on AX Engine and MLX-LM for that golden
  pack without hash changes.
- `optimize` rejects at least one documented infeasible memory+context
  request and accepts a feasible sibling with a recorded breakdown.
- Zero live certified repositories renamed.

## 8. Launch gates

- Full existing CI (tests, ruff, mypy, schema-contract check).
- `docs/releases/1.8.0.md` present and non-empty (release workflow).
- Certification Spec v1.0 and interchange spec checked in.
- No CUDA, GGUF, NVFP4, or backend-federation deliverable in the tag.

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| Dilution into “another quantizer brand” | Release notes lead with spec + verify, not new bit widths |
| Honesty backlash on `4bit` ≈ 5.4 BPW | Card H1 and cert title lead with measured BPW (ADR-0008) |
| Schema mutation under an old version | ADR-0001; new `schema_version` only |
| Scope creep into full 2.x KV optimizer | ADR-0009 cut line |
| CUDA theater via “neutral planner” | ADR-0007: hygiene only, no adapter |

## 10. Key decisions

1. 1.8.0, not 2.0 — standardise what already works.
2. B→D→C→A order (ADR-0007).
3. Apple-only shipping; CUDA is a future separate track (ADR-0006/0007).
4. Keep `4bit`/`6bit` SKUs; measured BPW is the claim (ADR-0008).
5. One memory budget over existing allocators (ADR-0009).
