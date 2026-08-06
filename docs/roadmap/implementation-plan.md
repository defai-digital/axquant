# Implementation plan — phased delivery

- **Status:** draft
- **Date:** 2026-08-06
- **Inputs:** `prd.md` (requirements), `tech-spec.md` (workstreams WS-1…10),
  ADRs 0001–0006

Phases are gates, not sprints: a phase exits on criteria, not dates. Phase 0
overlaps nothing; Phases 1–3 may overlap where dependencies allow; Phase 4
items land opportunistically once their gates open.

## Status snapshot (2026-08-06, post-wiring)

Toolkit code for this roadmap is done — CLI and gate wiring included — and the
full suite, ruff, and mypy pass on the `macstudio-m2u` development host.
What remains is measurement, certification, and deferred scope:

```text
[A] Toolkit code for this roadmap     ~ DONE
[B] Deferred features                 = NOT DONE by design (vision tower,
                                        per-expert MoE, domain LoRA/SFT)
[C] Real 27B measurement round        = NOT DONE  <- next battlefield
[D] Flagship certification (Phase 0)  = NOT DONE  <- official trust anchor
```

| Workstream | Code + wiring | Remaining to exit (real evidence, not unit tests) |
| --- | --- | --- |
| 1.1 act-order GPTQ (`gptq-act`) | ✅ probe/plan/convert/audit + `--methods gptq-act` | 27B measurement round, both profiles |
| 1.2 frontier + near-tie surfacing | ✅ | same measurement round |
| 1.3 interaction optimization | ✅ `refine-select --interaction` + role guard | real bound-candidate evaluations on development roles |
| 2.1 MTP bundle admissibility | ✅ enforced by flagship M7 | formal-host A/B run after Phase 0 exit |
| 2.2 measured-KV serving quality | ✅ `kv-serving-quality` + kv_exec binding | dual-profile short/long-context measurements |
| 3.1–3.3 kernel-latency planning | ✅ `benchmark-kernels` + `plan --latency-table` | tables on `mbp-m5` + dev hosts (3.2), re-plan decode measurement (3.4), kernel wishlist to AX Engine (3.5) |
| 4.1 quantized MTP sidecar | ✅ `quantize-mtp-sidecar` + live capability probe | an AX Engine build that actually reports the quantized MTP layout; see the open product question below |
| 4.3 2/3-bit robust trunk | ✅ | stability evidence per pack |
| Phase 0 certification | — (operational, untouched per ADR-0001) | the entire campaign → formal → M0–M8 chain |

**Open product question (owner decision).** The tech spec originally sketched
sidecar quantization as a `convert` flag; it landed as the standalone
`quantize-mtp-sidecar` command instead, because the capability gate needs a
live AX Engine probe that plain conversion environments may not have, and
ADR-0001 keeps the convert critical path frozen during the campaign. If the
product intent is "one-shot `convert` also emits the quantized sidecar," that
is a deliberate follow-up wiring decision, not an oversight — decide it
explicitly before adding the flag.

README's "Still incomplete" list stays the public scoreboard: entries close
only on met acceptance criteria with measured evidence, never on merged code.

```text
Phase 0  Certification closure          (RM-01/02)        ── blocks all claims
Phase 1  Quality pack                   (RM-10..14)       ── algo work, flags off
Phase 2  Speed evidence pack            (RM-20..22)       ── evidence work
Phase 3  Runtime co-design              (RM-30/31)        ── planner cost model
Phase 4  Deferred scope                 (RM-40..44)       ── gated items
```

---

## Phase 0 — Certification closure (now)

**Objective.** Close `qwen36-mtp-v2` on the real candidate: campaign-overlap →
campaign-frontier → campaign-freeze → campaign-preflight (`mbp-m5`) → formal
run → holdout completion → M0–M8 `release-audit`, then publication — or a
recorded `closed_no_go` / `formal_failed` with archived evidence.

**Rules (ADR-0001).**
- No algorithm, planner, or schema change on the campaign's critical path.
- Roadmap code may merge behind default-off flags only.
- Bug fixes touching `campaign.py` / `claims.py` / `lifecycle.py` /
  release-audit paths need independent review.

**Exit criteria.** `release-audit` returns 0 and publication packages
`release_audit.json`, **or** a no-go/failure record exists under the durable
campaign root. Either outcome exits the phase.

**Effort shape.** Operational (operator + custodian + reviewer time, formal
host time); engineering only on-call for defects.

---

## Phase 1 — Quality pack

**Objective (G2).** Better quality-at-BPW with portable packing intact.

| Order | Work | Spec | Depends on |
| --- | --- | --- | --- |
| 1.1 | Group-preserving GPTQ act-order + new candidate label | WS-1 | none (flag-gated, can merge during Phase 0) |
| 1.2 | Frontier exploitation: (bits, method, group) as one axis; extended Pareto points | WS-3 | none |
| 1.3 | Measured-holdout interaction optimization backend + role guard | WS-2 | 1.2 (variants come from frontier) |
| 1.4 | Floor tuning study → documented recommendations | WS-10 | 1.2 |
| 1.5 | DWQ×AWQ/GPTQ near-tie surfacing | WS-3 | 1.2 |

**Measurement round.** After 1.1–1.3 land: full measured sensitivity +
frontier on the reference model, both profiles, with act-order and mixed-group
candidates enabled. This is the round that proves or refutes RM-10/RM-12.

**Exit criteria.** Pareto front on the reference model dominates (or ties
with a recorded already-optimal finding) the pre-program front, both
profiles; interaction-opt smoke passes with role guard; no portable-packing
contract test regressions.

**Risk to watch.** Act-order shows no measurable win → keep the label,
record the negative result, do not force adoption (ADR-0002).

---

## Phase 2 — Speed evidence pack

**Objective (G3).** Convert existing speed machinery into hardware-scoped
evidence; push BPW at held quality.

| Order | Work | Spec | Depends on |
| --- | --- | --- | --- |
| 2.1 | MTP A/B bundle admissibility (host binding, acceptance rate, effective tok/s) | WS-5 | Phase 0 exit (frees formal host) |
| 2.2 | Measured-KV serving-quality report (report-only gate first) | WS-6 | none |
| 2.3 | KV gate threshold proposal from 2.2 data | WS-6 | 2.2 |
| 2.4 | BPW push: reference pack at lower measured BPW, unchanged eval verdicts | — | Phase 1 measurement round |

**Exit criteria.** MTP bundle digest-bound to `mbp-m5` and reproduced on a
clean host; KV-quality report exists for short+long context, both profiles;
at least one lower-BPW reference pack with unchanged verdicts, or a recorded
finding that current BPW is on the frontier.

---

## Phase 3 — Runtime co-design

**Objective (G4).** Planner prefers configurations that are measurably fast on
the shipped runtimes.

| Order | Work | Spec | Depends on |
| --- | --- | --- | --- |
| 3.1 | Kernel-latency harness + `axquant.kernel-latency.v1` artifact | WS-4 | none |
| 3.2 | Latency tables for MLX-LM and AX Engine on formal + one dev host | WS-4 | 3.1 |
| 3.3 | Planner latency provider + `cost_model` provenance + golden-file neutrality test | WS-4 | 3.1 |
| 3.4 | Re-plan reference model with latency table; measure decode latency of changed plan | WS-4 | 3.2, 3.3, Phase 1 round |
| 3.5 | Kernel wishlist report to AX Engine (quality-optimal but kernel-slow configs) | WS-4 | 3.2 |

**Exit criteria.** Neutrality golden test green (no table → identical plans);
≥1 real plan changes under the table and wins measured decode latency at equal
quality verdicts (RM-31 success metric), or a recorded finding that abstract
BPW already matches kernel reality on current runtimes.

**Note.** 3.5 is the cross-repo lever: latency evidence tells AX Engine which
kernels to build next; revisit ADR-0002 option 1 (permutation-aware kernels)
only after that loop exists.

---

## Phase 4 — Deferred scope (gated, opportunistic)

| Item | Spec | Gate that must open first |
| --- | --- | --- |
| 4.1 MTP sidecar quantization | WS-7 | AX Engine capability check executes quantized MTP layout |
| 4.2 Vision eval suite → tower quantization | WS-8 | Clean-room vision eval tasks integrated into `evaluate-quality` |
| 4.3 2/3-bit robust-trunk hardening | WS-9 | Phase 3 latency data (2/3-bit only pays if kernels are fast) |
| 4.4 Per-expert MoE precision | ADR-0005 | Upstream MLX-LM unfused-expert support (tracked, not built) |
| 4.5 Secondary family promotion (Nemotron) | — | Capacity after Phases 1–3 |

**Exit criteria.** Per item; each closes its README "Still incomplete" bullet
and its `known-issues.md` entry in the same PR (tech-spec cross-cutting rule).

---

## Program-level tracking

- Each RM-id maps to one issue; phase exit reviews check acceptance criteria
  from `prd.md` verbatim.
- README's incomplete list is the public scoreboard: it shrinks only on met
  acceptance criteria, never on merged-but-unproven code.
- Any deviation from an ADR requires editing the ADR (status: superseded)
  before the deviating code merges.
