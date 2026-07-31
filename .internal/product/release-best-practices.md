# AXQuant release best practices

**Status:** Active for v1 candidate cycles  
**Last reviewed:** 2026-07-31

These practices are derived from the formal M5 measurement outcome
(`.internal/tmp/qwen36-v1-recovered-fast-formal-failure-audit-v1.json`). They constrain how the
next candidate cycle is planned, measured, and published.

## 1. Separate software readiness from release readiness

| Layer | Meaning |
| --- | --- |
| Toolkit software | CLI/pipeline/tests green; schemas and gates exist |
| Release candidate | One immutable artifact that passes **all** dual-profile gates |
| Public v1.0 | `release-audit` green + explicit publish authorization |

Package version `1.0.0` and a green pytest suite are **not** a release claim. Do not publish or
preview Hub uploads while `release_ready=false`.

## 2. Enforce gate order (never skip ahead)

Solve blockers in this order. Later gates may not mask earlier failures.

1. **Exactness / integrity** — greedy A/B identity, zero kernel fallbacks  
2. **Metric completeness** — every profile emits every metric the validator requires  
3. **MTP quality** — acceptance, retention, token accuracy on both profiles  
4. **MTP speed** — effective speedup ≥ 1.20× on **agent-coding and general**  
5. **Size** — weight ratio ≤ 1.10 vs uniform-4 (or a governed exception)  
6. **Provenance** — AX Engine version and software pins present in every bundle  
7. **Audit / publish** — only after 1–6  

A size exception is **ineligible** while any non-size gate fails
(`size_exception_eligible=false` in the formal audit).

## 3. Dual-profile completeness by construction

Both `agent-coding` and `general` must:

- use disjoint evaluation data from calibration;
- emit governed structured metrics (`json_valid_rate`, `syntax_valid_rate`);
- carry MTP A/B evidence with complete horizons;
- pass under the same immutable candidate artifact.

If a suite cannot produce a metric the validator requires, fix the **suite or thresholds** before
the next formal run—do not treat missing pairs as model failure.

## 4. Use positive controls before blaming the planner

Uniform-6 on the formal host met the 1.20× MTP floor. That means:

- the host + AX Engine path can pass speed;
- a failing candidate is a **candidate** problem (plan, packing, layout, depth), not proof that
  MTP is impossible on that host.

Always retain uniform-4 size evidence and uniform-6 speed/quality controls in the same formal root.

## 5. Prefer gate-feasible plans over quality-max plans

Hard protection floors imply a structural BPW floor (~5.58 / ~114.6% of uniform-4 under current
policy). Candidates that spend optional 6-bit upgrades (cand-002 ~5.76 BPW, cand-003 ~5.96 BPW)
make size harder without guaranteeing MTP speed.

Default next-cycle bias:

- start from the **measured policy-floor** plan (or tighter);
- only add precision where complete-model holdout proves it is necessary;
- never lower protection floors without measured authorization and a product decision.

## 6. New cycle, new evidence root

- Do not rewrite historical formal bundles.  
- Fix probes/suites in software, then produce a **new** formal root.  
- Bind every artifact by SHA-256 in the decision/audit record.  
- Treat exit status 1 from governed finalizers as expected failure, not a crash.

## 7. Provenance must be machine-complete

Every evaluation bundle must carry `software_versions.ax_engine` when AX Engine is used.
Resolve version from:

1. doctor JSON when the schema provides it; else  
2. Homebrew Cellar path; else  
3. explicitly versioned standalone directory (`…/6.12.1/…`).

Never invent a version. Never claim notarization outside the bundle as a substitute for the field.

## 8. Exception discipline

Governed size exceptions require owner, approver, reference, effective/expiry times, and
checksum-bound tradeoff evidence. They:

- cover only the size gate;
- never waive MTP speed, exactness, structured metrics, or missing provenance;
- expire; they are not permanent policy.

## 9. Candidate-cycle runbook (minimum)

```text
replan near policy floor
  → convert (measured plan + calibration + chosen MTP layout)
  → runtime-check (AX Engine + MLX-LM)
  → prepare-suite (current suite version)
  → dual-profile quality + MTP A/B (complete metrics)
  → size-evidence vs uniform-4
  → validate both profiles
  → if only size fails → exception path; else replan
  → new formal suite root with versioned runtime probe
  → release-audit → publish only with explicit authorization
```

## 10. Repo hygiene before any public cut

- Record commits/tags for the exact toolkit revision that produced the wheel.  
- Keep clean-room and evidence-kind gates intact.  
- Prefer honest “not release-ready” status over optimistic version marketing.
