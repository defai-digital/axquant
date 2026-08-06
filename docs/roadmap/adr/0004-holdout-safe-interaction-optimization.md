# ADR-0004 — Holdout-safe complete-candidate interaction optimization

- **Status:** accepted
- **Date:** 2026-08-06

## Context

`refinement.py` already implements top-N plan generation, bounded coordinate
descent, beam refinement, and immutable candidate history — but driven by
proxy metrics from sensitivity reports. README lists the real thing as
incomplete: interaction optimization driven by *measured* results on a bound
candidate. The danger is obvious: an optimization loop that sees evaluation
results can silently become training on the test set. The campaign design
already treats the formal holdout as consumed-on-contact.

## Decision

1. Interaction optimization consumes measured results **only from
   development/validation dataset roles**. The optimizer API takes a
   measurement set whose provenance records the dataset role; construction
   fails closed if any measurement's role is a formal holdout role.
2. The loop remains what `refinement.py` defines — propose plan variants,
   evaluate, keep immutable history — with evaluation delegated to the real
   quality harness (`evaluate-quality` / `compare-quality` over the bound
   candidate) instead of proxy sensitivity aggregation.
3. One optimized candidate is then bound and frozen like any other candidate;
   the formal holdout sees it exactly once. A holdout failure archives the
   candidate; re-optimization requires freshly frozen datasets (existing
   campaign rule, restated here because the optimizer makes it tempting to
   violate).
4. The candidate history (all variants, their validation scores, and the
   selection rationale) ships inside the evidence bundle so the independent
   reviewer can check for validation overfitting.

## Alternatives rejected

- **Optimize on proxy metrics only** (status quo): leaves the documented gap;
  interaction effects between per-tensor choices are precisely what proxies
  miss.
- **Cross-validation over the holdout**: contradicts the consumed-on-contact
  holdout contract; rejected without further analysis.

## Consequences

- Interaction optimization cost is real evaluation cost; budgets must cap
  variant count (beam width / descent iterations already bounded).
- Validation-set overfitting risk is accepted, bounded by history transparency
  and the one-shot holdout.
