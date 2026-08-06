# ADR-0002 — Group-preserving act-order for GPTQ

- **Status:** accepted
- **Date:** 2026-08-06

## Context

`gptq.py` implements static-grid GPTQ: columns are processed in natural order
and rounded onto per-group asymmetric affine grids, with error propagated
through the Cholesky factor of the damped inverse Hessian. `docs/known-issues.md`
records act-order as not implemented.

Classic act-order (process columns in descending Hessian-diagonal order)
improves low-bit quality, but with grouped quantization it normally reassigns
group membership by processing order. Toolkits that do this ship a `g_idx`
permutation tensor so the kernel can map columns back to groups. **AXQuant's
portable affine packing contract has no permutation metadata** — MLX-LM's
affine format expects contiguous groups — so classic act-order would break
every downstream runtime.

## Options considered

1. **Global permutation + `g_idx` metadata.** Best quality on paper; breaks
   the portable contract and requires kernel support in MLX-LM and AX Engine.
   Rejected.
2. **Group-preserving act-order (chosen).** Two nested orderings that never
   change group membership:
   - *Inter-group ordering:* process whole groups in descending aggregate
     Hessian mass, so high-salience groups quantize early and push their error
     into lower-salience groups.
   - *Intra-group ordering:* within a group, process columns in descending
     Hessian-diagonal order.
   After quantization, columns are written back in original positions; the
   packed artifact is indistinguishable in layout from static-grid GPTQ.
3. **Do nothing.** Leaves a documented quality gap at low bits. Rejected.

## Decision

Implement option 2 in `gptq.py` behind an opt-in flag, surfaced as a distinct
candidate method label (per ADR-0001) so the probe measures it against
static-grid GPTQ rather than replacing it. If measurement shows no win on the
reference model, we keep the label and record the negative result — the
measured frontier, not the literature, decides adoption.

## Consequences

- Portable packing contract untouched; no runtime changes needed.
- Expected quality gain is smaller than `g_idx`-style act-order; that is the
  accepted price of portability. If a future AX Engine kernel adds permutation
  support, a separate ADR can revisit option 1 as an AX-Engine-only label.
- Sensitivity probing cost roughly doubles for GPTQ candidates when the flag
  is on (two labels probed); mitigated by targeting act-order probes at
  tensors where static GPTQ already trails the affine baseline.
