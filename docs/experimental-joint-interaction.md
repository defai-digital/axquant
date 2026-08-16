# Experimental joint allocation (1.9.0b1)

AXQuant 1.9 is a try at **smarter allocation** under one memory budget,
not faster convert. AX Engine is speed; AXQuant chooses the precision mix.

`axquant diagnose-joint` asks two questions that `axquant optimize` does not:

1. Do isolated weight and KV losses add, or is there a joint remainder
   \(I(W, KV) = \Delta Q(W_q, KV_q) - \Delta Q(W_q) - \Delta Q(KV_q)\)?
2. Under one memory budget, does the cheapest feasible
   `(weight BPW, KV bits)` pair flip when context length grows?

`axquant plan-joint` is the search that can change the plan. \(I\) is a
gate, not the objective: small \(I\) keeps the 1.8 independent `optimize`
plan; material \(I\) ranks feasible `WeightPlan × KVPlan` cells with a
coupled proxy and writes convert-ready `axquant_plan.json`.

## What it is not

- Not a certificate, not a Hub claim.
- Not faster decode, MTP, KV kernels, or convert (AX Engine).
- Not the full v2 optimizer in `docs/prd/weight-kv-joint-optimization.md`
  (task-score ranking, per-tensor joint search, method as a decision).
  Each grid cell still plans weights and KV independently; the 1.9 search
  only chooses among those cells.
- Isolated probe KL is a **proxy**. Only the quality quadruple is a
  measured interaction.

## Run

```bash
axquant diagnose-joint \
  --model /path/to/model-bf16 \
  --max-memory 18GB \
  --contexts 4096,32768 \
  --weight-bpws 4.0,4.8,6.0 \
  --kv-bits 4,8,16 \
  --sensitivity measured-sensitivity.json \
  --kv-analysis measured-kv.json \
  --output ./joint-beta
```

Architecture priors need `--allow-unmeasured`. Those runs can map
feasibility, but they cannot claim a winner, a crossover, or a measured
\(I\).

To compute \(I(W, KV)\), add four matched `axquant.quality-evaluation.v2`
files from the same suite (same dataset hash, seed, generation config, and
task IDs):

```bash
  --quality-baseline eval-bf16.json \
  --quality-weight-only eval-weight-only.json \
  --quality-kv-only eval-kv-only.json \
  --quality-joint eval-joint.json \
  --interaction-threshold 0.02
```

`ΔQ` is `baseline_score - treatment_score` (signed). Supplying only some of
the four files fails closed. All four evaluations must use the same
`--model-id` as the inspected BF16 source (converted checkpoints live at
another path). Winners and crossover are claimed only when `--kv-analysis`
provides a complete additive proxy; without it the grid is feasibility-only.

## Output

| File | Meaning |
| --- | --- |
| `joint-interaction.json` | `axquant.joint-interaction.v1` |
| `joint-interaction.md` | Operator summary |
| `weight-plan-*.json` | Weight plans used for the grid (hash-verifiable) |
| `kv-plan-*.json` | KV plans used for the grid (hash-verifiable) |

`verdict` is one of:

- `insufficient-measured-interaction` — no quality quadruple
- `interaction-small` — `|I|` below the threshold
- `interaction-material` — `|I|` at or above the threshold

The markdown also lists **estimated feasible cells** per context. That table
is only a static byte budget; it is not a winner and not a quality ranking.

`crossover.detected` is true only when two **rankable** context winners pick
different `(target_bpw, kv_default_bits)` pairs. Equal-proxy ties keep the
lower KV bit-width, then the lower target BPW, then more leftover memory.

`evidence_kind` is never release-quality `measured`. Priors stay
`architecture_prior`; probe-backed runs are `measured_development`.

## How to read a result

| Result | What to do |
| --- | --- |
| `interaction-small` and no crossover | Keep `optimize` as the product path |
| `interaction-material` | Isolated additivity is wrong; a joint planner has a research reason |
| crossover at long context | The 1.8 shared-budget check is too coarse for long context |
| prior-only / no quality quadruple | Useful as a dry run only |

If the first real-model pass is small and has no crossover, `plan-joint`
correctly emits the 1.8 plan. Do not invent a heavier search just to have
a paper; I only earns a different cell when it is material.

## Turning the question into a plan

`axquant plan-joint` uses the same grid, then **changes the plan**:

- I small: emit the 1.8 independent `optimize` plan
- I material: rank feasible cells at `--context` with
  `additive + I * u(weight) * u(KV)` and write `axquant_plan.json`

A constant I does not change ranking. The coupling term does: when I is
positive, compressing weights and KV together is penalized, so the search
can keep one side higher — a cell 1.8 would not pick. Convert that plan
with `axquant convert --plan axquant_plan.json`.
