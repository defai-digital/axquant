# Experimental joint interaction diagnostic (1.9.0b1)

`axquant diagnose-joint` is a **beta try**, not a product planner.

It answers two questions that `axquant optimize` does not:

1. Do isolated weight and KV losses add, or is there a joint remainder
   \(I(W, KV) = \Delta Q(W_q, KV_q) - \Delta Q(W_q) - \Delta Q(KV_q)\)?
2. Under one memory budget, does the cheapest feasible
   `(weight BPW, KV bits)` pair flip when context length grows?

If \(I\) is small and the winner does not flip, the current independent
`optimize` path is enough. If either signal is large, a later joint
search may be worth writing. That decision needs this measurement first.

## What it is not

- Not a certificate, not a Hub claim, not a convert.
- Not the v2 joint optimizer in `docs/prd/weight-kv-joint-optimization.md`.
  Each grid cell still plans weights and KV independently, then accounts
  memory. There is no `WeightPlan × KVPlan` search.
- Isolated probe KL is a **proxy**. Only the optional quality quadruple is a
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
the four files fails closed. Winners and crossover are claimed only when
`--kv-analysis` provides a complete additive proxy; without it the grid is
feasibility-only.

## Output

| File | Meaning |
| --- | --- |
| `joint-interaction.json` | `axquant.joint-interaction.v1` |
| `joint-interaction.md` | Operator summary |

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

Stop if the first real-model pass is small and has no crossover. Do not
invent a named allocator just to have a paper.
