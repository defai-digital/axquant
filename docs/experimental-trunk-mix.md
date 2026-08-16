# Experimental trunk mix (2/3/4-bit)

Development evidence only. Not a certificate. Not mlx-optiq. Not `plan-joint`.

Heuristic Flash-0731 recipes (uniform 2-bit, then hand-raised shared/edge/down-proj)
did not beat `mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit` on the factory short
QA suite. `plan-joint` cannot search 2-bit (its grid stays 4.0/4.8/6.0 × KV4/8/16).

`axquant plan-experimental-mix` is the next AXQ-owned lever:

- Robust trunk (MLP / expert) starts at 2-bit affine and may climb to 3- or 4-bit.
- Each fused MLX-LM switch module is **one** allocation unit. Every expert in
  `layers.N.ffn.experts.*.w1` (or the Qwen/Nemotron equivalent) shares one
  `(bits, method, group_size)` before any budget is spent.
- Upgrade order is measured ranking loss per extra storage bit from a
  sensitivity report. Architecture priors need `--allow-unmeasured`.
- Attention stays ≥ 4-bit (RM-42). Floors stay 8/16. Fused/packed stacks stay affine.

```bash
axquant plan-experimental-mix \
  --sensitivity sensitivity.json \
  --target-bpw 3.6 \
  --output mix-plan.json
```

Convert is unchanged. Experimental AX Engine gates still apply
(`AX_ENGINE_2BIT_EXPERIMENTAL=1`, `AX_ENGINE_3BIT_EXPERIMENTAL=1`).

A full Flash-0731 measured sensitivity is tens of thousands of tensors and is
not a CI or laptop job. Unit tests drive the shipped functions on a handful of
fused modules.

## Flash-0731 quality gap (factory)

Uniform 2-bit and heuristic mixed 2/3/4 lose because the model is **incoherent**,
not because one extra bit on down-proj was missing:

- AXQ 2-bit coding outputs echo constraints ("Do not use modulo...") instead of
  writing functions. Mean 0.133 vs OptiQ 0.500.
- AXQ 2-bit general often leaks the prompt or JSON scaffolding. Mean 0.487
  vs OptiQ 0.948 (14/15).
- Mixed 2/3/4 got worse (mean 0.300).

The follow-up pack is 4-bit affine on the robust trunk (optional DWQ clip
that no longer flattens fused stacks past MLX int32).

## Flash-0731 status (2026-08-16)

`df-macstudio-m2` is reachable and has the 0731 inventory plus both AXQ packs
and OptiQ. `plan-experimental-mix` runs on that inventory (72,317 tensors,
129 fused switch units, mixed fused signatures = 0) from an architecture
prior in about 10s. Policy minimum is **3.97 BPW** at group 32.

That prior uses the same role-level KL for every expert, so the upgrade
order is **not** measured. A Studio convert+short-QA vs OptiQ was **not**
started: it would repeat the failed heuristic mix under a new name. The
blocker is a measured Flash sensitivity (or a cheaper measured proxy that
is still AXQ-owned), not the mixer.
