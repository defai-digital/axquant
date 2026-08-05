# Known issues

AXQuant v1.2.0. Items here are documented limitations, not silent failures — each fails closed
or is gated behind an explicit flag.

## Quantization algorithms

- **GPTQ factorization cost is superlinear in the input dimension.** The damped Cholesky is
  O(in³) and peak memory is dominated by LAPACK internals (~3×in² fp32 of AXQuant buffers plus
  Accelerate scratch; measured 4.7 GB peak at in = 8192 on an M-series machine). Large
  `down_proj` modules (in = intermediate size) can take tens of seconds and GB-scale memory per
  module. This is inherent to second-order methods; plan for it on 27B-class checkpoints.
- **GPTQ has no activation ordering (act-order) yet.** v1 implements the static-grid baseline;
  act-order is a known quality improvement for low bits and may land in a future release.
- **GPTQ weight-space MSE can exceed RTN.** GPTQ minimizes the activation-weighted reconstruction
  objective, not weight distance; judge it by output metrics (`output_kl`,
  `hidden_state_error`), not `mean_quant_error`.
- **AWQ's reconstruction grid search uses at most 256 calibration rows** (channel magnitudes use
  all captured rows). This mirrors the reference AWQ cost model.
- **AWQ/GPTQ are unavailable for fused MoE expert modules** (`SwitchLinear`); fused groups remain
  affine-only by design. Non-fused expert Linears can use AWQ/GPTQ normally.
- **2-bit and 3-bit remain experimental**, gated by AX Engine's documented switches; GPTQ at 2/3
  bits is allowed but quality at 2-bit scalar grids is limited by the packing format, not the
  optimizer.

## Activation capture

- **Capture artifact size** before compression is roughly
  `modules × max_rows × in_features × 2 bytes` (fp16). Use `--max-rows`, `--target-module`, or
  `--modules-per-shard` to control footprint.
- **Capture resume is binding-sensitive by design.** Rerunning with a different model, cache,
  `--max-rows`, `--segment-batches`, or token budget against an existing `capture_progress.json`
  is rejected; use a fresh output directory.
- **Completed capture outputs are immutable.** A second capture will not overwrite a directory
  containing `completion.json`; choose a new output path. Committed resume chunks are shape- and
  row-accounting-verified before replay continues.
- **Hub capture resolution follows the cache revision.** For a Hub model, use a tokenized cache
  with an immutable revision (or pass the same value via `--revision`). Prepared BF16 sources with
  `axquant_source.json` fail closed when their source ID or revision differs from the cache.
- **Capture only wraps `nn.Linear` leaves.** Custom architectures that route through non-Linear
  projections are not captured (they are also outside the refinement predicate's scope).

## Evidence and state

- **Probe resume state does not survive backend version bumps** (v4 → v5 → v6 in v1.1.x). Saved
  `--state` files from older versions are rejected; rerun the analysis.
- **Architecture priors never emit AWQ/GPTQ candidates.** Prior-based plans stay affine/DWQ;
  AWQ/GPTQ require measured evidence (`analyze --calibration ... --calibration-activations ...`)
  or an explicit manual recipe.
- **Plain dense backbones expose logits only.** Measured probing on non-multimodal layouts
  (MiniCPM5, Qwen3 dense, Mistral, …) must pass `analyze --capture-points output`; the default
  `("output", "hidden")` fails closed with a hint, and reports produced this way carry no
  hidden-state metrics.
- **Release attestations begin with the first tag containing the release workflow.** The workflow
  verifies tag/package version equality, creates the GitHub Release when needed, uploads the wheel,
  sdist, and checksums, and emits keyless GitHub build provenance. Verify with
  `gh attestation verify` and `shasum -c SHA256SUMS.txt`.
- **Pre-fix Qwen3-Coder-Next artifacts must be regenerated.** The family adapter previously let
  `switch_mlp` fall through to the ordinary MLP role, so packed 3-D experts were preserved at
  BF16. Their low-bit repository names are not proof of low effective BPW; regenerate and rerun
  measured size/quality gates with the post-v1.1.1 fix before republishing.

## CI and release

- **Ubuntu non-MLX jobs are the hard gate, not macOS MLX.** Generation-smoke and gemma4
  shard-path contracts must pass with `.[dev]` only. Run `./scripts/ci-local.sh` before
  pushing; see [ci-root-causes.md](ci-root-causes.md) for the historical red-run analysis.
- **PyPI upload is opt-in.** The Release `pypi` job runs only when repository variable
  `ENABLE_PYPI_PUBLISH=true`. Without a Trusted Publisher on pypi.org for
  `defai-digital/axquant` / `release.yml`, leave the variable unset so tag Releases stay
  green (GitHub assets still publish). Steps to enable real uploads:
  [ci-root-causes.md](ci-root-causes.md).

## Validated scope

- **End-to-end quality has been validated on tiny/synthetic models only.** The 27B-class
  certification track requires a real-hardware run (capture → analyze → plan → convert →
  validate) before any quality claim; that run is deliberately outside the v1.2.0 scope.
