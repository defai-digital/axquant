# Known issues

As of AXQuant **v1.6.1**. Items here are documented limitations, not silent failures — each
fails closed or is gated behind an explicit flag. See
[GitHub Releases](https://github.com/defai-digital/axquant/releases) for what changed
between published versions.

## Quantization algorithms

- **GPTQ factorization cost is superlinear in the input dimension.** The damped Cholesky is
  O(in³) and peak memory is dominated by LAPACK internals (~3×in² fp32 of AXQuant buffers plus
  Accelerate scratch; measured 4.7 GB peak at in = 8192 on an M-series machine). Large
  `down_proj` modules (in = intermediate size) can take tens of seconds and GB-scale memory per
  module. This is inherent to second-order methods; plan for it on 27B-class checkpoints.
- **GPTQ act-order is group-preserving, not `g_idx`-style.** The `gptq-act` method (ADR-0002)
  orders whole groups by aggregate Hessian mass and columns within groups by Hessian diagonal,
  so the packed layout stays byte-compatible with portable affine packing. It therefore captures
  less of the classic act-order gain than toolchains that ship a `g_idx` permutation tensor —
  adoption is decided per tensor by the measured frontier, not assumed.
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

## Public Hub catalog

- **Some bases do not publish an AXQ-4bit pack.** When protection floors raise both the ~4.8 and
  ~6.0 BPW budgets to the same (or near-identical) artifact, publishing a separate `4bit` sibling
  is misleading. Removed from Hugging Face (use the `6bit` pack only):
  `AX-Qwen3.5-9B-MLX-AXQ-4bit-MTP`, `AX-MiniCPM5-1B-MLX-AXQ-4bit`, and
  `AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit`. See
  [model-fleet-v2.md](model-fleet-v2.md#floor-collapsed-4bit-retirement) and the Hub catalog
  section in the main README.

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

## Overlap and multilingual text

- **Coding-suite / general-holdout overlap use `axquant-token-5gram-v2` (from v1.5.1).**
  The same CJK-aware tokenizer as `campaign-overlap`. Manifests or reports produced under
  the older v1 algorithm must be regenerated with `prepare-coding-suite` /
  `prepare-general-overlap` before a formal freeze.
- **`campaign-overlap --id-field` is repeatable** (default order: `id`, then `task_id`) so
  one run can span calibration corpora and strict `QualityTask` suites.

## CI and release

- **Ubuntu non-MLX jobs are the hard gate, not macOS MLX.** Generation-smoke and gemma4
  shard-path contracts must pass with `.[dev]` only. Run `./scripts/ci-local.sh` before
  pushing; see [ci-root-causes.md](ci-root-causes.md) for the historical red-run analysis.
- **Install from PyPI, not the Packages tab.** Canonical install is
  [pypi.org/project/axquant](https://pypi.org/project/axquant/) (`pip install axquant`).
  The repo [Packages](https://github.com/defai-digital/axquant/packages) page is for
  npm/container/Maven-style registries and stays empty for this Python project; wheels
  also appear under [Releases](https://github.com/defai-digital/axquant/releases).
- **PyPI publish is gated and enabled for this project.** The Release `pypi` job runs only
  when repository variable `ENABLE_PYPI_PUBLISH` is `true` and a Trusted Publisher exists
  on pypi.org for `defai-digital/axquant` / `release.yml`. For axquant both are configured,
  so tagged releases publish to PyPI. Details: [ci-root-causes.md](ci-root-causes.md).

## Validated scope

- **Large-model certification is exact-revision scoped.** On `df-macbookpro-m5`, dense Qwen 3.6
  27B AXQ 6-bit v3 and 27B AXQ 4-bit (5.6 BPW) have checkpoint Tier 1 and scoped MTP Tier 2
  ([index](certifications/README.md)). 35B-A3B MoE packs have Tier 1 only; formal MTP
  exactness is achievable after engine MoE `experts.gate_up_proj` load support, but speedup
  gates (≥1.20× / ≥1.10×) are not met. Product default remains MTP direct fallback; the formal
  acceleration route is opt-in. Vision paths and short-answer universal speedup remain
  uncertified.
