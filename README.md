# AXQuant

[![CI](https://github.com/defai-digital/axquant/actions/workflows/ci.yml/badge.svg)](https://github.com/defai-digital/axquant/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/axquant.svg)](https://pypi.org/project/axquant/)
[![Python versions](https://img.shields.io/pypi/pyversions/axquant.svg)](https://pypi.org/project/axquant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Reproducible quantization and verifiable model packs for MLX and CUDA.**

AXQuant is an independent, [MIT-licensed](LICENSE) product from **AutomatosX /
DEFAI Digital**. It allocates per-tensor precision, preserves sensitive model
components, and binds the source, plan and output bytes to reproducible
validation evidence. It does not train the source model or add capabilities.

We want **AXQ** model packs to be easy to produce, inspect and adopt across
inference tools. Independent producers, runtime integrations and community
forks are welcome. See [AXQ formats and runtime targets](docs/guides/axq-ecosystem.md)
for existing contracts, integration guidance and current limitations.

| Product line | Runtime targets | Support policy |
| --- | --- | --- |
| MLX AXQ: **2-bit, 3-bit, 6-bit, MXFP4, MXFP8** | **MLX-LM, MTPLX, oMLX, AX Engine** | First-class compatibility targets where the runtime supports the model, physical format and requested features; each combination needs validation |
| CUDA AXQ: **NVFP4** | **vLLM** as the primary target | Experimental NVFP4 and mixed NVFP4/FP8 exports; qualification depends on the model, GPU, kernel and vLLM version |

AX Engine is optimized for AXQ on Apple Silicon. Portable model loading does
not require AX Engine; its additional optimizations and performance claims
need their own evidence. Runtime targets are a product commitment, not a claim
that every model works in every runtime today.

**Status:** official toolkit release `2.3.0` (not a prerelease). The
historical validated-toolkit pin is `axquant[mlx]==1.8.1`; new installations
should use the current package. Neither install choice certifies a model.
The 2.0 line opens the experimental
[native CUDA NVFP4 backend](docs/guides/cuda-nvfp4.md) — RTN NVFP4 W4A16 exports
plus an opt-in calibrated W4A4 path — alongside the MLX planning line and its
joint weight/KV diagnostics (`diagnose-joint`, `plan-joint`) as development
evidence. CUDA runtime and quality certification remain open. All per-pack
certificate records were withdrawn on 2026-10-03 pending re-certification, so
there is currently **no** certified pack revision on the live catalog; the
certification index is the source of truth for what is certified today.

AXDocLayout RT-DETRv2 has a separate [development storage exporter](docs/guides/doclayout.md)
for NVFP4, MXFP4 and MXFP8, with FP32 reference execution.

Version 2.1 adds [explicit oMLX/MTPLX runtime exports](docs/guides/runtime-exports.md)
and manifest-bound compatibility checks for future MLX conversions. Unknown
peer profiles fail closed; static format checks do not establish generation
or certification. Install the toolkit with
`python -m pip install 'axquant[mlx]==2.3.0'` when using these commands.

Install from PyPI, then convert. You do not need to clone this repository.

## Who it is for

- You ship or evaluate MLX models on Apple Silicon, or experimental CUDA
  models for vLLM, and need controlled precision allocation.
- You need reproducible artifacts: pinned source, exact plan, measured BPW,
  checksums, and runtime metadata in every output.
- You publish packs and want fail-closed gates between a development
  checkpoint and a public quality or speed claim.

The conversion workflow expects a supported checkpoint at the `convertible`
tier or above. Training, GGUF export and arbitrary model-family conversion
are outside this workflow. Inspection, planning and reporting work without MLX.

## How it compares

| Tool | Strength | What AXQuant adds |
| --- | --- | --- |
| `mlx_lm.convert` | Fast HF-to-MLX conversion, affine / MXFP recipes | Architecture roles, hard protection floors, budget allocation, coverage checks, provenance-bound artifacts |
| MLX learned quants | Dynamic per-layer bit selection, DWQ / AWQ / GPTQ | Per-tensor planning under policy, replayable artifact graph, governed claim labels |
| Community MLX scripts | Speed of experimentation, broad coverage | Reproducibility, completeness checks, refusal of ambiguous success |
| llama.cpp quantize | GGUF breadth, cross-platform serving | Native Apple/MLX packs, AX Engine metadata, MTP and multimodal sidecar protection |

## Design beliefs

1. **Open formats and independent adoption.** MLX exports use public MLX
   packing; the CUDA exporter writes documented NVFP4 layouts for vLLM's
   compressed-tensors interface. AXQ adds precision policy and verifiable
   provenance. Format conformance and runtime execution are separate checks.
2. **Floors are constraints, not suggestions.** Sensitive components are
   never sacrificed to hit a pretty BPW number; infeasible budgets adjust
   or refuse, with a record.
3. **Prior, measurement, and certification are different things.**
   Architecture priors suit fast development; calibrated measurement
   supports allocation; only complete quality and hardware evidence
   supports a public claim.
4. **Success must be complete.** Every module a plan claims to quantize
   must actually be processed by the backend — no silent fallback.
5. **Claims follow artifact identity.** Quality, size, and speed claims
   belong to one source revision, one plan, one set of bytes, and one
   runtime and profile — never to a family by association.
6. **Certification aims at reproducibility, not universality.** A single
   factory host class keeps comparisons honest; checkpoint (Tier 1) and
   MTP acceleration (Tier 2) claims stay separate.

## Install

**Python 3.11+**. The MLX quick start below needs an Apple Silicon Mac and the
`mlx` extra. For experimental CUDA conversion targeting vLLM, use the separate
[CUDA installation and conversion guide](docs/guides/cuda-nvfp4.md).

**Always install into a virtual environment.** A bare system-wide install
fails with `externally-managed-environment`; do **not** pass
`--break-system-packages`. Copy the block as a whole (plain ASCII quotes;
single-quote the extra so zsh does not treat `[mlx]` as a glob):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install 'axquant[mlx]'
axquant --help
```

| Goal | Command |
| --- | --- |
| Convert / analyze / evaluate (typical) | `python -m pip install 'axquant[mlx]'` inside a venv |
| Inspect / plan / report only (no Metal) | `python -m pip install axquant` inside a venv |
| Global CLI via Homebrew tooling | `brew install pipx && pipx install 'axquant[mlx]'` |

Wheels and checksums also ship on
[GitHub Releases](https://github.com/defai-digital/axquant/releases).

## Convert

Point `quantize` at a local BF16 Safetensors directory. `--target-bpw`
defaults to 4.8. The result is a **development** checkpoint — good for
trying the model locally, not a public quality or speed claim.

```bash
axquant quantize /path/to/model-bf16
```

### Choose an MLX format

**Available in 2.3.0:** `--format`, `axquant formats` and `axquant --version`.
Upgrade with `python -m pip install --upgrade 'axquant[mlx]'` to use these commands.

```bash
axquant --version
axquant formats --backend mlx
axquant quantize /path/to/model-bf16 --format mxfp4
axquant quantize /path/to/model-bf16 --format 2bit
axquant quantize /path/to/model-bf16 --format 3bit
axquant quantize /path/to/model-bf16 --format 6bit
axquant quantize /path/to/model-bf16 --format mxfp8
```

These are base-format presets with higher precision for protected tensors,
not uniform whole-model bit widths. Omitting `--format` retains the existing mixed 4.8-BPW
planner. Presets use architecture priors and remain development evidence;
2/3-bit runtime admission gates and MXFP8's unmeasured status still apply.
See [format behavior and runtime coverage](docs/guides/axq-ecosystem.md).

```bash
# Choose the bit budget and output folder
axquant quantize /path/to/model-bf16 --target-bpw 4.8 --output ./AXQuant-output

# Hugging Face id (downloads only when you pass --allow-download)
axquant quantize Qwen/Qwen3.6-27B --allow-download --revision COMMIT_SHA

# Confirm the family is convertible before spending the convert
axquant inspect --model /path/to/model-bf16 --output inventory.json
```

For a model and physical format supported by MLX-LM, load the output with:

```bash
python -m pip install -U mlx-lm
mlx_lm.generate --model ./AXQuant-output --prompt "Hello" --max-tokens 64 --temp 0.0
```

For MTPLX, oMLX and AX Engine, check the exact pack's runtime requirements in
the [runtime guide](docs/guides/axq-ecosystem.md#runtime-coverage).
Some models need a distinct [runtime export](docs/guides/runtime-exports.md).
Text generation does not establish vision, audio or MTP support.

Prefer a ready-made pack? The live catalog is on
[AutomatosX on Hugging Face](https://huggingface.co/AutomatosX) — packs,
model cards, and certification status. This README deliberately does not
mirror the pack list.

Conversion needs enough unified memory for the source model. Not every
Hugging Face model is convertible: run `axquant support-matrix` and
`axquant support-policy` for the registry-derived tier
(`certified` / `convertible` / `inspect-only`) before beginning work.

## How it works

```text
BF16 Safetensors checkpoint (pin a revision for measured/release evidence)
                         |
                         v
        inspect -> plan -> convert -> runtime-check / validate
           |                |
           |                +-- mixed-precision MLX checkpoint
           |                    + plan, manifest, checksums, provenance
           |                    + AX Engine / MLX runtime metadata
           +-- inventory with protection boundaries
```

Most users only need [Install](#install) and [Convert](#convert). The
staged journey (inspect → plan → convert → validate → publish) is for
measured releases and public claims; see the
[certification operator guide](docs/guides/flagship-certification.md).

## Input and output

For the MLX workflow above, **input** is an unquantized Safetensors checkpoint of a family at the
`convertible` tier or above, converted through the promoted public backend
(MLX-LM for text, MLX-VLM / MLX-Audio with BF16-protected modality towers
for vision and audio). A pinned source revision is mandatory for measured
sensitivity and release evidence; an unpinned local source is permitted
only for development.

**Output:** a portable MLX model directory with mixed-precision weights,
the exact quantization plan, a checksum-bound artifact manifest recording
authoritative parameters and measured BPW, runtime metadata for the
supported runtimes, and byte-preserved MTP / modality sidecars unless an
explicit validated transform is selected. The language-model output stays
usable as a standard MLX checkpoint; AX Engine reads the extra metadata.

CUDA exports have separate versioned plans and manifests targeting vLLM;
see [CUDA NVFP4](docs/guides/cuda-nvfp4.md). MLX and CUDA packs have distinct
physical layouts and are not interchangeable just because both use four-bit
values. AXDocLayout uses a separate custom development storage format.

## CLI workflow

Run `axquant COMMAND --help` for the full options of any command. The main
entry points are `quantize` (one-command development conversion), `inspect`,
`analyze`, `plan`, `convert`, and `verify-cert` (offline certificate
check). The complete command table lives in the
[CLI reference](docs/cli-reference.md).

## Certification

The existing certification workflow is AX Engine-bound and separates
checkpoint claims from acceleration claims. Making other runtimes first-class
compatibility targets does not extend this frozen certification contract:

- **Tier 1 (checkpoint):** size, matched quality, conversion integrity,
  and standard-runtime compatibility for one exact revision.
- **Tier 2 (MTP, scoped):** greedy exactness plus token-weighted ≥1.20×
  and prompt-median ≥1.10× decode speedup on named authorizing workloads —
  measured evidence for that pack, engine build, and host only.

Official MLX conversion, Tier 1, and Tier 2 evidence are produced on one factory
host class: **Mac Studio, M2 Ultra, 192 GB unified memory**. Records name
the hardware spec, never a machine identity.

The rules are frozen in the
[Certification Spec v1.0](docs/contracts/certification-spec-v1.0.md); the
verdicts live in [docs/certifications/](docs/certifications/README.md),
including the [full list](docs/certifications/full-list.md) of every
record. A certificate never promotes a family by association.

## Documentation

Index: [docs/README.md](docs/README.md).

| I want to … | Start with |
| --- | --- |
| Quantize a model | [Convert](#convert), then the [CLI reference](docs/cli-reference.md) |
| Adopt AXQ or integrate a runtime | [AXQ formats and runtime targets](docs/guides/axq-ecosystem.md), [environment compatibility](docs/guides/compatibility.md) |
| Understand or verify an artifact | [Affine-U32 v1 contract](docs/contracts/axq-pack-interchange-v1.md), [certification status](docs/certifications/README.md) |
| Operate certification | [Certification operator guide](docs/guides/flagship-certification.md), [Certification Spec v1.0](docs/contracts/certification-spec-v1.0.md) |
| Contribute code | [Development](#development), [CONTRIBUTING.md](CONTRIBUTING.md), [CI root causes](docs/guides/ci-root-causes.md) |

Product requirements, the architecture decision register, technical
specifications, and the independent-implementation policy are maintained
internally and are not published in this repository.

## Development

CI splits surfaces on purpose: Ubuntu runs the non-MLX gate, macOS runs
the MLX suite. Prefer the local CI mirror:

```bash
./scripts/ci-local.sh
```

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
```

Tests use small synthetic Safetensors fixtures and do not require real
model weights. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing

**Developers are welcome to help build AXQuant and grow the AXQ ecosystem.**
[Fork AXQuant](https://github.com/defai-digital/axquant/fork), try a focused
change, and open a pull request against `main`. First-time contributors and
independent runtime or model-tool authors are welcome.

Useful starting points include documentation, reproducible bug reports,
synthetic tests, CLI usability, MLX-LM / MTPLX / oMLX / AX Engine integrations,
and vLLM compatibility. Many contributions need no Apple hardware, GPU or
model downloads. Share the checks you can run; maintainers can help identify
the additional evidence needed for hardware-specific work.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the fork-to-PR workflow.
Small fixes can go straight to a PR. For a new architecture, format or major
API change, start a [GitHub issue](https://github.com/defai-digital/axquant/issues/new/choose)
to agree on scope and validation. Draft PRs are welcome for early feedback.

## License

AXQuant is released under the [MIT License](LICENSE). Dependencies, model
checkpoints, calibration datasets, and external tools retain their own
licenses.
