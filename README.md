# AXQuant

[![CI](https://github.com/defai-digital/axquant/actions/workflows/ci.yml/badge.svg)](https://github.com/defai-digital/axquant/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/axquant.svg)](https://pypi.org/project/axquant/)
[![Python versions](https://img.shields.io/pypi/pyversions/axquant.svg)](https://pypi.org/project/axquant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AXQuant is the offline precision and evidence control plane for Apple Silicon.
It allocates per-tensor precision under explicit policy, materializes
checkpoints through public MLX backends, and binds source identity, plan,
output bytes, runtime evidence, and — when certified — permitted claims into a
checksum-verifiable artifact graph. AX Engine is the primary inference
runtime; performance claims require bound measurements.

In short: MLX encodes, AX Engine runs, and AXQuant decides the precision mix —
then proves what the result may claim. It does not train the source model or
add new capabilities.

**Status:** toolkit `1.9.0`, packaging classifier **Beta**. The certified PyPI
pin stays `axquant[mlx]==1.8.1`; the 1.9 line adds joint weight/KV diagnostics
(`diagnose-joint`, `plan-joint`) as development evidence. CUDA / NVFP4 is 2.x.
All per-pack certificate records were withdrawn on 2026-10-03 pending
re-certification, so there is currently **no** certified pack revision on the
live catalog; the certification index is the source of truth for what is
certified today.

Install from PyPI, then convert. You do not need to clone this repository.

## Who it is for

- You ship or evaluate quantized models on Apple Silicon and want mixed
  precision with protection floors, not blind uniform quantization.
- You need reproducible artifacts: pinned source, exact plan, measured BPW,
  checksums, and runtime metadata in every output.
- You publish packs and want fail-closed gates between a development
  checkpoint and a public quality or speed claim.

Not for: training or fine-tuning, non-Apple platforms, GGUF workflows, or
any model family below the `convertible` support tier.

## How it compares

| Tool | Strength | What AXQuant adds |
| --- | --- | --- |
| `mlx_lm.convert` | Fast HF-to-MLX conversion, affine / MXFP recipes | Architecture roles, hard protection floors, budget allocation, coverage checks, provenance-bound artifacts |
| MLX learned quants | Dynamic per-layer bit selection, DWQ / AWQ / GPTQ | Per-tensor planning under policy, replayable artifact graph, governed claim labels |
| Community MLX scripts | Speed of experimentation, broad coverage | Reproducibility, completeness checks, refusal of ambiguous success |
| llama.cpp quantize | GGUF breadth, cross-platform serving | Native Apple/MLX packs, AX Engine metadata, MTP and multimodal sidecar protection |

## Design beliefs

1. **Precision allocation, not codec invention.** AXQuant chooses which
   tensors deserve which precision; public MLX backends do the encoding.
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

Apple Silicon Mac and **Python 3.11+**. Conversion needs the MLX extra; that
stack only runs on arm64 macOS.

**Always install into a virtual environment.** A bare system-wide install
fails with `externally-managed-environment`; do **not** pass
`--break-system-packages`. Copy the block as a whole (plain ASCII quotes;
single-quote the extra so zsh does not treat `[mlx]` as a glob):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install 'axquant[mlx]==1.8.1'
axquant --help
```

| Goal | Command |
| --- | --- |
| Convert / analyze / evaluate (typical) | `python -m pip install 'axquant[mlx]==1.8.1'` inside a venv |
| Inspect / plan / report only (no Metal) | `python -m pip install 'axquant==1.8.1'` inside a venv |
| Global CLI via Homebrew tooling | `brew install pipx && pipx install 'axquant[mlx]==1.8.1'` |

Wheels and checksums also ship on
[GitHub Releases](https://github.com/defai-digital/axquant/releases).

## Convert

Point `quantize` at a local BF16 Safetensors directory. `--target-bpw`
defaults to 4.8. The result is a **development** checkpoint — good for
trying the model locally, not a public quality or speed claim.

```bash
axquant quantize /path/to/model-bf16
```

```bash
# Choose the bit budget and output folder
axquant quantize /path/to/model-bf16 --target-bpw 4.8 --output ./AXQuant-output

# Hugging Face id (downloads only when you pass --allow-download)
axquant quantize Qwen/Qwen3.6-27B --allow-download --revision COMMIT_SHA

# Confirm the family is convertible before spending the convert
axquant inspect --model /path/to/model-bf16 --output inventory.json
```

Load the output with MLX-LM:

```bash
python -m pip install -U mlx-lm
mlx_lm.generate --model ./AXQuant-output --prompt "Hello" --max-tokens 64 --temp 0.0
```

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

**Input:** an unquantized Safetensors checkpoint of a family at the
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

## CLI workflow

Run `axquant COMMAND --help` for the full options of any command. The main
entry points are `quantize` (one-command development conversion), `inspect`,
`analyze`, `plan`, `convert`, and `verify-cert` (offline certificate
check). The complete command table lives in the
[CLI reference](docs/cli-reference.md).

## Certification

AXQuant separates checkpoint claims from acceleration claims:

- **Tier 1 (checkpoint):** size, matched quality, conversion integrity,
  and standard-runtime compatibility for one exact revision.
- **Tier 2 (MTP, scoped):** greedy exactness plus token-weighted ≥1.20×
  and prompt-median ≥1.10× decode speedup on named authorizing workloads —
  measured evidence for that pack, engine build, and host only.

Conversion, Tier 1, and Tier 2 evidence are all produced on one factory
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
| Understand or verify an artifact | [Pack interchange contract](docs/contracts/axq-pack-interchange-v1.md), [certified checkpoints](docs/certifications/README.md) |
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

Contributions are warmly welcome: bug fixes, documentation, tests,
usability improvements, runtime compatibility, architecture adapters, or
reproducible quantization research. Fork the repository and send a pull
request; for a substantial design change or new model family, open a
[GitHub issue](https://github.com/defai-digital/axquant/issues) first.

## License

AXQuant is released under the [MIT License](LICENSE). Dependencies, model
checkpoints, calibration datasets, and external tools retain their own
licenses.
