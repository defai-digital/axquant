# AXQuant

[![CI](https://github.com/defai-digital/axquant/actions/workflows/ci.yml/badge.svg)](https://github.com/defai-digital/axquant/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/axquant.svg)](https://pypi.org/project/axquant/)
[![Python versions](https://img.shields.io/pypi/pyversions/axquant.svg)](https://pypi.org/project/axquant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AXQuant is a command-line toolkit that converts a supported, unquantized Safetensors checkpoint
into an AXQuant-optimized MLX checkpoint for Apple Silicon.

It inspects the model, creates an auditable mixed-precision plan, converts the weights through
public MLX-LM, MLX-Audio, or MLX-VLM interfaces, and writes the manifests and validation metadata
needed by the selected runtime.
AXQuant can assign 4-bit, 6-bit, 8-bit, or BF16 precision per tensor while protecting sensitive
components such as normalization layers, output heads, routers, vision/audio tensors, and
multi-token-prediction (MTP) weights.

> AXQuant improves deployment efficiency; it does not train the source model or add new learned
> capabilities. Its goal is to reduce storage and unified-memory cost while preserving important
> model quality and runtime behavior.

**Ready-made development packs:** [AutomatosX on Hugging Face](https://huggingface.co/AutomatosX)
([MLX catalog](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog)) —
not certified releases; full table under [Current status](#current-status).

## Contents

**Start here**

- [Install (Mac / Apple Silicon)](#install-mac-apple-silicon)
- [At a glance](#at-a-glance)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Input and output](#input-and-output)
- [Why AXQuant](#why-axquant)

**Product status and packs**

- [Current status](#current-status) (tiers, Hub catalog, known gaps)

**Operators**

- [From-source install (Mac)](#from-source-install-mac)
- [Simple development conversion](#simple-development-conversion)
- [Staged development conversion](#staged-development-conversion)
- [CLI workflow](#cli-workflow)
- [Measured planning and validation](#measured-planning-and-validation)
- [Evidence and safety boundaries](#evidence-and-safety-boundaries)
- [Model naming](#model-naming)

**Contributors**

- [Development](#development)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)

## Install (Mac / Apple Silicon)

AXQuant is built for **Macs with Apple Silicon** (M1-M5). Conversion and measured analysis need
the MLX stack; that only runs on arm64 macOS. Use **Python 3.11+** (Homebrew
`brew install python@3.13` or [python.org](https://www.python.org/downloads/macos/) both work).

**Always install into a virtual environment.** Homebrew Python is
[PEP 668](https://peps.python.org/pep-0668/) externally managed: a bare
`python -m pip install axquant` against system Python fails with
`externally-managed-environment`. That is expected; do **not** pass
`--break-system-packages`.

Copy the block below as a whole (plain ASCII quotes only; single-quote the extra so zsh
does not treat `[mlx]` as a glob):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install 'axquant[mlx]'
axquant --help
```

With the venv active, `python` and `axquant` are both from `.venv`. Without activate, call
`.venv/bin/axquant` directly.

| Goal | Command |
| --- | --- |
| Full convert / analyze / evaluate (typical) | `python -m pip install 'axquant[mlx]'` inside a venv |
| Inspect / plan / report only (no Metal backends) | `python -m pip install axquant` inside a venv |
| Global CLI via Homebrew tooling | `brew install pipx && pipx install 'axquant[mlx]'` |

If zsh prints `missing end of string`, a curly/smart quote usually got pasted. Re-type the
line or paste only from the fenced block above (do not copy prose with `"` / `'` glyphs).

Package index: [pypi.org/project/axquant](https://pypi.org/project/axquant/). Wheels and checksums
also ship on [GitHub Releases](https://github.com/defai-digital/axquant/releases) (not the GitHub
Packages tab; that UI is for npm/containers, not pip).

## At a glance

- **What it does:** turns a supported BF16 checkpoint into a mixed-precision MLX checkpoint for
  Mac Apple Silicon, assigning bits per tensor instead of one flat width for the whole model.
- **Install and convert:**
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  python -m pip install 'axquant[mlx]'
  axquant quantize /path/to/model-bf16 --target-bpw 4.8
  ```
- **Naming isn't the bit budget:** existing `4bit`/`6bit` development-pack names are planning
  classes, not fixed-width claims — for example the public
  `AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP` pack measures ~5.42 BPW. A future certified flagship must
  use the measured-BPW form `AX-<Base>-MLX-AXQ-MP-<N>bpw[-MTP]`; manifests remain authoritative.
- **Support:** Qwen 3.6, Qwen 3.5, Qwen3 dense/Embeddings, Qwen3-Next/Coder-Next,
  Qwen3-ASR, Qwen3-VL, MiniCPM5, Gemma-4, Mistral/Devstral/Ministral, and Nemotron 3 Nano —
  see the tier matrix under [Current status](#current-status).
- **Where it stands:** the toolkit is feature-complete and tested, but no checkpoint has yet
  cleared the full M0–M8 release audit — every public pack today is development evidence, not a
  certified release. [Current status](#current-status) states exactly what remains open.

## Quickstart

This path creates a **development** conversion from a local, unquantized Safetensors checkpoint
on an Apple Silicon Mac. It is the quickest way to verify compatibility; it is not a substitute
for the measured calibration, evaluation, and release-audit evidence required for a public
quality or performance claim.

```bash
git clone https://github.com/defai-digital/axquant.git
cd axquant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[mlx]'

MODEL=/absolute/path/to/bf16-safetensors-model
axquant inspect --model "$MODEL" --output inventory.json
axquant quantize "$MODEL" --output ./AXQuant-output --target-bpw 4.8
```

Read `inventory.json` before conversion: it records the adapter, support tier, protection
boundaries, and any blocking source-layout issue. Use an explicit `--target-bpw` at or above the
reported floor when the budget must be fixed. For a measured or publishable workflow, continue
with [measured planning and validation](#measured-planning-and-validation) before making a
quality or performance claim.

## How it works

```text
BF16 Safetensors checkpoint supported by its promoted MLX backend
        (pin a revision for measured/release evidence)
                         │
                         ▼
        inspect → plan → convert → runtime-check / validate
           │                │
           │                └── AXQuant-optimized MLX checkpoint
           │                    + AX Engine runtime metadata
           │                    + plan, manifest, and provenance
           └── model inventory and protection boundaries
```

The intended user journey is:

1. provide a supported, unquantized LLM checkpoint;
2. inspect its architecture and tensors;
3. generate a manual or evidence-based mixed-precision plan;
4. convert it from the command line;
5. verify runtime compatibility, model quality, memory use, and speed;
6. publish only after the required validation gates pass.

## Input and output

### Input

AXQuant converts unquantized Safetensors checkpoints of families at the `convertible` tier or
above through the public runtime backend promoted for that architecture. Text families use
MLX-LM. Qwen3-VL uses MLX-VLM with its vision tower protected at BF16. Qwen3-ASR uses MLX-Audio
with its audio tower protected at BF16; the pinned upstream `thinker.*` checkpoint must first be
normalized with `scripts/hf_to_mlx_bf16.py`, which records `axquant_source.json`. Planning directly
from the unnormalized ASR export is rejected because it contains a duplicated tied LM head and
runtime-specific tensor layouts.

A pinned source revision is mandatory for measured sensitivity and release evidence; an unpinned
local source is permitted only for development workflows. MoE expert stacks quantize as fused
switch modules with a uniform per-group precision, and routers keep an 8-bit floor. The checkpoint
must use the expected configuration and indexed Safetensors layout. Remaining recognized families
(for example Nemotron Super/Ultra) stay inspect-only until promotion evidence exists.

### Output

A successful conversion produces a new portable MLX model directory containing:

- mixed-precision model weights and standard MLX configuration files;
- the exact quantization plan used for the conversion;
- an AXQuant artifact manifest with checksums and provenance;
- architecture-specific runtime metadata for AX Engine, MLX-LM, MLX-Audio, or MLX-VLM;
- an AX Engine native manifest when the runtime tool is available;
- a byte-preserved external MTP sidecar by default, or an explicitly prepared development
  sidecar with transform-level provenance;
- a raw, checksummed BF16 sidecar for protected vision tensors when MLX-LM excludes them, or
  protected modality tensors in the main MLX-Audio/MLX-VLM checkpoint.

The artifact manifest records authoritative main-model and total logical parameters, physical
Safetensors bytes, and measured BPW. The language-model output remains usable as a standard MLX
checkpoint. AX Engine consumes the additional AXQuant metadata for runtime-specific behavior;
MLX-LM may ignore that metadata and use ordinary decode.

## Why AXQuant

Uniform quantization gives every eligible tensor the same precision; rule-based per-module
overrides assign precision by name pattern. AXQuant instead allocates precision per tensor from
a budget-constrained solve over measured sensitivity, so the model spends more bits where the
measurement shows it matters and fewer where it does not.

Its design centers on:

- **mixed precision:** 4-bit, 6-bit, 8-bit, and BF16 assignments, with an experimental
  2/3-bit range for robust trunk tensors (AX Engine gates them behind
  `AX_ENGINE_2BIT_EXPERIMENTAL` / `AX_ENGINE_3BIT_EXPERIMENTAL`);
- **quality protection:** hard precision floors for sensitive model components;
- **MTP awareness:** explicit MTP detection, protection, validation, and runtime metadata;
- **workload awareness:** separate objectives for general and agent/coding workloads;
- **real deployment cost:** actual artifact bytes, unified memory, latency, and throughput;
- **reproducibility:** revision-pinned release inputs, deterministic artifacts, checksums, and
  manifests;
- **fail-closed conversion:** incomplete plans or unmatched modules stop conversion;
- **independent implementation:** public APIs and research without reused quantizer internals.

## Current status

The latest tagged toolkit version is `1.5.1` (packaging classifier: **Beta**). Its inspection,
planning, conversion, runtime-check, validation, and publication-gating commands are implemented
and covered by the test suite. Certification is checkpoint- and evidence-specific; a working
command does not by itself certify an output.

### v1.5.x at a glance

- Coding-suite and general-holdout overlap share `campaign-overlap`'s CJK-aware
  `axquant-token-5gram-v2` tokenizer; regenerate coding-suite manifests built under v1.
  GPTQ column codes use the joint `round(w/s + z)` form shared with AWQ (v1.5.1).
- The flagship formal-host identifier is `df-macbookpro-m5` (was `mbp-m5`), matching the
  certification machine's canonical DNS identity; the id is a schema literal on the host
  contract, preflight, and certified-claim hardware scope. No campaign or claim ever bound
  the old id (v1.5.0).

### v1.4.x at a glance

- `campaign-overlap` normalization is Unicode-aware (`axquant-token-5gram-v2`): CJK and other
  non-ASCII scripts now produce real shingles instead of failing closed, unblocking flagship
  campaign freezes over multilingual datasets; ASCII-only reports are byte-identical to v1.
- `campaign-overlap --id-field` is repeatable with ordered fallback (default `id`, then
  `task_id`), so one overlap run spans calibration corpora and strict `QualityTask` suites
  (v1.4.1).
- Quantized MTP sidecars can emit AX Engine's MLX-packed layout (`mlx-affine-packed-u32`)
  with round-trip verification and `--runtime-json` `mtp_sidecar_bits` stamping. Capability
  and contract gates still fail closed until an AX Engine build reports that layout as
  executable, so **shipped public packs keep byte-preserved sidecars** (see still-incomplete
  list below).
- `benchmark-kernels --from-ax-engine` ingests the engine's raw kernel-latency documents into
  host-scoped tables that plug directly into `plan --latency-table`.

See the [v1.5.1 release notes](https://github.com/defai-digital/axquant/releases/tag/v1.5.1)
for the complete change list and download verification instructions. Past tags keep their
notes on [GitHub Releases](https://github.com/defai-digital/axquant/releases); the next
tag's curated body is prepared under [docs/releases/](docs/releases/README.md).

### Support snapshot

| Scope | Use today | Public certification status |
| --- | --- | --- |
| Qwen 3.6 language paths | `convertible`; primary certification track | No certified public pack yet |
| Qwen 3.5, Qwen3 dense/Embedding/Next, MiniCPM5, Gemma-4, Mistral/Devstral/Ministral | `convertible` through their promoted MLX text paths | Development evidence only |
| Qwen3-ASR 1.7B and Qwen3-VL 8B Instruct | `convertible` with protected modality towers and their MLX-Audio/MLX-VLM backends | Development evidence only |
| Nemotron 3 Nano | `convertible` thin path | Development evidence only |
| Other or unmatched checkpoints | `inspect-only` | Not eligible for conversion or certification |

The detailed registry-derived matrix below is authoritative. Run `axquant support-matrix` for the
exact tier of a checkpoint before beginning work.

Further reading: [AXQ model fleet v2 migration and audit](docs/model-fleet-v2.md),
[migration guide (v1.1.x → v1.2.0)](docs/migration-v1.2.md),
[migration guide (v1.0.x → v1.1.x)](docs/migration-v1.1.md),
[environment compatibility matrix](docs/compatibility.md), and
[known issues](docs/known-issues.md).

Release artifacts are built and signed (keyless Sigstore attestation) by the release workflow;
verify a downloaded dist with `gh attestation verify <file> --repo defai-digital/axquant` and
`shasum -a 256 -c SHA256SUMS.txt`.

What remains evidence-gated is AXQuant's own **certified public model release**: publishing a
checkpoint under an AXQuant quality/performance claim requires every M0–M8 gate to pass on
formal hardware. There is **no** certified public AXQuant model release claimed here yet.
The public Qwen 3.6 packs are development artifacts, and their release evidence chains are not
closed. Same-candidate dual-profile quality comparison, AX Engine evidence, MTP speed, Pareto and
hardware-registry evidence, compatibility coverage, and the full M0–M8 audit remain required. Do
not treat any single metric as the sole remaining certification blocker.

AXQuant records an evidence-backed **support tier** for every recognized model family
(`certified` / `convertible` / `inspect-only`). Conversion requires at least the `convertible`
tier; tier promotion requires recorded promotion evidence, and certification requires the full
release audit. The current tier matrix:

| Family | Adapter | Tier |
| --- | --- | --- |
| Qwen 3.6 (27B dense + 35B-A3B MoE language paths) | `qwen36-v1` | `convertible`; primary certification track |
| Qwen 3.5 dense | `qwen35-dense-v1` | `convertible`; development claims only |
| **Qwen3-Next / Coder-Next** (hybrid MoE) | `qwen3-next-v1` | `convertible`; development claims only; fused experts |
| **Qwen3 dense + Embeddings** (`model_type=qwen3`) | `qwen3-dense-v1` | `convertible`; includes Qwen3-Embedding-0.6B/4B/8B |
| **Qwen3-ASR 1.7B** | `qwen3-asr-v1` | `convertible` after pinned MLX-Audio BF16 normalization; audio tower protected |
| **Qwen3-VL 8B Instruct** | `qwen3-vl-v1` | `convertible` through MLX-VLM; vision tower protected |
| MiniCPM5 dense | `minicpm5-dense-v1` | `convertible`; development claims only |
| Gemma-4 dense / unified | `gemma4-dense-v1` | `convertible` — `gemma4_unified` prepared at convert time to `gemma4` text path; multimodal sidecars preserved |
| **Nemotron 3** (thin) | `nemotron3-v1` | **`convertible` only for Nano-30B-A3B** hybrid MoE; Super/Ultra **inspect-only** (no SSD-stream product path) |
| **Mistral / Devstral dense** | `mistral-devstral-dense-v1` | **`convertible`** — `model_type=mistral` (MLX remaps to llama) or llama exports named Mistral/Devstral/Ministral |
| **Mistral 3 / Ministral-3 shell** | `mistral3-dense-v1` | **`convertible`** — language path via nested `text_config`; vision stripped by MLX sanitize |

New families start at `inspect-only` until promotion evidence exists. Run
`axquant support-matrix` and `axquant support-policy` for the registry-derived source of truth.

| Area | Current support |
| --- | --- |
| Platform | macOS on Apple Silicon (M-series) with MLX |
| Conversion input | Unquantized Safetensors checkpoint supported by the promoted MLX backend; revision pin required for measured/release evidence |
| Conversion targets | Qwen 3.6 27B/35B-A3B; Qwen 3.5; Qwen3 dense + Embeddings; Qwen3-Next/Coder-Next MoE; Qwen3-ASR 1.7B; Qwen3-VL 8B Instruct; MiniCPM5; Gemma-4; Nemotron Nano only (thin); Mistral/Devstral/Ministral and Mistral3 shells (development evidence) |
| Family support tiers | `certified` / `convertible` / `inspect-only`, recorded in every inventory and plan |
| Precision choices | 4-bit, 6-bit, 8-bit, and BF16 (plus experimental 2-bit and 3-bit behind AX Engine's documented gates); measured affine, DWQ-clipped affine, portable AWQ, and GPTQ |
| Planning | Manual recipes and a planner that consumes measured sensitivity artifacts |
| MTP | Detection, byte-preserved sidecars, and an opt-in Qwen 3.6 AX Engine layout backend |
| Primary runtime | AX Engine for text tracks; MLX-Audio for Qwen3-ASR; MLX-VLM for Qwen3-VL |
| Compatibility runtime | Architecture-specific standard inference; generic text artifacts use MLX-LM |
| Output integrity | Atomic conversion, exact parameter coverage, measured BPW, checksums, manifests, and runtime metadata |

### Development evidence

Conversion and generation smokes establish artifact compatibility, not model quality or release
certification. The public model cards and manifests record each checkpoint's exact source
revision, plan, achieved BPW, sidecars, and evidence limits. Keep hardware names, network
addresses, and local artifact paths in local operational records rather than public docs.

The default 4.8 BPW budget can be infeasible when protection floors raise the policy minimum
(for example, Gemma-4, Devstral, and Mistral3). The simple `quantize` path raises the requested
budget once to the computed minimum and records that decision. Use an explicit `--target-bpw` at
or above the floor when the budget must be fixed.

### AutomatosX Hub catalog (AXQ, development)

Public **development** packs on [AutomatosX](https://huggingface.co/AutomatosX)
(BF16 source → `axquant quantize` → Hub upload; **not** certified releases).
The stable repository names below now serve the audited v2 artifacts on `main`, so existing model
identifiers continue to work. Each exact v2 revision is also tagged `v2`; the artifact previously
served by each repository remains recoverable at `legacy-pre-v2`. Temporary migration repositories
with edition suffixes are not part of the public catalog.

Each repo ships a full model card (`README.md`) plus public AXQuant provenance
(`axquant_manifest.json`, `axquant_plan.json`, runtime metadata, sidecars when
present). Cards are multi-family aware and state evidence limits explicitly.

The table below lists 32 audited packs under their stable names; the
[AutomatosX MLX model catalog](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog)
provides a browsable catalog. The BPW values are rounded from each current public manifest's
`measured_main_bpw`; the linked model card and manifest remain authoritative.

| Pack | Main-model BPW | Notes |
| --- | --- | --- |
| [`AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP) | 5.418315 | primary dense; MTP + vision sidecars |
| [`AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP) | 5.844833 | primary dense; MTP + vision sidecars |
| [`AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP) | 4.878782 | primary MoE; MTP + vision sidecars |
| [`AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP) | 5.759473 | primary MoE; MTP + vision sidecars |
| [`AX-Qwen3.5-9B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-4bit-MTP) | 6.736665 | secondary; protection floors dominate |
| [`AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP) | 6.736665 | secondary; same effective floor |
| [`AX-gemma-4-12b-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit) | 4.890033 | secondary; vision sidecar preserved |
| [`AX-gemma-4-12b-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit) | 6.000088 | secondary; vision sidecar preserved |
| [`AX-Devstral-Small-2505-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-4bit) | 4.949963 | secondary coding/agent |
| [`AX-Devstral-Small-2505-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-6bit) | 5.999989 | secondary coding/agent |
| [`AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit) | 5.150021 | secondary; vision sidecar preserved |
| [`AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit) | 5.999949 | secondary; vision sidecar preserved |
| [`AX-MiniCPM5-1B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-MiniCPM5-1B-MLX-AXQ-4bit) | 7.380428 | secondary fixture; protection floors dominate |
| [`AX-MiniCPM5-1B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-MiniCPM5-1B-MLX-AXQ-6bit) | 7.380428 | secondary fixture; same effective floor |
| [`AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit) | 4.799310 | thin Nano support |
| [`AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit) | 5.990219 | thin Nano support |
| [`AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit) | 5.550330 | embedding; `feature-extraction` card |
| [`AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit) | 8.000275 | embedding |
| [`AX-Qwen3-Embedding-4B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-4B-MLX-AXQ-4bit) | 4.890183 | embedding |
| [`AX-Qwen3-Embedding-4B-MLX-AXQ-8bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-4B-MLX-AXQ-8bit) | 7.999979 | embedding |
| [`AX-Qwen3-Embedding-8B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-8B-MLX-AXQ-4bit) | 4.830057 | embedding |
| [`AX-Qwen3-Embedding-8B-MLX-AXQ-8bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-8B-MLX-AXQ-8bit) | 7.999911 | embedding |
| [`AX-Qwen3-Coder-Next-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-4bit) | 4.797752 | corrected indexed-expert packing |
| [`AX-Qwen3-Coder-Next-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit) | 5.998996 | corrected indexed-expert packing |
| [`AX-Qwen3-ASR-1.7B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-4bit) | 6.910001 | MLX-Audio; protected BF16 audio tower |
| [`AX-Qwen3-ASR-1.7B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-6bit) | 8.350084 | MLX-Audio; protected BF16 audio tower |
| [`AX-Qwen3-VL-8B-Instruct-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-4bit) | 6.359976 | MLX-VLM; protected BF16 vision tower |
| [`AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit) | 7.999975 | MLX-VLM; protected BF16 vision tower |
| [`AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit) | 5.990115 | Mistral3 language path |
| [`AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit) | 5.999992 | Mistral3 language path |
| [`AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit) | 5.610033 | Mistral3 language path |
| [`AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit) | 5.999912 | Mistral3 language path |

**Development naming:** `AX-<Base>-MLX-AXQ-<4bit|6bit|8bit>[-MTP]` (MLX-style bit labels,
not GGUF `q4`). Artifact editions are recorded in the model card and immutable Hub tags instead
of changing the repository identifier. The class is a planning budget, not a claim that every
tensor uses that width.

**Certified naming:** `AX-<Base>-MLX-AXQ-MP-<measured-main-BPW>bpw[-MTP]`, rounded to two
decimal places with decimal half-up rules (for example `MP-5p30bpw-MTP`). This name is generated
from audited measured bytes; `target_class` remains metadata.

**Quick load (MLX-LM):**

```bash
python -m pip install -U mlx-lm
mlx_lm.generate --model AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP \
  --prompt "Hello" --max-tokens 64 --temp 0.0
```

**Investment policy:** `axquant support-policy` (primary certification track = Qwen 3.6;
Nemotron = thin Nano only).

Regenerate a public card from a local pack:

```bash
python scripts/prepare_development_model_card.py \
  --artifact /path/to/AX-...-MLX-AXQ-6bit-MTP \
  --repo-id AutomatosX/AX-...-MLX-AXQ-6bit-MTP \
  --artifact-edition 2
```

Implemented now:

- indexed Safetensors inspection and logical parameter reconstruction;
- deterministic, provenance-bound tokenized calibration caches;
- resumable per-tensor MLX probes with 4/6/8/BF16 affine candidates and targeted DWQ/AWQ/GPTQ refinement;
- portable AWQ activation-scale search and GPTQ Hessian error compensation with convert-time refinement and affine packing;
- checksum-bound per-module activation capture (`capture-activations`) feeding AWQ/GPTQ probes and conversion;
- Qwen 3.6 tensor classification, MTP detection, and vision protection;
- Qwen3-ASR and Qwen3-VL text-path quantization through public MLX-Audio/MLX-VLM backends,
  with BF16 modality-tower protection and real media runtime smokes;
- auditable manual recipes with mandatory precision floors;
- mixed-precision planning from compatible sensitivity reports;
- architecture-specific MLX conversion with plan-to-module coverage checks;
- atomic output staging that prevents partial final checkpoints;
- AX Engine manifest generation and runtime readiness checks;
- identical-checkpoint AX Engine MTP off/on benchmarking with greedy-output equality;
- deterministic quality/benchmark suites and complete-model MLX quality evaluation;
- validation gates for externally measured quality and performance evidence;
- guarded Hugging Face publication;
- tiered family support with declarative adapters (Qwen 3.6 primary; Qwen 3.5, Qwen3-ASR,
  Qwen3-VL, MiniCPM5, Gemma-4, Mistral/Devstral, Mistral3, and Nemotron Nano at
  `convertible`; Nemotron Super/Ultra remain inspect-only), including byte-preserving extraction
  of integrated MTP heads and protected vision into canonical checksummed sidecars;
- development Hub model cards (`axquant.model_card` / `scripts/prepare_development_model_card.py`)
  that sanitize provenance and document evidence limits for public packs;
- `axquant quantize`: one-command development conversion with explicit development-evidence
  labeling;
- checksummed recipe bundles (`recipe-export`, `quantize --recipe`) that bind published plans
  to user conversions without upgrading their evidence kind, resolvable locally or from
  revision-pinned `hf://` references; prepared releases package their bundle automatically;
- a registry-derived support matrix (`support-matrix`) with investment posture and
  `support-policy` best practices (primary Qwen cert track; thin Nemotron Nano only);
- per-layer KV-cache precision planning **and runtime execution**: prior-based
  (`--kv-cache prior`) and measured (`analyze-kv` + `plan --kv-cache measured`, digest-bound
  to the sensitivity report) planning, and `runtime-check --runtime mlx-lm-kv` executes the
  plan's exact per-layer table at runtime — one cache object per layer through MLX-LM's
  public `prompt_cache`/`QuantizedKVCache` API, with per-layer mixed precisions (e.g.
  8-bit boundary + 4-bit interior layers) verified active on real artifacts. The ordinary
  generation smoke also applies the advisory global KV values. Families whose attention
  implementation rejects quantized caches fail closed (the hybrid Qwen 3.6 path awaits
  AX Engine-native KV, the scoped engine project);
- a fail-closed measured-KV release chain: conversion packages the bound `kv_sensitivity.json`
  (`convert --kv-sensitivity`) and publication re-verifies the digest and reproduces the exact
  per-layer allocation from the packaged report;
- an evidence-bound head-to-head page renderer that loads only checksum-verified evaluation
  bundles and always lists unavailable mandatory baselines with their reasons;
- a bundled, clean-room-authored reference calibration dataset (160 samples across 7 domains —
  coding, json, tool, multilingual, long-context, reasoning, general) with a
  `validate-calibration-dataset` command, so a user without their own domain-representative
  calibration text can still run the full measured pipeline; an integration test proves the
  complete chain (inspect → tokenize-calibration → analyze → plan) closes end to end on it;
- a repository evaluation task suite (`data/eval/`) with 60 clean-room-authored tasks across four
  categories (coding, reasoning, json-tool, instruction) for `evaluate-quality` and
  `compare-quality`, covering python-syntax, JSON validity, exact match, regex, and token-F1
  scoring.

Still incomplete (external evidence / runtime / deferred scope — not missing toolkit commands):

- **Qwen 3.6 certification is not closed.** Public packs are development artifacts. The additive
  `qwen36-mtp-v2` tooling now enforces one semantic candidate, frozen disjoint data roles,
  authorizing performance only on `df-macbookpro-m5`, holdout consumption, durable evidence, independent
  review, lifecycle state, measured-BPW claims, and a final M0–M8 audit. Those controls are
  implemented; the real candidate and formal evidence have not yet passed them;
- interaction-optimization evidence: the toolkit path exists
  (`refine-select --interaction`, holdout-safe by construction), but no bound candidate has yet
  been optimized against real measured development-role evaluations;
- validated conversion evidence for any future official dense Qwen 3.6 sizes beyond current smokes;
- certification evidence for secondary families (Nemotron Super/Ultra remain inspect-only);
- quantized external MTP sidecars in production: the toolkit can emit the engine's
  `mlx-affine-packed-u32` layout via `quantize-mtp-sidecar` (with a fail-closed capability
  probe), but no AX Engine build yet reports that layout as executable, so every shipped
  public pack keeps a byte-preserved sidecar;
- measured KV serving-quality evidence: the report-only artifact and `kv-serving-quality`
  command exist, but the dual-profile short/long-context measurements that would fill them have
  not been run;
- vision-tower quantization (deferred scope — Qwen3-VL language paths convert, but vision towers
  remain BF16 until vision-specific evaluation evidence exists);
- per-expert (unfused) MoE precision (deferred scope — packed expert stacks quantize as fused
  switch modules with one precision per group; finer splits need MLX-LM-side support).

The `validation-index`, `hardware-registry`, `compatibility-matrix`, and `release-audit` commands
enforce release gate order, dual-profile completeness, and evidence binding.

Architecture-prior analysis, smoke probes, and manual plans are explicitly marked as
non-release development evidence. They cannot support production-quality or performance claims.

## From-source install (Mac)

PyPI install is under [Install](#install). For an editable checkout on Apple Silicon
(conversion needs `.[mlx]`; AX Engine manifest generation needs `ax-engine-bench` on `PATH`):

```bash
cd axquant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[mlx]'           # conversion path
# python -m pip install -e '.[dev,mlx]'     # plus tests and lint
axquant --help
```

## Simple development conversion

AXQuant uses a **two-door** model:

| Door | When | Command |
| --- | --- | --- |
| **Simple (dev)** | local trials, fit-check, smoke | `axquant quantize MODEL --target-bpw 4.8` |
| **Release** | public quality/speed claims | staged analyze → plan → convert → validate → scoreboard |

Simple convert is **always development evidence**. It never upgrades to a certified claim.

### Minimal commands

```bash
# Local BF16 checkpoint — one command from source to development artifact
axquant quantize /models/Qwen3.6-27B-bf16 --target-bpw 4.8

# Explicit flags still work
axquant quantize \
  --model /models/Qwen3.6-27B-bf16 \
  --model-id Qwen/Qwen3.6-27B \
  --revision REVISION_SHA \
  --target-bpw 4.8 \
  --runtime-smoke mlx-lm \
  --json quantize-summary.json

# Hub id (download opt-in; pin a revision for reproducibility)
axquant quantize Qwen/Qwen3.6-27B --target-bpw 4.8 --allow-download --revision REVISION_SHA
```

Qwen3-ASR requires one pinned BF16 normalization step before inspection or planning:

```bash
python scripts/hf_to_mlx_bf16.py \
  --hf-id Qwen/Qwen3-ASR-1.7B \
  --revision REVISION_SHA \
  --mlx-path /models/Qwen3-ASR-1.7B-MLX-BF16 \
  --work /models/.axquant-source-work

axquant quantize /models/Qwen3-ASR-1.7B-MLX-BF16 \
  --target-bpw 6.91 \
  --runtime-smoke mlx-audio \
  --audio-input ./sample.wav
```

Qwen3-VL converts from its pinned upstream BF16 checkpoint through MLX-VLM:

```bash
axquant quantize /models/Qwen3-VL-8B-Instruct \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --revision REVISION_SHA \
  --target-bpw 6.36 \
  --runtime-smoke mlx-vlm \
  --image-input ./sample.png
```

Defaults on the simple path:

- ladder `prior` with multi-group grid `(32, 64)`;
- output directory `./AX-<model>-MLX-AXQ-4bit` when `--output` is omitted;
- development-evidence banner in logs and summary notes;
- family tier gates (inspect-only still fails closed).

```bash
axquant simple-convert-help          # two-door best practices
axquant ladders --markdown-output convert-ladders.md
axquant probe-capacity --inventory architecture_report.json --output probe-capacity.json
axquant scoreboard --plan plan-01.json --output scoreboard.json --markdown-output scoreboard.md
```

To reuse published planning evidence, pass a checksummed recipe bundle with `--recipe` — either a
local path or a revision-pinned Hub reference such as
`--recipe hf://AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit@COMMIT_SHA/recipe/axquant_recipe_bundle.json`
(the revision pin is mandatory and the payload checksum is always verified). Expert memory-tier
development recipes live under `examples/expert-memory-tier-v0.1.yaml` (2-bit fused experts, 8-bit
routers; requires AX Engine experimental 2-bit flags). To add prior-based per-layer KV-cache
metadata, pass `--kv-cache prior`.

## Staged development conversion

The staged path below uses the reviewed manual recipe. This proves the
conversion workflow stage by stage, but its output remains unmeasured development evidence.

### 1. Inspect the source checkpoint

```bash
axquant inspect \
  --model /models/Qwen3.6-27B-bf16 \
  --model-id Qwen/Qwen3.6-27B \
  --revision REVISION_SHA \
  --output inventory.json
```

Inspection verifies the checkpoint layout, identifies the supported architecture, classifies
each tensor, detects MTP, and records which components must remain protected.

### 2. Create a mixed-precision plan

```bash
axquant plan-manual \
  --inventory inventory.json \
  --recipe examples/qwen36-27b-manual-v0.1.yaml \
  --output manual-plan.json \
  --markdown-output manual-plan.md
```

The included recipe applies 4-bit defaults, keeps attention weights at 6-bit, and preserves
protected components at their required precision. The generated plan records every assignment
and its reason.

### 3. Convert the model

```bash
axquant convert \
  --model /models/Qwen3.6-27B-bf16 \
  --revision REVISION_SHA \
  --plan manual-plan.json \
  --allow-unmeasured \
  --ax-engine-manifest if-available \
  --output AX-Qwen3.6-27B-MLX-AXQ-4bit
```

If the plan preserves MTP as an external bundle, conversion requires:

```bash
--mtp-sidecar /models/Qwen3.6-27B-bf16/mtp.safetensors
```

The default `--mtp-layout byte-preserved` path never changes tensor payloads; the copied bundle's
`mtplx_runtime.json` declares `mtp_norm_layout: raw_hf_delta` so AX Engine converts every MTP
norm deterministically at load time instead of guessing from tensor statistics. The explicit
development path:

```bash
--mtp-layout ax-engine-qwen36-v1
```

accepts only a checksum-bound raw Qwen 3.6 bundle with the exact 15-tensor BF16 contract. It adds
one, with BF16 rounding, to the seven named MTP RMSNorm tensors, proves the eight projection
payloads unchanged, and writes a new provenance manifest plus a depth-1 AX Engine runtime
contract. This opt-in layout is not a release waiver: identical-checkpoint MTP exactness,
acceptance, and throughput must still pass the ordinary validation gates.

The `--allow-unmeasured` and `--ax-engine-manifest if-available` options are development-only.
Omit them from a release workflow: release conversion requires measured evidence and a valid AX
Engine manifest. A measured plan must also pass
`--calibration-manifest calibration_manifest.json`; conversion verifies its checksum and
provenance against the plan and packages it with the artifact.

### 4. Check the converted model

```bash
axquant runtime-check \
  --model AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --model-id AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --revision candidate-revision \
  --runtime ax-engine \
  --output runtime-check.json
```

Use `--runtime mlx-lm` to perform the MLX-LM generation smoke. Qwen3-ASR uses
`--runtime mlx-audio --audio-input ./sample.wav`; Qwen3-VL uses
`--runtime mlx-vlm --image-input ./sample.png`. `--static-only` is an MLX-LM development
diagnostic.

## CLI workflow

Run `axquant COMMAND --help` for the full options of any command.

| Command | Purpose | Current maturity |
| --- | --- | --- |
| `feasibility` | Audit source and comparison checkpoints before conversion | Implemented |
| `source-checkpoint-manifest` | Derive and bind the immutable source revision, tokenizer, architecture, and file digests for exact-checkpoint certification | Implemented |
| `certification-policy` | Emit the frozen Qwen3-Next non-MTP certification policy and policy digest | Implemented |
| `prepare-coding-suite` | Build the checksum-bound 128-task Qwen3-Next coding suite, toolchain manifest, and calibration-overlap report (`axquant-token-5gram-v2`) | Implemented; formal use requires all pinned toolchains; regenerate manifests after v1.5.1 |
| `evaluate-coding-suite` | Run resumable generation and network-disabled executable scoring for coding-suite v2 | Implemented; Apple Silicon/Seatbelt execution evidence required |
| `verify-coding-suite` | Self-test every coding oracle and scorer by requiring the reference to pass and an empty mutant to fail | Implemented; run before suite freeze |
| `evaluate-general-quality` | Evaluate the disjoint direct-track general holdout and archive every raw model output | Implemented; BF16 and candidate runs must use matched settings |
| `direct-validation-index` | Recompute policy-bound BF16/candidate quality retention for both direct-track profiles | Implemented; emits a fail-closed index for N4 |
| `prepare-general-overlap` | Recompute exact/near-duplicate separation between general holdout and calibration (`axquant-token-5gram-v2`) | Implemented; any match blocks direct validation |
| `inspect` | Inventory tensors, architecture, quantization, and MTP | Implemented |
| `calibrate` | Validate calibration input, record provenance, and build a tokenized cache (`--manifest-only` skips tokenization) | Implemented |
| `validate-calibration-dataset` | Check a calibration JSONL against the toolkit's domain/size/format bar (defaults to the bundled reference dataset) | Implemented |
| `tokenize-calibration` | Build and verify a deterministic tokenized cache | Implemented |
| `capture-activations` | Capture per-module Linear input activations from a verified tokenized cache into a checksum-bound artifact | Implemented |
| `analyze` | Generate architecture priors or measure resumable affine/DWQ/AWQ/GPTQ/BF16 sensitivity from a calibration cache | Implemented |
| `analyze-kv` | Measure per-layer KV-cache sensitivity over a tokenized calibration cache | Implemented; development evidence |
| `plan` | Allocate 4/6/8/BF16 from a sensitivity report | Implemented; release use requires measured evidence |
| `plan-manual` | Apply an explicit YAML precision recipe | Implemented for development |
| `quantize` | Simple development convert: positional `MODEL`, optional `--target-bpw` / `--output` / `--allow-download`; ladder `prior` multi-group default | Implemented; always development evidence (two-door) |
| `simple-convert-help` | Print simple-convert best practices (two-door model) | Implemented |
| `ladders` | List convert ladders (`prior` → `measured-lite` → `measured-full` → `refine-awq-dwq`) with cost/evidence | Implemented |
| `probe-capacity` | Recommend sensitivity probe mode under host memory (bf16-full / measured-lite / streaming / prior-only) | Implemented |
| `scoreboard` | Certification scoreboard from plan + optional size/quality/MTP evidence (MTP speed owned by AX Engine) | Implemented |
| `bind-sensitivity` | Bind weight (+ optional KV) sensitivity digests into one lineage artifact | Implemented |
| `recovery-rank` | Rank quantized tensors for opt-in recovery by sensitivity (not implied by convert) | Implemented |
| `deferred-features` | List fail-closed deferred expansion features (vision-tower quant, per-expert unfused, domain LoRA) | Implemented |
| `recipe-export` | Export a revision-pinned plan as a checksummed recipe bundle | Implemented |
| `support-matrix` | List families with tier, investment posture, priority, and policy notes | Implemented |
| `support-policy` | Print family investment best practices (primary/secondary/thin) | Implemented |
| `head-to-head` | Render the public comparison page from a bound benchmark evidence index | Implemented |
| `convert` | Create the mixed-precision MLX checkpoint and metadata | Implemented for checkpoints at the `convertible` tier or above |
| `runtime-check` | Run AX Engine readiness or actual MLX-LM, MLX-Audio, or MLX-VLM generation | Implemented |
| `prepare-suite` | Materialize deterministic disjoint benchmark inputs | Implemented |
| `evaluate-quality` | Run MLX perplexity and scored generation tasks | Implemented |
| `compare-quality` | Compare matched quality runs with per-task visibility | Implemented |
| `benchmark` | Collect AX Engine runtime evidence | Implemented |
| `benchmark-ab` | Compare one checkpoint with MTP disabled/enabled | Implemented |
| `benchmark-kernels` | Measure host-scoped decode/prefill kernel latency per (bits, group size) for `plan --latency-table` | Implemented |
| `quantize-mtp-sidecar` | Emit an opt-in quantized MTP sidecar next to the untouched byte-preserved default, gated on a live or recorded AX Engine capability check | Implemented |
| `kv-serving-quality` | Bind executed per-layer KV precisions to dual-profile quality retention as a report-only artifact | Implemented |
| `mtp-diagnose` | Run the MTP kill-switch diagnostic matrix | Implemented; diagnostic evidence only |
| `benchmark-index` | Bind every required baseline or record why it is unavailable | Implemented |
| `validation-index` | Require disjoint passing agent-coding and general evidence | Implemented |
| `refine` | Generate proxy-ranked bounded precision swaps | Development only |
| `recover` | Record optional post-PTQ recovery provenance | Implemented as identity-copy provenance; no weight mutation |
| `refine-measure` | Build checksum-bound complete-candidate evidence | Implemented |
| `refine-select` | Select only from checksum-bound, validated complete candidates | Implemented |
| `refine-export` | Export standalone executable plans from a refinement result | Implemented |
| `refine-run` | Resume complete conversion, quality, MTP, validation, and selection runs | Implemented |
| `pareto` | Report non-dominated validated candidates on named hardware | Implemented |
| `hardware-registry` | Certify checksum-bound kernel, version, power, and shape coverage | Implemented |
| `campaign-overlap` | Build privacy-preserving exact/5-gram overlap evidence (`axquant-token-5gram-v2`; repeatable `--id-field`, default `id` then `task_id`) | Implemented |
| `campaign-frontier` | Verify every cheapest-failure-first candidate gate and derive the eligible frontier | Implemented |
| `campaign-freeze` | Freeze one exact `qwen36-mtp-v2` source/candidate/evidence graph | Implemented |
| `campaign-preflight` | Verify frozen bindings, durable storage, and exact `df-macbookpro-m5` host identity | Implemented |
| `campaign-start-formal` | Start one budgeted formal cycle only after matching preflight | Implemented |
| `campaign-complete-formal` | Derive pass/fail from the bound completion and consume both formal holdouts | Implemented |
| `campaign-close-no-go` | Close a pre-formal campaign without consuming its blind holdout | Implemented |
| `campaign-record-publication` | Bind downloaded Hub bytes, revision, audit, claim, lifecycle, and runtime re-verification | Implemented |
| `artifact-lifecycle` | Append legal development → candidate → frozen → certified/superseded/revoked transitions | Implemented |
| `claim-render` | Generate measured-BPW public claims and the certified model card from bound evidence | Implemented |
| `release-audit` | Dispatch historical Qwen 3.6 v4, Qwen3-Next N0–N8, or additive `qwen36-mtp-v2` M0–M8 proof | Implemented |
| `compatibility-matrix` | Bind family-wide artifact, runtime, and validation evidence | Implemented |
| `validate` | Apply release thresholds to external benchmark evidence | Implemented |
| `size-evidence` | Bind authoritative candidate/uniform-4 or uniform-6 artifact sizes | Implemented |
| `release-exception` | Record an approved, expiring, evidence-bound size exception | Implemented |
| `report` | Render plan and validation reports | Implemented |
| `publish-prepare` | Assemble a release only after validation | Implemented |
| `publish` | Preview or execute a guarded Hugging Face upload | Implemented |
| `verify-reproduction` | Verify regenerated weight bytes and bound provenance | Implemented |
| `name` | Generate the recommended AXQuant model name | Implemented |

## Measured planning and validation

DWQ release evidence uses the same deterministic 0.1/99.9-percentile clipping implementation
during sensitivity probing and conversion. A targeted run adds measured DWQ candidates to an
existing complete affine report without rewriting any base candidate:

```bash
axquant analyze \
  --model Qwen/Qwen3.6-27B \
  --revision pinned-source-revision \
  --calibration calibration-cache \
  --base-sensitivity measured-affine-sensitivity.json \
  --methods dwq \
  --target-tensor model.language_model.layers.4.mlp.up_proj.weight \
  --state dwq-probe-progress.json \
  --output measured-affine-dwq-sensitivity.json
```

The merged report records the base report's semantic digest, inventory digest, probe backend,
target count, and method set. Release audit requests list every ancestor under
`sensitivity_lineage`; M3 replays the chain and rejects removed or modified base candidates,
undeclared additions, protocol drift, cycles, missing parents, and unused reports.

Once a measured sensitivity report is available, create a plan without the development override:

```bash
axquant plan \
  --analysis measured-analysis.json \
  --target-bpw 4.8 \
  --bits 4,6,8,16 \
  --mtp protected \
  --output quantization-plans
```

`--lm-head-floor 8bit` is the governed size-gate path: it lowers the LM-head weight
floor from BF16 to 8-bit for that plan only, records the deviation in
`constraints.lm_head_min_bits`, and requires a measured 8-bit LM-head sensitivity candidate
before the release audit accepts the plan. The default floor stays BF16.

Validate externally collected benchmark bundles:

```bash
axquant size-evidence \
  --artifact-manifest candidate/axquant_manifest.json \
  --model-id AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --revision candidate-revision \
  --output candidate-size-evidence.json

axquant validate \
  --reference-evaluation reference-evaluation.json \
  --candidate-direct-evaluation candidate-mtp-off.json \
  --candidate-evaluation candidate-mtp-on.json \
  --mtp-ab candidate-mtp-ab.json \
  --size-reference uniform4-size-evidence.json \
  --candidate-size candidate-size-evidence.json \
  --profile agent-coding \
  --output validation.json
```

For an MTP speed claim, `--mtp-ab` binds the matched AX Engine direct/MTP comparison used for
token-weighted decode speedup, prompt-median speedup, and greedy-output exactness. AXQuant rejects
the bundle when its model identity, workload, software, hardware, controls, or environment do not
match the candidate evidence.

For a `6bit` certification, freeze the class explicitly and derive the size reference from the
matching complete uniform-6 baseline. The same `max_weight_size_ratio` threshold is then applied
to the uniform-6 denominator; a 4-bit candidate cannot switch denominators opportunistically:

```bash
axquant size-evidence \
  --feasibility-report feasibility.json \
  --reference-kind uniform-6bit \
  --output uniform6-size-evidence.json

axquant validate \
  --reference-evaluation reference-evaluation.json \
  --candidate-direct-evaluation candidate-mtp-off.json \
  --candidate-evaluation candidate-mtp-on.json \
  --mtp-ab candidate-mtp-ab.json \
  --size-reference uniform6-size-evidence.json \
  --candidate-size candidate-size-evidence.json \
  --target-class 6bit \
  --profile agent-coding \
  --output validation.json
```

If a measured Pareto candidate misses both the BPW target and the uniform-4 size-ratio gate, a
release authority can record a time-bounded exception. The command computes the observed values
from the two size artifacts; it does not accept caller-authored observed values:

```bash
axquant release-exception \
  --exception-id AXQ-SIZE-001 \
  --plan selected-plan.json \
  --candidate-size candidate-size-evidence.json \
  --size-reference uniform4-size-evidence.json \
  --tradeoff-evidence measured-tradeoff.json \
  --measured-tradeoff "Measured quality, speed, and memory tradeoff approved for release." \
  --owner "AutomatosX release owner" \
  --approved-by "Named release authority" \
  --approval-reference "release-decision-001" \
  --approved-at 2026-07-30T12:00:00Z \
  --expires-at 2027-01-31T00:00:00Z \
  --output release-exception.json

axquant validate \
  --reference-evaluation reference-evaluation.json \
  --candidate-direct-evaluation candidate-mtp-off.json \
  --candidate-evaluation candidate-mtp-on.json \
  --size-reference uniform4-size-evidence.json \
  --candidate-size candidate-size-evidence.json \
  --plan selected-plan.json \
  --release-exception release-exception.json \
  --exception-evidence tradeoff=measured-tradeoff.json \
  --profile agent-coding \
  --output validation.json
```

The exception can downgrade only `artifact.weight_size_ratio`; it must also disclose the failed
measured-BPW target. Quality, speed, memory, fallback, integrity, and provenance failures remain
errors. Release audit requests that use an exception must list its file under
`release_exceptions` and provide the exact `plan`, `candidate_size`, `size_reference`, and
`tradeoff` paths under `release_exception_evidence`. M4 reloads and hashes every file, checks both
validation profiles, verifies approval and expiry, and compares the packaged
`release_exception.json` with the approved record.

For a refinement candidate, derive its selection record from the converted manifest, matched
quality comparison, and release validation rather than authoring measurement values:

```bash
axquant refine-measure \
  --refinement refinement.json \
  --candidate-id cand-0000-000 \
  --measurement-id cand-0000-000-m3-max \
  --artifact-manifest candidate/axquant_manifest.json \
  --quality-comparison candidate/quality-comparison.json \
  --validation candidate/validation.json \
  --output measurements.json
```

Use `--existing measurements.json` with a new output path to accumulate another candidate or a
second named-host result for the same candidate. Measurement IDs must be unique. `refine-select`
uses the worst measured objective and BPW across every host record for a candidate, so adding
hardware evidence cannot make selection less conservative. The complete objective combines task
retention and perplexity with MTP acceptance, peak memory, and effective speed. Refinement
parentage is a precision-only monotonic chain: a child may upgrade formats but cannot downgrade
or exchange an unrelated tensor.

Prepare the exact complete-candidate run without executing expensive model work:

```bash
axquant refine-run \
  --request examples/refinement-execution-request.yaml \
  --output-dir run/complete-candidates
```

Review `execution-manifest.json`, then add `--execute`. The runner resumes checksum-verified
completed outputs, skips the remainder of a candidate after an execution failure, treats
validation exit `1` as measured failed-gate evidence, merges complete measurements, and runs
`refine-select` plus `pareto`.

Every release benchmark must name its power mode and quantizer/version. `refine-run` reads
`benchmark_power_mode` from its request, derives the AXQuant identity from each plan, and includes
both raw A/B logs in the resumable output contract. Standalone baseline runs use
`--power-mode`, `--quantizer`, and `--quantizer-version`. `benchmark-ab` derives adjacent-token
repetition directly from emitted token IDs, records depth-one proposal accuracy, and derives
greedy divergence from the matched A/B outputs. Its release speed gate defaults to
`token-weighted-decode-tps`: total output tokens divided by total generation wall time, with the
same calculation applied to both arms. The artifact also records the legacy prompt-median TPS
ratio and requires it to remain at or above `1.10x`, preventing a long decode from hiding a
typical-prompt regression. Release MTP evidence therefore requires token-weighted decode speedup
`>=1.20x`, prompt-median speedup `>=1.10x`, and exact greedy outputs. Use
`--speedup-metric prompt-median-tps` only when reproducing the
legacy protocol. For a uniform-6 reference A/B, use
`--direct-baseline-kind uniform-6bit --mtp-baseline-kind uniform-6bit`; the default kinds remain
the AXQuant MTP-off/on release pair. Use `--record-failed-speedup` for an evidence sweep that must
retain both evaluation bundles when only the speed floor fails: the command writes the complete
evidence, returns status `1`, and leaves exactness and matched-control invariants fail-closed.
Build the M7 hardware registry only from the resulting raw
logs, evaluation bundles, validation, plan, converted artifact manifest, sensitivity report,
quality comparison, and quantizer execution manifest:

```bash
axquant hardware-registry \
  --request examples/hardware-registry-request.yaml \
  --output hardware-profile-registry.json
```

The command returns `1` while validation is failing, any runtime or conversion fallback is
present, provenance is inconsistent, the complete objective cannot be rebuilt from the artifact,
quality, and validation files, or the claimed bit/group/role/shape coverage is not measured. The
registry records both the semantic and file digest of its complete-candidate measurement set.
Publication verifies that file, packages it as `refinement_measurements.json`, packages every
objective input, and rewrites the registry to packaged relative paths. Each registry entry
identifies the exact measurement ID, allowing one candidate and plan to be certified on multiple
named hosts.

### Flagship campaign closure

The certified Qwen 3.6 path starts from the exact source
`Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. It is separate from the
historical v4 development audit:

```bash
axquant campaign-freeze \
  --request flagship-campaign-request.json \
  --output flagship-campaign.json

# This authorizing preflight must run on the exact host id df-macbookpro-m5.
axquant campaign-preflight \
  --campaign flagship-campaign.json \
  --output flagship-campaign-preflight.json

axquant release-audit \
  --request flagship-release-audit-request.json \
  --output flagship-authorization-audit.json
```

An authorization-ready audit proves the frozen campaign and current M0–M8 evidence but is
deliberately not publication-ready until the independent lifecycle and claim closure is present.
The campaign request and every transition, raw-evidence, review, no-go, and publication record
must remain inside the declared non-symlinked durable root. Formal preflight also requires fresh
doctor, Metal, zero-fallback, storage, power, and thermal results bound to the exact frozen
`df-macbookpro-m5` contract.
After the legal `frozen → certified` event, `claim-render` creates `public-claim.json` and the
measured-BPW `README.md`. An independent final publication review binds those exact files and the
authorization audit under the durable campaign root; the final flagship request must pass M0–M8
again. Preview and executed publication both rerun that exact final request. A v4 audit cannot
authorize a package containing flagship claims or lifecycle metadata.

Prepare the release directory locally, then run the aggregate proof before publishing a certified
checkpoint:

```bash
axquant publish-prepare \
  --model AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --repo AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --validation-index release-validation-index.json \
  --hardware-registry hardware-profile-registry.json \
  --pareto-report pareto-report.json
```

```bash
axquant release-audit \
  --request examples/release-audit-request.yaml \
  --output release-audit.json
```

This revalidates indexed evaluation, complete-refinement, and hardware file checksums; binds the
selected interaction improvement to the packaged measurement set; reruns reproduction
verification; inspects the wheel metadata, contents, and every `RECORD` member hash/size; and
requires the packaged plan, validation/benchmark evidence, hardware registry/evidence,
refinement measurements, Pareto report, and recipe to match the external evidence graph. The
M0 check recomputes checkpoint completeness, parameter/architecture equivalence, revisions, MTP,
and baseline runtime results rather than trusting the feasibility status label. M1 requires every
artifact Safetensors file to have one safe, size- and checksum-valid manifest record. M2 reloads
the indexed evaluations and rechecks complete trials, matched controls and hardware, provenance,
fallbacks, identical-checkpoint MTP pairing, one cross-profile candidate/reference pair, and
disjoint datasets. M3 reloads the checksum-bound calibration manifest, verifies separation and
provenance, requires finite tensor-scoped measurements, and verifies every targeted-sensitivity
ancestor. M6 reloads the bound artifact, quality comparison, and validation for every
measurement, recomputes the versioned complete objective, and requires a measured, validated,
monotonic parent/child gain. Complete-measurement construction also rejects non-authoritative
profile thresholds, an inconsistent validation pass label, core release metrics below their
active thresholds, nonzero kernel fallbacks, and a passing size overage without its governed
plan-bound exception. M7 rebuilds every Pareto point and frontier member from the bound
measurement set. The
compatibility matrix must bind that same candidate manifest, runtime checks, and validation. The
audit also reloads every checkpoint from the original compatibility request and re-hashes its
manifest, plan, runtime checks, and validation. The wheel must declare Python 3.11+, MIT, and all
runtime dependencies; the artifact, plan, recipe, and wheel must identify the same AXQuant
version. It reports M0 through M8 separately and returns `0` only when all nine milestones pass;
an alpha or pre-1.0 wheel, including one still carrying an Alpha distribution classifier, cannot
pass M8. An executed publication packages that exact authorizing result as `release_audit.json`
and refuses to overwrite a different existing audit.

Preview publication first. Add `--yes` only when the release should be uploaded; an executed
upload also requires the matching audit and its original request so the full M0–M8 proof can be
rerun from current evidence:

```bash
axquant publish \
  --model AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --repo AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit \
  --validation-index release-validation-index.json \
  --hardware-registry hardware-profile-registry.json \
  --pareto-report pareto-report.json \
  --release-audit release-audit.json \
  --release-audit-request examples/release-audit-request.yaml
```

Before publication, build the complete comparison index. BF16, uniform 4-bit, uniform 6-bit,
and the identical AXQuant MTP-off/on pair are mandatory. Mixed-precision, AWQ, and DWQ entries
may be unavailable, but they cannot be omitted and must state why:

```bash
axquant benchmark-index \
  --request examples/benchmark-evidence-request.yaml \
  --output benchmark-evidence-index.json
```

Build one benchmark index and validation report for each required profile, using distinct
evaluation datasets. Then bind them into the publication gate:

```bash
axquant validation-index \
  --request examples/release-validation-request.yaml \
  --output release-validation-index.json
```

Publication rejects a missing profile, a reused dataset, differing candidate/reference
identities, a failed validation, or a non-ready benchmark index.

Every prepared release includes `reproduction_recipe.yaml` with argument-array commands for
downloading the pinned source, converting it, checking both runtimes, and verifying every
regenerated Safetensors file. Prepared MTP layouts additionally checksum-bind the provenance and
runtime companion files required to reuse the transformed sidecar without applying the transform
again. After running those commands, verification can also be invoked directly:

```bash
axquant verify-reproduction \
  --recipe reproduction_recipe.yaml \
  --artifact regenerated-model \
  --output reproduction-verification.json
```

Build the M5 family matrix from checksum-bound artifact, AX Engine, MLX-LM, and validation
evidence:

```bash
axquant compatibility-matrix \
  --request examples/qwen36-compatibility-request.yaml \
  --output compatibility-matrix.json
```

The request declares the complete official dense catalog as verified at a timezone-qualified
timestamp. The command returns `1` and still writes the matrix when any declared official dense
Qwen 3.6 model is absent, uses inconsistent candidate evidence, or lacks a compatible
`agent-coding` or `general` validation profile. The checked-in example lists 27B as the only dense
size in the linked catalog; refresh `catalog_verified_at` and `required_dense_models` before every
release. FP8 is a representation of a parameter size, not a second model size.

## Evidence and safety boundaries

- Architecture priors are never described as measured sensitivity.
- `--allow-unmeasured` is restricted to development conversion.
- Conversion fails if the plan does not cover every module it claims to quantize.
- External MTP sidecars remain byte-for-byte unchanged unless the explicit, provenance-checked
  Qwen 3.6 AX Engine layout backend is selected.
- Output is staged and atomically renamed only after conversion succeeds.
- Release claims require complete-model quality and hardware evidence.
- Credentials and Hugging Face tokens are never written to logs or manifests.

## Model naming

Recommended model names use:

```text
OWNER/AX-BASE-MODEL-MLX-AXQ-TARGET
```

For example:

```text
AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit
```

The target suffix describes the checkpoint class, not a claim that every tensor uses that bit
width. The manifest contains the actual precision distribution and effective bits per weight.

## Development

A Mac with MLX installed will **not** catch Ubuntu CI failures. CI splits surfaces on
purpose: **Ubuntu = non-MLX** (`.[dev]` only; MLX cannot run on Linux runners) and
**macOS = MLX** (`.[dev,mlx]`). Prefer the local CI mirror:

```bash
./scripts/ci-local.sh
```

That runs ruff, format, mypy, then a **non-MLX** venv with a sanitized `PATH` (matching
GitHub Actions non-MLX jobs), and the host MLX suite when available. See
[docs/ci-root-causes.md](docs/ci-root-causes.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
```

Tests use small synthetic Safetensors fixtures and do not require real model weights.

## Contributing

Contributions are warmly welcome. AXQuant is built to help Mac users get more reliable,
efficient local inference and a better experience on Apple Silicon. Fork the repository and
send us a pull request for bug fixes, documentation, tests, usability improvements, runtime
compatibility, architecture adapters, or reproducible quantization research.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the fork-and-pull-request workflow, development setup,
validation commands, and evidence requirements. For a substantial design change or new model
family, open a [GitHub issue](https://github.com/defai-digital/axquant/issues) first so the scope
and required promotion evidence are clear.

## Documentation

| Doc | Audience |
| --- | --- |
| [Known issues](docs/known-issues.md) | Operators — documented limitations and fail-closed gates |
| [Environment compatibility](docs/compatibility.md) | Operators — platforms, Python, MLX extras |
| [Flagship certification](docs/flagship-certification.md) | Certification operators — `qwen36-mtp-v2` sequence |
| [AXQ model fleet v2](docs/model-fleet-v2.md) | Hub pack maintainers — stable names and editions |
| [Migration v1.1](docs/migration-v1.1.md) / [v1.2](docs/migration-v1.2.md) | Upgraders from earlier toolkit releases |
| [CI root causes and prevention](docs/ci-root-causes.md) | Contributors — Ubuntu non-MLX vs macOS MLX, PyPI gate |
| [Roadmap](docs/roadmap/README.md) | Contributors — open workstreams and ADRs |
| [Release notes convention](docs/releases/README.md) | Maintainers — curated GitHub Release body per version |
| [GitHub Releases](https://github.com/defai-digital/axquant/releases) | Everyone — published version history |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Legal — research and dependency attribution |

Product requirements, the architecture decision register, technical specifications, and the
independent-implementation policy are maintained internally and are not published in this
repository.

## License

AXQuant is released under the [MIT License](LICENSE). Dependencies, model checkpoints,
calibration datasets, and external tools retain their own licenses.
