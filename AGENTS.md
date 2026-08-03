# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Summary

AXQuant is an independent, MLX-native post-training quantization (PTQ) toolkit for supported LLM
checkpoints on Apple Silicon. Qwen 3.6 is the primary certification track; promoted secondary
adapters cover Qwen 3.5, Gemma-4, MiniCPM5, Mistral/Devstral, Mistral3 language paths, and
Nemotron 3 Nano. AXQuant plans MTP-aware, workload-aware mixed-precision assignments and converts
them through public MLX-LM APIs. It does not reuse mlx-optiq code, data, or metadata.

## Commands

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                          # full suite
.venv/bin/pytest tests/test_planner.py    # single file
.venv/bin/pytest tests/test_planner.py::test_name -x   # single test
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
```

Optional MLX execution backend (Apple Silicon only):

```bash
.venv/bin/pip install -e ".[dev,mlx]"
```

## Architecture

### Pipeline Stages

The CLI (`src/axquant/cli/`) orchestrates the core artifact pipeline. Staged release commands
consume checksum-bound JSON artifacts, and every schema artifact carries a `schema_version`
literal:

1. **feasibility** — audits reference baselines (4-bit, 6-bit, mixed, BF16 source) and runtime
   availability before any conversion work begins (`feasibility.py`).
2. **inspect** — reads Safetensors index files directly (no MLX import), reconstructs logical
   tensors, classifies roles via architecture adapters, detects protected MTP/vision tensors, and
   emits an `Inventory` (`inspector.py`).
3. **calibrate / tokenize-calibration** — records dataset provenance in a
   `CalibrationManifest` and optionally creates a deterministic tokenized cache
   (`calibration.py`, `activation_cache.py`).
4. **analyze / analyze-kv** — without a calibration cache, weight analysis emits explicitly
   unmeasured architecture priors; with a verified tokenized cache, the lazy MLX backends produce
   measured weight or KV sensitivity (`analyzer.py`, `probe.py`, `kv_probe.py`).
5. **plan / plan-manual** — budget-constrained bit allocation. `planner.py` solves from measured
   sensitivity; `manual.py` applies explicit YAML recipes. Both enforce protection floors
   (norms ≥ 16-bit, embeddings/routers ≥ 8-bit, MTP ≥ policy minimum, and LM-head ≥ 16-bit by
   default). The governed `--lm-head-floor 8bit` path requires measured support. Both planners emit
   the same `QuantizationPlan` schema.
6. **convert** — atomic conversion: preflight verifies plan↔model module coverage via
   `PlanPredicate`, runs `mlx_lm.convert` into a temp staging dir, preserves protected MTP/vision
   sidecars (or applies the explicit validated Qwen 3.6 MTP transform), generates AX Engine
   manifest + runtime metadata, then renames staging→final so a partial checkpoint never appears
   at the output path (`converter.py`, `predicate.py`, `runtime.py`).
7. **runtime / evaluate / benchmark / validate** — collects runtime, quality, size, and MTP
   evidence and applies profile-specific thresholds to matched reference/candidate bundles
   (`runtime.py`, `quality.py`, `benchmark.py`, `validator.py`, `profiles.py`).
8. **publish-prepare / release-audit / publish** — assembles checksum-bound release evidence,
   proves M0–M8, and performs a guarded Hugging Face upload; `publish` is a preview unless `--yes`
   is supplied (`reporting.py`, `release_audit.py`, `publisher.py`).

### Key Design Patterns

- **`schema/` is the single source of truth**: all Pydantic models use `extra="forbid"` via
  `StrictModel`. Artifact cross-references use `stable_sha256` (canonical sorted-key JSON) from
  `serde.py`.
- **Architecture adapters** (`architectures/`): a `Protocol`-based registry covers Qwen 3.6,
  promoted declarative dense families, and thin Nemotron 3 Nano support. An adapter profile
  declares optimization scope and a `certified` / `convertible` / `inspect-only` tier; unmatched
  or out-of-scope checkpoints remain inventory-only.
- **Evidence gating**: `EvidenceKind.ARCHITECTURE_PRIOR` blocks `plan` and `convert` unless
  `--allow-unmeasured` is passed. This is intentional — never weaken it.
- **Fail-closed conversion**: `PlanPredicate` tracks which plan modules MLX-LM actually visited;
  unmatched quantized modules abort the conversion.
- **Activation capture** (`capture.py`): `capture-activations` replays a verified tokenized
  cache with recording wrappers on eligible `nn.Linear` modules and writes checksum-bound
  per-module fp16 activations (`ActivationCaptureManifest`) for weight-refinement backends;
  `load_capture_activations` fails closed on model identity, shape, or checksum drift.
- **MLX is a lazy optional dependency**: execution backends import it only inside conversion,
  sensitivity, quality, and runtime paths. Inspection, planning, reporting, and tests do not
  require MLX.
- **serde.py writes atomically**: temp file + `os.replace`; JSON output is deterministic
  (sorted keys, 2-space indent) so artifact diffs are stable.
- **profiles.py** maps `ProfileName` → planner `ObjectiveWeights` and validation
  `ValidationThresholds`; adding a profile requires entries in both tables.
- **Errors** (`errors.py`): `AxquantError` hierarchy; `cli.py` catches it plus
  `ValidationError`/`OSError`/`ValueError` and exits 2.

### Testing Conventions

Tests build tiny Safetensors fixtures in `tmp_path` via `conftest.py` (`tiny_model_dir`,
`qwen36_model_dir`, `packed_model_dir`) using `safetensors.numpy.save_file` + numpy. No real
model weights or network access are needed.

## Clean-Room Boundary

- Do not import from, vendor, translate, or copy mlx-optiq implementation code, tests,
  documentation, calibration data, or generated metadata.
- Build against public MLX and MLX-LM interfaces and independently defined AXQuant schemas.
- Record every calibration input, source revision, objective, and planner decision in manifests.
- Never label architecture-prior output as measured sensitivity.
- Conversion must fail closed when a plan does not cover the modules it claims to quantize.
- External MTP sidecars remain byte-preserved unless a dedicated, validated backend handles their
  tensor layout.

## Project Rules

- Python 3.11+, Ruff formatting at 100 columns, strict mypy, and pytest.
- Use Pydantic models at file and CLI boundaries.
- Use structured logging (structlog) and never log credentials or Hub tokens.
- Put temporary reports under `.internal/tmp/`.
- Do not commit unless explicitly requested.
