# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Summary

AXQuant is an independent, MLX-native post-training quantization (PTQ) toolkit for Qwen 3.6 on
Apple Silicon. It plans MTP-aware, workload-aware mixed-precision weight assignments and converts
them through the public MLX-LM API. It does not reuse mlx-optiq code, data, or metadata.

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

The CLI (`cli.py`) orchestrates a linear pipeline; each stage consumes the previous stage's
JSON artifact and every artifact carries a `schema_version` literal:

1. **feasibility** — audits reference baselines (4-bit, 6-bit, mixed, BF16 source) and runtime
   availability before any conversion work begins (`feasibility.py`).
2. **inspect** — reads Safetensors index files directly (no MLX import), reconstructs logical
   tensors, classifies roles via architecture adapters, detects MTP sidecars, and emits an
   `Inventory` (`inspector.py`).
3. **calibrate** — records dataset provenance into a `CalibrationManifest`; the actual MLX
   forward probe is Phase 2 and raises `BackendUnavailableError` if requested (`calibration.py`).
4. **analyze** — produces a `SensitivityReport` from architecture priors only; always marked
   `evidence_kind=architecture_prior` (`analyzer.py`).
5. **plan / plan-manual** — budget-constrained bit allocation. `planner.py` solves from measured
   sensitivity; `manual.py` applies explicit YAML recipes. Both enforce protection floors
   (norms/LM-head ≥ 16-bit, embeddings/routers ≥ 8-bit, MTP ≥ policy minimum) and emit the same
   `QuantizationPlan` schema.
6. **convert** — atomic conversion: preflight verifies plan↔model module coverage via
   `PlanPredicate`, runs `mlx_lm.convert` into a temp staging dir, byte-copies external MTP
   sidecars, generates AX Engine manifest + runtime metadata, then renames staging→final so a
   partial checkpoint never appears at the output path (`converter.py`, `predicate.py`,
   `runtime.py`).
7. **validate** — gates release on profile-specific thresholds comparing reference vs candidate
   evaluation bundles (`validator.py`, `profiles.py`).
8. **publish** — guarded Hugging Face upload; dry-run unless `--yes` (`publisher.py`,
   `reporting.py`).

### Key Design Patterns

- **schema.py is the single source of truth**: all Pydantic models use `extra="forbid"` via
  `StrictModel`. Artifact cross-references use `stable_sha256` (canonical sorted-key JSON) from
  `serde.py`.
- **Architecture adapters** (`architectures/`): a `Protocol`-based registry; `Qwen36Adapter`
  matches by config `model_type`/`architectures`, classifies tensor roles, and declares
  optimization scope. Non-Qwen checkpoints are inventory-only.
- **Evidence gating**: `EvidenceKind.ARCHITECTURE_PRIOR` blocks `plan` and `convert` unless
  `--allow-unmeasured` is passed. This is intentional — never weaken it.
- **Fail-closed conversion**: `PlanPredicate` tracks which plan modules MLX-LM actually visited;
  unmatched quantized modules abort the conversion.
- **MLX is a lazy optional dependency**: imported only inside `converter.py` via `importlib`;
  everything else runs without it. Tests never require MLX.
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
