# Contributing to AXQuant

AXQuant is an independent MIT-licensed AutomatosX / DEFAI Digital product.
We welcome developers, forks, independent AXQ integrations and pull requests.
Our priorities are easy pip installation and CLI conversion, MLX 2/3/6-bit,
MXFP4 and MXFP8, and CUDA NVFP4. MLX-LM, MTPLX, oMLX and AX Engine are MLX
compatibility targets; vLLM is the primary CUDA target. Support claims need
model-, format- and runtime-specific evidence.

You can start with documentation, CLI usability, synthetic tests, schemas or
bug reproduction without Apple Silicon, a GPU or model downloads. Runtime
adapters, conformance fixtures and reproducible quantization research are also
welcome. AX Engine is not required to contribute or to use portable AXQ packs.

First-time contributors are welcome too. If you are unsure where a change belongs, open a
[GitHub issue](https://github.com/defai-digital/axquant/issues) and describe what you want to
improve.

## Fork and submit a pull request

1. Fork [AXQuant](https://github.com/defai-digital/axquant) on GitHub.
2. Create a focused branch in your fork.
3. Set up the development environment and make your change.
4. Add or update tests and documentation where appropriate.
5. Run the checks below.
6. Push your branch and open a pull request against `defai-digital/axquant`.

Small, well-scoped fixes can go directly to a pull request. For a substantial design change or a
new model family, please open an issue first so contributors can agree on scope, runtime support,
and the evidence needed for promotion.

## Development setup

```bash
# Clone your own fork; end users install the released package with pip.
git clone https://github.com/YOUR_USERNAME/axquant.git
cd axquant
git switch -c improve-cli-format-help
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Apple Silicon, install the optional MLX execution dependencies when your change needs them:

```bash
python -m pip install -e ".[dev,mlx]"
```

Most tests use small synthetic Safetensors fixtures and do not require model downloads or real
weights.

## Run the checks

CI uses two install surfaces on purpose:

| Surface | Where | Install | Purpose |
|---------|--------|---------|---------|
| **non-MLX** | Ubuntu Actions + local isolated venv | `.[dev]` only | Package works without Apple Silicon / MLX |
| **MLX** | macOS Actions + host Mac with `.[mlx]` | `.[dev,mlx]` | Real MLX execution and integration |

MLX cannot run on Linux GitHub runners. Ubuntu is not “wrong” — it is the gate that
catches Mac-only assumptions. A green host pytest with MLX installed is not enough.

The local CI mirror is the preferred full check (non-MLX venv + host MLX when present):

```bash
./scripts/ci-local.sh
```

You can also run the checks individually:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
.venv/bin/pytest -m "not integration"
```

Real-hardware tests require Apple Silicon and the MLX dependencies:

```bash
.venv/bin/pytest -m integration
```

## Project layout and style

Source and CLI code live in `src/axquant/`; Pydantic models in `src/axquant/schema/`
are the source of truth for frozen `schemas/` contracts. Tests live in `tests/`,
operator guides in `docs/`, and recipes in `examples/`. Keep optional MLX imports
lazy so portable commands work without Metal.

Use four-space Python indentation, snake_case functions/modules and PascalCase
classes. Follow Ruff formatting and mypy checks configured in `pyproject.toml`.
Name regression tests `test_*.py`; use tiny Safetensors fixtures and `tmp_path`.
Run a focused test with `.venv/bin/pytest tests/test_conversion_formats.py`.
There is no fixed coverage percentage gate; cover meaningful changed behavior.

## Pull request guidance

Draft PRs and first contributions are welcome. Target `main` and link a related
issue when there is one. Recent commit subjects use short imperative descriptions
such as `Fix fused Qwen GDN NVFP4 scales`; use that style for focused commits.
Screenshots are optional for visual changes; CLI changes benefit from before/after
commands and output. Forks and downstream integrations need not become upstream PRs.

A helpful pull request:

- explains what changed, why it helps users, and any known limitations;
- stays focused enough to review and test;
- includes tests for changed behavior and documentation for user-facing changes;
- reports the commands and hardware used for validation;
- labels architecture priors, measured evidence, runtime smokes, and certification claims
  accurately; and
- does not include credentials, private paths, downloaded model weights, or local-only evidence.

Maintainers may ask for additional evidence when a change affects conversion correctness,
checkpoint compatibility, quality claims, or release gates.

## Clean-room and evidence boundary

AXQuant is independently implemented against public MLX, MLX-LM, MLX-Audio, and MLX-VLM
interfaces. Contributions must not import, vendor, translate, or copy mlx-optiq implementation
code, tests, documentation, calibration data, or generated metadata.

New model families begin at the `inspect-only` support tier. Promotion requires real-checkpoint
classification, conversion, and runtime evidence. Architecture-prior output must never be
presented as measured sensitivity, and a successful conversion or runtime smoke must not be
presented as quality certification.

## License

By submitting a contribution, you agree that it may be distributed under the project's
[MIT License](LICENSE).
