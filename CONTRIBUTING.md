# Contributing to AXQuant

Contributions are warmly welcome. AXQuant exists to help Mac users get more reliable, efficient
local inference and a better overall experience on Apple Silicon. Bug fixes, documentation,
tests, usability improvements, runtime compatibility work, architecture adapters, and
reproducible quantization research can all move that goal forward.

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

The local CI mirror is the preferred full check:

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

## Pull request guidance

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
