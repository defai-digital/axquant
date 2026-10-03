# AXQuant public documentation

Operator and user docs. Product PRDs, ADRs, and internal tech specs are not
published here.

| Folder | What belongs here |
| --- | --- |
| [`contracts/`](contracts/axq-pack-interchange-v1.md) | Published freeze contracts (pack interchange, certification spec) |
| [`guides/`](guides/known-issues.md) | How to operate the toolkit (known issues, compatibility, CI, experimental 1.9) |
| [`migrations/`](migrations/migration-v1.8.md) | Toolkit version upgrades |
| [`runbooks/`](runbooks/holo3-35b-axq-dev-runbook.md) | Family convert / cert procedures |
| [`reports/`](reports/ax-engine-72h-endurance.md) | Evidence write-ups. Not certificates. |
| [`certifications/`](certifications/README.md) | Public checkpoint certificates (JSON + markdown) |
| [`hub-cards/`](hub-cards/) | Hub model-card drafts |
| [`releases/`](releases/README.md) | Curated GitHub Release notes |
| [`eval/`](eval/) | Raw eval dumps bound by reports |
| [`cli-reference.md`](cli-reference.md) | Every `axquant` subcommand in one table |

## Start here

Pick the journey that matches your goal:

- **I want to quantize a model** — repository [README](../README.md)
  (install, convert), then the [CLI reference](cli-reference.md) and
  [known issues](guides/known-issues.md).
- **I want to understand or verify an artifact** — the
  [pack interchange contract](contracts/axq-pack-interchange-v1.md),
  [public certificates](certifications/README.md), and
  [microscaling formats](guides/microscaling.md) and
  [native CUDA NVFP4 conversion](guides/cuda-nvfp4.md).
- **I want to operate certification** — the
  [certification operator guide](guides/flagship-certification.md) and the
  [certification rules](contracts/certification-spec-v1.0.md).
- **I want to contribute code** — [CONTRIBUTING.md](../CONTRIBUTING.md) and
  [CI root causes](guides/ci-root-causes.md).

Other entry points:

- Super-class SSD stream: [guides/expert-ssd-stream.md](guides/expert-ssd-stream.md)
- Compatibility: [guides/compatibility.md](guides/compatibility.md)

Do not add PRDs, ADRs, or product-planning tech specs under `docs/`.
