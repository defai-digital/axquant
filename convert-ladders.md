# AXQuant convert ladders

Progress from fast architecture-prior development converts to measured, refined release candidates. Evidence labels never upgrade automatically.

| Ladder | Evidence | Target BPW | Bits | Groups | Methods | Rel. cost | Needs cal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `prior` | `architecture_prior` | 4.8 | 4,6,8,16 | 32,64 | affine,bf16 | 0.01 | no |
| `measured-lite` | `measured_development` | 5.0 | 4,8,16 | 64 | affine,bf16 | 0.25 | yes |
| `measured-full` | `measured` | 4.8 | 4,6,8,16 | 32,64,128 | affine,dwq,bf16 | 1.00 | yes |
| `refine-awq-dwq` | `measured` | 4.8 | 4,6,8,16 | 32,64,128 | affine,awq,dwq,gptq,bf16 | 1.60 | yes |

## `prior`

Architecture-prior multi-group plan. Development evidence only; release claims require measured sensitivity or a measured recipe bundle.

- Always available (no forward probes).
- Planner grid includes group sizes 32 and 64 (AXQ-028 / P0).

## `measured-lite`

Lightweight measured probes: fewer bit widths and a single group size. Produces measured_development evidence suitable for iteration, not certification.

- Prefer when probe capacity is measured-lite or streaming-partial.
- Does not run AWQ/DWQ refine.

## `measured-full`

Full measured grid over bits x groups x affine/DWQ methods. Release-quality when the probe backend records measured evidence.

- Requires probe capacity bf16-full (or an explicit measured protocol).
- Use refine-awq-dwq after this when channel scales are desired.

## `refine-awq-dwq`

Measured full grid plus AWQ/DWQ refinement candidates. Highest convert cost; best scale/outlier strategy coverage.

- Run after measured-full or bind via refine-select / refine-run.
- Still subject to EvidenceKind and release gates.
