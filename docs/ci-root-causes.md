# GitHub Actions root causes (historical red runs)

This document explains **why**
[github.com/defai-digital/axquant/actions](https://github.com/defai-digital/axquant/actions)
showed a long streak of red runs, what was fixed in code, and what still needs
operator configuration.

Historical runs stay red in the UI forever. Only the tip of `main` and future
tags can be made green.

## Snapshot (Aug 2026)

Across the last ~20 workflow runs on `main` / tags:

| Workflow | Pattern |
|----------|---------|
| **CI** | ~16 consecutive failures, then a green tip after product + mypy fixes |
| **Release** | GitHub Release / dist often succeed; **PyPI publish fails** on Trusted Publishing |

macOS **`test`** (MLX) was frequently green while Ubuntu **`lint`** and
**`python-compatibility`** were red — a classic “works on my Mac” split.

## Root cause clusters

### 1. Mac MLX install masked Ubuntu non-MLX test contracts

**What CI does**

- Ubuntu `python-compatibility`: `pip install -e ".[dev]"` — **no** `mlx` / `mlx-lm`
- macOS `test`: `pip install -e ".[dev,mlx]"`

**What broke**

Several tests call `check_mlx_lm_generation` with a **fake executable + mock runner**.
`check_mlx_lm_generation` used to require `check_mlx_lm_static(...).passed`, and
static “passed” required `mlx_lm` to be **importable or on `PATH`**.

On developer Macs and the macOS CI runner, `mlx_lm` is present → static passes →
generation smoke and advisory-KV checks run.

On Ubuntu CI, `mlx_lm` is absent → static fails early → smoke returns
`passed=False` without raising `ArtifactError` for bad `axquant_runtime.json`.
Tests expecting a successful mock run or an invalid-metadata error then fail.

**Fix (shipped)**

- Generation-smoke validates advisory KV / runtime metadata **before** install gates.
- Artifact readiness for smoke is **config + weights**, not Python package install.
- See `src/axquant/runtime.py` (`check_mlx_lm_generation`, `_advisory_kv_execution`).

### 2. Gemma4 shard validation ran after the MLX import gate

**What broke**

`source_prep._filter_sharded` called `_mlx_core()` before checking
`weight_map` path safety / types. On non-MLX installs, unsafe or non-string shard
references raised “requires the MLX backend” instead of the intended
`ArtifactError` (`unsafe shard path`, `non-string shard reference`).

**Fix (shipped)**

- Index/path validation first; `mx.load` only after the map is safe.
- See `src/axquant/source_prep.py`.

### 3. mypy lint job assumed optional backends were installed

**What CI does**

- `lint` installs only `.[dev]` and runs `mypy src` in **strict** mode.

**What broke**

Lazy imports of `mlx`, `mlx_lm`, and `transformers` are intentional (optional
`axquant[mlx]`). Without stubs on the Ubuntu runner, mypy reported
`import-not-found` for those modules. Ruff failures often short-circuited the
lint job **before** mypy ran, so this stayed latent for many commits.

Local `mypy` on a Mac with MLX installed often passed while CI lint failed.

**Fix (shipped)**

```toml
[[tool.mypy.overrides]]
module = ["mlx", "mlx.*", "mlx_lm", "mlx_lm.*", "transformers", "transformers.*"]
ignore_missing_imports = true
```

### 4. Process gap: no local non-MLX mirror before push

Developers and agents routinely ran pytest under a venv that already had MLX
(or host `PATH` contained `mlx_lm.generate`). That never exercised the Ubuntu
matrix. There was also no single script that ran **ruff + format + mypy +
PATH-isolated non-MLX pytest** the way CI does.

**Fix (this change)**

- `scripts/ci-local.sh` mirrors the CI gates, including a temporary venv without
  MLX and a sanitized `PATH` for the non-MLX suite.
- Regression tests monkeypatch `find_spec` / `which` so generation-smoke contracts
  hold even when MLX is installed on the host.

### 5. Release: PyPI Trusted Publishing (operator config)

**Status (Aug 2026)**

- **PyPI is live:** https://pypi.org/project/axquant/ (`pip install axquant`).
- **GitHub Releases** carry the same wheel/sdist (+ SHA256SUMS) for each tag.
- **GitHub Packages tab is empty on purpose.** That UI is for npm, containers,
  Maven, NuGet, RubyGems — not the public Python index. Install Python packages
  from PyPI (or download Release assets). Do not expect
  `github.com/.../packages` to list pip wheels.

**Historical failure (before Trusted Publisher + gate)**

```text
invalid-publisher: valid token, but no corresponding publisher
(Publisher with matching claims was not found)
environment: MISSING
```

**Workflow behaviour (in-repo)**

The `pypi` job runs only when the repository variable `ENABLE_PYPI_PUBLISH` is
exactly `true` (now set). If that variable is unset, tag Releases still complete
green with GitHub assets only. Turning the variable on without a matching
Trusted Publisher on pypi.org reds the Release.

**Operator steps (already done for axquant; keep for new repos)**

1. Sign in and open https://pypi.org/manage/account/publishing/.
2. Add a GitHub publisher:
   - PyPI project name: `axquant`
   - Owner: `defai-digital`, Repository: `axquant`
   - Workflow: `release.yml`
   - Environment: leave empty (the `pypi` job does not set one)
3. Repeat for TestPyPI if you use `v*rc*` tags.
4. Repo variable `ENABLE_PYPI_PUBLISH` = `true`.
5. Dispatch Release for the tag (or push a new tag).

## What does *not* need “fixing” in git history

- **Past red runs** — immutable; ignore them once tip is green.
- **Node 20 deprecation annotations** — warnings only; workflows now pin current
  action majors that run on Node 24 where available.
- **macOS-only flakiness** — not the dominant historical pattern; Ubuntu non-MLX
  and lint were.

## Prevention checklist (before every push to main)

```bash
./scripts/ci-local.sh
```

Or the expanded form documented in the script header / README Development
section. Do not treat “pytest green on a Mac with MLX” as sufficient for main.
