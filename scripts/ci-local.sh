#!/usr/bin/env bash
# Mirror GitHub Actions CI gates as closely as practical on a developer machine.
#
# Surfaces (same split as .github/workflows/ci.yml):
#   non-MLX — isolated venv with `.[dev]` only + sanitized PATH  ≈ Ubuntu jobs
#   MLX     — host interpreter with mlx importable               ≈ macOS job
#
# Why: MLX is Apple Silicon only and optional (axquant[mlx]). Ubuntu CI is the
# hard gate for the non-MLX package contract; a green Mac pytest with MLX on
# PATH does not prove Ubuntu will pass. Always run this before pushing to main.
# See docs/ci-root-causes.md and CONTRIBUTING.md.
#
# Usage:
#   ./scripts/ci-local.sh              # lint + non-MLX suite (+ MLX suite if importable)
#   ./scripts/ci-local.sh --lint-only
#   ./scripts/ci-local.sh --no-mlx     # skip host MLX suite even if mlx is installed
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LINT_ONLY=0
RUN_MLX=1
for arg in "$@"; do
  case "$arg" in
    --lint-only) LINT_ONLY=1 ;;
    --no-mlx) RUN_MLX=0 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
  BIN="$ROOT/.venv/bin"
else
  PY="${PYTHON:-python3}"
  BIN="$(dirname "$(command -v "$PY")")"
fi

echo "==> ruff check"
"$BIN/ruff" check .
echo "==> ruff format --check"
"$BIN/ruff" format --check .
echo "==> mypy src"
"$BIN/mypy" src
echo "==> schema contract freeze"
"$PY" scripts/render_schema_contracts.py --check
echo "==> certification matrix SSOT"
"$PY" scripts/render_certification_docs.py --check

if [[ "$LINT_ONLY" -eq 1 ]]; then
  echo "OK (lint-only)"
  exit 0
fi

# Isolated non-MLX env (matches Ubuntu python-compatibility install surface).
# Default lives under the user's cache dir — never under a published tree path.
# Override with AXQUANT_CI_NONMLX_VENV if you want a local-only location.
_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
NONMLX_VENV="${AXQUANT_CI_NONMLX_VENV:-$_CACHE_HOME/axquant/ci-nonmlx-venv}"
mkdir -p "$(dirname "$NONMLX_VENV")"
if [[ ! -x "$NONMLX_VENV/bin/python" ]]; then
  echo "==> creating non-MLX venv at $NONMLX_VENV"
  "$PY" -m venv "$NONMLX_VENV"
fi
echo "==> ensuring non-MLX editable install (.[dev] only)"
"$NONMLX_VENV/bin/pip" install -e ".[dev]" -q
AXQUANT_CI_NONMLX_VENV="$NONMLX_VENV" "$NONMLX_VENV/bin/python" - <<'PY'
import importlib.util
import os
import sys

venv = os.environ["AXQUANT_CI_NONMLX_VENV"]
for name in ("mlx", "mlx_lm", "mlx_audio", "mlx_vlm"):
    if importlib.util.find_spec(name) is not None:
        sys.exit(f"non-MLX venv unexpectedly has {name}; remove {venv} and rerun")
print("non-MLX venv clean")
PY

echo "==> mypy src (non-MLX venv)"
"$NONMLX_VENV/bin/mypy" src

echo "==> pytest -m 'not integration' (non-MLX, sanitized PATH)"
# Prefer the non-MLX venv bin; drop host Homebrew bins so which('mlx_lm.*') cannot
# resurrect a host install.
env PATH="$NONMLX_VENV/bin:/usr/bin:/bin" \
  "$NONMLX_VENV/bin/pytest" -m "not integration"

if [[ "$RUN_MLX" -eq 1 ]] && "$PY" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('mlx') else 1)" 2>/dev/null; then
  echo "==> pytest -m 'not integration' (host MLX)"
  "$BIN/pytest" -m "not integration"
else
  echo "==> skipping host MLX suite (mlx not importable or --no-mlx)"
fi

echo "OK — local CI mirrors passed"
