#!/usr/bin/env bash
# Mirror GitHub Actions CI gates as closely as practical on a developer machine.
#
# Why this exists: a Mac with MLX on PATH/site-packages will pass generation-smoke
# and mypy paths that fail on Ubuntu `.[dev]`-only jobs. Always run this before
# pushing to main. See docs/ci-root-causes.md.
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

if [[ "$LINT_ONLY" -eq 1 ]]; then
  echo "OK (lint-only)"
  exit 0
fi

# Isolated non-MLX env (matches Ubuntu python-compatibility install surface).
NONMLX_VENV="${AXQUANT_CI_NONMLX_VENV:-$ROOT/.internal/tmp/ci-nonmlx-venv}"
mkdir -p "$(dirname "$NONMLX_VENV")"
if [[ ! -x "$NONMLX_VENV/bin/python" ]]; then
  echo "==> creating non-MLX venv at $NONMLX_VENV"
  "$PY" -m venv "$NONMLX_VENV"
fi
echo "==> ensuring non-MLX editable install (.[dev] only)"
"$NONMLX_VENV/bin/pip" install -e ".[dev]" -q
"$NONMLX_VENV/bin/python" - <<'PY'
import importlib.util
import sys
for name in ("mlx", "mlx_lm"):
    if importlib.util.find_spec(name) is not None:
        sys.exit(f"non-MLX venv unexpectedly has {name}; recreate {sys.argv[0]}")
print("non-MLX venv clean")
PY

echo "==> mypy src (non-MLX venv)"
"$NONMLX_VENV/bin/mypy" src

echo "==> pytest -m 'not integration' (non-MLX, sanitized PATH)"
# Drop host Homebrew/local bins so shutil.which('mlx_lm.*') cannot resurrect MLX.
env PATH="/usr/bin:/bin:$NONMLX_VENV/bin" \
  "$NONMLX_VENV/bin/pytest" -m "not integration"

if [[ "$RUN_MLX" -eq 1 ]] && "$PY" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('mlx') else 1)" 2>/dev/null; then
  echo "==> pytest -m 'not integration' (host MLX)"
  "$BIN/pytest" -m "not integration"
else
  echo "==> skipping host MLX suite (mlx not importable or --no-mlx)"
fi

echo "OK — local CI mirrors passed"
