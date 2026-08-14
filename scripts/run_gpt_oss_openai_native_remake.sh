#!/usr/bin/env bash
# Driver skeleton for GPT-OSS OpenAI-native remake.
# Full narrative: docs/gpt-oss-openai-native-remake-runbook.md
#
# Usage:
#   export OSS20_REV=... OSS120_REV=...
#   ./scripts/run_gpt_oss_openai_native_remake.sh plan|convert-20b|convert-120b|card
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"

: "${OSS20_REV:?set OSS20_REV to openai/gpt-oss-20b commit sha}"
: "${OSS120_REV:?set OSS120_REV to openai/gpt-oss-120b commit sha}"

WORK="${WORK:-/Volumes/Ext4T/axquant/work/gpt-oss-openai-native-remake}"
PUB="${PUB:-/Volumes/Ext4T/axquant/axq-publish}"
CERT="${CERT:-/Volumes/Ext4T/axquant-certification/gpt-oss-openai-native-remake}"
mkdir -p "$WORK" "$PUB" "$CERT"/{inventories,plans,quality,logs,size}

export AXQUANT_FORCE_CPU="${AXQUANT_FORCE_CPU:-1}"

resolve_snapshot() {
  local mid="$1" rev="$2"
  python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("${mid}", revision="${rev}", local_files_only=True))
PY
}

cmd="${1:-}"
case "$cmd" in
  plan)
    OSS20_DIR="$(resolve_snapshot openai/gpt-oss-20b "$OSS20_REV")"
    OSS120_DIR="$(resolve_snapshot openai/gpt-oss-120b "$OSS120_REV")"
    axquant inspect --model "$OSS20_DIR" --model-id openai/gpt-oss-20b \
      --revision "$OSS20_REV" --allow-quantized \
      --output "$CERT/inventories/gpt-oss-20b.inventory.json"
    axquant inspect --model "$OSS120_DIR" --model-id openai/gpt-oss-120b \
      --revision "$OSS120_REV" --allow-quantized \
      --output "$CERT/inventories/gpt-oss-120b.inventory.json"
    axquant plan-manual \
      --inventory "$CERT/inventories/gpt-oss-20b.inventory.json" \
      --recipe "$ROOT/examples/gpt-oss-20b-axq4-agent-v0.1.yaml" \
      --output "$CERT/plans/gpt-oss-20b-axq4.plan.json"
    axquant plan-manual \
      --inventory "$CERT/inventories/gpt-oss-20b.inventory.json" \
      --recipe "$ROOT/examples/gpt-oss-20b-axq6-agent-v0.1.yaml" \
      --output "$CERT/plans/gpt-oss-20b-axq6.plan.json"
    axquant plan-manual \
      --inventory "$CERT/inventories/gpt-oss-120b.inventory.json" \
      --recipe "$ROOT/examples/gpt-oss-120b-axq4-agent-v0.1.yaml" \
      --output "$CERT/plans/gpt-oss-120b-axq4.plan.json"
    axquant plan-manual \
      --inventory "$CERT/inventories/gpt-oss-120b.inventory.json" \
      --recipe "$ROOT/examples/gpt-oss-120b-axq6-agent-v0.1.yaml" \
      --output "$CERT/plans/gpt-oss-120b-axq6.plan.json"
    echo "Plans written under $CERT/plans"
    ;;
  convert-20b)
    OSS20_DIR="$(resolve_snapshot openai/gpt-oss-20b "$OSS20_REV")"
    OUT4="$PUB/AX-gpt-oss-20b-MLX-AXQ-4bit-openai-native"
    OUT6="$PUB/AX-gpt-oss-20b-MLX-AXQ-6bit-openai-native"
    rm -rf "$OUT4" "$OUT6"
    axquant convert --model "$OSS20_DIR" --revision "$OSS20_REV" \
      --plan "$CERT/plans/gpt-oss-20b-axq4.plan.json" --allow-unmeasured \
      --output "$OUT4" 2>&1 | tee "$CERT/logs/convert-20b-axq4.log"
    axquant convert --model "$OSS20_DIR" --revision "$OSS20_REV" \
      --plan "$CERT/plans/gpt-oss-20b-axq6.plan.json" --allow-unmeasured \
      --output "$OUT6" 2>&1 | tee "$CERT/logs/convert-20b-axq6.log"
    echo "20B outputs: $OUT4 $OUT6"
    ;;
  convert-120b)
    OSS120_DIR="$(resolve_snapshot openai/gpt-oss-120b "$OSS120_REV")"
    OUT4="$PUB/AX-gpt-oss-120b-MLX-AXQ-4bit-openai-native"
    OUT6="$PUB/AX-gpt-oss-120b-MLX-AXQ-6bit-openai-native"
    rm -rf "$OUT4" "$OUT6"
    axquant convert --model "$OSS120_DIR" --revision "$OSS120_REV" \
      --plan "$CERT/plans/gpt-oss-120b-axq4.plan.json" --allow-unmeasured \
      --output "$OUT4" 2>&1 | tee "$CERT/logs/convert-120b-axq4.log"
    axquant convert --model "$OSS120_DIR" --revision "$OSS120_REV" \
      --plan "$CERT/plans/gpt-oss-120b-axq6.plan.json" --allow-unmeasured \
      --output "$OUT6" 2>&1 | tee "$CERT/logs/convert-120b-axq6.log"
    echo "120B outputs: $OUT4 $OUT6"
    ;;
  card)
    python "$ROOT/scripts/prepare_development_model_card.py" \
      --artifact "$PUB/AX-gpt-oss-20b-MLX-AXQ-4bit-openai-native" \
      --repo-id AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit --product-class 4bit \
      --no-public-certification
    python "$ROOT/scripts/prepare_development_model_card.py" \
      --artifact "$PUB/AX-gpt-oss-20b-MLX-AXQ-6bit-openai-native" \
      --repo-id AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit --product-class 6bit \
      --no-public-certification
    python "$ROOT/scripts/prepare_development_model_card.py" \
      --artifact "$PUB/AX-gpt-oss-120b-MLX-AXQ-6bit-openai-native" \
      --repo-id AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit --product-class 6bit \
      --no-public-certification
    if [[ -d "$PUB/AX-gpt-oss-120b-MLX-AXQ-4bit-openai-native" ]]; then
      python "$ROOT/scripts/prepare_development_model_card.py" \
        --artifact "$PUB/AX-gpt-oss-120b-MLX-AXQ-4bit-openai-native" \
        --repo-id AutomatosX/AX-gpt-oss-120b-MLX-AXQ-4bit --product-class 4bit \
        --no-public-certification
    fi
    ;;
  *)
    echo "Usage: $0 plan|convert-20b|convert-120b|card" >&2
    echo "See docs/gpt-oss-openai-native-remake-runbook.md" >&2
    exit 2
    ;;
esac
