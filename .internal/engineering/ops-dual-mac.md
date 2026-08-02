# Dual-Mac operations (macstudio-m2u + mbp-m5)

**Confirmed layout (2026-08-02).** Factory work on Studio external SSD; control plane on M5.

## Machine roles

| Host | Hardware | Role |
| --- | --- | --- |
| **macstudio-m2u** | M2 Ultra 192 GB; internal Data; **Ext4T-02 3.6 TB** USB SSD; NAS `llm-models` / `home` | **Factory**: HF download cache, BF16 sources, AXQ convert, generation QA |
| **mbp-m5** (`akdf-m3-max`) | M5 Max ~128 GB; roomy internal disk; primary git tree | **Control**: code, measured/cert lineage, AutomatosX HF upload |

Do **not** run identical full convert jobs on both hosts for the same pack. Do **not** upload to the same Hub repo from two machines at once.

## macstudio-m2u paths

| Purpose | Path |
| --- | --- |
| HF hub cache | `~/.cache/huggingface` → `/Volumes/Ext4T-02/hf-data` |
| BF16 sources | `~/models/<name>` → `/Volumes/Ext4T-02/axquant/models/<name>` |
| AXQ outputs | `~/axquant-artifacts/axq-publish` → `/Volumes/Ext4T-02/axquant/axq-publish` |
| Dev smokes | `~/axquant-artifacts/*-dev-smoke` → `/Volumes/Ext4T-02/axquant/smokes/*` |
| Logs | `~/axquant-artifacts/logs` → `/Volumes/Ext4T-02/axquant/logs` |
| Prep work | `/Volumes/Ext4T-02/axquant/work` |
| Code | `~/code/axquant` (rsync from M5 git) |

Internal Data is for OS, conda, code, and small files. **Large weights stay on Ext4T-02.**

Verified: models + axq-publish + logs migrated 2026-08-02; all `*-dev-smoke` trees
live under `smokes/` on Ext4T (qwen36 copies completed after an interrupted rsync was discarded).

### NAS

- `/Volumes/llm-models`, `/Volumes/home`: **cold backup / cross-machine share**, not the hot convert scratch.
- Prefer Ext4T for active `quantize` read/write (local bus vs SMB latency).

### Before convert

```bash
df -h /System/Volumes/Data /Volumes/Ext4T-02
test -d /Volumes/Ext4T-02/axquant/models
readlink ~/.cache/huggingface   # expect .../Ext4T-02/hf-data
```

If internal free &lt; ~100 GB, do not write new large trees under `~/models` as real directories (only symlinks).

## mbp-m5 paths

| Purpose | Path |
| --- | --- |
| Git / primary tree | `/Users/akiralam/code/axquant` |
| Measured evidence | `.internal/tmp/e5-evidence-m5/`, formal stage models |
| Publish staging (optional) | `.internal/tmp/axq-publish/` after rsync from Studio |
| HF token for org upload | Local token must identify as **AutomatosX** (not a personal account without org write) |

## Pipeline (flagship AXQ 4bit + 6bit)

```text
1. Studio: BF16 on Ext4T (hub cache or models/)
2. Studio: axquant quantize → axq-publish/...-MLX-AXQ-4bit-MTP
3. Studio: axquant quantize → ...-MLX-AXQ-6bit-MTP
4. Studio: runtime-check --runtime mlx-lm (generation-smoke)
5. M5: rsync packs (if upload from M5) + model card polish
6. M5: huggingface_hub upload_folder → AutomatosX/<pack>
7. M5: measured/cert lineage only when closing a release candidate
```

Naming: `AX-<Base>-MLX-AXQ-<4bit|6bit|8bit>[-MTP]` with **MTP last**.

## Sync commands

```bash
# Code M5 → Studio
rsync -az --delete \
  --exclude '.venv' --exclude '.git' --exclude '.internal' \
  --exclude '__pycache__' --exclude '.pytest_cache' \
  /Users/akiralam/code/axquant/ macstudio-m2u:~/code/axquant/

# Finished packs Studio → M5 (example)
rsync -az \
  macstudio-m2u:/Volumes/Ext4T-02/axquant/axq-publish/ \
  /Users/akiralam/code/axquant/.internal/tmp/axq-publish/
```

On Studio after code sync:

```bash
conda activate axquant
cd ~/code/axquant && pip install -e ".[mlx]" -q
```

## HF token rule

| Action | Token |
| --- | --- |
| Download public models | Any valid token (rate limits) |
| `create_repo` / `upload` under `AutomatosX/` | Token for user/org with **write** on AutomatosX |

Export for a one-shot remote upload:

```bash
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"  # on machine that is AutomatosX
ssh macstudio-m2u "export HF_TOKEN='$HF_TOKEN'; ..."
```

## Finalize / repair layout (Studio)

After reboot, unplug, or interrupted rsync, run on **macstudio-m2u**:

```bash
bash ~/code/axquant/scripts/studio-finalize-ext4t-layout.sh
# or from M5 / laptop:
ssh macstudio-m2u 'bash -s' < scripts/studio-finalize-ext4t-layout.sh
```

The script: ensures Ext4T trees, re-links `axq-publish`/`logs`, migrates any real
`~/models/*` or smoke dirs still on internal disk, and **discards incomplete**
`smokes/qwen36-*` copies (too few safetensors / too small) before re-sync.

## Failure modes

| Symptom | Fix |
| --- | --- |
| Ext4T missing | Re-plug SSD; do not convert to internal Data |
| `~/models/X` is real dir again | Re-migrate to Ext4T + symlink |
| Hub 403 under AutomatosX | Wrong token identity |
| SMB NAS slow convert | Move source to Ext4T |
| Dual upload race | Single publisher host only |
| Studio SSH timeout | Fix LAN; M5 can still QA local 27B packs and hold git |

## Related

- Product naming / Hub catalog: root `README.md` (AXQ section)
- Qwen 3.6 27B evidence honesty: `qwen36-27b-development-evidence-status.md`
