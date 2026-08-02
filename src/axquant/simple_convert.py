"""Simple development convert UX (OptiQ-like ergonomics, AXQuant evidence rules).

Best practices encoded here
---------------------------
1. **Two doors, never one.** Simple convert is always *development evidence*.
   Release claims stay on the staged pipeline (measured sensitivity → plan →
   convert → validate → scoreboard → audit). Never upgrade evidence labels.
2. **Minimal required inputs.** Prefer ``model`` + optional ``target_bpw``.
   Default output name, infer ``model_id`` from Hub refs, auto-discover MTP.
3. **Network is opt-in.** Hub downloads require ``allow_download`` (or the CLI
   flag). Cache-only resolution is the safe default for CI and air-gapped hosts.
4. **Family gates stay hard.** inspect-only families fail closed; do not widen
   conversion just to match OptiQ breadth.
5. **Say the truth loudly.** Print development banner, ladder name, achieved
   BPW, and that certification is a separate path.
6. **Plan degrees of freedom.** Default ladder ``prior`` uses multi-group
   (32, 64) so simple converts still benefit from group/method productization.
7. **Do not hide measured ladders.** ``measured-*`` ladders must refuse the
   simple path without a recipe or staged analyze artifacts.

Pros of a simple path (why ship it)
-----------------------------------
- Matches user expectation set by OptiQ: one command → MLX checkpoint.
- Lowers first-run friction for convertible Qwen/MiniCPM families.
- Still reuses fail-closed convert, plan predicate, and atomic staging.

Cons / risks (why keep the second door)
---------------------------------------
- Architecture-prior mixes are not release-grade; easy path can over-promise.
- Hub downloads and unpinned revisions hurt reproducibility if not warned.
- Broad family support without promotion evidence would violate AXQ-017.
- Collapsing release gates into simple convert would destroy auditability.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from axquant.errors import PlanningError
from axquant.ladders import get_ladder
from axquant.naming import model_name
from axquant.quantize import DEVELOPMENT_NOTE, RuntimeSmoke
from axquant.schema import ConvertLadderName, ProfileName, QuickConversionSummary

_HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)+$")
_SAFE_DIR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def looks_like_hub_id(model: str) -> bool:
    """True when *model* looks like ``org/name`` rather than a filesystem path."""
    if not model or model.startswith((".", "/", "~")):
        return False
    path = Path(model).expanduser()
    if path.is_dir() or path.is_file():
        return False
    # Drive-style or multi-segment OS paths are not Hub ids.
    if "\\" in model or model.count("/") > 1:
        # Allow org/name only (exactly one slash for classic Hub ids; nested ok via regex).
        pass
    return bool(_HUB_ID.fullmatch(model))


def target_class_for_bpw(target_bpw: float) -> str:
    """Map a target BPW to a short name suffix for default output directories."""
    if target_bpw <= 0:
        raise PlanningError("target_bpw must be positive")
    # Common product labels first.
    centers = (("2bit", 2.0), ("3bit", 3.0), ("4bit", 4.0), ("6bit", 6.0), ("8bit", 8.0))
    for label, center in centers:
        if abs(target_bpw - center) <= 0.35:
            return label
    if 4.4 <= target_bpw <= 5.2:
        return "4bit"  # typical mixed ~4.8 BPW ships as 4bit-class name
    # Filesystem-safe continuous label, e.g. 5p5bpw.
    return f"{target_bpw:.1f}bpw".replace(".", "p")


def default_output_dir(
    model: str,
    *,
    model_id: str | None = None,
    target_bpw: float = 4.8,
    mtp: bool = False,
    parent: str | Path = ".",
) -> Path:
    """Derive ``./AX-<base>-MLX-AXQuant-<class>`` under *parent*."""
    base = model_id or model
    name = model_name(base, target_class=target_class_for_bpw(target_bpw), mtp=mtp)
    if not _SAFE_DIR.fullmatch(name):
        raise PlanningError(f"refusing unsafe default output name: {name}")
    return Path(parent).expanduser().resolve() / name


def infer_model_id(model: str, model_id: str | None = None) -> str | None:
    """Prefer explicit model_id; else Hub-like model strings."""
    if model_id:
        return model_id
    if looks_like_hub_id(model):
        return model
    return None


def resolve_download_policy(
    model: str,
    *,
    allow_download: bool | None,
) -> tuple[bool, list[str]]:
    """Decide whether Hub download is allowed; return (allow, notes).

    Policy: local directories never download. Hub ids download only when
    ``allow_download`` is True. ``None`` means false (safe default).
    """
    notes: list[str] = []
    local = Path(model).expanduser()
    if local.is_dir():
        return False, notes
    if allow_download is True:
        if looks_like_hub_id(model):
            notes.append("Hub download enabled for model id (development convert).")
        return True, notes
    if looks_like_hub_id(model):
        notes.append(
            "Hub id provided without --allow-download; resolving from local Hub cache only."
        )
    return False, notes


def simple_convert(
    model: str,
    *,
    output: str | Path | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    profile: ProfileName = ProfileName.GENERAL,
    target_bpw: float | None = None,
    ladder: ConvertLadderName | str = ConvertLadderName.PRIOR,
    kv_cache: str = "off",
    recipe: str | Path | None = None,
    allow_download: bool = False,
    runtime_smoke: RuntimeSmoke = "none",
    ax_engine: str = "ax-engine",
    mlx_lm: str = "mlx_lm.generate",
    ax_engine_manifest: Literal["required", "if-available", "skip"] = "if-available",
    mtp_sidecar: str | Path | None = None,
    calibration_manifest: str | Path | None = None,
    kv_sensitivity: str | Path | None = None,
) -> QuickConversionSummary:
    """One-command development convert with OptiQ-like defaults.

    Always labels output as development evidence when using architecture priors.
    Measured ladders without a recipe fail closed (staged pipeline required).
    """
    from axquant.inspector import resolve_model_dir
    from axquant.quantize import quick_convert as _quick

    resolved_ladder = get_ladder(ladder)
    effective_bpw = resolved_ladder.default_target_bpw if target_bpw is None else float(target_bpw)
    if effective_bpw <= 0 or effective_bpw > 16:
        raise PlanningError("target_bpw must be in (0, 16]")

    download, download_notes = resolve_download_policy(model, allow_download=allow_download)
    # Resolve early so Hub cache / download failures are clear before planning.
    model_dir = resolve_model_dir(model, revision=revision, allow_download=download)
    resolved_model_id = infer_model_id(model, model_id)

    if output is None:
        output_path = default_output_dir(
            model,
            model_id=resolved_model_id,
            target_bpw=effective_bpw,
        )
    else:
        output_path = Path(output).expanduser()

    if kv_cache not in {"off", "prior"}:
        raise PlanningError("simple convert supports --kv-cache off|prior only")
    kv_mode: Literal["off", "prior"] = "prior" if kv_cache == "prior" else "off"

    # Prefer resolved directory for conversion so Hub ids work end-to-end.
    # MTP auto-discovery in quick_convert uses inventory.local_path / model path.
    summary = _quick(
        model=str(model_dir),
        output=output_path,
        model_id=resolved_model_id or str(model),
        revision=revision,
        profile=profile,
        target_bpw=effective_bpw,
        ladder=ladder,
        kv_cache=kv_mode,
        recipe=recipe,
        calibration_manifest=calibration_manifest,
        kv_sensitivity=kv_sensitivity,
        mtp_sidecar=mtp_sidecar,
        runtime_smoke=runtime_smoke,
        ax_engine=ax_engine,
        mlx_lm=mlx_lm,
        ax_engine_manifest=ax_engine_manifest,
        allow_download=False,  # already resolved above
    )
    extra_notes = [
        "Simple convert path: development evidence only (best practice: two doors).",
        *download_notes,
        f"Source directory: {model_dir}",
    ]
    if output is None:
        extra_notes.append(f"Default output directory: {output_path}")
    # Preserve order: core notes first, practice notes after, keep DEVELOPMENT_NOTE last if present.
    notes = list(summary.notes)
    if DEVELOPMENT_NOTE in notes:
        notes = [n for n in notes if n != DEVELOPMENT_NOTE]
        notes.extend(extra_notes)
        notes.append(DEVELOPMENT_NOTE)
    else:
        notes.extend(extra_notes)
    return summary.model_copy(update={"notes": notes})


def simple_convert_help_markdown() -> str:
    """Operator-facing best-practice cheat sheet."""
    return """# Simple convert best practices

## Two doors

| Path | Command | Evidence | Use when |
| --- | --- | --- | --- |
| **Simple (dev)** | `axquant quantize MODEL --target-bpw 4.8` | prior / recipe | local trials |
| **Release** | analyze → plan → convert → validate | measured + gates | public claims |

## Simple path (OptiQ-like)

```bash
# Local BF16 checkpoint (safest)
axquant quantize /models/Qwen3.6-27B-bf16 --target-bpw 4.8

# Hub id from cache, or with explicit download
axquant quantize Qwen/Qwen3.6-27B --target-bpw 4.8 --allow-download --revision <sha>

# Optional: profile, KV prior, smoke
axquant quantize /models/Qwen3.6-27B-bf16 --target-bpw 4.8 --kv-cache prior --runtime-smoke mlx-lm
```

Defaults: ladder `prior` (groups 32,64), auto output name
`AX-<model>-MLX-AXQuant-4bit`, development evidence banner.

## Do not

- Claim release quality from a simple convert.
- Drop family tier gates to convert inspect-only models.
- Pin production deployments to unpinned Hub floating revisions.
- Use measured ladders without calibration + analyze artifacts.

## Release path (unchanged)

```bash
axquant tokenize-calibration ...
axquant analyze ...
axquant plan --ladder measured-full ...
axquant convert --plan ... --calibration-manifest ...
axquant scoreboard --plan ... --quality-comparison ... --mtp-ab ...
```
"""
