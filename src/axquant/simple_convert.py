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

import math
import re
from pathlib import Path
from typing import Literal

from axquant.errors import PlanningError
from axquant.ladders import get_ladder
from axquant.naming import model_name, target_class_for_bpw
from axquant.quantize import DEVELOPMENT_NOTE, RuntimeSmoke
from axquant.recipes import load_recipe_bundle
from axquant.schema import (
    ConvertLadderName,
    ManualPlanRecipe,
    ProfileName,
    QuantizationPlan,
    QuickConversionSummary,
)
from axquant.serde import load_model

_HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)+$")
_SAFE_DIR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def looks_like_hub_id(model: str) -> bool:
    """True when *model* looks like ``org/name`` rather than a filesystem path."""
    if not model or model.startswith((".", "/", "~")):
        return False
    path = Path(model).expanduser()
    if path.is_dir() or path.is_file():
        return False
    # Hub repository ids have exactly one owner/name separator.
    if "\\" in model or model.count("/") != 1:
        return False
    return bool(_HUB_ID.fullmatch(model))


def default_output_dir(
    model: str,
    *,
    model_id: str | None = None,
    target_bpw: float = 4.8,
    mtp: bool = False,
    parent: str | Path = ".",
) -> Path:
    """Derive ``./AX-<base>-MLX-AXQ-<class>`` under *parent*."""
    if not math.isfinite(target_bpw) or target_bpw <= 0 or target_bpw > 16:
        raise PlanningError("target_bpw must be finite and in (0, 16]")
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
    profile: ProfileName | None = None,
    target_bpw: float | None = None,
    ladder: ConvertLadderName | str = ConvertLadderName.PRIOR,
    kv_cache: str | None = None,
    recipe: str | Path | None = None,
    allow_download: bool = False,
    allow_quantized: bool = False,
    runtime_smoke: RuntimeSmoke = "none",
    ax_engine: str = "ax-engine",
    mlx_lm: str = "mlx_lm.generate",
    python: str = "python3",
    audio_input: str | Path | None = None,
    image_input: str | Path | None = None,
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
    if recipe is not None and target_bpw is not None:
        raise PlanningError(
            "simple convert --target-bpw cannot be combined with --recipe; "
            "the recipe bundle already fixes its target BPW"
        )
    if recipe is not None and kv_cache is not None:
        raise PlanningError(
            "simple convert --kv-cache cannot be combined with --recipe; "
            "the recipe bundle already fixes its KV-cache plan"
        )
    if recipe is not None:
        recipe_record, recipe_payload = load_recipe_bundle(recipe)
        recipe_model = (
            load_model(recipe_payload, QuantizationPlan)
            if recipe_record.payload_kind == "plan"
            else load_model(recipe_payload, ManualPlanRecipe)
        )
        if profile is not None and profile != recipe_model.profile:
            raise PlanningError(
                "simple convert --profile does not match the profile fixed by --recipe"
            )
        effective_bpw = recipe_model.target_bpw
        effective_profile = recipe_model.profile
    else:
        effective_bpw = (
            resolved_ladder.default_target_bpw if target_bpw is None else float(target_bpw)
        )
        effective_profile = profile or ProfileName.GENERAL
    if not math.isfinite(effective_bpw) or effective_bpw <= 0 or effective_bpw > 16:
        raise PlanningError("target_bpw must be finite and in (0, 16]")

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
    resolved_output = output_path.resolve()
    if (
        resolved_output == model_dir
        or resolved_output.is_relative_to(model_dir)
        or model_dir.is_relative_to(resolved_output)
    ):
        raise PlanningError("simple convert output must not overlap the source checkpoint")

    effective_kv_cache = kv_cache or "off"
    if effective_kv_cache not in {"off", "prior"}:
        raise PlanningError("simple convert supports --kv-cache off|prior only")
    kv_mode: Literal["off", "prior"] = "prior" if effective_kv_cache == "prior" else "off"

    # Prefer resolved directory for conversion so Hub ids work end-to-end.
    # MTP auto-discovery in quick_convert uses inventory.local_path / model path.
    summary = _quick(
        model=str(model_dir),
        output=output_path,
        model_id=resolved_model_id or str(model),
        revision=revision,
        profile=effective_profile,
        target_bpw=None if recipe is not None else effective_bpw,
        ladder=ladder,
        kv_cache=kv_mode,
        recipe=recipe,
        calibration_manifest=calibration_manifest,
        kv_sensitivity=kv_sensitivity,
        mtp_sidecar=mtp_sidecar,
        runtime_smoke=runtime_smoke,
        ax_engine=ax_engine,
        mlx_lm=mlx_lm,
        python=python,
        audio_input=audio_input,
        image_input=image_input,
        ax_engine_manifest=ax_engine_manifest,
        allow_download=False,  # already resolved above
        allow_quantized=allow_quantized,
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

Install the toolkit from PyPI, then convert. Do not clone this repository
just to run a conversion.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install 'axquant[mlx]'
axquant quantize /path/to/model-bf16
```

## Two doors

| Path | Command | Evidence | Use when |
| --- | --- | --- | --- |
| **Simple (dev)** | `axquant quantize MODEL` | prior / recipe | local trials |
| **Release** | analyze → plan → convert → validate | measured + gates | public claims |

## Simple path

```bash
# Local BF16 checkpoint (safest). --target-bpw defaults to 4.8
axquant quantize /models/Qwen3.6-27B-bf16

# Hub id from cache, or with explicit download
axquant quantize Qwen/Qwen3.6-27B --allow-download --revision <sha>

# Optional: profile, KV prior, smoke
axquant quantize /models/Qwen3.6-27B-bf16 --target-bpw 4.8 --kv-cache prior --runtime-smoke mlx-lm

# Architecture-specific media smokes
axquant quantize /models/Qwen3-ASR-1.7B-MLX-BF16 --target-bpw 6.91 \\
  --runtime-smoke mlx-audio --audio-input ./sample.wav
axquant quantize /models/Qwen3-VL-8B-Instruct --target-bpw 6.36 \\
  --runtime-smoke mlx-vlm --image-input ./sample.png
```

Defaults: ladder `prior` (groups 32,64), auto output name
`AX-<model>-MLX-AXQ-4bit`, development evidence banner.

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
