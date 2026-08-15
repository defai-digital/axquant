#!/usr/bin/env python3
"""Recertify non-DeepSeek Tier 1 packs under the vision/audio standard.

Inspects each certified pack (local artifact or Hub listing), optionally runs
mlx-vlm / mlx-audio smoke when supported, writes capability-gated ``modalities``
blocks via ``derive_modality_claim``, and emits evidence JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from axquant.modality_certification import (  # noqa: E402
    ArtifactModalityInspect,
    _sidecar_key_prefixes,
    build_modalities_block,
    format_modalities_card_section,
    inspect_hub_listing,
    modalities_to_public_dict,
)
from axquant.schema.public_certification import (  # noqa: E402
    load_public_checkpoint_certification,
)

CERT_DIR = _ROOT / "docs" / "certifications"
EVIDENCE_DIR = CERT_DIR / "evidence" / "modality-recert-capability-gated"
_M2_SMOKE = CERT_DIR / "evidence" / "modality-recert-macstudio-m2" / "results"
DEEPSEEK_PREFIX = "deepseek-v4-flash"
REUSED_SMOKES = {
    "qwen3-vl-30b-axq4-tier1": {
        "passed": True,
        "host": "df-macstudio-m2",
        "runtime": "mlx-vlm",
        "evidence": str(_M2_SMOKE / "qwen3-vl-30b-axq4.json"),
    },
    "qwen3-vl-30b-axq6-tier1": {
        "passed": True,
        "host": "df-macstudio-m2",
        "runtime": "mlx-vlm",
        "evidence": str(_M2_SMOKE / "qwen3-vl-30b-axq6.json"),
    },
    "qwen38-27b-axq4-mtp-tier1": {
        "passed": False,
        "host": "df-macstudio-m2",
        "runtime": "mlx-vlm",
        "evidence": str(_M2_SMOKE / "qwen38-27b-axq4-mtp.json"),
    },
    "qwen36-27b-axq4-tier1": {
        "passed": False,
        "host": "df-macstudio-m2",
        "runtime": "mlx-vlm",
        "evidence": str(_M2_SMOKE / "qwen36-27b-axq4-mtp.json"),
    },
}
LOCAL_PACK_ROOTS = (
    Path("/Volumes/Ext4T/models"),
    Path("/Volumes/Ext4T/axquant/axq-publish"),
)
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
NOTE_TAG = "Modality recert (capability-gated redo)"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _certified_tier1() -> list[Path]:
    return sorted(CERT_DIR.glob("*-tier1.json"))


def _is_deepseek(path: Path) -> bool:
    return path.name.startswith(DEEPSEEK_PREFIX)


def _hub_snapshot(repo_id: str) -> Path | None:
    slug = "models--" + repo_id.replace("/", "--")
    base = HF_HUB / slug
    refs = base / "refs" / "main"
    if refs.is_file():
        snap = base / "snapshots" / refs.read_text(encoding="utf-8").strip()
        if snap.is_dir():
            return snap
    return None


def _pack_complete(path: Path) -> bool:
    shards = list(path.glob("model-*.safetensors"))
    return bool(shards) and sum(p.stat().st_size for p in shards) > 10_000_000


def resolve_local_pack(hub_repo_id: str) -> Path | None:
    name = hub_repo_id.rsplit("/", 1)[-1]
    candidates: list[Path] = []
    for root in LOCAL_PACK_ROOTS:
        cand = root / name
        if cand.is_dir():
            candidates.append(cand)
    snap = _hub_snapshot(hub_repo_id)
    if snap is not None:
        candidates.append(snap)
    complete = [path for path in candidates if _pack_complete(path)]
    if complete:
        return max(
            complete,
            key=lambda p: sum(x.stat().st_size for x in p.glob("model-*.safetensors")),
        )
    return candidates[0] if candidates else None


def inspect_pack(
    hub_repo_id: str, filenames: list[str], config: dict[str, Any] | None
) -> tuple[ArtifactModalityInspect, Path | None]:
    local = resolve_local_pack(hub_repo_id)
    local_names: list[str] = []
    local_config = config
    prefixes: tuple[str, ...] = ()
    if local is not None:
        local_names = [path.name for path in local.iterdir()]
        if (local / "config.json").is_file():
            local_config = json.loads((local / "config.json").read_text(encoding="utf-8"))
        if (local / "vision.safetensors").is_file():
            prefixes = _sidecar_key_prefixes(local / "vision.safetensors")
    merged = inspect_hub_listing(
        filenames=list(filenames) + local_names,
        config=local_config,
        source=str(local) if local is not None else hub_repo_id,
    )
    if prefixes:
        merged = ArtifactModalityInspect(
            vision_declared=merged.vision_declared,
            audio_declared=merged.audio_declared,
            vision_weight_files=merged.vision_weight_files,
            audio_weight_files=merged.audio_weight_files,
            vision_key_prefixes=prefixes,
            source=merged.source,
            notes=merged.notes,
        )
    return merged, local


def _reason(
    inspect: ArtifactModalityInspect,
    *,
    modality: str,
    smoke: dict[str, Any] | None,
) -> str:
    supported = inspect.vision_supported if modality == "vision" else inspect.audio_supported
    if not supported:
        return f"{modality} not supported (no tower config and no sidecar weights)"
    files = inspect.vision_weight_files if modality == "vision" else inspect.audio_weight_files
    prefixes = inspect.vision_key_prefixes
    layout = f" sidecar={list(files)}"
    if prefixes:
        layout += f" keys={list(prefixes)}"
    if smoke is None:
        return (
            f"{modality} present{layout}; smoke not run on this host "
            "(weights protected; no multimodal quality claim)"
        )
    if smoke.get("passed"):
        return (
            f"{modality} runtime smoke passed on {smoke.get('host')} "
            f"({smoke.get('runtime')}); quality suite not certified. "
            f"Evidence: {smoke.get('evidence')}"
        )
    return (
        f"{modality} present{layout}; {smoke.get('runtime')} smoke failed "
        f"on {smoke.get('host')} ({smoke.get('error') or 'see evidence'}). "
        f"Text Tier 1 unchanged. Evidence: {smoke.get('evidence')}"
    )


def write_cert(path: Path, inspect: ArtifactModalityInspect, smoke: dict[str, Any] | None) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    block = build_modalities_block(
        vision_supported=inspect.vision_supported,
        audio_supported=inspect.audio_supported,
        vision_smoke_passed=(
            True if smoke and smoke.get("modality") == "vision" and smoke.get("passed") else None
        ),
        audio_smoke_passed=(
            True if smoke and smoke.get("modality") == "audio" and smoke.get("passed") else None
        ),
        vision_reason=_reason(
            inspect,
            modality="vision",
            smoke=smoke if inspect.vision_supported else None,
        ),
        audio_reason=_reason(inspect, modality="audio", smoke=None),
        vision_runtime=(smoke or {}).get("runtime") if inspect.vision_supported else None,
    )
    # Failed smoke still means present-not-certified: do not pass smoke_passed=False
    # as that is already the default when supported and not passed.
    data["modalities"] = modalities_to_public_dict(block)
    notes = list(data.get("notes") or [])
    notes = [note for note in notes if not note.startswith(NOTE_TAG)]
    notes.append(
        f"{NOTE_TAG} {_now()}: {format_modalities_card_section(block).splitlines()[0]} "
        f"vision={block.vision.status} audio={block.audio.status}. "
        f"Evidence: evidence/modality-recert-capability-gated/."
    )
    data["notes"] = notes
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = path.with_suffix(".md")
    if md_path.is_file():
        text = md_path.read_text(encoding="utf-8")
        marker = "## Modalities (capability-gated)"
        section = format_modalities_card_section(block).rstrip() + "\n"
        if marker in text:
            head, _rest = text.split(marker, 1)
            # drop old section through next H2 or EOF
            tail = _rest.split("\n## ", 1)
            remainder = ("\n## " + tail[1]) if len(tail) == 2 else ""
            text = head.rstrip() + "\n\n" + section + remainder
        else:
            text = text.rstrip() + "\n\n" + section
        md_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def run_mlx_vlm_smoke(model_dir: Path, image: Path, output: Path, python: str) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "-m",
        "axquant",
        "runtime-check",
        "--runtime",
        "mlx-vlm",
        "--model",
        str(model_dir),
        "--image-input",
        str(image),
        "--output",
        str(output),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    passed = False
    error = None
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        passed = bool(payload.get("passed"))
        error = (payload.get("stderr") or "")[:240]
    else:
        error = (completed.stderr or completed.stdout or "no output")[:240]
    return {
        "modality": "vision",
        "passed": passed,
        "host": __import__("socket").gethostname(),
        "runtime": "mlx-vlm",
        "evidence": str(output),
        "exit_code": completed.returncode,
        "error": error,
        "command": cmd,
    }


def patch_hub_readme(readme: str, block_md: str) -> str:
    marker = "## Modalities (capability-gated)"
    section = block_md.rstrip() + "\n"
    if marker in readme:
        head, rest = readme.split(marker, 1)
        tail = rest.split("\n## ", 1)
        remainder = ("\n## " + tail[1]) if len(tail) == 2 else ""
        return head.rstrip() + "\n\n" + section + remainder
    if "## Evidence and validation status" in readme:
        parts = readme.split("## Evidence and validation status", 1)
        after = parts[1]
        nxt = after.find("\n## ")
        if nxt == -1:
            return (
                parts[0] + "## Evidence and validation status" + after.rstrip() + "\n\n" + section
            )
        return (
            parts[0]
            + "## Evidence and validation status"
            + after[:nxt]
            + "\n"
            + section
            + after[nxt:]
        )
    return readme.rstrip() + "\n\n" + section


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-certs", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only")
    parser.add_argument(
        "--image",
        type=Path,
        default=_ROOT / "data/eval/practical-qwen38-vs-qwen36/images/vl-ocr-code.png",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--hub-file-lists", type=Path)
    args = parser.parse_args()

    hub_lists: dict[str, dict[str, Any]] = {}
    if args.hub_file_lists and args.hub_file_lists.is_file():
        for row in json.loads(args.hub_file_lists.read_text(encoding="utf-8")):
            hub_lists[row["cert"]] = row

    results: list[dict[str, Any]] = []
    inventory: list[str] = []
    for path in _certified_tier1():
        stem = path.stem
        if _is_deepseek(path):
            inventory.append(f"EXCLUDE {path.name}")
            continue
        record = load_public_checkpoint_certification(path)
        if record.status != "certified":
            inventory.append(f"SKIP {path.name} status={record.status}")
            continue
        if args.only and args.only not in path.name:
            continue
        hub = record.artifact.hub_repo_id
        listing = hub_lists.get(path.name, {})
        filenames = listing.get("files") or []
        local = resolve_local_pack(hub)
        config = None
        if local is not None and (local / "config.json").is_file():
            config = json.loads((local / "config.json").read_text(encoding="utf-8"))
        else:
            cached = (
                Path(
                    "/var/folders/_k/7sc0bwc55zq_t81br51f6xn40000gn/T/"
                    "grok-goal-4a71b64dffef/implementer/inspect-downloads"
                )
                / stem.replace("-tier1", "")
                / "config.json"
            )
            if cached.is_file():
                config = json.loads(cached.read_text(encoding="utf-8"))
        inspect, local = inspect_pack(hub, filenames, config)
        smoke = None
        reused = REUSED_SMOKES.get(stem)
        if inspect.vision_supported and reused is not None:
            smoke = {
                "modality": "vision",
                "passed": reused["passed"],
                "host": reused["host"],
                "runtime": reused["runtime"],
                "evidence": reused["evidence"],
                "reused": True,
            }
        elif (
            inspect.vision_supported and args.smoke and local is not None and _pack_complete(local)
        ):
            out = args.evidence_dir / "results" / f"{stem}.json"
            smoke = run_mlx_vlm_smoke(local, args.image, out, args.python)
        row = {
            "cert": path.name,
            "hub": hub,
            "local_path": str(local) if local else None,
            "complete": bool(local and _pack_complete(local)),
            "vision_supported": inspect.vision_supported,
            "audio_supported": inspect.audio_supported,
            "vision_declared": inspect.vision_declared,
            "audio_declared": inspect.audio_declared,
            "vision_weight_files": list(inspect.vision_weight_files),
            "vision_key_prefixes": list(inspect.vision_key_prefixes),
            "source": inspect.source,
            "smoke": smoke,
        }
        results.append(row)
        inventory.append(
            f"RECERT {path.name} hub={hub} vis={inspect.vision_supported} "
            f"aud={inspect.audio_supported} complete={row['complete']}"
        )
        if args.write_certs:
            write_cert(path, inspect, smoke)

    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    (args.evidence_dir / "results").mkdir(exist_ok=True)
    report = {
        "schema_version": "axquant.modality-recert.v1",
        "created_at": _now(),
        "policy": "capability-gated-v1",
        "results": results,
    }
    (args.evidence_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("\n".join(inventory))
    print(f"wrote {args.evidence_dir / 'REPORT.json'} n={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
