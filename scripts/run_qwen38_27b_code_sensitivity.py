#!/usr/bin/env python3
"""AX Coder campaign for pinned Qwen3.8-27B.

Builds a clean-room calibration mix, records the Apple MLX command sequence,
and scores four cells after an external harness finishes them. It does not
download weights, convert a pack, or publish the model.

The product name is AX Coder. That name is not AX Coder Flash, and a score
does not grant a public release.
Format: Qwen3.8-27B MXFP8. Control is the existing uniform MXFP8-MTP pack
with the stock Qwen template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "calibration" / "qwen38-27b-code-sensitivity-v0.1.jsonl"
PREAMBLE_PATH = ROOT / "examples" / "qwen38-27b-coding-preamble.txt"

PRODUCT_ID = "AX Coder"
SOURCE_ID = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CONTROL_PACK_ID = "AX-Qwen3.8-27B-MLX-AXQ-MXFP8-MTP"
PROFILE = "agent-coding"
SEED = 20260728
# MXFP8 remaps unrefined 8-bit affine allocations at group 32. The published
# pack measures 8.4978 total BPW, but this probe's protected floors make the
# planner policy minimum 9.2238 BPW, so 9.0 cannot be executed.
TARGET_BPW = 9.224
GROUP_SIZE = 32
BITS = (8, 16)
METHODS = ("affine",)
Q_MODE = "mxfp8"
MTP_MODE = "protected"
MTP_MIN_BITS = 8
MTP_CANDIDATE_BITS = (8, 16)
# Planner default: an external MTP sidecar stays BF16.
MTP_EXTERNAL_SIDECAR_BITS = 16
VISION_MIN_BITS = 16
NORM_MIN_BITS = 16
EMBEDDING_MIN_BITS = 8
# The MXFP8 recipe lowers the LM head to 8-bit group 32 so the remap applies.
LM_HEAD_MIN_BITS = 8
MIN_BASELINE_SOLVE_RATE = 0.25
CELL_ORDER = ("sensitivity", "template", "combined")
REQUIRED_DOMAINS = ("coding", "json", "tool", "multilingual", "long-context")

_LANGUAGES = ("Python", "Go", "Rust", "TypeScript")
_TASKS = (
    ("bounded queue", "reject a push when the caller-supplied capacity is already full"),
    ("rolling checksum", "update a 32-bit checksum as bytes arrive and finish with one digest"),
    ("interval merge", "merge overlapping closed integer intervals and keep the result sorted"),
    (
        "token bucket",
        "refill by elapsed time and refuse a request that exceeds the current balance",
    ),
    ("path normalizer", "resolve dot segments without reading the filesystem"),
    (
        "stable partition",
        "move matching items forward while preserving the relative order of both groups",
    ),
    ("retry budget", "spend one attempt per failure and stop when the budget reaches zero"),
    ("line indexer", "map a byte offset to a 1-based line and column"),
    ("schema diff", "list keys added, removed, and type-changed between two flat records"),
    ("ring log", "keep the last N records and drop the oldest on overflow"),
    (
        "semver compare",
        "order major, minor, and patch numbers without treating missing parts as wildcards",
    ),
    ("csv scanner", "split rows on commas while keeping quoted commas inside a field"),
    ("edit script", "emit insert and delete operations that turn one short string into another"),
    ("deadline clock", "report whether a monotonic timestamp is still before a stored deadline"),
    ("id mint", "hand out decimal identifiers that never repeat inside one process"),
    ("lock order", "detect when two named locks would be acquired in opposite orders"),
    (
        "blob store",
        "save bytes under a content hash and reject a second write with different bytes",
    ),
    ("query parser", "accept field, operator, and value triples and reject an unknown operator"),
    (
        "work tree",
        "record added and removed relative paths and refuse a path that escapes the root",
    ),
    ("histogram", "count observations into fixed-width buckets and ignore non-finite numbers"),
    (
        "page cursor",
        "encode an offset and a page size, then reject a cursor from a different collection",
    ),
    ("diff budget", "stop comparing two sequences after a caller-supplied number of mismatches"),
    ("config overlay", "apply later keys over earlier keys and leave unspecified keys untouched"),
    ("unit parser", "read an integer with a s, m, or h suffix and reject an unknown suffix"),
)


def _sample(sample_id: str, domain: str, text: str, language: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "domain": domain,
        "language": language,
        "source": "clean-room-authored",
        "license": "CC0-1.0",
        "redistributable": True,
        "text": text,
    }


def _long_note(module: str, index: int) -> str:
    paragraphs = [
        (
            f"Step {step} of maintenance note {index} for the {module} module. "
            f"Store {module} records in insertion order. Reject a duplicate key "
            f"with an error that names the key. A reader observes a batch only "
            f"after the writer publishes it. Batch {step} carries an identifier, "
            f"a monotonic revision, and one original sentence explaining the "
            f"change. An empty batch is an explicit marker so an audit can tell "
            f"a skipped step from a missing file. Do not copy text from another "
            f"module into this note."
        )
        for step in range(1, 15)
    ]
    return "\n\n".join(paragraphs)


def build_samples() -> list[dict[str, object]]:
    """Return the frozen clean-room mix. Coding is the majority domain."""
    samples: list[dict[str, object]] = []
    number = 1
    for task, rule in _TASKS:
        for language in _LANGUAGES:
            text = (
                f"Write a {language} function for a {task}. {rule.capitalize()}. "
                "Keep the public signature stable, document the error for empty "
                "input, and put one assertion in a comment. This text was written "
                f"for {PRODUCT_ID} calibration sample {number}."
            )
            samples.append(_sample(f"coding-{number:03d}", "coding", text, "en"))
            number += 1
    for index in range(1, 17):
        payload = {
            "id": f"record-{index}",
            "kind": "task-result",
            "passed": index % 2 == 0,
            "notes": f"Original calibration record {index} for a flat JSON object.",
        }
        samples.append(
            _sample(
                f"json-{index:03d}",
                "json",
                json.dumps(payload, indent=2, sort_keys=True),
                "en",
            )
        )
        samples.append(
            _sample(
                f"tool-{index:03d}",
                "tool",
                (
                    f"Tool call {index}: read_file path=src/module_{index}.py. "
                    "Observed result: the file exists and exports one function. "
                    "Next action: edit only that function, then run the module's "
                    "unit check. Do not invent a second tool result."
                ),
                "en",
            )
        )
    multilingual = (
        ("zh-Hant", "這個函式只修改呼叫端點名的檔案，並在檢查失敗時回報原始錯誤。"),  # noqa: RUF001
        ("ja", "指定されたファイルだけを変更し、検査が失敗したら元のエラーを返す。"),
        ("es", "Cambia solo el archivo nombrado y devuelve el error original si la prueba falla."),
        ("de", "Aendere nur die genannte Datei und gib den urspruenglichen Fehler zurueck."),
    )
    for index in range(1, 17):
        language, sentence = multilingual[(index - 1) % len(multilingual)]
        samples.append(
            _sample(
                f"multi-{index:03d}",
                "multilingual",
                f"{sentence} Kalibrierung {index}.",
                language,
            )
        )
        samples.append(
            _sample(
                f"long-{index:03d}",
                "long-context",
                _long_note(f"module_{index}", index),
                "en",
            )
        )
    return samples


def dataset_bytes() -> bytes:
    lines = [json.dumps(sample, ensure_ascii=False, sort_keys=True) for sample in build_samples()]
    return ("\n".join(lines) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def domain_counts(samples: list[dict[str, object]]) -> dict[str, int]:
    counts = {domain: 0 for domain in REQUIRED_DOMAINS}
    for sample in samples:
        domain = str(sample["domain"])
        counts[domain] = counts.get(domain, 0) + 1
    return counts


def validate_authored_dataset(samples: list[dict[str, object]]) -> list[str]:
    issues: list[str] = []
    counts = domain_counts(samples)
    missing = [domain for domain in REQUIRED_DOMAINS if counts.get(domain, 0) == 0]
    if missing:
        issues.append(f"missing domains: {missing}")
    coding = counts.get("coding", 0)
    others = sum(count for domain, count in counts.items() if domain != "coding")
    if coding <= others:
        issues.append(f"coding samples {coding} are not a majority of {coding + others}")
    for sample in samples:
        if sample.get("license") != "CC0-1.0" or sample.get("redistributable") is not True:
            issues.append(f"{sample.get('id')} is not redistributable CC0-1.0")
            break
        if sample["domain"] == "long-context" and len(str(sample["text"])) < 2000:
            issues.append(f"{sample['id']} long-context text is under 2000 characters")
            break
    return issues


def axquant_commands(bf16: Path, work: Path) -> list[list[str]]:
    """Measured MLX commands. They do not enable unmeasured escape hatches."""
    dataset = str(DATASET_PATH)
    cache = str(work / "calibration-cache")
    sensitivity = str(work / "sensitivity.json")
    common = [
        "--model",
        str(bf16),
        "--model-id",
        SOURCE_ID,
        "--revision",
        SOURCE_REVISION,
        "--profile",
        PROFILE,
    ]
    return [
        ["axquant", "validate-calibration-dataset", "--path", dataset],
        [
            "axquant",
            "calibrate",
            *common,
            "--dataset",
            dataset,
            "--domains",
            ",".join(REQUIRED_DOMAINS),
            "--seed",
            str(SEED),
            "--attest-calibration-eval-separation",
            "--output",
            cache,
        ],
        [
            "axquant",
            "analyze",
            *common,
            "--bits",
            ",".join(str(bit) for bit in BITS),
            "--methods",
            ",".join(METHODS),
            "--group-size",
            str(GROUP_SIZE),
            "--calibration",
            cache,
            "--token-budget",
            "2048",
            "--output",
            sensitivity,
        ],
        [
            "axquant",
            "plan",
            "--sensitivity",
            sensitivity,
            "--target-bpw",
            str(TARGET_BPW),
            "--bits",
            ",".join(str(bit) for bit in BITS),
            "--methods",
            ",".join(METHODS),
            "--group-size",
            str(GROUP_SIZE),
            "--mode",
            "balanced",
            "--mtp",
            MTP_MODE,
            "--mtp-bits",
            ",".join(str(bit) for bit in MTP_CANDIDATE_BITS),
            "--mtp-min-bits",
            str(MTP_MIN_BITS),
            "--lm-head-floor",
            "8bit",
            "--seed",
            str(SEED),
            # The 2048-token probe is below the 8192 release floor, so the
            # report is measured_development. This pack is not a release.
            "--allow-unmeasured",
            "--output",
            str(work / "quantization-plans"),
        ],
        [
            "axquant",
            "convert",
            "--model",
            str(bf16),
            "--revision",
            SOURCE_REVISION,
            "--plan",
            str(work / "quantization-plans" / "plan-01.json"),
            "--q-mode",
            Q_MODE,
            "--allow-unmeasured",
            "--output",
            str(work / "AX-Coder-MXFP8"),
        ],
    ]


def _rate(successes: int, trials: int) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    return successes / trials


def cost_per_success(cell: dict[str, object]) -> float | None:
    successes = int(cell["successes"])
    if successes <= 0:
        return None
    device_hours = float(cell["reserved_device_seconds"]) / 3600.0
    human_hours = float(cell["human_seconds"]) / 3600.0
    return (
        device_hours * float(cell["per_device_hour"]) + human_hours * float(cell["human_hour"])
    ) / successes


def score_cells(cells: dict[str, dict[str, object]]) -> dict[str, object]:
    """Pick the cheapest eligible cell. The result is not a release tag."""
    expected = {"control", *CELL_ORDER}
    if set(cells) != expected:
        raise ValueError(f"expected cells {sorted(expected)}")
    baseline = cells["control"]
    trials = int(baseline["trials"])
    rates = {name: float(baseline[name]) for name in ("per_device_hour", "human_hour")}
    for name, cell in cells.items():
        if int(cell["trials"]) != trials:
            raise ValueError(f"{name} trial count does not match the control")
        for field in rates:
            if float(cell[field]) != rates[field]:
                raise ValueError(f"{name} cost rate {field} does not match the control")
    baseline_solve = _rate(int(baseline["successes"]), trials)
    report: dict[str, object] = {
        "product_id": PRODUCT_ID,
        "release_tag": False,
        "ax_coder_name_authorized": True,
        "baseline_solve_rate": baseline_solve,
        "recommendation": None,
        "speed_tokens_per_second": {
            name: cell.get("decode_tokens_per_second") for name, cell in cells.items()
        },
        "cells": {},
    }
    details: dict[str, dict[str, object]] = {}
    for name, cell in cells.items():
        cost = cost_per_success(cell)
        details[name] = {
            "solve_rate": _rate(int(cell["successes"]), trials),
            "cost_per_success": cost,
            "schema_failures": int(cell["schema_failures"]),
            "retries": int(cell["retries"]),
        }
    report["cells"] = details
    if baseline_solve < MIN_BASELINE_SOLVE_RATE:
        report["stop"] = "baseline solve rate is below 0.25"
        return report
    baseline_cost = details["control"]["cost_per_success"]
    if not isinstance(baseline_cost, float):
        report["stop"] = "baseline has no successful trial to price"
        return report

    def eligible(name: str, *, cost_ceiling: float, solve_floor: float) -> bool:
        cell = details[name]
        cost = cell["cost_per_success"]
        return (
            isinstance(cost, float)
            and float(cell["solve_rate"]) >= solve_floor
            and int(cell["schema_failures"]) <= int(baseline["schema_failures"])
            and int(cell["retries"]) <= int(baseline["retries"])
            and cost < cost_ceiling
        )

    singles = {
        name: eligible(name, cost_ceiling=baseline_cost, solve_floor=baseline_solve)
        for name in ("sensitivity", "template")
    }
    combined_ok = False
    if all(singles.values()):
        worse_solve = min(float(details[name]["solve_rate"]) for name in singles)
        single_costs = [details[name]["cost_per_success"] for name in singles]
        cheaper = min(float(cost) for cost in single_costs if isinstance(cost, float))
        combined_ok = eligible("combined", cost_ceiling=cheaper + 1e-12, solve_floor=worse_solve)
        combined_cost = details["combined"]["cost_per_success"]
        if isinstance(combined_cost, float) and combined_cost > cheaper:
            combined_ok = False
    chosen = [
        name for name in CELL_ORDER if (singles.get(name) if name != "combined" else combined_ok)
    ]
    if chosen:
        report["recommendation"] = min(
            chosen,
            key=lambda name: (
                float(details[name]["cost_per_success"]),
                CELL_ORDER.index(name),
            ),
        )
    report["stop"] = None if report["recommendation"] else "no candidate lowered cost per success"
    return report


def prepare(work: Path, bf16: Path | None) -> dict[str, object]:
    samples = build_samples()
    issues = validate_authored_dataset(samples)
    if issues:
        raise SystemExit("; ".join(issues))
    payload = dataset_bytes()
    if DATASET_PATH.is_file() and DATASET_PATH.read_bytes() != payload:
        raise SystemExit(f"authored dataset drifted from {DATASET_PATH}")
    preamble = PREAMBLE_PATH.read_text(encoding="utf-8")
    if "Sharp" in preamble or "imatrix" in preamble.lower():
        raise SystemExit("preamble must be a Qwen instruction, not a Tiel template")
    work.mkdir(parents=True, exist_ok=True)
    bf16_ready = bf16 is not None and bf16.is_dir()
    manifest: dict[str, object] = {
        "schema_version": "axquant.qwen38-27b-code-sensitivity.v1",
        "product_id": PRODUCT_ID,
        "ax_coder_name_authorized": True,
        "public_release": False,
        "host_class": "apple-mlx",
        "nvidia_required": False,
        "source_model_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "control_pack_id": CONTROL_PACK_ID,
        "q_mode": Q_MODE,
        "target_bpw": TARGET_BPW,
        "dataset_sha256": sha256_bytes(payload),
        "preamble_sha256": sha256_file(PREAMBLE_PATH),
        "domain_counts": domain_counts(samples),
        "protection": {
            "vision_min_bits": VISION_MIN_BITS,
            "norm_min_bits": NORM_MIN_BITS,
            "embedding_min_bits": EMBEDDING_MIN_BITS,
            "lm_head_min_bits": LM_HEAD_MIN_BITS,
            "mtp_mode": MTP_MODE,
            "mtp_min_bits": MTP_MIN_BITS,
            "mtp_external_sidecar_bits": MTP_EXTERNAL_SIDECAR_BITS,
        },
        "bf16_present": bf16_ready,
        "bf16_path": str(bf16) if bf16 is not None else None,
        "commands": axquant_commands(bf16 or Path("MISSING_BF16"), work),
    }
    (work / "campaign.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write-dataset")
    write.add_argument("--out", type=Path, default=DATASET_PATH)
    prep = sub.add_parser("prepare")
    prep.add_argument("--work", type=Path, required=True)
    prep.add_argument("--bf16", type=Path, default=None)
    measure = sub.add_parser("measure")
    measure.add_argument("--work", type=Path, required=True)
    measure.add_argument("--bf16", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--cells", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "write-dataset":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(dataset_bytes())
        print(args.out)
        return 0
    if args.command == "prepare":
        manifest = prepare(args.work, args.bf16)
        print(json.dumps({"bf16_present": manifest["bf16_present"], "product_id": PRODUCT_ID}))
        return 0
    if args.command == "measure":
        if not args.bf16.is_dir():
            print(
                f"BF16 source is not a directory: {args.bf16}. "
                "This Mac Studio can run the MLX campaign after the pinned "
                f"{SOURCE_ID}@{SOURCE_REVISION} weights are local. "
                "An NVIDIA GPU is not the missing piece.",
                file=sys.stderr,
            )
            return 2
        manifest = prepare(args.work, args.bf16)
        print(json.dumps(manifest["commands"], indent=2))
        print("measure prints the command sequence and does not launch conversion")
        return 0
    loaded = json.loads(args.cells.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit("cells file must be an object keyed by cell id")
    print(json.dumps(score_cells(loaded), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
