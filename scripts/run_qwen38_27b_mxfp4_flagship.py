#!/usr/bin/env python3
"""Measured Qwen3.8-27B MXFP4 study with protected MTP and required-MTP serving.

Runs existing AXQuant calibration, probing, allocation, and conversion stages.
This development study does not change historical certification policies or
authorize a release. Coding/general holdouts stay outside calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.artifact_paths import artifact_member_path  # noqa: E402
from axquant.calibration_dataset import validate_calibration_dataset  # noqa: E402
from axquant.certification.common import (  # noqa: E402
    build_source_checkpoint_manifest,
    tokenizer_sha256,
)
from axquant.coding_calibration import DOMAINS, agent_coding_dataset_bytes  # noqa: E402
from axquant.dataset_overlap import build_campaign_overlap_report  # noqa: E402
from axquant.errors import AxquantError  # noqa: E402
from axquant.inspector import inspect_model  # noqa: E402
from axquant.schema import EvidenceKind, QuantMethod  # noqa: E402
from axquant.schema.loading import (  # noqa: E402
    load_artifact_manifest,
    load_quantization_plan,
    load_sensitivity_report,
)
from axquant.serde import file_sha256, read_data, stable_sha256, write_data  # noqa: E402

SOURCE_ID = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SEED = 20260728


def commands(source: Path, work: Path, budget: float) -> list[list[str]]:
    common = [
        "--model",
        str(source),
        "--model-id",
        SOURCE_ID,
        "--revision",
        SOURCE_REVISION,
        "--profile",
        "agent-coding",
    ]
    cache = work / "calibration-cache"
    return [
        [
            "calibrate",
            *common,
            "--dataset",
            str(work / "calibration.jsonl"),
            "--domains",
            ",".join(DOMAINS),
            "--max-seq-length",
            "8192",
            "--seed",
            str(SEED),
            "--attest-calibration-eval-separation",
            "--output",
            str(cache),
        ],
        [
            "analyze",
            *common,
            "--calibration",
            str(cache),
            "--bits",
            "4,6,8,16",
            "--methods",
            "mxfp4,affine",
            "--candidate-group-sizes",
            "32,64",
            "--token-budget",
            "32768",
            "--measure-trunk-mtp",
            "--state",
            str(work / "probe-progress.json"),
            "--output",
            str(work / "sensitivity.json"),
        ],
        [
            "plan",
            "--sensitivity",
            str(work / "sensitivity.json"),
            "--target-bpw",
            str(budget),
            "--bits",
            "4,6,8,16",
            "--methods",
            "mxfp4,affine",
            "--candidate-group-sizes",
            "32,64",
            "--mtp",
            "protected",
            "--mtp-bits",
            "16",
            "--mtp-min-bits",
            "16",
            "--lm-head-floor",
            "8bit",
            "--seed",
            str(SEED),
            "--output",
            str(work / "plan.json"),
        ],
        [
            "convert",
            "--model",
            str(source),
            "--revision",
            SOURCE_REVISION,
            "--plan",
            str(work / "plan.json"),
            "--q-mode",
            "mxfp4",
            "--calibration-manifest",
            str(cache / "calibration_manifest.json"),
            "--output",
            str(work / "candidate"),
        ],
    ]


def write_once(path: Path, payload: bytes) -> None:
    """Resume identical inputs; never replace an existing study input."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing study input differs: {path}; use a new work directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source, control, work = (
        getattr(args, name).expanduser().resolve() for name in ("bf16", "control", "work")
    )
    if not source.is_dir():
        raise ValueError("pinned Qwen3.8-27B BF16 source is not available locally")
    inventory = inspect_model(source, model_id=SOURCE_ID, revision=SOURCE_REVISION)
    if inventory.quantized_source or not inventory.mtp_present:
        raise ValueError("flagship requires an unquantized source with MTP weights")
    control_manifest = load_artifact_manifest(control / "axquant_manifest.json")
    control_plan = load_quantization_plan(control / "axquant_plan.json")
    for model in (control_manifest.source_model, control_plan.source_model):
        if model.model_id != SOURCE_ID or model.revision != SOURCE_REVISION:
            raise ValueError("control must use the exact pinned flagship BF16 source")
    if not control_manifest.mtp_present:
        raise ValueError("control must contain MTP weights")
    if tokenizer_sha256(source) != tokenizer_sha256(control):
        raise ValueError("control and BF16 source must use the same tokenizer and chat template")
    if control_manifest.plan_sha256 != stable_sha256(control_plan):
        raise ValueError("control manifest does not bind its plan")
    if not any(a.method == QuantMethod.MXFP4 for a in control_plan.assignments):
        raise ValueError("control must be an MXFP4 pack")
    # Check the actual control, not just its model-card numbers.
    control_inputs = {}
    for member in control_manifest.files:
        path = artifact_member_path(control, member.path)
        if (
            not path.is_file()
            or path.stat().st_size != member.size_bytes
            or (file_sha256(path) != member.sha256)
        ):
            raise ValueError(f"control file failed integrity check: {member.path}")
        control_inputs[str(path)] = member.sha256
    dataset = work / "calibration.jsonl"
    payload = args.calibration.read_bytes() if args.calibration else agent_coding_dataset_bytes()
    write_once(dataset, payload)
    issues = validate_calibration_dataset(dataset)
    if issues:
        raise ValueError(f"invalid calibration dataset: {issues}")
    holdouts = [args.coding_holdout.resolve(), args.general_holdout.resolve()]
    overlap = build_campaign_overlap_report(dataset_path=dataset, compared_paths=holdouts)
    separate_holdouts = build_campaign_overlap_report(
        dataset_path=holdouts[0], compared_paths=holdouts[1:]
    )
    if not overlap.passed or not separate_holdouts.passed:
        raise ValueError("calibration and coding/general holdouts must be disjoint")
    source_manifest = build_source_checkpoint_manifest(source, inventory=inventory)
    inputs = [
        dataset,
        *holdouts,
        control / "axquant_manifest.json",
        control / "axquant_plan.json",
        source / "tokenizer.json",
        source / "tokenizer_config.json",
    ]
    study = {
        "schema_version": "axquant.qwen38-mxfp4-study.v1",
        "source_model_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_checkpoint_sha256": stable_sha256(source_manifest),
        "inputs": {
            **{str(source / member.path): member.sha256 for member in source_manifest.files},
            **control_inputs,
            **{str(path): file_sha256(path) for path in inputs},
        },
        "control": str(control),
        "work": str(work),
        "target_bpw": control_plan.effective_bpw,
        "commands": commands(source, work, control_plan.effective_bpw),
        "serving": {"mtp_policy": "required", "dflash2": "runtime-auto-if-local"},
        "certified": False,
    }
    rendered = (json.dumps(study, sort_keys=True, indent=2) + "\n").encode()
    write_once(work / "study.json", rendered)
    write_data(work / "calibration-overlap.json", overlap)
    write_data(work / "holdout-overlap.json", separate_holdouts)
    return study


def verify_inputs(study: dict[str, Any]) -> None:
    for path, expected in study["inputs"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"frozen study input changed: {path}")


def check_measured_plan(work: Path) -> None:
    plan = load_quantization_plan(work / "plan.json")
    report = load_sensitivity_report(work / "sensitivity.json")
    if plan.evidence_kind != EvidenceKind.MEASURED or report.evidence_kind != EvidenceKind.MEASURED:
        raise ValueError("flagship conversion requires measured evidence; no development override")
    if not report.calibration or report.calibration.metadata.get("measure_trunk_mtp") is not True:
        raise ValueError("flagship requires trunk MTP sensitivity")
    if plan.analysis_sha256 != stable_sha256(report):
        raise ValueError("plan does not bind the measured sensitivity report")
    if plan.mtp.mode != "protected" or plan.mtp.min_bits != 16:
        raise ValueError("flagship MTP must remain protected at BF16")
    entries = {entry.tensor.name: entry for entry in report.entries}
    for allocation in plan.assignments:
        if allocation.role.is_mtp and allocation.bits != 16:
            raise ValueError("flagship plan must preserve all MTP tensors at BF16")
        if allocation.bits == 4 and allocation.method != QuantMethod.MXFP4:
            raise ValueError("flagship 4-bit allocations must use measured native MXFP4")
        if allocation.bits < 16:
            entry = entries.get(allocation.tensor)
            if entry is None or not any(
                c.supported
                and c.bits == allocation.bits
                and c.method == allocation.method
                and c.group_size == allocation.group_size
                and c.metrics.mtp_acceptance_loss > 0
                for c in entry.candidates
            ):
                raise ValueError(f"selected allocation lacks MTP sensitivity: {allocation.tensor}")


def execute(command: list[str], work: Path, stage: str) -> None:
    print(f"Running {stage}", flush=True)
    with (work / f"{stage}.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-m", "axquant", *command],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
                ),
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    write_data(
        work / f"{stage}-receipt.json", {"command": command, "returncode": result.returncode}
    )
    if result.returncode:
        raise ValueError(f"{stage} failed ({result.returncode}); see {work / f'{stage}.log'}")


def verify_candidate(work: Path) -> None:
    completion = read_data(work / "measurement.json")
    if completion.get("matched_within_one_percent") is not True:
        raise ValueError("candidate has not completed matched-size measurement")
    for name in ("plan.json", "sensitivity.json", "candidate/axquant_manifest.json"):
        if file_sha256(work / name) != completion["files"].get(name):
            raise ValueError(f"measured candidate changed: {name}")
    candidate = load_artifact_manifest(work / "candidate" / "axquant_manifest.json")
    plan = load_quantization_plan(work / "plan.json")
    if candidate.plan_sha256 != stable_sha256(plan) or not candidate.mtp_present:
        raise ValueError("candidate does not bind the measured MTP plan")
    for member in candidate.files:
        path = artifact_member_path(work / "candidate", member.path)
        if path.stat().st_size != member.size_bytes or file_sha256(path) != member.sha256:
            raise ValueError(f"measured candidate file changed: {member.path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "measure"):
        p = sub.add_parser(name)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--bf16", type=Path, required=True)
        p.add_argument("--control", type=Path, required=True)
        p.add_argument("--coding-holdout", type=Path, required=True)
        p.add_argument("--general-holdout", type=Path, required=True)
        p.add_argument("--calibration", type=Path)
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--work", type=Path, required=True)
    benchmark.add_argument("--prompts", type=Path, required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--work", type=Path, required=True)
    serve.add_argument("--ax-engine", default="ax-engine")
    args = parser.parse_args(argv)
    work = args.work.expanduser().resolve()
    try:
        if args.command in {"prepare", "measure"}:
            study = prepare(args)
            if args.command == "prepare":
                print(work / "study.json")
                return 0
            verify_inputs(study)
            for command in study["commands"]:
                if command[0] == "convert":
                    check_measured_plan(work)
                execute(command, work, command[0])
            candidate = load_artifact_manifest(work / "candidate" / "axquant_manifest.json")
            control = load_artifact_manifest(Path(study["control"]) / "axquant_manifest.json")
            ratio = candidate.weight_file_size_bytes / control.weight_file_size_bytes
            write_data(
                work / "size-comparison.json",
                {
                    "actual_weight_byte_ratio": ratio,
                    "matched_within_one_percent": abs(ratio - 1) <= 0.01,
                    "certified": False,
                },
            )
            if abs(ratio - 1) > 0.01:
                raise ValueError(
                    "actual pack sizes differ by over 1%; not a matched-size comparison"
                )
            write_data(
                work / "measurement.json",
                {
                    "matched_within_one_percent": True,
                    "actual_weight_byte_ratio": ratio,
                    "files": {
                        name: file_sha256(work / name)
                        for name in (
                            "plan.json",
                            "sensitivity.json",
                            "candidate/axquant_manifest.json",
                        )
                    },
                    "certified": False,
                },
            )
            return 0
        study = read_data(work / "study.json")
        verify_inputs(study)
        check_measured_plan(work)
        verify_candidate(work)
        if args.command == "serve":
            return subprocess.call(
                [args.ax_engine, "serve", str(work / "candidate"), "--mlx-mtp-policy", "required"]
            )
        prompts = work / "benchmark-prompts.jsonl"
        write_once(prompts, args.prompts.expanduser().read_bytes())
        overlap = build_campaign_overlap_report(
            dataset_path=work / "calibration.jsonl", compared_paths=[prompts]
        )
        if not overlap.passed:
            raise ValueError("benchmark prompts must be separate from calibration")
        write_data(work / "benchmark-overlap.json", overlap)
        execute(
            [
                "benchmark-ab",
                "--model",
                str(work / "candidate"),
                "--model-id",
                SOURCE_ID,
                "--revision",
                SOURCE_REVISION,
                "--prompts",
                str(prompts),
                "--trials",
                "5",
                "--warmup",
                "2",
                "--prompt-format",
                "chat-template",
                "--seed",
                str(SEED),
                "--runtime-env",
                "AX_MLX_DFLASH2=0",
                "--record-failed-speedup",
                "--output-dir",
                str(work / "mtp-ab"),
            ],
            work,
            "benchmark-ab",
        )
        return 0
    except (OSError, ValueError, AxquantError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
