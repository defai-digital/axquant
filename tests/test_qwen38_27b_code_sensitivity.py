from __future__ import annotations

import json
from pathlib import Path

from axquant.calibration_dataset import validate_calibration_dataset
from axquant.schema.enums import TensorRole
from axquant.schema.planning import PROTECTED_MIN_BITS
from scripts.run_qwen38_27b_code_sensitivity import (
    DATASET_PATH,
    EMBEDDING_MIN_BITS,
    LM_HEAD_MIN_BITS,
    NORM_MIN_BITS,
    PREAMBLE_PATH,
    VISION_MIN_BITS,
    axquant_commands,
    build_samples,
    dataset_bytes,
    main,
    prepare,
    score_cells,
    validate_authored_dataset,
)


def _cell(
    successes: int,
    *,
    schema_failures: int = 1,
    retries: int = 2,
    seconds: int = 3600,
) -> dict:
    return {
        "trials": 40,
        "successes": successes,
        "schema_failures": schema_failures,
        "retries": retries,
        "reserved_device_seconds": seconds,
        "human_seconds": 600,
        "per_device_hour": 2.0,
        "human_hour": 30.0,
        "decode_tokens_per_second": 40.0,
    }


def test_authored_dataset_passes_toolkit_validation(tmp_path: Path) -> None:
    path = tmp_path / "calibration.jsonl"
    path.write_bytes(dataset_bytes())
    assert validate_calibration_dataset(path) == []
    assert validate_authored_dataset(build_samples()) == []


def test_committed_dataset_matches_builder() -> None:
    assert DATASET_PATH.read_bytes() == dataset_bytes()


def test_coding_is_the_majority_and_preamble_is_qwen_specific() -> None:
    samples = build_samples()
    coding = sum(sample["domain"] == "coding" for sample in samples)
    assert coding > len(samples) - coding
    preamble = PREAMBLE_PATH.read_text(encoding="utf-8")
    assert "repository" in preamble
    assert "Sharp" not in preamble


def test_protection_floors_match_planner_policy() -> None:
    assert VISION_MIN_BITS == PROTECTED_MIN_BITS[TensorRole.VISION] == 16
    assert NORM_MIN_BITS == PROTECTED_MIN_BITS[TensorRole.NORM] == 16
    assert EMBEDDING_MIN_BITS == PROTECTED_MIN_BITS[TensorRole.EMBEDDING] == 8
    assert LM_HEAD_MIN_BITS == 8
    assert PROTECTED_MIN_BITS[TensorRole.LM_HEAD] == 16


def test_commands_stay_on_mlx_and_keep_mtp_protected(tmp_path: Path) -> None:
    commands = axquant_commands(Path("/models/Qwen3.8-27B-bf16"), tmp_path)
    flat = [" ".join(command) for command in commands]
    assert all(command[0] == "axquant" for command in commands)
    assert not any("plan-cuda" in command or "convert-cuda" in command for command in flat)
    plan = next(command for command in flat if command.startswith("axquant plan "))
    convert = flat[-1]
    assert "--mtp protected" in plan
    assert "--bits 8,16" in plan
    assert "--target-bpw 9.224" in plan
    assert "--methods affine" in plan
    assert "--lm-head-floor 8bit" in plan
    assert "mxfp4" not in plan
    assert "--allow-unmeasured" in plan
    assert "--allow-mtp-unmeasured" not in plan
    assert "--q-mode mxfp8" in convert
    assert "--allow-unmeasured" in convert


def test_prepare_records_missing_bf16_without_nvidia(tmp_path: Path) -> None:
    manifest = prepare(tmp_path / "work", tmp_path / "missing-bf16")
    assert manifest["bf16_present"] is False
    assert manifest["nvidia_required"] is False
    assert manifest["product_id"] == "AX Coder"
    assert manifest["ax_coder_name_authorized"] is True
    assert manifest["host_class"] == "apple-mlx"


def test_measure_refuses_a_missing_bf16_directory(tmp_path: Path) -> None:
    code = main(["measure", "--work", str(tmp_path / "work"), "--bf16", str(tmp_path / "missing")])
    assert code == 2


def test_low_baseline_stops_before_a_recommendation() -> None:
    cells = {
        "control": _cell(5),
        "sensitivity": _cell(30, seconds=1000),
        "template": _cell(30, seconds=1000),
        "combined": _cell(30, seconds=900),
    }
    report = score_cells(cells)
    assert report["recommendation"] is None
    assert report["release_tag"] is False


def test_cheapest_eligible_single_wins_and_is_not_a_release() -> None:
    cells = {
        "control": _cell(20, seconds=4000),
        "sensitivity": _cell(22, seconds=2000),
        "template": _cell(21, seconds=3000),
        "combined": _cell(19, seconds=1500),
    }
    report = score_cells(cells)
    assert report["recommendation"] == "sensitivity"
    assert report["release_tag"] is False
    assert report["product_id"] == "AX Coder"
    assert report["ax_coder_name_authorized"] is True


def test_extra_retries_remove_a_candidate() -> None:
    cells = {
        "control": _cell(20, seconds=4000),
        "sensitivity": _cell(22, seconds=2000, retries=3),
        "template": _cell(21, seconds=3000),
        "combined": _cell(22, seconds=1000),
    }
    report = score_cells(cells)
    assert report["recommendation"] == "template"


def test_score_command_reads_a_cells_file(tmp_path: Path) -> None:
    path = tmp_path / "cells.json"
    path.write_text(
        json.dumps(
            {
                "control": _cell(20, seconds=4000),
                "sensitivity": _cell(22, seconds=2000),
                "template": _cell(21, seconds=3000),
                "combined": _cell(22, seconds=1800),
            }
        ),
        encoding="utf-8",
    )
    assert main(["score", "--cells", str(path)]) == 0
