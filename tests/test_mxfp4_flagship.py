from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from axquant.calibration_dataset import validate_calibration_dataset
from axquant.cli._parser import _build_parser
from axquant.coding_calibration import DOMAINS, agent_coding_dataset_bytes, agent_coding_samples
from axquant.schema import EvidenceKind, QuantMethod, TensorRole
from scripts import run_qwen38_27b_mxfp4_flagship as flagship


def test_agent_calibration_covers_workloads_and_valid_tool_sequences(tmp_path: Path) -> None:
    path = tmp_path / "calibration.jsonl"
    path.write_bytes(agent_coding_dataset_bytes())
    assert validate_calibration_dataset(path) == []
    samples = agent_coding_samples()
    assert len({s["id"] for s in samples}) == len(samples)
    assert {s["domain"] for s in samples} == set(DOMAINS)
    assert sum(s["domain"] == "coding" for s in samples) > len(samples) / 2
    assert all(s["license"] == "CC0-1.0" and s["redistributable"] for s in samples)
    for sample in samples:
        assert json.loads(sample["text"]) == sample["messages"]
        pending = set()
        for message in sample["messages"]:
            for call in message.get("tool_calls", []):
                pending.add(call["id"])
                assert isinstance(json.loads(call["function"]["arguments"]), dict)
            if message["role"] == "tool":
                assert message["tool_call_id"] in pending
                pending.remove(message["tool_call_id"])
        assert not pending


def test_authored_repair_actually_fixes_capacity_boundary() -> None:
    sample = next(s for s in agent_coding_samples() if s["id"] == "repair-00-0")
    source = sample["messages"][2]["content"]
    original = {}
    exec(source, original)
    assert original["append_record"]([1, 2], 99, 2) == [1, 2, 99]
    repaired = {}
    exec(source.replace(" > capacity", " >= capacity"), repaired)
    with pytest.raises(ValueError, match="full"):
        repaired["append_record"]([1, 2], 99, 2)
    assert repaired["append_record"]([], 7, 1) == [7]


def test_flagship_commands_are_executable_and_use_measured_mxfp4(tmp_path: Path) -> None:
    commands = flagship.commands(tmp_path / "bf16", tmp_path / "work", 5.6)
    parser = _build_parser()
    parsed = [parser.parse_args(command) for command in commands]
    assert [p.command for p in parsed] == ["calibrate", "analyze", "plan", "convert"]
    assert parsed[1].measure_trunk_mtp
    assert parsed[1].token_budget >= 8192
    assert QuantMethod.MXFP4 in parsed[1].methods
    assert parsed[2].target_bpw == 5.6
    assert parsed[2].mtp == "protected"
    assert parsed[2].mtp_min_bits == 16
    assert parsed[2].mtp_bits == (16,)
    assert parsed[3].q_mode == "mxfp4"
    assert not any("--allow-unmeasured" in c or "--allow-mtp-unmeasured" in c for c in commands)


def test_cli_rejects_trunk_mtp_without_measured_calibration() -> None:
    from axquant.cli import _run

    args = _build_parser().parse_args(["analyze", "--model", "missing", "--measure-trunk-mtp"])
    with pytest.raises(ValueError, match="requires measured --calibration probing"):
        _run(args)


def test_write_once_rejects_changed_study_input(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    flagship.write_once(path, b"original")
    flagship.write_once(path, b"original")
    with pytest.raises(ValueError, match="existing study input differs"):
        flagship.write_once(path, b"changed")
    assert path.read_bytes() == b"original"


def test_frozen_input_changes_stop_execution(tmp_path: Path) -> None:
    path = tmp_path / "holdout.jsonl"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen study input changed"):
        flagship.verify_inputs({"inputs": {str(path): "0" * 64}})


def test_measure_fails_before_creating_outputs_without_bf16(tmp_path: Path) -> None:
    work = tmp_path / "work"
    assert (
        flagship.main(
            [
                "measure",
                "--work",
                str(work),
                "--bf16",
                str(tmp_path / "missing"),
                "--control",
                str(tmp_path / "control"),
                "--coding-holdout",
                str(tmp_path / "coding"),
                "--general-holdout",
                str(tmp_path / "general"),
            ]
        )
        == 2
    )
    assert not work.exists()


def test_selected_allocation_requires_measured_mtp_proxy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = Mock()
    report.evidence_kind = EvidenceKind.MEASURED
    report.calibration.metadata = {"measure_trunk_mtp": True}
    report.entries = []
    plan = SimpleNamespace(
        evidence_kind=EvidenceKind.MEASURED,
        analysis_sha256="a" * 64,
        mtp=SimpleNamespace(mode="protected", min_bits=16),
        assignments=[
            SimpleNamespace(
                tensor="layer.weight",
                role=TensorRole.MLP,
                bits=4,
                method=QuantMethod.MXFP4,
                group_size=32,
            )
        ],
    )
    monkeypatch.setattr(flagship, "load_sensitivity_report", lambda _: report)
    monkeypatch.setattr(flagship, "load_quantization_plan", lambda _: plan)
    monkeypatch.setattr(flagship, "stable_sha256", lambda _: "a" * 64)
    with pytest.raises(ValueError, match="selected allocation lacks MTP sensitivity"):
        flagship.check_measured_plan(tmp_path)
    report.entries = [
        SimpleNamespace(
            tensor=SimpleNamespace(name="layer.weight"),
            candidates=[
                SimpleNamespace(
                    supported=True,
                    bits=4,
                    method=QuantMethod.MXFP4,
                    group_size=32,
                    metrics=SimpleNamespace(mtp_acceptance_loss=0.1),
                ),
            ],
        )
    ]
    flagship.check_measured_plan(tmp_path)
    plan.assignments[0].role = TensorRole.MTP_BLOCK
    with pytest.raises(ValueError, match="preserve all MTP tensors at BF16"):
        flagship.check_measured_plan(tmp_path)


def test_execution_stops_on_failed_stage_and_retains_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        flagship.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 1)
    )
    with pytest.raises(ValueError, match="analyze failed"):
        flagship.execute(["analyze"], tmp_path, "analyze")
    receipt = json.loads((tmp_path / "analyze-receipt.json").read_text())
    assert receipt["returncode"] == 1


def test_benchmark_runtime_records_dflash_isolation() -> None:
    args = _build_parser().parse_args(
        [
            "benchmark-ab",
            "--model",
            "candidate",
            "--prompts",
            "prompts.jsonl",
            "--runtime-env",
            "AX_MLX_DFLASH2=0",
        ]
    )
    from axquant.benchmark import parse_runtime_env_items

    assert parse_runtime_env_items(args.runtime_env) == {"AX_MLX_DFLASH2": "0"}


def test_serve_defaults_to_required_mtp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(flagship, "read_data", lambda _: {"inputs": {}})
    monkeypatch.setattr(flagship, "check_measured_plan", lambda _: None)
    monkeypatch.setattr(flagship, "verify_candidate", lambda _: None)
    call = Mock(return_value=0)
    monkeypatch.setattr(flagship.subprocess, "call", call)
    assert flagship.main(["serve", "--work", str(tmp_path)]) == 0
    command = call.call_args.args[0]
    assert command[-2:] == ["--mlx-mtp-policy", "required"]


def test_incomplete_measurement_cannot_serve(tmp_path: Path) -> None:
    from axquant.errors import ArtifactError

    with pytest.raises(ArtifactError):
        flagship.verify_candidate(tmp_path)


def test_changed_candidate_cannot_reuse_measurement_receipt(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text("changed", encoding="utf-8")
    (tmp_path / "measurement.json").write_text(
        json.dumps({"matched_within_one_percent": True, "files": {"plan.json": "0" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="measured candidate changed"):
        flagship.verify_candidate(tmp_path)
