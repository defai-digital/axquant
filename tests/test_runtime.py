from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.errors import ArtifactError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.runtime import (
    assert_qwen36_conversion_scope,
    build_runtime_metadata,
    check_ax_engine,
    check_mlx_lm_generation,
    check_mlx_lm_static,
    generate_ax_engine_manifest,
)
from axquant.schema import ModelIdentity, PlanRequest, ProfileName, RuntimeName


def _qwen_plan(model_dir: Path):
    inventory = inspect_model(
        model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="abc",
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
    )
    return plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=16.0,
            allow_unmeasured=True,
        ),
    )


def test_ax_engine_manifest_uses_native_generator_contract(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "ax-engine-bench"
    executable.write_text("", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        assert command[1:4] == [
            "generate-manifest",
            "--json",
            "--validate",
        ]
        (Path(command[-1]) / "model-manifest.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            list(command),
            0,
            stdout='{"schema_version":"ax.generate_manifest.v1"}',
            stderr="",
        )

    result = generate_ax_engine_manifest(
        qwen36_model_dir,
        executable=str(executable),
        runner=runner,
    )
    assert result.passed is True
    assert result.report["schema_version"] == "ax.generate_manifest.v1"


def test_ax_engine_manifest_rejects_zero_exit_validation_failure(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "ax-engine-bench"
    executable.write_text("", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        (Path(command[-1]) / "model-manifest.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(
            list(command),
            0,
            stdout="",
            stderr="generated manifest validation failed: invalid native model manifest",
        )

    result = generate_ax_engine_manifest(
        qwen36_model_dir,
        executable=str(executable),
        runner=runner,
    )

    assert result.passed is False


def test_runtime_metadata_declares_ax_primary_and_mlx_fallback(
    qwen36_model_dir: Path,
) -> None:
    metadata = build_runtime_metadata(_qwen_plan(qwen36_model_dir), qwen36_model_dir)
    assert metadata.primary_runtime.name == RuntimeName.AX_ENGINE
    assert metadata.primary_runtime.compatibility_level == "A"
    assert metadata.compatible_runtimes[0].name == RuntimeName.MLX_LM
    assert metadata.compatible_runtimes[0].compatibility_level == "B"
    assert metadata.mtp.draft_tokens == 2
    assert metadata.mtp.head_precision == "8bit"
    assert metadata.ax_engine.decode_kernel is None
    assert metadata.ax_engine.kernel_evidence == "unmeasured"


def test_ax_engine_doctor_contract_is_machine_readable(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "ax-engine"
    executable.write_text("", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        assert command[1:3] == ["doctor", "--json"]
        assert command[3] == "--mlx-model-artifacts-dir"
        return subprocess.CompletedProcess(
            list(command),
            0,
            stdout='{"schema_version":"ax.engine.doctor.v1","result":"ready"}',
            stderr="",
        )

    result = check_ax_engine(
        qwen36_model_dir,
        executable=str(executable),
        runner=runner,
    )
    assert result.available is True
    assert result.passed is True


def test_ax_engine_doctor_accepts_current_status_field(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "ax-engine"
    executable.write_text("", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(command),
            0,
            stdout='{"schema_version":"ax.engine_bench.doctor.v1","status":"ready"}',
            stderr="",
        )

    result = check_ax_engine(
        qwen36_model_dir,
        executable=str(executable),
        runner=runner,
    )

    assert result.passed is True


def test_ax_engine_doctor_rejects_conflicting_status_fields(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "ax-engine"
    executable.write_text("", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(command),
            0,
            stdout='{"result":"ready","status":"blocked"}',
            stderr="",
        )

    result = check_ax_engine(
        qwen36_model_dir,
        executable=str(executable),
        runner=runner,
    )

    assert result.passed is False


def test_ax_engine_doctor_does_not_accept_unknown_success_payload(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "ax-engine"
    executable.write_text("", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(command),
            0,
            stdout='{"schema_version":"ax.engine.doctor.v1"}',
            stderr="",
        )

    result = check_ax_engine(
        qwen36_model_dir,
        executable=str(executable),
        runner=runner,
    )
    assert result.passed is False


def test_mlx_lm_static_check_accepts_installed_cli(
    qwen36_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("axquant.runtime.importlib.util.find_spec", lambda _: None)
    monkeypatch.setattr(
        "axquant.runtime.shutil.which",
        lambda executable: f"/opt/bin/{executable}",
    )
    result = check_mlx_lm_static(qwen36_model_dir)
    assert result.available is True
    assert result.passed is True
    assert result.report["mlx_lm_importable"] is False
    assert result.report["mlx_lm_executable"] == "/opt/bin/mlx_lm.generate"


def test_mlx_lm_generation_check_requires_successful_output(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        assert command[1:3] == ["--model", str(qwen36_model_dir.resolve())]
        return subprocess.CompletedProcess(
            list(command),
            0,
            stdout="OK",
            stderr="",
        )

    identity = ModelIdentity(
        model_id="AutomatosX/candidate",
        revision="candidate-revision",
        local_path=str(qwen36_model_dir.resolve()),
    )
    result = check_mlx_lm_generation(
        qwen36_model_dir,
        executable=str(executable),
        runner=runner,
        model_identity=identity,
    )
    assert result.passed
    assert result.check_kind == "generation-smoke"
    assert result.model == identity
    assert result.report["scope"] == "load-and-generation"


def test_conversion_scope_rejects_generic_inventory(tiny_model_dir: Path) -> None:
    inventory = inspect_model(tiny_model_dir)
    report = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=16.0,
            allow_unmeasured=True,
        ),
    )
    with pytest.raises(ArtifactError, match=r"inspect-only"):
        assert_qwen36_conversion_scope(plan)


def test_runtime_metadata_emits_kv_cache_table(qwen36_model_dir: Path, tmp_path: Path) -> None:
    from axquant.planner import allocate_kv_cache

    plan = _qwen_plan(qwen36_model_dir)
    plan.kv_cache = allocate_kv_cache(20, default_bits=4, group_size=plan.group_size)
    output = tmp_path / "artifact"
    output.mkdir()
    metadata = build_runtime_metadata(plan, output)
    assert metadata.kv_cache is not None
    assert metadata.kv_cache.allocation_basis == "architecture-prior"
    assert len(metadata.kv_cache.layer_bits) == 20
    assert metadata.kv_cache.advisory_mlx_lm_kv_bits == 4
    assert metadata.kv_cache.advisory_mlx_lm_kv_group_size == plan.group_size
    assert metadata.kv_cache.advisory is True
    assert metadata.memory_policy["kv_cache_precision"] == "planned-per-layer"


def test_runtime_metadata_without_kv_plan_is_unchanged(
    qwen36_model_dir: Path, tmp_path: Path
) -> None:
    plan = _qwen_plan(qwen36_model_dir)
    output = tmp_path / "artifact"
    output.mkdir()
    metadata = build_runtime_metadata(plan, output)
    assert metadata.kv_cache is None
    assert metadata.memory_policy["kv_cache_precision"] == "runtime-default"


def test_conversion_scope_rejects_measured_kv_basis(qwen36_model_dir: Path) -> None:
    from axquant.planner import allocate_kv_cache

    plan = _qwen_plan(qwen36_model_dir)
    layer_count = plan.architecture_profile.text_layer_count
    assert layer_count is not None
    kv = allocate_kv_cache(layer_count, default_bits=4)
    plan.kv_cache = kv.model_copy(update={"allocation_basis": "measured"})
    with pytest.raises(ArtifactError, match="measured KV-cache allocation is not yet supported"):
        assert_qwen36_conversion_scope(plan)


def test_conversion_scope_rejects_kv_layer_count_mismatch(qwen36_model_dir: Path) -> None:
    from axquant.planner import allocate_kv_cache

    plan = _qwen_plan(qwen36_model_dir)
    layer_count = plan.architecture_profile.text_layer_count
    assert layer_count is not None
    plan.kv_cache = allocate_kv_cache(layer_count + 1, default_bits=4)
    with pytest.raises(ArtifactError, match="does not cover the text layer count"):
        assert_qwen36_conversion_scope(plan)
