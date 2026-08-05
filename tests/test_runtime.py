from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.errors import ArtifactError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.runtime import (
    assert_conversion_scope,
    build_runtime_metadata,
    check_ax_engine,
    check_mlx_lm_generation,
    check_mlx_lm_static,
    generate_ax_engine_manifest,
)
from axquant.schema import (
    KvCachePlan,
    KvLayerAllocation,
    ModelIdentity,
    PlanRequest,
    ProfileName,
    RuntimeName,
)
from axquant.serde import write_data


def _simulate_ubuntu_without_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Match Ubuntu CI: no mlx_lm import and no mlx_lm.* on PATH.

    Host Macs often have both; without this, generation-smoke tests pass locally
    while the same code fails on python-compatibility jobs. See
    docs/ci-root-causes.md.
    """

    monkeypatch.setattr("axquant.runtime.importlib.util.find_spec", lambda _name: None)

    def _which(name: str) -> str | None:
        if name.startswith("mlx_lm"):
            return None
        return None

    monkeypatch.setattr("axquant.runtime.shutil.which", _which)


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
            stdout='{"schema_version":"ax.generate_manifest.v1","status":"ready"}',
            stderr="",
        )

    result = generate_ax_engine_manifest(
        qwen36_model_dir,
        executable=str(executable),
        runner=runner,
    )
    assert result.passed is True
    assert result.report["schema_version"] == "ax.generate_manifest.v1"


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "not-json",
        '{"schema_version":"ax.generate_manifest.v1"}',
        '{"status":"ok"}',
        '{"result":"ready","status":"blocked"}',
    ],
)
def test_ax_engine_manifest_requires_parseable_ready_status(
    qwen36_model_dir: Path,
    tmp_path: Path,
    stdout: str,
) -> None:
    executable = tmp_path / "ax-engine-bench"
    executable.write_text("", encoding="utf-8")

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        (Path(command[-1]) / "model-manifest.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(list(command), 0, stdout=stdout, stderr="")

    result = generate_ax_engine_manifest(
        qwen36_model_dir,
        executable=str(executable),
        runner=runner,
    )

    assert result.passed is False
    assert "parseable ready status" in result.stderr


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
    assert result.report["config_valid"] is True
    assert result.report["weights_valid"] is True
    assert result.report["main_weight_files"] == ["model.safetensors"]


@pytest.mark.parametrize("config_text", ["{", "[]"])
def test_mlx_lm_static_check_requires_json_object_config(
    qwen36_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_text: str,
) -> None:
    (qwen36_model_dir / "config.json").write_text(config_text, encoding="utf-8")
    monkeypatch.setattr("axquant.runtime.importlib.util.find_spec", lambda _: object())

    result = check_mlx_lm_static(qwen36_model_dir)

    assert result.available is True
    assert result.passed is False
    assert result.report["config_present"] is True
    assert result.report["config_valid"] is False


def test_mlx_lm_static_check_rejects_sidecars_without_main_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "sidecar-only"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    for name in ("mtp.safetensors", "mtp_head.safetensors", "vision.safetensors"):
        (model_dir / name).write_bytes(b"sidecar")
    monkeypatch.setattr("axquant.runtime.importlib.util.find_spec", lambda _: object())

    result = check_mlx_lm_static(model_dir)

    assert result.available is True
    assert result.passed is False
    assert result.report["weights_present"] is False
    assert result.report["weights_valid"] is False
    assert result.report["main_weight_files"] == []


def test_mlx_lm_static_check_rejects_invalid_main_safetensors(
    qwen36_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (qwen36_model_dir / "model.safetensors").write_bytes(b"not-safetensors")
    monkeypatch.setattr("axquant.runtime.importlib.util.find_spec", lambda _: object())

    result = check_mlx_lm_static(qwen36_model_dir)

    assert result.available is True
    assert result.passed is False
    assert result.report["weights_present"] is True
    assert result.report["weights_valid"] is False
    assert result.report["invalid_main_weight_files"] == ["model.safetensors"]


def test_mlx_lm_generation_check_requires_successful_output(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simulate_ubuntu_without_mlx(monkeypatch)
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
    assert result.report["kv_cache_execution"] == "runtime-default"


def test_generation_smoke_executes_advisory_kv_quantization(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planned KV artifact runs its advisory bits through QuantizedKVCache."""
    from axquant.planner import allocate_kv_cache

    _simulate_ubuntu_without_mlx(monkeypatch)
    plan = _qwen_plan(qwen36_model_dir)
    plan.kv_cache = allocate_kv_cache(
        20,
        default_bits=4,
        group_size=64,
    )
    write_data(
        qwen36_model_dir / "axquant_runtime.json",
        build_runtime_metadata(plan, qwen36_model_dir),
    )
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\necho OK\n", encoding="utf-8")
    executable.chmod(0o755)
    seen: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        seen.append(list(command))
        return subprocess.CompletedProcess(list(command), 0, stdout="OK", stderr="")

    result = check_mlx_lm_generation(
        qwen36_model_dir,
        executable=str(executable),
        runner=runner,
    )
    assert result.passed
    command = seen[0]
    assert command[command.index("--kv-bits") + 1] == "4"
    assert command[command.index("--kv-group-size") + 1] == "64"
    assert result.report["kv_cache_execution"] == {
        "kv_bits": 4,
        "kv_group_size": 64,
        "source": "axquant_runtime.json advisory values",
    }


@pytest.mark.parametrize("runtime_text", ["{", "[]"])
def test_generation_smoke_rejects_malformed_runtime_metadata(
    qwen36_model_dir: Path,
    tmp_path: Path,
    runtime_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simulate_ubuntu_without_mlx(monkeypatch)
    (qwen36_model_dir / "axquant_runtime.json").write_text(
        runtime_text,
        encoding="utf-8",
    )
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("", encoding="utf-8")

    with pytest.raises(ArtifactError, match=r"axquant_runtime\.json is invalid"):
        check_mlx_lm_generation(
            qwen36_model_dir,
            executable=str(executable),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("advisory_mlx_lm_kv_bits", 1),
        ("advisory_mlx_lm_kv_bits", 5),
        ("advisory_mlx_lm_kv_bits", "4"),
        ("advisory_mlx_lm_kv_group_size", 0),
        ("advisory_mlx_lm_kv_group_size", 7),
        ("advisory_mlx_lm_kv_group_size", "64"),
        ("advisory", False),
    ],
)
def test_generation_smoke_rejects_invalid_advisory_kv_fields(
    qwen36_model_dir: Path,
    tmp_path: Path,
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from axquant.planner import allocate_kv_cache

    _simulate_ubuntu_without_mlx(monkeypatch)
    plan = _qwen_plan(qwen36_model_dir)
    plan.kv_cache = allocate_kv_cache(20, default_bits=4, group_size=64)
    payload = build_runtime_metadata(plan, qwen36_model_dir).model_dump(mode="json")
    assert isinstance(payload["kv_cache"], dict)
    payload["kv_cache"][field] = value
    write_data(qwen36_model_dir / "axquant_runtime.json", payload)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("", encoding="utf-8")

    with pytest.raises(ArtifactError, match=r"axquant_runtime\.json is invalid"):
        check_mlx_lm_generation(
            qwen36_model_dir,
            executable=str(executable),
        )


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
        assert_conversion_scope(plan)


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


def test_runtime_metadata_advisory_kv_values_are_an_observed_pair(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    plan = _qwen_plan(qwen36_model_dir)
    pairs = [(4, 32)] * 3 + [(8, 64)] * 2 + [(8, 128)] * 2
    plan.kv_cache = KvCachePlan(
        allocation_basis="architecture-prior",
        min_bits=4,
        default_bits=4,
        default_group_size=32,
        layers=[
            KvLayerAllocation(
                layer_index=index,
                bits=bits,
                group_size=group_size,
                reason="test allocation",
            )
            for index, (bits, group_size) in enumerate(pairs)
        ],
    )
    output = tmp_path / "artifact"
    output.mkdir()

    metadata = build_runtime_metadata(plan, output)

    assert metadata.kv_cache is not None
    advisory = (
        metadata.kv_cache.advisory_mlx_lm_kv_bits,
        metadata.kv_cache.advisory_mlx_lm_kv_group_size,
    )
    assert advisory == (4, 32)
    assert advisory in pairs


def test_runtime_metadata_schema_rejects_unobserved_advisory_pair(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    plan = _qwen_plan(qwen36_model_dir)
    plan.kv_cache = KvCachePlan(
        allocation_basis="architecture-prior",
        min_bits=4,
        default_bits=4,
        default_group_size=32,
        layers=[
            KvLayerAllocation(
                layer_index=0,
                bits=4,
                group_size=32,
                reason="test allocation",
            ),
            KvLayerAllocation(
                layer_index=1,
                bits=8,
                group_size=64,
                reason="test allocation",
            ),
        ],
    )
    output = tmp_path / "artifact"
    output.mkdir()
    kv_metadata = build_runtime_metadata(plan, output).kv_cache
    assert kv_metadata is not None
    payload = kv_metadata.model_dump(mode="json")
    payload["advisory_mlx_lm_kv_bits"] = 6

    with pytest.raises(ValueError, match="must be an observed layer allocation"):
        type(kv_metadata).model_validate(payload)


def test_runtime_metadata_advisory_kv_tie_prefers_safer_pair(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    plan = _qwen_plan(qwen36_model_dir)
    pairs = [(4, 32)] * 2 + [(8, 64)] * 2
    plan.kv_cache = KvCachePlan(
        allocation_basis="architecture-prior",
        min_bits=4,
        default_bits=4,
        default_group_size=32,
        layers=[
            KvLayerAllocation(
                layer_index=index,
                bits=bits,
                group_size=group_size,
                reason="test allocation",
            )
            for index, (bits, group_size) in enumerate(pairs)
        ],
    )
    output = tmp_path / "artifact"
    output.mkdir()

    metadata = build_runtime_metadata(plan, output)

    assert metadata.kv_cache is not None
    assert (
        metadata.kv_cache.advisory_mlx_lm_kv_bits,
        metadata.kv_cache.advisory_mlx_lm_kv_group_size,
    ) == (8, 64)


def test_runtime_metadata_does_not_infer_mtp_capability_from_policy_mode(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    plan = _qwen_plan(qwen36_model_dir)
    assert plan.mtp.mode == "protected"
    assert plan.mtp_distribution
    output = tmp_path / "artifact"
    output.mkdir()

    metadata = build_runtime_metadata(plan, output)

    assert metadata.mtp.detected is False
    assert metadata.primary_runtime.mtp_support == "none"
    assert metadata.compatible_runtimes[0].mtp_support == "none"
    assert metadata.memory_policy["mtp_buffers"] == "not-required"


def test_runtime_metadata_detects_alternate_mtp_sidecar_filename(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "mtp_head.safetensors").write_bytes(
        (qwen36_model_dir / "mtp.safetensors").read_bytes()
    )

    metadata = build_runtime_metadata(_qwen_plan(qwen36_model_dir), output)

    assert metadata.mtp.detected is True
    assert metadata.mtp.sidecar_file == "mtp_head.safetensors"
    assert metadata.primary_runtime.mtp_support == "native"
    assert metadata.memory_policy["mtp_buffers"] == "preallocate-when-enabled"


def test_runtime_metadata_rejects_invalid_mtp_sidecar(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "mtp_head.safetensors").write_bytes(b"not-safetensors")

    with pytest.raises(ArtifactError, match="invalid MTP Safetensors sidecar"):
        build_runtime_metadata(_qwen_plan(qwen36_model_dir), output)


def test_runtime_metadata_prefers_structured_mtp_sidecar_bits(
    qwen36_model_dir: Path,
) -> None:
    (qwen36_model_dir / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "mtp_depth_max": 3,
                "mtp_sidecar_bits": 4,
                "mtp_sidecar": "INT8 quantized projections",
            }
        ),
        encoding="utf-8",
    )

    metadata = build_runtime_metadata(_qwen_plan(qwen36_model_dir), qwen36_model_dir)

    assert metadata.mtp.draft_tokens == 3
    assert metadata.mtp.head_precision == "4bit"


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        ({"mtp_sidecar_bits": True}, "mtp_sidecar_bits"),
        ({"mtp_sidecar_bits": 3}, "mtp_sidecar_bits"),
        ({"mtp_sidecar_bits": "8"}, "mtp_sidecar_bits"),
        ({"mtp_depth_max": 0}, "mtp_depth_max"),
        ({"mtp_depth_max": True}, "mtp_depth_max"),
        ({"mtp_sidecar": {"bits": 8}}, "mtp_sidecar must be a string"),
    ],
)
def test_runtime_metadata_rejects_invalid_mtp_runtime_contract_fields(
    qwen36_model_dir: Path,
    contract: dict[str, object],
    message: str,
) -> None:
    plan = _qwen_plan(qwen36_model_dir)
    (qwen36_model_dir / "mtplx_runtime.json").write_text(
        json.dumps(contract),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match=message):
        build_runtime_metadata(plan, qwen36_model_dir)


@pytest.mark.parametrize("contract_text", ["{", "[]"])
def test_runtime_metadata_rejects_malformed_mtp_runtime_contract(
    qwen36_model_dir: Path,
    contract_text: str,
) -> None:
    plan = _qwen_plan(qwen36_model_dir)
    (qwen36_model_dir / "mtplx_runtime.json").write_text(
        contract_text,
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match=r"mtplx_runtime\.json"):
        build_runtime_metadata(plan, qwen36_model_dir)


def test_runtime_metadata_without_kv_plan_is_unchanged(
    qwen36_model_dir: Path, tmp_path: Path
) -> None:
    plan = _qwen_plan(qwen36_model_dir)
    output = tmp_path / "artifact"
    output.mkdir()
    metadata = build_runtime_metadata(plan, output)
    assert metadata.kv_cache is None
    assert metadata.memory_policy["kv_cache_precision"] == "runtime-default"


def test_conversion_scope_rejects_unbound_measured_kv_basis(qwen36_model_dir: Path) -> None:
    from axquant.planner import allocate_kv_cache

    plan = _qwen_plan(qwen36_model_dir)
    layer_count = plan.architecture_profile.text_layer_count
    assert layer_count is not None
    kv = allocate_kv_cache(layer_count, default_bits=4)
    plan.kv_cache = kv.model_copy(update={"allocation_basis": "measured"})
    with pytest.raises(ArtifactError, match="bind its sensitivity report digest"):
        assert_conversion_scope(plan)
    plan.kv_cache = kv.model_copy(
        update={"allocation_basis": "measured", "sensitivity_sha256": "a" * 64}
    )
    assert_conversion_scope(plan)


def test_conversion_scope_rejects_kv_layer_count_mismatch(qwen36_model_dir: Path) -> None:
    from axquant.planner import allocate_kv_cache

    plan = _qwen_plan(qwen36_model_dir)
    layer_count = plan.architecture_profile.text_layer_count
    assert layer_count is not None
    plan.kv_cache = allocate_kv_cache(layer_count + 1, default_bits=4)
    with pytest.raises(ArtifactError, match="does not cover the text layer count"):
        assert_conversion_scope(plan)


def test_kv_layered_check_parses_execution_report(qwen36_model_dir: Path) -> None:
    from axquant.runtime import check_mlx_lm_kv_layered

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        assert command[1:3] == ["-m", "axquant.kv_exec"]
        return subprocess.CompletedProcess(
            list(command),
            0,
            stdout=(
                '{"ok": true, "output_characters": 4, '
                '"planned_layer_bits": [8, 4], "executed_layer_bits": [8, 4], '
                '"quantized_layers_active": 2, "per_layer_execution": true}'
            ),
            stderr="",
        )

    result = check_mlx_lm_kv_layered(qwen36_model_dir, runner=runner)
    assert result.passed
    assert result.check_kind == "kv-layered-generation-smoke"
    assert result.report["executed_layer_bits"] == [8, 4]
    assert result.report["per_layer_execution"] is True

    def failing_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(command),
            1,
            stdout='{"ok": false, "error": "no layer accepted a quantized KV cache"}',
            stderr="",
        )

    failed = check_mlx_lm_kv_layered(qwen36_model_dir, runner=failing_runner)
    assert not failed.passed
    assert "no layer accepted" in failed.report["error"]
