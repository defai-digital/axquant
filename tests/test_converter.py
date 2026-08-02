from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

import axquant.converter as converter
from axquant.analyzer import architecture_prior_report
from axquant.errors import ArtifactError, PlanningError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.predicate import build_quant_predicate
from axquant.schema import (
    ArtifactManifest,
    CalibrationEvidence,
    CalibrationManifest,
    EvidenceKind,
    PlanRequest,
    ProfileName,
    QuantizationPlan,
    QuantizerExecutionManifest,
    QuantMethod,
    TensorRole,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data


def _plan(model_dir: Path) -> QuantizationPlan:
    inventory = inspect_model(
        model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="source-revision",
    )
    report = architecture_prior_report(
        inventory,
        profile=ProfileName.AGENT_CODING,
    )
    return plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
        ),
    )


def _measured_plan(
    model_dir: Path,
    calibration_path: Path,
) -> QuantizationPlan:
    plan = _plan(model_dir)
    calibration = CalibrationManifest(
        model=plan.source_model,
        profile=plan.profile,
        dataset_id="internal/calibration-v1",
        dataset_sha256="a" * 64,
        samples=128,
        domains=["coding", "json", "tool", "multilingual", "long-context"],
        sequence_length=2048,
        random_seed=7,
        calibration_evaluation_separation_attested=True,
    )
    write_data(calibration_path, calibration)
    plan.evidence_kind = EvidenceKind.MEASURED
    plan.calibration = CalibrationEvidence(
        dataset_id=calibration.dataset_id,
        dataset_sha256=calibration.dataset_sha256,
        samples=calibration.samples,
        domains=calibration.domains,
        sequence_length=calibration.sequence_length,
        backend="test-probe",
        reference="calibration_manifest.json",
        metadata={"calibration_manifest_sha256": file_sha256(calibration_path)},
    )
    return plan


def test_awq_convert_preflight_and_predicate_are_executable(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real convert preflight/predicate entry points with an AWQ plan."""
    plan = _plan(qwen36_model_dir)
    awq_targets = [
        allocation
        for allocation in plan.assignments
        if allocation.bits < 16 and allocation.role == TensorRole.MLP
    ]
    assert awq_targets
    for allocation in awq_targets:
        allocation.method = QuantMethod.AWQ
    plan.hardware = plan.hardware.model_copy(
        update={
            "supported_methods": (
                *plan.hardware.supported_methods,
                QuantMethod.AWQ,
            )
        }
    )

    # Preflight path must admit AWQ without activations (refinement disabled).
    preflight = build_quant_predicate(plan, execute_refinement=False)
    for allocation in plan.assignments:
        if allocation.bits < 16:
            config = preflight(allocation.module_path, object())
            assert config == {
                "group_size": allocation.group_size,
                "bits": allocation.bits,
                "mode": "affine",
            }
    assert preflight.unmatched_quantized_modules() == set()

    # Match the plan group size (64) so portable AWQ scale search can run.
    awq_activations = {
        allocation.module_path: np.random.default_rng(i).standard_normal(
            (16, 64),
            dtype=np.float32,
        )
        for i, allocation in enumerate(awq_targets)
    }

    class FakeModule:
        def __init__(self) -> None:
            self.weight = np.random.default_rng(1).standard_normal((8, 64), dtype=np.float32)

    class FakeModel:
        def named_modules(self):
            return [
                (allocation.module_path, FakeModule())
                for allocation in plan.assignments
                if allocation.bits < 16
            ]

    def fake_load(*args, **kwargs):
        return FakeModel(), {}, {}

    def fake_convert(model, *, mlx_path, quant_predicate, **kwargs):
        del model, kwargs
        for path, module in FakeModel().named_modules():
            quant_predicate(path, module)
        output = Path(mlx_path)
        output.mkdir()
        converted_config = json.loads(
            (qwen36_model_dir / "config.json").read_text(encoding="utf-8")
        )
        converted_config.pop("vision_config")
        (output / "config.json").write_text(json.dumps(converted_config), encoding="utf-8")
        with safe_open(qwen36_model_dir / "model.safetensors", framework="numpy") as source:
            save_file(
                {
                    name: source.get_tensor(name)
                    for name in list(source.keys())
                    if not name.startswith("visual.")
                },
                output / "model.safetensors",
            )

    # Use the real portable refine path for AWQ modules (numpy weights; no MLX required).
    import axquant.predicate as predicate_module
    from axquant.awq import refine_weight_with_awq

    def _apply_numpy_awq(
        module: FakeModule,
        *,
        activations: object,
        bits: int,
        group_size: int,
        alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    ) -> dict[str, float | int | list[float]]:
        refined, metadata = refine_weight_with_awq(
            module.weight,
            activations,
            bits=bits,
            group_size=group_size,
            alpha_grid=alpha_grid,
        )
        module.weight = refined
        return metadata

    monkeypatch.setattr(predicate_module, "_apply_awq_scale", _apply_numpy_awq)
    monkeypatch.setattr(converter, "_mlx_api", lambda: (fake_convert, fake_load))

    output = tmp_path / "awq-candidate"
    manifest = converter.convert_model(
        model=str(qwen36_model_dir),
        plan=plan,
        output=output,
        mtp_sidecar=qwen36_model_dir,
        awq_activations=awq_activations,
        allow_unmeasured=True,
        ax_engine_manifest="skip",
    )
    assert (output / "axquant_quantizer_execution.json").is_file()
    assert any(record.path == "axquant_quantizer_execution.json" for record in manifest.files)
    execution = load_model(output / "axquant_quantizer_execution.json", QuantizerExecutionManifest)
    awq_records = [record for record in execution.records if record.method == QuantMethod.AWQ]
    assert awq_records
    assert all(record.success for record in awq_records)
    assert all(
        record.note == "AWQ activation scaling followed by portable affine packing"
        for record in awq_records
    )
    assert all("awq_channel_scales" in record.metadata for record in awq_records)

    with pytest.raises(PlanningError, match="AWQ conversion requires calibration activations"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            plan=plan,
            output=tmp_path / "awq-missing-activations",
            mtp_sidecar=qwen36_model_dir,
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )


def test_measured_conversion_requires_bound_calibration_manifest(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.json"
    plan = _measured_plan(qwen36_model_dir, calibration_path)
    assert converter._validated_calibration_source(plan, calibration_path) == calibration_path
    with pytest.raises(PlanningError, match="requires --calibration-manifest"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            plan=plan,
            output=tmp_path / "candidate",
            ax_engine_manifest="skip",
        )
    plan.calibration.metadata["calibration_manifest_sha256"] = "0" * 64
    with pytest.raises(PlanningError, match="checksum"):
        converter._validated_calibration_source(plan, calibration_path)


def test_measured_conversion_accepts_canonical_calibration_manifest_hash(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.json"
    plan = _measured_plan(qwen36_model_dir, calibration_path)
    calibration = load_model(calibration_path, CalibrationManifest)
    plan.calibration.metadata["calibration_manifest_sha256"] = stable_sha256(
        calibration.model_dump(mode="json", exclude={"created_at"})
    )

    assert converter._validated_calibration_source(plan, calibration_path) == calibration_path


def test_measured_conversion_accepts_calibration_model_without_architecture(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.json"
    plan = _measured_plan(qwen36_model_dir, calibration_path)
    calibration = load_model(calibration_path, CalibrationManifest)
    calibration.model.architecture = None
    write_data(calibration_path, calibration)
    plan.calibration.metadata["calibration_manifest_sha256"] = file_sha256(calibration_path)

    assert converter._validated_calibration_source(plan, calibration_path) == calibration_path


def test_measured_conversion_accepts_legacy_cache_directory_reference(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.json"
    plan = _measured_plan(qwen36_model_dir, calibration_path)
    plan.calibration.dataset_id = str(calibration_path.parent)

    assert converter._validated_calibration_source(plan, calibration_path) == calibration_path


def test_conversion_preserves_mtp_bundle_and_runtime_contract(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(qwen36_model_dir)

    class FakeModel:
        def named_modules(self):
            return [
                (allocation.module_path, object())
                for allocation in plan.assignments
                if allocation.bits < 16
            ]

    def fake_load(*args, **kwargs):
        return FakeModel(), {}, {}

    def fake_convert(model, *, mlx_path, quant_predicate, **kwargs):
        del model, kwargs
        for path, module in FakeModel().named_modules():
            quant_predicate(path, module)
        output = Path(mlx_path)
        output.mkdir()
        converted_config = json.loads(
            (qwen36_model_dir / "config.json").read_text(encoding="utf-8")
        )
        converted_config.pop("vision_config")
        (output / "config.json").write_text(
            json.dumps(converted_config),
            encoding="utf-8",
        )
        with safe_open(qwen36_model_dir / "model.safetensors", framework="numpy") as source:
            save_file(
                {
                    name: source.get_tensor(name)
                    for name in list(source.keys())
                    if not name.startswith("visual.")
                },
                output / "model.safetensors",
            )

    monkeypatch.setattr(
        converter,
        "_mlx_api",
        lambda: (fake_convert, fake_load),
    )
    output = tmp_path / "candidate"
    manifest = converter.convert_model(
        model=str(qwen36_model_dir),
        plan=plan,
        output=output,
        mtp_sidecar=qwen36_model_dir,
        allow_unmeasured=True,
        ax_engine_manifest="skip",
    )
    loaded = load_model(output / "axquant_manifest.json", ArtifactManifest)
    assert loaded == manifest
    assert (output / "mtp.safetensors").read_bytes() == (
        qwen36_model_dir / "mtp.safetensors"
    ).read_bytes()
    assert (output / "mtplx_runtime.json").is_file()
    assert loaded.runtime.primary_runtime.name.value == "ax-engine"
    assert loaded.runtime.compatible_runtimes[0].name.value == "mlx-lm"
    assert loaded.runtime.mtp.draft_tokens == 2
    assert loaded.logical_parameters == sum(item.parameters for item in plan.assignments)
    assert loaded.measured_total_bpw > 0.0
    assert loaded.measured_main_bpw > 0.0
    assert loaded.weight_file_size_bytes == (
        loaded.main_weight_file_size_bytes + loaded.mtp_weight_file_size_bytes
    )
    assert any(record.path == "axquant_runtime.json" for record in loaded.files)
    assert any(record.path == "axquant_quantizer_execution.json" for record in loaded.files)
    assert (output / "vision.safetensors").is_file()
    assert (output / "axquant_vision_sidecar_manifest.json").is_file()
    source_inventory = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="source-revision",
    )
    with safe_open(output / "vision.safetensors", framework="numpy") as sidecar:
        assert sorted(sidecar.keys()) == sorted(
            tensor.name for tensor in source_inventory.tensors if tensor.role == TensorRole.VISION
        )
    vision = next(
        allocation for allocation in plan.assignments if allocation.role == TensorRole.VISION
    )
    assert vision.bits == 16
    converted_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    source_config = json.loads((qwen36_model_dir / "config.json").read_text(encoding="utf-8"))
    assert converted_config["vision_config"] == source_config["vision_config"]


def test_restore_protected_vision_config_rejects_conflicting_contract(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "converted"
    output.mkdir()
    source_config = json.loads((qwen36_model_dir / "config.json").read_text(encoding="utf-8"))
    source_config["vision_config"] = {"hidden_size": 1152}
    (qwen36_model_dir / "config.json").write_text(
        json.dumps(source_config),
        encoding="utf-8",
    )
    converted_config = dict(source_config)
    converted_config["vision_config"] = {"hidden_size": 2048}
    (output / "config.json").write_text(
        json.dumps(converted_config),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="conflicts with protected source field vision_config"):
        converter._restore_protected_vision_config(qwen36_model_dir, output)


def test_mtp_sidecar_provenance_rejects_transformed_bundle(tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    (sidecar / "mtp.safetensors").write_bytes(b"mtp")
    (sidecar / "ax_mtp_sidecar_manifest.json").write_text(
        json.dumps(
            {
                "output": {
                    "mtp": {
                        "path": "mtp.safetensors",
                        "sha256": file_sha256(sidecar / "mtp.safetensors"),
                        "size_bytes": 3,
                    }
                },
                "transform": {"norm_policy": "shift_mtp_norm_weights_by_1"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match=r"transform\.mode=byte_preserved"):
        converter._copy_external_mtp_bundle(sidecar, tmp_path / "output")


def test_mtp_sidecar_provenance_binds_byte_preserved_bundle(tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    source = sidecar / "mtp.safetensors"
    source.write_bytes(b"mtp")
    (sidecar / "ax_mtp_sidecar_manifest.json").write_text(
        json.dumps(
            {
                "output": {
                    "mtp": {
                        "path": "mtp.safetensors",
                        "sha256": file_sha256(source),
                        "size_bytes": source.stat().st_size,
                    }
                },
                "transform": {"mode": "byte_preserved"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()

    converter._copy_external_mtp_bundle(sidecar, output)

    assert (output / "mtp.safetensors").read_bytes() == b"mtp"
    # The byte-preserved sidecar keeps raw HF norm deltas; the copied bundle
    # must declare that layout so AX Engine shifts every norm deterministically
    # instead of guessing from tensor statistics.
    runtime = json.loads((output / "mtplx_runtime.json").read_text(encoding="utf-8"))
    assert runtime["mtp_norm_layout"] == "raw_hf_delta"


def test_mtp_sidecar_provenance_binds_alternate_filename(tmp_path: Path) -> None:
    """``mtp_head.safetensors`` is a recognized external sidecar filename
    alongside ``mtp.safetensors`` (converter/probe/release_audit all accept
    both); a directory that ships only the alternate name must still resolve
    and copy correctly rather than failing closed with "does not exist"."""
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    source = sidecar / "mtp_head.safetensors"
    source.write_bytes(b"mtp")
    (sidecar / "ax_mtp_sidecar_manifest.json").write_text(
        json.dumps(
            {
                "output": {
                    "mtp": {
                        "path": "mtp_head.safetensors",
                        "sha256": file_sha256(source),
                        "size_bytes": source.stat().st_size,
                    }
                },
                "transform": {"mode": "byte_preserved"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()

    converter._copy_external_mtp_bundle(sidecar, output)

    assert (output / "mtp.safetensors").read_bytes() == b"mtp"


def test_byte_preserved_bundle_preserves_explicit_norm_layout(tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    source = sidecar / "mtp.safetensors"
    source.write_bytes(b"mtp")
    (sidecar / "ax_mtp_sidecar_manifest.json").write_text(
        json.dumps(
            {
                "output": {
                    "mtp": {
                        "path": "mtp.safetensors",
                        "sha256": file_sha256(source),
                        "size_bytes": source.stat().st_size,
                    }
                },
                "transform": {"mode": "byte_preserved"},
            }
        ),
        encoding="utf-8",
    )
    (sidecar / "mtplx_runtime.json").write_text(
        json.dumps({"mtp_depth_max": 2, "mtp_norm_layout": "mlx_multiplier"}),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()

    converter._copy_external_mtp_bundle(sidecar, output)

    runtime = json.loads((output / "mtplx_runtime.json").read_text(encoding="utf-8"))
    # An explicit declaration from the source bundle always wins over the
    # byte-preserved default.
    assert runtime["mtp_norm_layout"] == "mlx_multiplier"
    assert runtime["mtp_depth_max"] == 2


def test_conversion_rejects_plan_without_quantized_assignments(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    plan = _plan(qwen36_model_dir)
    for allocation in plan.assignments:
        allocation.bits = 16
        allocation.method = QuantMethod.BF16
        allocation.group_size = None
    with pytest.raises(PlanningError, match="no quantized assignments"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            plan=plan,
            output=tmp_path / "candidate",
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )


def test_converted_weight_verification_rejects_missing_mtp_parameters(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    plan = _plan(qwen36_model_dir)
    staging = tmp_path / "incomplete"
    staging.mkdir()
    (staging / "config.json").write_bytes((qwen36_model_dir / "config.json").read_bytes())
    (staging / "model.safetensors").write_bytes(
        (qwen36_model_dir / "model.safetensors").read_bytes()
    )
    with pytest.raises(ArtifactError, match="logical parameter coverage mismatch"):
        converter._verify_converted_weights(staging, plan)


def test_conversion_requires_declared_mtp_sidecar(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(PlanningError, match="requires --mtp-sidecar"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            plan=_plan(qwen36_model_dir),
            output=tmp_path / "candidate",
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )


def test_failed_conversion_does_not_leave_partial_output(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(qwen36_model_dir)

    class FakeModel:
        def named_modules(self):
            return [
                (allocation.module_path, object())
                for allocation in plan.assignments
                if allocation.bits < 16
            ]

    def fake_load(*args, **kwargs):
        return FakeModel(), {}, {}

    def failing_convert(model, *, mlx_path, **kwargs):
        del model, kwargs
        output = Path(mlx_path)
        output.mkdir()
        (output / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("conversion interrupted")

    monkeypatch.setattr(
        converter,
        "_mlx_api",
        lambda: (failing_convert, fake_load),
    )
    output = tmp_path / "candidate"
    with pytest.raises(ArtifactError, match="conversion interrupted"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            plan=plan,
            output=output,
            mtp_sidecar=qwen36_model_dir,
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )
    assert output.exists() is False
    assert list(tmp_path.glob(".candidate.*")) == []
