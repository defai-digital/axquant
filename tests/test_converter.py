from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from _capture_helpers import load_test_activation_capture
from safetensors import safe_open
from safetensors.numpy import save_file

import axquant.converter as converter
import axquant.multimodal_backend as multimodal_backend
from axquant.analyzer import architecture_prior_report
from axquant.capture_binding import (
    CAPTURE_MANIFEST_SHA256_KEY,
    LoadedActivationCapture,
    activation_capture_metadata,
)
from axquant.errors import ArtifactError, BackendUnavailableError, PlanningError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.predicate import build_quant_predicate
from axquant.schema import (
    ActivationCaptureManifest,
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


def test_multimodal_backend_dispatch_and_vlm_public_convert_contract(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(qwen36_model_dir)
    assert multimodal_backend.conversion_backend(plan) == "mlx-lm"
    asr_plan = plan.model_copy(
        update={
            "architecture_profile": plan.architecture_profile.model_copy(
                update={"adapter_id": "qwen3-asr-v1"}
            )
        }
    )
    assert multimodal_backend.conversion_backend(asr_plan) == "mlx-audio"
    vlm_plan = plan.model_copy(
        update={
            "architecture_profile": plan.architecture_profile.model_copy(
                update={"adapter_id": "qwen3-vl-v1"}
            )
        }
    )
    assert multimodal_backend.conversion_backend(vlm_plan) == "mlx-vlm"
    predicate = build_quant_predicate(vlm_plan, execute_refinement=False)
    observed: dict[str, object] = {}

    def fake_convert(source: str, **kwargs) -> None:
        observed["source"] = source
        observed.update(kwargs)

    monkeypatch.setattr(
        multimodal_backend,
        "_import",
        lambda module, *, extra: SimpleNamespace(convert=fake_convert),
    )
    destination = tmp_path / "converted"
    multimodal_backend.convert_multimodal(
        qwen36_model_dir,
        destination,
        vlm_plan,
        predicate,
        4,
    )

    assert observed["source"] == str(qwen36_model_dir)
    assert observed["mlx_path"] == destination
    assert observed["quantize"] is True
    assert observed["q_group_size"] == vlm_plan.group_size
    assert observed["q_bits"] == 4
    assert observed["q_mode"] == "affine"
    assert observed["quant_method"] == "rtn"
    assert observed["quant_predicate"] is predicate


def test_audio_backend_uses_public_stt_model_and_mlx_quantization_contract(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(qwen36_model_dir)
    plan.architecture_profile = plan.architecture_profile.model_copy(
        update={"adapter_id": "qwen3-asr-v1"}
    )
    quantized = [allocation.module_path for allocation in plan.assignments if allocation.bits < 16]
    observed: dict[str, object] = {"loaded_dtypes": []}

    class FakeModelConfig:
        def __init__(self) -> None:
            self.model_path: Path | None = None

        @classmethod
        def from_dict(cls, config: dict[str, object]) -> FakeModelConfig:
            observed["model_config"] = config
            return cls()

    class FakeModel:
        def __init__(self, config: FakeModelConfig) -> None:
            observed["model_path"] = config.model_path

        def named_modules(self):
            return [(path, object()) for path in quantized]

        def sanitize(self, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            observed["sanitized"] = True
            return weights

        def load_weights(self, weights: list[tuple[str, np.ndarray]]) -> None:
            loaded_dtypes = observed["loaded_dtypes"]
            assert isinstance(loaded_dtypes, list)
            loaded_dtypes.append([value.dtype for _, value in weights])

        def parameters(self) -> dict[str, np.ndarray]:
            return {"decoder.weight": np.ones((2, 2), dtype=np.float32)}

    fake_domain = SimpleNamespace(value="stt")
    fake_audio = SimpleNamespace(
        Domain=SimpleNamespace(STT=fake_domain),
        MODEL_CONVERSION_DTYPES={"float16"},
        load_config=lambda source: {"torch_dtype": "float16"},
        get_model_type=lambda config, source, domain: "qwen3_asr",
        get_model_class=lambda model_type, domain: SimpleNamespace(
            ModelConfig=FakeModelConfig,
            Model=FakeModel,
        ),
        load_weights=lambda source: {"decoder.weight": np.ones((2, 2), dtype=np.float32)},
        copy_model_files=lambda source, destination: (destination / "tokenizer.json").write_text(
            "{}", encoding="utf-8"
        ),
    )
    fake_mlx = SimpleNamespace(float16=np.float16)
    fake_mlx_utils = SimpleNamespace(
        tree_flatten=lambda parameters: list(parameters.items()),
    )

    def quantize_model(
        model,
        config,
        group_size,
        bits,
        *,
        mode,
        quant_predicate,
    ):
        observed.update(
            {
                "group_size": group_size,
                "bits": bits,
                "mode": mode,
            }
        )
        for path in quantized:
            quant_predicate(path, object())
        return model, dict(config)

    fake_mlx_lm_utils = SimpleNamespace(
        quantize_model=quantize_model,
        save_model=lambda destination, model, *, donate_model: (
            destination / "model.safetensors"
        ).write_bytes(b"fake"),
        save_config=lambda config, *, config_path: config_path.write_text(
            json.dumps(config), encoding="utf-8"
        ),
    )

    def fake_import(module: str, *, extra: str):
        return {
            "mlx_audio.convert": fake_audio,
            "mlx.core": fake_mlx,
            "mlx.utils": fake_mlx_utils,
            "mlx_lm.utils": fake_mlx_lm_utils,
        }[module]

    monkeypatch.setattr(multimodal_backend, "_import", fake_import)
    preflight = build_quant_predicate(plan, execute_refinement=False)
    multimodal_backend.preflight_multimodal(qwen36_model_dir, plan, preflight)
    assert preflight.unmatched_quantized_modules() == set()

    predicate = build_quant_predicate(plan, execute_refinement=False)
    destination = tmp_path / "converted-audio"
    multimodal_backend.convert_multimodal(
        qwen36_model_dir,
        destination,
        plan,
        predicate,
        6,
    )

    assert observed["model_path"] == qwen36_model_dir
    assert observed["sanitized"] is True
    assert observed["group_size"] == plan.group_size
    assert observed["bits"] == 6
    assert observed["mode"] == "affine"
    assert observed["loaded_dtypes"][-1] == [np.dtype(np.float16)]
    assert predicate.unmatched_quantized_modules() == set()
    assert (
        json.loads((destination / "config.json").read_text(encoding="utf-8"))["model_type"]
        == "qwen3_asr"
    )
    assert (destination / "tokenizer.json").is_file()
    assert (destination / "model.safetensors").is_file()


def test_multimodal_preflight_visits_all_vlm_plan_modules_and_fails_closed(
    qwen36_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(qwen36_model_dir)
    plan.architecture_profile = plan.architecture_profile.model_copy(
        update={"adapter_id": "qwen3-vl-v1"}
    )
    quantized = [allocation.module_path for allocation in plan.assignments if allocation.bits < 16]
    assert quantized

    class FakeModel:
        def __init__(self, paths: list[str]) -> None:
            self._paths = paths

        def named_modules(self):
            return [(path, object()) for path in self._paths]

    loaded_paths = list(quantized)
    monkeypatch.setattr(
        multimodal_backend,
        "_import",
        lambda module, *, extra: SimpleNamespace(
            load_model=lambda source, *, lazy: FakeModel(loaded_paths)
        ),
    )
    predicate = build_quant_predicate(plan, execute_refinement=False)
    multimodal_backend.preflight_multimodal(qwen36_model_dir, plan, predicate)
    assert predicate.unmatched_quantized_modules() == set()

    loaded_paths.pop()
    predicate = build_quant_predicate(plan, execute_refinement=False)
    with pytest.raises(PlanningError, match="plan modules do not match the MLX-VLM model"):
        multimodal_backend.preflight_multimodal(qwen36_model_dir, plan, predicate)


def test_multimodal_backend_rejects_wrong_calls_and_wraps_backend_failures(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(qwen36_model_dir)
    predicate = build_quant_predicate(plan, execute_refinement=False)
    with pytest.raises(PlanningError, match="multimodal preflight called for the MLX-LM"):
        multimodal_backend.preflight_multimodal(qwen36_model_dir, plan, predicate)
    with pytest.raises(PlanningError, match="multimodal conversion called for the MLX-LM"):
        multimodal_backend.convert_multimodal(
            qwen36_model_dir,
            tmp_path / "wrong-backend",
            plan,
            predicate,
            4,
        )

    vlm_plan = plan.model_copy(
        update={
            "architecture_profile": plan.architecture_profile.model_copy(
                update={"adapter_id": "qwen3-vl-v1"}
            )
        }
    )
    predicate = build_quant_predicate(vlm_plan, execute_refinement=False)
    monkeypatch.setattr(
        multimodal_backend,
        "_import",
        lambda module, *, extra: SimpleNamespace(
            convert=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("conversion boom"))
        ),
    )
    with pytest.raises(ArtifactError, match="mlx-vlm conversion failed: conversion boom"):
        multimodal_backend.convert_multimodal(
            qwen36_model_dir,
            tmp_path / "failed-vlm",
            vlm_plan,
            predicate,
            4,
        )


def test_multimodal_backend_reports_missing_dependency_and_bad_audio_resolution(
    qwen36_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(module: str):
        raise ModuleNotFoundError(module)

    monkeypatch.setattr(multimodal_backend.importlib, "import_module", missing_module)
    with pytest.raises(BackendUnavailableError, match="install mlx-vlm"):
        multimodal_backend._import("mlx_vlm.utils", extra="mlx-vlm")

    domain = SimpleNamespace(value="stt")
    fake_audio = SimpleNamespace(
        load_config=lambda source: {},
        Domain=SimpleNamespace(STT=domain),
        get_model_type=lambda config, source, selected_domain: "qwen3",
    )
    monkeypatch.setattr(
        multimodal_backend,
        "_import",
        lambda module, *, extra: fake_audio,
    )
    with pytest.raises(
        ArtifactError,
        match="requires MLX-Audio to resolve domain=stt and model_type=qwen3_asr",
    ):
        multimodal_backend._audio_model(qwen36_model_dir)


def _write_fake_converted_checkpoint(
    source_dir: Path,
    output: Path,
    plan: QuantizationPlan,
    *,
    pack_weights: bool = True,
) -> None:
    """Write a tiny MLX-shaped result for converter contract tests."""

    output.mkdir()
    converted_config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    converted_config.pop("vision_config", None)
    allocations = {allocation.tensor: allocation for allocation in plan.assignments}
    quantization: dict[str, dict[str, int | str]] = {}
    tensors: dict[str, np.ndarray] = {}
    with safe_open(source_dir / "model.safetensors", framework="numpy") as source:
        for name in list(source.keys()):
            if name.startswith("visual."):
                continue
            value = source.get_tensor(name)
            allocation = allocations[name]
            if not pack_weights or allocation.bits >= 16:
                tensors[name] = value
                continue
            assert allocation.group_size is not None
            assert value.shape[-1] * allocation.bits % 32 == 0
            packed_shape = (
                *value.shape[:-1],
                value.shape[-1] * allocation.bits // 32,
            )
            metadata_shape = (
                *value.shape[:-1],
                max(1, (value.shape[-1] + allocation.group_size - 1) // allocation.group_size),
            )
            tensors[name] = np.zeros(packed_shape, dtype=np.uint32)
            tensors[f"{allocation.module_path}.scales"] = np.ones(
                metadata_shape,
                dtype=np.float32,
            )
            tensors[f"{allocation.module_path}.biases"] = np.zeros(
                metadata_shape,
                dtype=np.float32,
            )
            quantization[allocation.module_path] = {
                "bits": allocation.bits,
                "group_size": allocation.group_size,
                "mode": "affine",
            }
    if quantization:
        converted_config["quantization"] = quantization
    (output / "config.json").write_text(json.dumps(converted_config), encoding="utf-8")
    save_file(tensors, output / "model.safetensors")


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


def _bound_capture(
    plan: QuantizationPlan,
    activations: dict[str, np.ndarray],
    source_dir: Path,
) -> LoadedActivationCapture:
    manifest = ActivationCaptureManifest(
        model=plan.source_model.model_id,
        revision=plan.source_model.revision,
        tokenized_cache_manifest_sha256="c" * 64,
        cache_key_sha256="d" * 64,
        calibration_dataset_id=(
            plan.calibration.dataset_id if plan.calibration is not None else "development-cache"
        ),
        max_rows=max(rows.shape[0] for rows in activations.values()),
    )
    capture = load_test_activation_capture(
        source_dir / "capture",
        manifest=manifest,
        activations=activations,
    )
    if plan.calibration is not None:
        plan.calibration.metadata.update(activation_capture_metadata(capture))
    return capture


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
    activation_arrays = {
        allocation.module_path: np.random.default_rng(i).standard_normal(
            (16, 64),
            dtype=np.float32,
        )
        for i, allocation in enumerate(awq_targets)
    }
    calibration_activations = _bound_capture(plan, activation_arrays, tmp_path)

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
        _write_fake_converted_checkpoint(qwen36_model_dir, Path(mlx_path), plan)

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
        calibration_activations=calibration_activations,
        allow_unmeasured=True,
        ax_engine_manifest="skip",
    )
    assert (output / "axquant_quantizer_execution.json").is_file()
    assert (output / "activation_capture_manifest.json").is_file()
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

    with pytest.raises(PlanningError, match="unbound activation mapping"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            plan=plan,
            output=tmp_path / "awq-unbound-activations",
            mtp_sidecar=qwen36_model_dir,
            calibration_activations=activation_arrays,
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )

    with pytest.raises(PlanningError, match="requires calibration activations"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            plan=plan,
            output=tmp_path / "awq-missing-activations",
            mtp_sidecar=qwen36_model_dir,
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )


def test_gptq_convert_preflight_and_predicate_are_executable(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real convert preflight/predicate entry points with a GPTQ plan."""
    plan = _plan(qwen36_model_dir)
    gptq_targets = [
        allocation
        for allocation in plan.assignments
        if allocation.bits < 16 and allocation.role == TensorRole.MLP
    ]
    assert gptq_targets
    for allocation in gptq_targets:
        allocation.method = QuantMethod.GPTQ
    plan.hardware = plan.hardware.model_copy(
        update={
            "supported_methods": (
                *plan.hardware.supported_methods,
                QuantMethod.GPTQ,
            )
        }
    )

    # Preflight path must admit GPTQ without activations (refinement disabled).
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

    # Match the plan group size (64) so portable GPTQ refinement can run.
    activation_arrays = {
        allocation.module_path: np.random.default_rng(i).standard_normal(
            (16, 64),
            dtype=np.float32,
        )
        for i, allocation in enumerate(gptq_targets)
    }
    calibration_activations = _bound_capture(plan, activation_arrays, tmp_path)

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
        _write_fake_converted_checkpoint(qwen36_model_dir, Path(mlx_path), plan)

    # Use the real portable refine path for GPTQ modules (numpy weights; no MLX required).
    import axquant.predicate as predicate_module
    from axquant.gptq import learn_gptq_refined_weight

    def _apply_numpy_gptq(
        module: FakeModule,
        *,
        activations: object,
        bits: int,
        group_size: int,
        damping: float = 0.01,
        act_order: bool = False,
    ) -> dict[str, float | int]:
        refined, metadata = learn_gptq_refined_weight(
            module.weight,
            activations,
            bits=bits,
            group_size=group_size,
            damping=damping,
            act_order=act_order,
        )
        module.weight = refined
        return metadata

    monkeypatch.setattr(predicate_module, "_apply_gptq_refine", _apply_numpy_gptq)
    monkeypatch.setattr(converter, "_mlx_api", lambda: (fake_convert, fake_load))

    output = tmp_path / "gptq-candidate"
    manifest = converter.convert_model(
        model=str(qwen36_model_dir),
        plan=plan,
        output=output,
        mtp_sidecar=qwen36_model_dir,
        calibration_activations=calibration_activations,
        allow_unmeasured=True,
        ax_engine_manifest="skip",
    )
    assert (output / "axquant_quantizer_execution.json").is_file()
    assert (output / "activation_capture_manifest.json").is_file()
    assert any(record.path == "axquant_quantizer_execution.json" for record in manifest.files)
    execution = load_model(output / "axquant_quantizer_execution.json", QuantizerExecutionManifest)
    gptq_records = [record for record in execution.records if record.method == QuantMethod.GPTQ]
    assert gptq_records
    assert all(record.success for record in gptq_records)
    assert all(
        record.note == "GPTQ Hessian error compensation followed by portable affine packing"
        for record in gptq_records
    )
    assert all("gptq_damping" in record.metadata for record in gptq_records)

    with pytest.raises(PlanningError, match="requires calibration activations"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            plan=plan,
            output=tmp_path / "gptq-missing-activations",
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


def test_measured_awq_conversion_binds_exact_activation_capture(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    plan = _measured_plan(qwen36_model_dir, tmp_path / "calibration.json")
    target = next(
        allocation
        for allocation in plan.assignments
        if allocation.bits < 16 and allocation.role == TensorRole.MLP
    )
    target.method = QuantMethod.AWQ
    activations = {
        target.module_path: np.zeros((4, 64), dtype=np.float32),
    }
    capture = _bound_capture(plan, activations, tmp_path)

    assert converter._validated_activation_capture(plan, capture) is capture
    assert plan.calibration is not None
    plan.calibration.metadata[CAPTURE_MANIFEST_SHA256_KEY] = "0" * 64
    with pytest.raises(PlanningError, match="does not match the plan"):
        converter._validated_activation_capture(plan, capture)


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
        _write_fake_converted_checkpoint(qwen36_model_dir, Path(mlx_path), plan)

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


def test_restore_protected_vision_config_mirrors_tie_word_embeddings_into_text_config(
    tmp_path: Path,
) -> None:
    """mlx-lm qwen3_vl_moe requires text_config.tie_word_embeddings."""
    source = tmp_path / "source"
    output = tmp_path / "converted"
    source.mkdir()
    output.mkdir()
    source_config = {
        "model_type": "qwen3_vl_moe",
        "tie_word_embeddings": False,
        "vision_config": {"hidden_size": 1152},
        "text_config": {"hidden_size": 2048, "vocab_size": 100},
        "image_token_id": 1,
    }
    converted_config = {
        "model_type": "qwen3_vl_moe",
        "tie_word_embeddings": False,
        "text_config": {"hidden_size": 2048, "vocab_size": 100},
        # vision_config omitted by convert — restored from source
    }
    (source / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(converted_config), encoding="utf-8")

    converter._restore_protected_vision_config(source, output)

    restored = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert restored["vision_config"] == source_config["vision_config"]
    assert restored["text_config"]["tie_word_embeddings"] is False


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


def test_mtp_sidecar_contract_gains_structured_bits_from_free_text(tmp_path: Path) -> None:
    """AX Engine's structured mtp_sidecar_bits beats its free-text heuristic."""
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
        json.dumps({"mtp_sidecar": "Qwen3.6 MTP head INT8 sidecar"}),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()

    converter._copy_external_mtp_bundle(sidecar, output)

    runtime = json.loads((output / "mtplx_runtime.json").read_text(encoding="utf-8"))
    assert runtime["mtp_sidecar_bits"] == 8
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


def test_resolve_external_mtp_sidecar_file_prefers_canonical_name_deterministically(
    tmp_path: Path,
) -> None:
    # When a sidecar directory ships both recognized filenames, the choice
    # must be deterministic (a fixed preference order), not dependent on
    # Python's per-process string-hash randomization of set/frozenset
    # iteration order.
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    (sidecar / "mtp.safetensors").write_bytes(b"canonical")
    (sidecar / "mtp_head.safetensors").write_bytes(b"alternate")

    resolved = converter._resolve_external_mtp_sidecar_file(sidecar)

    assert resolved == sidecar / "mtp.safetensors"


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
    with pytest.raises(ArtifactError, match="tensor coverage mismatch"):
        converter._verify_converted_weights(staging, plan)


def test_converted_tensor_binding_accepts_only_one_to_one_mlx_wrapper_aliases() -> None:
    expected = {
        "lm_head.weight": object(),
        "model.language_model.layers.0.linear_attn.A_log": object(),
    }
    head = object()
    a_log = object()
    actual = {
        "language_model.lm_head.weight": head,
        "language_model.model.layers.0.linear_attn.A_log": a_log,
    }

    assert converter._bind_converted_tensors(expected, actual) == {
        "lm_head.weight": (head,),
        "model.language_model.layers.0.linear_attn.A_log": (a_log,),
    }

    with pytest.raises(ArtifactError, match="tensor coverage mismatch"):
        converter._bind_converted_tensors(
            expected,
            {**actual, "lm_head.weight": object()},
        )

    with pytest.raises(ArtifactError, match="tensor coverage mismatch"):
        converter._bind_converted_tensors(
            expected,
            {"language_model.lm_head.weight": head},
        )


def test_converted_tensor_binding_proves_qwen_packed_gate_up_split() -> None:
    tensor = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    gate = object()
    up = object()
    actual = {
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": gate,
        "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight": up,
    }

    assert converter._bind_converted_tensors({tensor: object()}, actual) == {tensor: (gate, up)}
    with pytest.raises(ArtifactError, match="tensor coverage mismatch"):
        converter._bind_converted_tensors(
            {tensor: object()},
            {"language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": gate},
        )


def test_converted_tensor_binding_proves_indexed_expert_stack() -> None:
    names = [f"backbone.layers.1.mixer.experts.{index}.up_proj.weight" for index in range(3)]
    fused = object()
    actual = {"backbone.layers.1.mixer.switch_mlp.fc1.weight": fused}

    assert converter._bind_converted_tensors(dict.fromkeys(names, object()), actual) == {
        name: (fused,) for name in names
    }
    with pytest.raises(ArtifactError, match="contiguous source indices"):
        converter._bind_converted_tensors(
            {names[0]: object(), names[2]: object()},
            actual,
        )
    with pytest.raises(ArtifactError, match="tensor coverage mismatch"):
        converter._bind_converted_tensors(
            dict.fromkeys(names, object()),
            {**actual, "backbone.layers.1.mixer.experts.0.up_proj.weight": object()},
        )


def test_expected_converted_shapes_conserve_qwen_packed_gate_up_parameters() -> None:
    gate_up = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    down = "model.language_model.layers.0.mlp.experts.down_proj"

    assert converter._expected_converted_shapes(gate_up, (256, 1024, 2048), 6) == (
        (256, 512, 384),
        (256, 512, 384),
    )
    assert converter._expected_converted_shapes(gate_up, (256, 1024, 2048), 16) == (
        (256, 512, 2048),
        (256, 512, 2048),
    )
    assert converter._expected_converted_shapes(down, (256, 2048, 512), 6) == ((256, 2048, 96),)


def test_expected_converted_shapes_qwen3_vl_moe_sanitize_layout() -> None:
    """mlx-vlm splits gate_up on the last axis then transpose(0,2,1) before pack."""

    gate_up = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    down = "model.language_model.layers.0.mlp.experts.down_proj"
    # Real 30B-A3B shapes: gate_up (E,H,2I)=(128,2048,1536), down (E,I,H)=(128,768,2048).
    assert converter._expected_converted_shapes(
        gate_up, (128, 2048, 1536), 4, model_type="qwen3_vl_moe"
    ) == (
        (128, 768, 256),
        (128, 768, 256),
    )
    assert converter._expected_converted_shapes(
        down, (128, 768, 2048), 4, model_type="qwen3_vl_moe"
    ) == ((128, 2048, 96),)
    assert converter._expected_converted_shapes(
        gate_up, (128, 2048, 1536), 6, model_type="qwen3_vl_moe"
    ) == (
        (128, 768, 384),
        (128, 768, 384),
    )
    assert converter._expected_converted_shapes(
        down, (128, 768, 2048), 16, model_type="qwen3_vl_moe"
    ) == ((128, 2048, 768),)


def test_native_gpt_oss_expert_biases_are_not_bound_as_quantized_weights() -> None:
    assert converter._is_converted_quant_sidecar_name(
        "model.layers.0.mlp.experts.gate_up_proj_bias"
    )
    assert converter._is_converted_quant_sidecar_name("model.layers.0.mlp.experts.down_proj_bias")
    # Router correction bias is a real protected parameter and must retain its
    # stricter one-to-one converted coverage.
    assert not converter._is_converted_quant_sidecar_name(
        "model.layers.0.ffn.gate.e_score_correction_bias"
    )

    blocks = "model.layers.0.mlp.experts.gate_up_proj_blocks"
    assert converter._expected_converted_shapes(blocks, (2, 8, 64), 6) == (
        (2, 4, 12),
        (2, 4, 12),
    )


def test_expected_fused_shapes_conserve_nemotron_expert_parameters() -> None:
    up = "backbone.layers.1.mixer.experts.0.up_proj.weight"
    down = "backbone.layers.1.mixer.experts.0.down_proj.weight"

    assert converter._expected_fused_converted_shapes(
        up,
        ((1856, 2688),) * 128,
        4,
    ) == ((128, 1856, 336),)
    assert converter._expected_fused_converted_shapes(
        down,
        ((2688, 1856),) * 128,
        6,
    ) == ((128, 2688, 348),)
    with pytest.raises(ArtifactError, match="mixes source shapes"):
        converter._expected_fused_converted_shapes(
            up,
            ((1856, 2688), (1856, 1024)),
            4,
        )


def test_qwen_even_expert_count_is_not_deepseek_dual_gate_concat() -> None:
    """Even expert counts must not trigger DeepSeek w1+w3 gate concat."""

    gate = "model.layers.3.mlp.experts.0.gate_proj.weight"
    # 64 Qwen experts (even) stack 1:1 — not 32 experts with doubled out dim.
    assert converter._expected_fused_converted_shapes(
        gate,
        ((8192, 4096),) * 64,
        4,
        multiplicity=1,
    ) == ((64, 8192, 512),)


def test_deepseek_dual_w1_w3_gate_concat_shape() -> None:
    """DeepSeek FusedSwitchGLU concatenates w1+w3 into gate_proj (multiplicity 2)."""

    w1 = "layers.0.ffn.experts.0.w1.weight"
    members = tuple(
        f"layers.0.ffn.experts.{index}.{proj}.weight" for index in range(4) for proj in ("w1", "w3")
    )
    assert converter._fused_member_multiplicity(members) == 2
    assert converter._expected_fused_converted_shapes(
        w1,
        ((2048, 4096),) * 8,
        2,
        multiplicity=2,
    ) == ((4, 4096, 256),)


def test_protected_shape_matching_allows_only_documented_conv1d_sanitize_transforms(
    qwen36_model_dir: Path,
) -> None:
    plan = _plan(qwen36_model_dir)
    tensor = "model.language_model.layers.0.linear_attn.conv1d.weight"

    assert converter._protected_shape_matches(
        plan,
        tensor,
        (8192, 1, 4),
        (8192, 4, 1),
    )
    assert not converter._protected_shape_matches(
        plan,
        "model.language_model.layers.0.linear_attn.A_log",
        (8192, 1, 4),
        (8192, 4, 1),
    )
    assert not converter._protected_shape_matches(
        plan,
        tensor,
        (8192, 1, 4),
        (8192, 2, 2),
    )

    qwen3_next_plan = plan.model_copy(
        update={
            "architecture_profile": plan.architecture_profile.model_copy(
                update={"config_model_type": "qwen3_next"}
            )
        }
    )
    assert converter._protected_shape_matches(
        qwen3_next_plan,
        tensor,
        (8192, 1, 4),
        (8192, 4, 1),
    )

    nemotron_plan = plan.model_copy(
        update={
            "architecture_profile": plan.architecture_profile.model_copy(
                update={"config_model_type": "nemotron_h"}
            )
        }
    )
    assert converter._protected_shape_matches(
        nemotron_plan,
        "backbone.layers.0.mixer.conv1d.weight",
        (6144, 1, 4),
        (6144, 4, 1),
    )
    assert not converter._protected_shape_matches(
        nemotron_plan,
        tensor,
        (8192, 1, 4),
        (8192, 4, 1),
    )


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_moe"])
def test_protected_shape_matching_allows_qwen3_vl_conv3d_sanitize_transform(
    qwen36_model_dir: Path,
    model_type: str,
) -> None:
    plan = _plan(qwen36_model_dir)
    qwen3_vl_plan = plan.model_copy(
        update={
            "architecture_profile": plan.architecture_profile.model_copy(
                update={"config_model_type": model_type}
            )
        }
    )
    tensor = "model.visual.patch_embed.proj.weight"

    assert converter._protected_shape_matches(
        qwen3_vl_plan,
        tensor,
        (1152, 3, 2, 16, 16),
        (1152, 2, 16, 16, 3),
    )
    assert not converter._protected_shape_matches(
        qwen3_vl_plan,
        "model.visual.blocks.0.attn.qkv.weight",
        (1152, 3, 2, 16, 16),
        (1152, 2, 16, 16, 3),
    )
    assert not converter._protected_shape_matches(
        qwen3_vl_plan,
        tensor,
        (1152, 3, 2, 16, 16),
        (1152, 16, 16, 2, 3),
    )


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


def test_conversion_rejects_revision_that_differs_from_plan(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(PlanningError, match="revision does not match"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            revision="different-revision",
            plan=_plan(qwen36_model_dir),
            output=tmp_path / "candidate",
            mtp_sidecar=qwen36_model_dir,
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )


def test_conversion_rejects_local_source_that_differs_from_plan(
    qwen36_model_dir: Path,
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(PlanningError, match="does not match the plan source path"):
        converter.convert_model(
            model=str(tiny_model_dir),
            plan=_plan(qwen36_model_dir),
            output=tmp_path / "candidate",
            mtp_sidecar=qwen36_model_dir,
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )


def test_conversion_rejects_declared_mtp_without_plan_allocations(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-with-missing-mtp"
    source.mkdir()
    for name in ("config.json", "model.safetensors"):
        (source / name).write_bytes((qwen36_model_dir / name).read_bytes())
    plan = _plan(source)
    assert plan.architecture_profile.mtp_declared
    assert not any(allocation.role.is_mtp for allocation in plan.assignments)

    with pytest.raises(PlanningError, match="contains no MTP tensor allocations"):
        converter.convert_model(
            model=str(source),
            plan=plan,
            output=tmp_path / "candidate",
            allow_unmeasured=True,
            ax_engine_manifest="skip",
        )


def test_conversion_rejects_backend_that_does_not_pack_planned_weights(
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
        _write_fake_converted_checkpoint(
            qwen36_model_dir,
            Path(mlx_path),
            plan,
            pack_weights=False,
        )

    monkeypatch.setattr(converter, "_mlx_api", lambda: (fake_convert, fake_load))

    with pytest.raises(ArtifactError, match="packing does not match the plan"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            plan=plan,
            output=tmp_path / "candidate",
            mtp_sidecar=qwen36_model_dir,
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
