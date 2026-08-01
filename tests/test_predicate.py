from __future__ import annotations

import numpy as np
import pytest

import axquant.predicate as predicate_module
from axquant.analyzer import architecture_prior_report
from axquant.awq import refine_weight_with_awq
from axquant.errors import PlanningError
from axquant.module_paths import mlx_module_aliases
from axquant.planner import plan_quantization
from axquant.predicate import build_quant_predicate
from axquant.schema import (
    Inventory,
    ModelIdentity,
    PlanRequest,
    ProfileName,
    QuantMethod,
    TensorRole,
    TensorSpec,
)


def _mlp_plan(*, shape: tuple[int, int] = (64, 64), method: QuantMethod = QuantMethod.AFFINE):
    rows, cols = shape
    tensor = TensorSpec(
        name="model.layers.0.mlp.down_proj.weight",
        module_path="model.layers.0.mlp.down_proj",
        shape=(rows, cols),
        dtype="BF16",
        parameters=rows * cols,
        role=TensorRole.MLP,
        quantizable=True,
        file="model.safetensors",
        current_precision="bf16",
    )
    inventory = Inventory(
        model=ModelIdentity(model_id="org/model"),
        tensors=[tensor],
        total_parameters=rows * cols,
        quantizable_parameters=rows * cols,
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
    )
    plan = plan_quantization(
        architecture_prior_report(inventory, profile=ProfileName.GENERAL),
        PlanRequest(profile=ProfileName.GENERAL, target_bpw=4.5, allow_unmeasured=True),
    )
    plan.assignments[0].method = method
    if method not in plan.hardware.supported_methods:
        plan.hardware = plan.hardware.model_copy(
            update={
                "supported_methods": (
                    *plan.hardware.supported_methods,
                    method,
                )
            }
        )
    return plan


def test_predicate_maps_plan_to_mlx_quantization_config() -> None:
    tensor = TensorSpec(
        name="model.layers.0.mlp.down_proj.weight",
        module_path="model.layers.0.mlp.down_proj",
        shape=(1000, 1),
        dtype="BF16",
        parameters=1000,
        role=TensorRole.MLP,
        quantizable=True,
        file="model.safetensors",
        current_precision="bf16",
    )
    inventory = Inventory(
        model=ModelIdentity(model_id="org/model"),
        tensors=[tensor],
        total_parameters=1000,
        quantizable_parameters=1000,
        mtp_present=False,
        quantized_source=False,
        source_files=["model.safetensors"],
        config_sha256="a" * 64,
    )
    report = architecture_prior_report(inventory, profile=ProfileName.GENERAL)
    plan = plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.GENERAL,
            target_bpw=4.5,
            allow_unmeasured=True,
        ),
    )
    predicate = build_quant_predicate(plan)
    config = predicate("layers.0.mlp.down_proj", object())
    assert config == {"group_size": 64, "bits": 4, "mode": "affine"}
    assert predicate.unmatched_quantized_modules() == set()


def test_qwen_checkpoint_paths_map_to_mlx_lm_module_paths() -> None:
    assert "language_model.model.layers.0.mlp.down_proj" in mlx_module_aliases(
        "model.language_model.layers.0.mlp.down_proj"
    )
    assert "language_model.lm_head" in mlx_module_aliases("lm_head")


def test_dwq_refinement_executes_before_affine_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _mlp_plan(method=QuantMethod.DWQ)
    monkeypatch.setattr(
        predicate_module,
        "_apply_dwq_clip",
        lambda module: {"sample_count": 64, "clip_lower": -1.0, "clip_upper": 1.0},
    )
    predicate = build_quant_predicate(plan)
    config = predicate("layers.0.mlp.down_proj", object())
    assert config == {"group_size": 64, "bits": 4, "mode": "affine"}
    assert predicate.dwq_metadata[plan.assignments[0].module_path]["sample_count"] == 64


def test_awq_plan_is_admitted_by_predicate_allowlist() -> None:
    plan = _mlp_plan(method=QuantMethod.AWQ)
    # Preflight / coverage path must not reject solely because the method is AWQ.
    predicate = build_quant_predicate(plan, execute_refinement=False)
    config = predicate("layers.0.mlp.down_proj", object())
    assert config == {"group_size": 64, "bits": 4, "mode": "affine"}
    assert predicate.unmatched_quantized_modules() == set()
    assert predicate.awq_metadata == {}


def test_awq_refinement_requires_calibration_activations() -> None:
    plan = _mlp_plan(method=QuantMethod.AWQ)
    with pytest.raises(PlanningError, match="AWQ conversion requires calibration activations"):
        build_quant_predicate(plan, execute_refinement=True)
    with pytest.raises(PlanningError, match="cannot execute methods"):
        plan.assignments[0].method = QuantMethod.GPTQ
        plan.hardware = plan.hardware.model_copy(
            update={
                "supported_methods": (
                    *plan.hardware.supported_methods,
                    QuantMethod.GPTQ,
                )
            }
        )
        build_quant_predicate(plan, execute_refinement=False)


def test_awq_refinement_executes_before_affine_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _mlp_plan(method=QuantMethod.AWQ)
    captured: dict[str, object] = {}

    def _fake_awq(
        module: object,
        *,
        activations: object,
        bits: int,
        group_size: int,
        alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    ) -> dict[str, float | int | list[float]]:
        del alpha_grid
        captured["module"] = module
        captured["activations"] = activations
        captured["bits"] = bits
        captured["group_size"] = group_size
        return {
            "awq_alpha": 0.5,
            "awq_channel_scales": [1.0] * 64,
            "activation_reconstruction_mse": 0.01,
            "calibration_rows": 32,
            "bits": bits,
            "group_size": group_size,
        }

    monkeypatch.setattr(predicate_module, "_apply_awq_scale", _fake_awq)
    activations = np.random.default_rng(0).standard_normal((32, 64), dtype=np.float32)
    predicate = build_quant_predicate(
        plan,
        awq_activations={plan.assignments[0].module_path: activations},
    )
    config = predicate("layers.0.mlp.down_proj", object())
    assert config == {"group_size": 64, "bits": 4, "mode": "affine"}
    assert captured["bits"] == 4
    assert captured["group_size"] == 64
    assert np.allclose(captured["activations"], activations)
    meta = predicate.awq_metadata[plan.assignments[0].module_path]
    assert meta["awq_alpha"] == 0.5
    assert len(meta["awq_channel_scales"]) == 64


def test_portable_awq_refinement_matches_plugin_contract() -> None:
    rng = np.random.default_rng(7)
    weight = rng.standard_normal((16, 64), dtype=np.float32)
    activations = rng.standard_normal((48, 64), dtype=np.float32)
    refined, metadata = refine_weight_with_awq(
        weight,
        activations,
        bits=4,
        group_size=64,
    )
    assert refined.shape == weight.shape
    assert metadata["bits"] == 4
    assert metadata["group_size"] == 64
    assert metadata["awq_alpha"] in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert len(metadata["awq_channel_scales"]) == 64
    # AWQ refinement stays near the source matrix after unscaled reconstruction.
    assert float(np.mean((weight - refined) ** 2)) < float(np.mean(weight**2))


def test_fused_expert_group_requires_uniform_precision() -> None:
    """Per-expert allocations fuse into one MLX switch module (AXQ MoE v1)."""
    from axquant.module_paths import fused_expert_module

    assert (
        fused_expert_module("model.language_model.layers.3.mlp.experts.17.gate_proj")
        == "model.language_model.layers.3.mlp.switch_mlp.gate_proj"
    )
    assert fused_expert_module("model.language_model.layers.3.mlp.gate_proj") is None

    plan = _mlp_plan()
    template = plan.assignments[0]
    members = []
    for index in (0, 1):
        members.append(
            template.model_copy(
                update={
                    "tensor": (
                        f"model.language_model.layers.0.mlp.experts.{index}.gate_proj.weight"
                    ),
                    "module_path": (f"model.language_model.layers.0.mlp.experts.{index}.gate_proj"),
                    "role": TensorRole.EXPERT,
                    "bits": 4,
                    "method": QuantMethod.AFFINE,
                    "group_size": 64,
                }
            )
        )
    uniform = plan.model_copy(update={"assignments": [*plan.assignments, *members]})
    predicate = build_quant_predicate(uniform, execute_refinement=False)
    # Visiting the fused MLX module marks every member expert as covered.
    result = predicate("language_model.model.layers.0.mlp.switch_mlp.gate_proj", object())
    assert isinstance(result, dict) and result["bits"] == 4
    assert not {member.module_path for member in members} - predicate.matched

    mixed_members = [
        members[0],
        members[1].model_copy(update={"bits": 8}),
    ]
    mixed = plan.model_copy(update={"assignments": [*plan.assignments, *mixed_members]})
    with pytest.raises(PlanningError, match="mixes precisions"):
        build_quant_predicate(mixed, execute_refinement=False)
