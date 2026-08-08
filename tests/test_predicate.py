from __future__ import annotations

import numpy as np
import pytest

import axquant.predicate as predicate_module
from axquant.analyzer import architecture_prior_report
from axquant.awq import refine_weight_with_awq
from axquant.errors import PlanningError
from axquant.module_paths import (
    fused_expert_tensor_target,
    mlx_module_aliases,
    mlx_tensor_aliases,
    mlx_tensor_binding_groups,
)
from axquant.planner import plan_quantization
from axquant.predicate import build_quant_predicate
from axquant.schema import (
    Allocation,
    Inventory,
    ModelIdentity,
    PlanRequest,
    PrecisionShare,
    ProfileName,
    QuantizationPlan,
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


def test_qwen_checkpoint_tensor_paths_map_to_mlx_lm_output_paths() -> None:
    assert "language_model.model.layers.0.linear_attn.A_log" in mlx_tensor_aliases(
        "model.language_model.layers.0.linear_attn.A_log"
    )
    assert "language_model.lm_head.weight" in mlx_tensor_aliases("lm_head.weight")
    assert mlx_tensor_binding_groups("model.language_model.layers.0.mlp.experts.gate_up_proj") == (
        (
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.switch_mlp.gate_proj.weight",
        ),
        (
            "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight",
            "model.language_model.layers.0.mlp.switch_mlp.up_proj.weight",
        ),
    )


def test_deepseek_v4_hc_tensor_aliases_compose_model_prefix_and_rename() -> None:
    """deepseek_v4.sanitize renames layers.* + hc_* into model.layers.*.*_hc.*."""

    assert "model.layers.0.attn_hc.base" in mlx_tensor_aliases("layers.0.hc_attn_base")
    assert "model.layers.0.attn_hc.fn" in mlx_tensor_aliases("layers.0.hc_attn_fn")
    assert "model.layers.0.attn_hc.scale" in mlx_tensor_aliases("layers.0.hc_attn_scale")
    assert "model.layers.0.ffn_hc.base" in mlx_tensor_aliases("layers.0.hc_ffn_base")
    assert "model.layers.0.ffn_hc.fn" in mlx_tensor_aliases("layers.0.hc_ffn_fn")
    assert "model.layers.0.ffn_hc.scale" in mlx_tensor_aliases("layers.0.hc_ffn_scale")
    assert "model.hc_head.scale" in mlx_tensor_aliases("hc_head_scale")
    assert "model.hc_head.base" in mlx_tensor_aliases("hc_head_base")
    assert "model.hc_head.fn" in mlx_tensor_aliases("hc_head_fn")
    assert "model.layers.0.ffn.shared_experts.gate_proj.weight" in mlx_tensor_aliases(
        "layers.0.ffn.shared_experts.w1.weight"
    )
    assert "model.layers.3.ffn.gate.e_score_correction_bias" in mlx_tensor_aliases(
        "layers.3.ffn.gate.bias"
    )
    assert "model.norm.weight" in mlx_tensor_aliases("norm.weight")


def test_deepseek_mtp_experts_are_not_fused_for_sidecar_binding() -> None:
    """MTP is byte-copied unfused; main-layer experts still fuse to switch_mlp."""

    assert fused_expert_tensor_target("mtp.0.ffn.experts.0.w1.weight") is None
    assert fused_expert_tensor_target("model.mtp.0.ffn.experts.1.w2.weight") is None
    assert "mtp.0.ffn.experts.0.w1.weight" in mlx_tensor_aliases("mtp.0.ffn.experts.0.w1.weight")
    fused = fused_expert_tensor_target("layers.3.ffn.experts.0.w1.weight")
    assert fused is not None
    assert fused[0].endswith("switch_mlp.gate_proj.weight")


def test_indexed_expert_tensor_paths_map_to_exact_fused_output() -> None:
    qwen = "model.language_model.layers.3.mlp.experts.17.gate_proj.weight"
    nemotron = "backbone.layers.8.mixer.experts.20.up_proj.weight"

    assert fused_expert_tensor_target(qwen) == (
        "model.language_model.layers.3.mlp.switch_mlp.gate_proj.weight",
        17,
    )
    assert "language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight" in mlx_tensor_aliases(
        qwen
    )
    assert fused_expert_tensor_target(nemotron) == (
        "backbone.layers.8.mixer.switch_mlp.fc1.weight",
        20,
    )
    assert mlx_tensor_binding_groups(nemotron) == (
        ("backbone.layers.8.mixer.switch_mlp.fc1.weight",),
    )
    assert fused_expert_tensor_target(nemotron.removesuffix(".weight") + ".bias") is None


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
    assert predicate.method_metadata == {}


def test_gptq_plan_is_admitted_by_predicate_allowlist() -> None:
    plan = _mlp_plan(method=QuantMethod.GPTQ)
    # Preflight / coverage path must not reject solely because the method is GPTQ.
    predicate = build_quant_predicate(plan, execute_refinement=False)
    config = predicate("layers.0.mlp.down_proj", object())
    assert config == {"group_size": 64, "bits": 4, "mode": "affine"}
    assert predicate.unmatched_quantized_modules() == set()
    assert predicate.method_metadata == {}


def test_awq_refinement_requires_calibration_activations() -> None:
    plan = _mlp_plan(method=QuantMethod.AWQ)
    with pytest.raises(PlanningError, match="requires calibration activations"):
        build_quant_predicate(plan, execute_refinement=True)


def test_gptq_refinement_requires_calibration_activations() -> None:
    plan = _mlp_plan(method=QuantMethod.GPTQ)
    with pytest.raises(PlanningError, match="requires calibration activations"):
        build_quant_predicate(plan, execute_refinement=True)


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
        calibration_activations={plan.assignments[0].module_path: activations},
    )
    config = predicate("layers.0.mlp.down_proj", object())
    assert config == {"group_size": 64, "bits": 4, "mode": "affine"}
    assert captured["bits"] == 4
    assert captured["group_size"] == 64
    assert np.allclose(captured["activations"], activations)
    meta = predicate.method_metadata[plan.assignments[0].module_path]
    assert meta["awq_alpha"] == 0.5
    assert len(meta["awq_channel_scales"]) == 64


def test_gptq_refinement_executes_before_affine_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _mlp_plan(method=QuantMethod.GPTQ)
    captured: dict[str, object] = {}

    def _fake_gptq(
        module: object,
        *,
        activations: object,
        bits: int,
        group_size: int,
        damping: float = 0.01,
        act_order: bool = False,
    ) -> dict[str, float | int]:
        del damping
        captured["module"] = module
        captured["activations"] = activations
        captured["bits"] = bits
        captured["group_size"] = group_size
        captured["act_order"] = act_order
        return {
            "gptq_damping": 0.01,
            "calibration_rows": 32,
            "bits": bits,
            "group_size": group_size,
            "mean_quant_error": 0.001,
        }

    monkeypatch.setattr(predicate_module, "_apply_gptq_refine", _fake_gptq)
    activations = np.random.default_rng(0).standard_normal((32, 64), dtype=np.float32)
    predicate = build_quant_predicate(
        plan,
        calibration_activations={plan.assignments[0].module_path: activations},
    )
    config = predicate("layers.0.mlp.down_proj", object())
    assert config == {"group_size": 64, "bits": 4, "mode": "affine"}
    assert captured["bits"] == 4
    assert captured["group_size"] == 64
    assert np.allclose(captured["activations"], activations)
    meta = predicate.method_metadata[plan.assignments[0].module_path]
    assert meta["gptq_damping"] == 0.01
    assert meta["calibration_rows"] == 32


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

    def with_assignments(assignments: list[Allocation]) -> QuantizationPlan:
        parameters_by_label: dict[str, int] = {}
        total = 0
        for assignment in assignments:
            parameters = assignment.parameters
            label = "bf16" if assignment.bits == 16 else f"{assignment.bits}bit"
            parameters_by_label[label] = parameters_by_label.get(label, 0) + parameters
            total += parameters
        distribution = {
            label: PrecisionShare(parameters=parameters, fraction=parameters / total)
            for label, parameters in parameters_by_label.items()
        }
        return plan.model_copy(
            update={
                "assignments": assignments,
                "weight_distribution": distribution,
            }
        )

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
    uniform = with_assignments([*plan.assignments, *members])
    predicate = build_quant_predicate(uniform, execute_refinement=False)
    # Visiting the fused MLX module marks every member expert as covered.
    result = predicate("language_model.model.layers.0.mlp.switch_mlp.gate_proj", object())
    assert isinstance(result, dict) and result["bits"] == 4
    assert not {member.module_path for member in members} - predicate.matched

    mixed_members = [
        members[0],
        members[1].model_copy(update={"bits": 8}),
    ]
    mixed = with_assignments([*plan.assignments, *mixed_members])
    with pytest.raises(PlanningError, match="mixes precisions"):
        build_quant_predicate(mixed, execute_refinement=False)

    gptq_members = [member.model_copy(update={"method": QuantMethod.GPTQ}) for member in members]
    gptq_fused = with_assignments([*plan.assignments, *gptq_members])
    gptq_fused.hardware = gptq_fused.hardware.model_copy(
        update={
            "supported_methods": (
                *gptq_fused.hardware.supported_methods,
                QuantMethod.GPTQ,
            )
        }
    )
    with pytest.raises(PlanningError, match="requires the affine method"):
        build_quant_predicate(gptq_fused, execute_refinement=False)


def test_packed_expert_requires_every_split_runtime_module() -> None:
    """One packed gate/up tensor is not covered until both MLX modules are visited."""
    plan = _mlp_plan()
    packed = plan.assignments[0].model_copy(
        update={
            "tensor": "model.layers.0.mlp.experts.gate_up_proj.weight",
            "module_path": "model.layers.0.mlp.experts.gate_up_proj",
            "role": TensorRole.EXPERT,
            "bits": 4,
            "method": QuantMethod.AFFINE,
            "group_size": 64,
        }
    )
    packed_plan = plan.model_copy(update={"assignments": [packed]})
    predicate = build_quant_predicate(packed_plan, execute_refinement=False)

    gate = predicate("model.layers.0.mlp.switch_mlp.gate_proj", object())
    assert isinstance(gate, dict)
    assert predicate.unmatched_quantized_modules() == {packed.module_path}

    up = predicate("model.layers.0.mlp.switch_mlp.up_proj", object())
    assert isinstance(up, dict)
    assert predicate.unmatched_quantized_modules() == set()


def test_packed_expert_rejects_non_affine_refinement() -> None:
    plan = _mlp_plan(method=QuantMethod.GPTQ)
    packed = plan.assignments[0].model_copy(
        update={
            "tensor": "model.layers.0.mlp.experts.gate_up_proj.weight",
            "module_path": "model.layers.0.mlp.experts.gate_up_proj",
            "role": TensorRole.EXPERT,
        }
    )
    packed_plan = plan.model_copy(update={"assignments": [packed]})

    with pytest.raises(PlanningError, match=r"packed expert tensor.*requires the affine method"):
        build_quant_predicate(packed_plan, execute_refinement=False)


def test_weight_suffixed_module_paths_keep_packed_and_fused_tracking() -> None:
    """Lookup keys are normalized in __init__; __call__ must normalize the same way."""
    plan = _mlp_plan()
    packed = plan.assignments[0].model_copy(
        update={
            "tensor": "model.layers.0.mlp.experts.gate_up_proj.weight",
            "module_path": "model.layers.0.mlp.experts.gate_up_proj.weight",
            "role": TensorRole.EXPERT,
            "bits": 4,
            "method": QuantMethod.AFFINE,
            "group_size": 64,
        }
    )
    packed_plan = plan.model_copy(update={"assignments": [packed]})
    predicate = build_quant_predicate(packed_plan, execute_refinement=False)

    gate = predicate("model.layers.0.mlp.switch_mlp.gate_proj", object())
    assert isinstance(gate, dict)
    # One split runtime module visited is not complete coverage.
    assert predicate.unmatched_quantized_modules() == {packed.module_path}

    up = predicate("model.layers.0.mlp.switch_mlp.up_proj", object())
    assert isinstance(up, dict)
    assert predicate.unmatched_quantized_modules() == set()

    members = [
        plan.assignments[0].model_copy(
            update={
                "tensor": f"model.layers.0.mlp.experts.{index}.gate_proj.weight",
                "module_path": f"model.layers.0.mlp.experts.{index}.gate_proj.weight",
                "role": TensorRole.EXPERT,
                "bits": 4,
                "method": QuantMethod.AFFINE,
                "group_size": 64,
            }
        )
        for index in (0, 1)
    ]
    fused_plan = plan.model_copy(update={"assignments": members})
    fused_predicate = build_quant_predicate(fused_plan, execute_refinement=False)
    result = fused_predicate("model.layers.0.mlp.switch_mlp.gate_proj", object())
    assert isinstance(result, dict)
    assert fused_predicate.unmatched_quantized_modules() == set()


def test_unwrapped_and_nemotron_packed_expert_tensors_bind_their_switch_modules() -> None:
    # Converted-output binding must accept every packed form the predicate,
    # planner, and manual recipes accept — including checkpoints without the
    # Qwen language-model wrapper and Nemotron mixer packs — or a valid MoE
    # conversion aborts at verification after the full conversion has run.
    groups = mlx_tensor_binding_groups("model.layers.0.mlp.experts.gate_up_proj.weight")
    flattened = {alias for group in groups for alias in group}
    assert "model.layers.0.mlp.switch_mlp.gate_proj.weight" in flattened
    assert "model.layers.0.mlp.switch_mlp.up_proj.weight" in flattened
    assert mlx_tensor_binding_groups("backbone.layers.3.mixer.experts.up_proj.weight") == (
        ("backbone.layers.3.mixer.switch_mlp.fc1.weight",),
    )
    scales_aliases = mlx_tensor_aliases("model.layers.0.mlp.experts.down_proj.scales")
    assert "model.layers.0.mlp.switch_mlp.down_proj.scales" in scales_aliases
