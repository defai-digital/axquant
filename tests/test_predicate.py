from __future__ import annotations

import numpy as np
import pytest

import axquant.predicate as predicate_module
from axquant.analyzer import architecture_prior_report
from axquant.awq import refine_weight_with_awq
from axquant.errors import PlanningError
from axquant.module_paths import (
    fused_expert_module,
    fused_expert_tensor_target,
    mlx_module_aliases,
    mlx_tensor_aliases,
    mlx_tensor_binding_groups,
)
from axquant.planner import plan_quantization
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

PlanPredicate = predicate_module.PlanPredicate
build_quant_predicate = predicate_module.build_quant_predicate

MXFP4_PARAMS = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
MXFP8_PARAMS = {"group_size": 32, "bits": 8, "mode": "mxfp8"}


def _mlp_plan(
    *,
    shape: tuple[int, int] = (64, 64),
    method: QuantMethod = QuantMethod.MXFP4,
    bits: int = 4,
    group_size: int = 32,
) -> QuantizationPlan:
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
        PlanRequest(profile=ProfileName.GENERAL, target_bpw=5.0, allow_unmeasured=True),
    )
    allocation = plan.assignments[0].model_copy(
        update={"method": method, "bits": bits, "group_size": group_size}
    )
    hardware = plan.hardware
    if method not in hardware.supported_methods:
        hardware = hardware.model_copy(
            update={"supported_methods": (*hardware.supported_methods, method)}
        )
    return plan.model_copy(update={"assignments": [allocation], "hardware": hardware})


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
            target_bpw=5.0,
            allow_unmeasured=True,
        ),
    )
    predicate = build_quant_predicate(plan)
    # AXQ-047: the default 4-bit rung is MXFP4 (group 32), not affine.
    assert plan.assignments[0].method == QuantMethod.MXFP4
    config = predicate("layers.0.mlp.down_proj", object())
    assert config == MXFP4_PARAMS
    assert predicate.unmatched_quantized_modules() == set()


@pytest.mark.parametrize(
    ("method", "bits", "group_size"),
    [
        (QuantMethod.MXFP4, 4, 32),
        (QuantMethod.AFFINE, 4, 32),
    ],
)
def test_four_bit_allocations_are_stored_as_mxfp4(
    method: QuantMethod, bits: int, group_size: int
) -> None:
    # AXQ-070: a 4-bit slot is MXFP4 storage whether the plan names it affine
    # or mxfp4; there is no affine 4-bit packing and no legacy opt-in.
    plan = _mlp_plan(method=method, bits=bits, group_size=group_size)
    allocation = plan.assignments[0]
    assert predicate_module.allocation_physical_mode(allocation) == "mxfp4"
    assert build_quant_predicate(plan)("layers.0.mlp.down_proj", object()) == MXFP4_PARAMS


def test_eight_bit_allocation_is_stored_as_mxfp8() -> None:
    plan = _mlp_plan(method=QuantMethod.AFFINE, bits=8, group_size=32)
    assert predicate_module.allocation_physical_mode(plan.assignments[0]) == "mxfp8"
    assert build_quant_predicate(plan)("layers.0.mlp.down_proj", object()) == MXFP8_PARAMS


def test_affine_plan_under_mxfp4_q_mode_stays_mxfp4() -> None:
    plan = _mlp_plan(method=QuantMethod.AFFINE)
    predicate = build_quant_predicate(plan, q_mode="mxfp4")
    assert predicate("layers.0.mlp.down_proj", object()) == MXFP4_PARAMS


@pytest.mark.parametrize(
    ("method", "bits", "group_size", "reason"),
    [
        (QuantMethod.AFFINE, 6, 64, "6-bit storage"),
        (QuantMethod.AFFINE, 2, 32, "2-bit storage"),
        (QuantMethod.AFFINE, 3, 32, "3-bit storage"),
        (QuantMethod.AFFINE, 8, 64, "8-bit group 64"),
        (QuantMethod.AFFINE, 4, 64, "4-bit group 64"),
        (QuantMethod.AFFINE, 4, 128, "4-bit group 128"),
        (QuantMethod.MXFP4, 8, 32, "mxfp4 at 8-bit"),
        (QuantMethod.AWQ, 4, 32, "awq refinement"),
        (QuantMethod.GPTQ, 4, 32, "gptq refinement"),
        (QuantMethod.GPTQ_ACT, 4, 32, "gptq-act refinement"),
        (QuantMethod.DWQ, 4, 32, "dwq refinement"),
    ],
)
def test_non_product_storage_plan_is_refused_by_axq070(
    method: QuantMethod, bits: int, group_size: int, reason: str
) -> None:
    plan = _mlp_plan(method=method, bits=bits, group_size=group_size)
    with pytest.raises(PlanningError, match="AXQ-070") as excinfo:
        build_quant_predicate(plan)
    assert reason in str(excinfo.value)
    with pytest.raises(PlanningError, match="AXQ-070"):
        PlanPredicate(plan)
    with pytest.raises(PlanningError, match="AXQ-070"):
        predicate_module.allocation_physical_mode(plan.assignments[0])


def test_affine_q_mode_is_refused() -> None:
    plan = _mlp_plan()
    with pytest.raises(PlanningError, match="AXQ-070 removed affine storage"):
        build_quant_predicate(plan, q_mode="affine")
    with pytest.raises(PlanningError, match="AXQ-070 removed affine storage"):
        predicate_module.allocation_physical_mode(plan.assignments[0], "affine")


def test_predicate_exposes_no_refinement_metadata() -> None:
    predicate = build_quant_predicate(_mlp_plan())
    predicate("layers.0.mlp.down_proj", object())
    assert predicate.dwq_metadata == {}
    assert predicate.method_metadata == {}
    assert frozenset({"affine", "mxfp4"}) == predicate_module.FUSED_STACK_METHODS


def test_qwen_checkpoint_paths_map_to_mlx_lm_module_paths() -> None:
    assert "language_model.model.layers.0.mlp.down_proj" in mlx_module_aliases(
        "model.language_model.layers.0.mlp.down_proj"
    )
    assert "language_model.lm_head" in mlx_module_aliases("lm_head")


def test_gemma4_protected_vision_paths_bind_normalized_mlx_vlm_outputs() -> None:
    for source_name, output_name in (
        (
            "model.vision_tower.encoder.layers.0.input_layernorm.weight",
            "vision_tower.encoder.layers.0.input_layernorm.weight",
        ),
        (
            "model.embed_vision.embedding_projection.weight",
            "embed_vision.embedding_projection.weight",
        ),
        (
            "model.vision_embedder.patch_dense.weight",
            "vision_embedder.patch_dense.weight",
        ),
        (
            "model.embed_audio.embedding_projection.weight",
            "embed_audio.embedding_projection.weight",
        ),
    ):
        assert output_name in mlx_tensor_binding_groups(source_name)[0]


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
    # Module-path fusion (quant predicate) must match tensor fusion for MTP.
    assert fused_expert_module("mtp.0.ffn.experts.0.w1") is None
    assert fused_expert_module("model.mtp.0.ffn.experts.1.w2") is None
    assert "mtp.0.ffn.experts.0.w1.weight" in mlx_tensor_aliases("mtp.0.ffn.experts.0.w1.weight")
    fused = fused_expert_tensor_target("layers.3.ffn.experts.0.w1.weight")
    assert fused is not None
    assert fused[0].endswith("switch_mlp.gate_proj.weight")
    up = fused_expert_tensor_target("layers.3.ffn.experts.0.w3.weight")
    assert up is not None
    assert up[0].endswith("switch_mlp.up_proj.weight")
    assert fused_expert_module("layers.3.ffn.experts.0.w1") is not None
    assert fused_expert_module("layers.3.ffn.experts.0.w3") == ("layers.3.ffn.switch_mlp.up_proj")
    from axquant.module_paths import packed_expert_runtime_modules

    # Qwen4-exp packed MTP is identity; DeepSeek indexed MTP is still unfused.
    assert packed_expert_runtime_modules("mtp.layers.0.mlp.experts.gate_up_proj") == ()
    assert packed_expert_runtime_modules("mtp.0.ffn.experts.0.w1") == ()
    mtp_groups = mlx_tensor_binding_groups("mtp.layers.0.mlp.experts.gate_up_proj")
    assert len(mtp_groups) == 1
    assert "mtp.layers.0.mlp.experts.gate_up_proj" in mtp_groups[0]
    assert "mtp.layers.0.mlp.experts.gate_up_proj.weight" in mtp_groups[0]
    assert not any("switch_mlp" in name for name in mtp_groups[0])


@pytest.mark.parametrize(
    "prefix", ["language_model.mtp", "language_model.model.mtp", "model.language_model.mtp"]
)
def test_wrapped_mtp_experts_keep_identity_bindings(prefix: str) -> None:
    from axquant.module_paths import packed_expert_runtime_modules

    indexed = f"{prefix}.layers.0.mlp.experts.0.gate_proj"
    assert fused_expert_module(indexed) is None
    assert fused_expert_tensor_target(f"{indexed}.weight") is None
    for suffix in ("gate_up_proj", "down_proj"):
        packed = f"{prefix}.layers.0.mlp.experts.{suffix}"
        assert packed_expert_runtime_modules(packed) == ()
        groups = mlx_tensor_binding_groups(packed)
        assert len(groups) == 1
        assert packed in groups[0] and f"{packed}.weight" in groups[0]
        assert not any("switch_mlp" in alias for alias in groups[0])


def test_hc_learnable_scale_does_not_alias_to_quant_scales() -> None:
    """HC scale vectors must not invent ``.scales`` aliases (ambiguous binding risk)."""

    aliases = set(mlx_tensor_aliases("layers.0.hc_attn_scale"))
    assert "model.layers.0.attn_hc.scale" in aliases
    assert "model.layers.0.attn_hc.scales" not in aliases
    assert "layers.0.attn_hc.scales" not in aliases
    head = set(mlx_tensor_aliases("hc_head_scale"))
    assert "model.hc_head.scale" in head
    assert "model.hc_head.scales" not in head
    # Real quant sidecar still expands .scale ↔ .scales
    quant = set(mlx_tensor_aliases("layers.0.attn.q_a_proj.scale"))
    assert "layers.0.attn.q_a_proj.scales" in quant


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


def _with_assignments(plan: QuantizationPlan, assignments: list[Allocation]) -> QuantizationPlan:
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
    return plan.model_copy(update={"assignments": assignments, "weight_distribution": distribution})


def test_fused_expert_group_requires_uniform_precision() -> None:
    """Per-expert allocations fuse into one MLX switch module (AXQ MoE v1)."""

    assert (
        fused_expert_module("model.language_model.layers.3.mlp.experts.17.gate_proj")
        == "model.language_model.layers.3.mlp.switch_mlp.gate_proj"
    )
    assert fused_expert_module("model.language_model.layers.3.mlp.gate_proj") is None

    plan = _mlp_plan()
    template = plan.assignments[0]
    members = [
        template.model_copy(
            update={
                "tensor": (f"model.language_model.layers.0.mlp.experts.{index}.gate_proj.weight"),
                "module_path": (f"model.language_model.layers.0.mlp.experts.{index}.gate_proj"),
                "role": TensorRole.EXPERT,
                "bits": 4,
                "method": QuantMethod.MXFP4,
                "group_size": 32,
            }
        )
        for index in (0, 1)
    ]
    uniform = _with_assignments(plan, [*plan.assignments, *members])
    predicate = build_quant_predicate(uniform)
    # Visiting the fused MLX module marks every member expert as covered.
    result = predicate("language_model.model.layers.0.mlp.switch_mlp.gate_proj", object())
    assert result == MXFP4_PARAMS
    assert not {member.module_path for member in members} - predicate.matched

    mixed_members = [
        members[0],
        members[1].model_copy(update={"bits": 8, "method": QuantMethod.AFFINE}),
    ]
    mixed = _with_assignments(plan, [*plan.assignments, *mixed_members])
    with pytest.raises(PlanningError, match="mixes precisions"):
        build_quant_predicate(mixed)

    # Refined fused stacks are no longer packable at all (AXQ-070).
    for method in (QuantMethod.GPTQ, QuantMethod.DWQ):
        refined_members = [member.model_copy(update={"method": method}) for member in members]
        refined = _with_assignments(plan, [*plan.assignments, *refined_members])
        with pytest.raises(PlanningError, match="AXQ-070"):
            build_quant_predicate(refined)


def test_mxfp8_allows_validated_mlx_switchlinear_fused_experts() -> None:
    plan = _mlp_plan()
    template = plan.assignments[0]
    members = [
        template.model_copy(
            update={
                "tensor": f"backbone.layers.0.mixer.experts.{index}.up_proj.weight",
                "module_path": f"backbone.layers.0.mixer.experts.{index}.up_proj",
                "role": TensorRole.EXPERT,
                "bits": 8,
                "method": QuantMethod.AFFINE,
                "group_size": 32,
            }
        )
        for index in (0, 1)
    ]
    total = sum(item.parameters for item in members)
    plan = plan.model_copy(
        update={
            "assignments": members,
            "weight_distribution": {
                "8bit": PrecisionShare(parameters=total, fraction=1.0),
            },
        }
    )
    for q_mode in (None, "mxfp8"):
        predicate = PlanPredicate(plan, q_mode=q_mode)
        assert predicate("backbone.layers.0.mixer.switch_mlp.fc1", object()) == MXFP8_PARAMS


def _packed_plan(
    method: QuantMethod = QuantMethod.MXFP4,
    *,
    bits: int = 4,
    module_path: str = "model.layers.0.mlp.experts.gate_up_proj",
) -> QuantizationPlan:
    plan = _mlp_plan(method=method)
    packed = plan.assignments[0].model_copy(
        update={
            "tensor": "model.layers.0.mlp.experts.gate_up_proj.weight",
            "module_path": module_path,
            "role": TensorRole.EXPERT,
            "bits": bits,
            "method": method,
            "group_size": 32,
        }
    )
    return plan.model_copy(update={"assignments": [packed]})


def test_mxfp4_method_matches_affine_slot_on_packed_experts() -> None:
    """An mxfp4 and a 4-bit affine expert pack execute the same MXFP4 params."""

    native = build_quant_predicate(_packed_plan(QuantMethod.MXFP4))
    affine_slot = build_quant_predicate(_packed_plan(QuantMethod.AFFINE), q_mode="mxfp4")
    for runtime_module in (
        "model.layers.0.mlp.switch_mlp.gate_proj",
        "model.layers.0.mlp.switch_mlp.up_proj",
    ):
        assert native(runtime_module, object()) == MXFP4_PARAMS
        assert affine_slot(runtime_module, object()) == MXFP4_PARAMS


def test_mxfp8_packed_expert_requires_every_split_runtime_module() -> None:
    """MXFP8 packed experts quantize once both split switch modules visit."""

    packed_plan = _packed_plan(QuantMethod.AFFINE, bits=8)
    packed = packed_plan.assignments[0]
    predicate = build_quant_predicate(packed_plan, q_mode="mxfp8")

    assert predicate("model.layers.0.mlp.switch_mlp.gate_proj", object()) == MXFP8_PARAMS
    assert predicate.unmatched_quantized_modules() == {packed.module_path}
    assert predicate("model.layers.0.mlp.switch_mlp.up_proj", object()) == MXFP8_PARAMS
    assert predicate.unmatched_quantized_modules() == set()


def test_allocation_quant_params_returns_complete_to_quantized_dict() -> None:
    """Every physical mode emits all three to_quantized keys explicitly.

    MLX passes a predicate dict to ``to_quantized`` verbatim; omitted
    group_size/bits silently fall back to 64/4, which would pack an invalid
    microscaling tensor. The predicate must never rely on those defaults.
    """

    base = _mlp_plan().assignments[0]
    cases = [
        (QuantMethod.AFFINE, 4, None, MXFP4_PARAMS),
        (QuantMethod.AFFINE, 4, "mxfp4", MXFP4_PARAMS),
        (QuantMethod.MXFP4, 4, None, MXFP4_PARAMS),
        (QuantMethod.MXFP4, 4, "mxfp8", MXFP4_PARAMS),
        (QuantMethod.AFFINE, 8, "mxfp8", MXFP8_PARAMS),
        (QuantMethod.AFFINE, 8, None, MXFP8_PARAMS),
    ]
    for method, bits, q_mode, expected in cases:
        allocation = base.model_copy(update={"method": method, "bits": bits, "group_size": 32})
        params = predicate_module.allocation_quant_params(allocation, q_mode)
        assert set(params) == {"group_size", "bits", "mode"}
        assert params == expected
    with pytest.raises(PlanningError, match="AXQ-070"):
        predicate_module.allocation_quant_params(base.model_copy(update={"group_size": 64}))


def test_packed_expert_requires_every_split_runtime_module() -> None:
    """One packed gate/up tensor is not covered until both MLX modules are visited."""
    packed_plan = _packed_plan(QuantMethod.AFFINE)
    packed = packed_plan.assignments[0]
    predicate = build_quant_predicate(packed_plan)

    gate = predicate("model.layers.0.mlp.switch_mlp.gate_proj", object())
    assert gate == MXFP4_PARAMS
    assert predicate.unmatched_quantized_modules() == {packed.module_path}

    up = predicate("model.layers.0.mlp.switch_mlp.up_proj", object())
    assert up == MXFP4_PARAMS
    assert predicate.unmatched_quantized_modules() == set()


@pytest.mark.parametrize("method", [QuantMethod.GPTQ, QuantMethod.AWQ, QuantMethod.DWQ])
def test_packed_expert_rejects_refinement(method: QuantMethod) -> None:
    with pytest.raises(PlanningError, match=rf"AXQ-070.*{method.value} refinement"):
        build_quant_predicate(_packed_plan(method))


def test_weight_suffixed_module_paths_keep_packed_and_fused_tracking() -> None:
    """Lookup keys are normalized in __init__; __call__ must normalize the same way."""
    packed_plan = _packed_plan(module_path="model.layers.0.mlp.experts.gate_up_proj.weight")
    packed = packed_plan.assignments[0]
    predicate = build_quant_predicate(packed_plan)

    gate = predicate("model.layers.0.mlp.switch_mlp.gate_proj", object())
    assert isinstance(gate, dict)
    # One split runtime module visited is not complete coverage.
    assert predicate.unmatched_quantized_modules() == {packed.module_path}

    up = predicate("model.layers.0.mlp.switch_mlp.up_proj", object())
    assert isinstance(up, dict)
    assert predicate.unmatched_quantized_modules() == set()

    plan = _mlp_plan()
    members = [
        plan.assignments[0].model_copy(
            update={
                "tensor": f"model.layers.0.mlp.experts.{index}.gate_proj.weight",
                "module_path": f"model.layers.0.mlp.experts.{index}.gate_proj.weight",
                "role": TensorRole.EXPERT,
            }
        )
        for index in (0, 1)
    ]
    fused_plan = plan.model_copy(update={"assignments": members})
    fused_predicate = build_quant_predicate(fused_plan)
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
    # Single-output renames expose accepted names as one OR alias set.
    nemo_groups = mlx_tensor_binding_groups("backbone.layers.3.mixer.experts.up_proj.weight")
    assert len(nemo_groups) == 1
    assert "backbone.layers.3.mixer.switch_mlp.fc1.weight" in nemo_groups[0]
    scales_aliases = mlx_tensor_aliases("model.layers.0.mlp.experts.down_proj.scales")
    assert "model.layers.0.mlp.switch_mlp.down_proj.scales" in scales_aliases


def test_gemma4_packed_experts_alias_switch_glu_runtime_modules() -> None:
    """Gemma-4 A4B packs experts under layer.experts → switch_glu (mlx_lm sanitize)."""
    from axquant.module_paths import mlx_module_aliases, packed_expert_runtime_modules

    gate_up = "model.language_model.layers.0.experts.gate_up_proj"
    down = "model.language_model.layers.0.experts.down_proj"
    assert packed_expert_runtime_modules(gate_up) == (
        "model.language_model.layers.0.experts.switch_glu.gate_proj",
        "model.language_model.layers.0.experts.switch_glu.up_proj",
    )
    # down_proj is a single-output rename: lookup aliases only (no multi-visit).
    assert packed_expert_runtime_modules(down) == ()
    down_aliases = set(mlx_module_aliases(down))
    assert "model.language_model.layers.0.experts.switch_glu.down_proj" in down_aliases
    aliases = set(mlx_module_aliases(gate_up))
    assert "language_model.model.layers.0.experts.switch_glu.gate_proj" in aliases
    assert "language_model.model.layers.0.experts.switch_glu.up_proj" in aliases
    # Qwen mlp.experts packing must still resolve to switch_mlp, not switch_glu.
    qwen = packed_expert_runtime_modules("model.language_model.layers.0.mlp.experts.gate_up_proj")
    assert qwen[0].endswith("switch_mlp.gate_proj")
    groups = mlx_tensor_binding_groups("model.language_model.layers.0.experts.gate_up_proj.weight")
    flat = {a for g in groups for a in g}
    assert "model.language_model.layers.0.experts.switch_glu.gate_proj.weight" in flat
    assert "language_model.model.layers.0.experts.switch_glu.up_proj.weight" in flat


def test_gpt_oss_mxfp4_blocks_bind_experts_switch_glu_modules() -> None:
    """Native openai/gpt-oss MXFP4 *_blocks sanitize into mlp.experts.{gate,up,down}_proj."""
    from axquant.module_paths import packed_expert_runtime_modules

    blocks = "model.layers.0.mlp.experts.gate_up_proj_blocks"
    assert packed_expert_runtime_modules(blocks) == (
        "model.layers.0.mlp.experts.gate_proj",
        "model.layers.0.mlp.experts.up_proj",
    )
    # Already-sanitized mlx-community packs keep experts.down_proj as identity;
    # multi-visit packing is not required (aliases cover Qwen switch_mlp rename).
    assert packed_expert_runtime_modules("model.layers.0.mlp.experts.down_proj") == ()
    groups = mlx_tensor_binding_groups("model.layers.0.mlp.experts.gate_up_proj_blocks")
    flat = {a for g in groups for a in g}
    assert "model.layers.0.mlp.experts.gate_proj.weight" in flat
    assert "model.layers.0.mlp.experts.up_proj.weight" in flat
    down_groups = mlx_tensor_binding_groups("model.layers.0.mlp.experts.down_proj.weight")
    assert len(down_groups) == 1
    down_flat = set(down_groups[0])
    assert "model.layers.0.mlp.switch_mlp.down_proj.weight" in down_flat
    assert "model.layers.0.mlp.experts.down_proj.weight" in down_flat


def test_mxfp4_q_mode_keeps_four_bit_trunk_and_refuses_group_64() -> None:
    plan = _mlp_plan(method=QuantMethod.AFFINE)
    predicate = build_quant_predicate(plan, q_mode="mxfp4")
    assert predicate("model.layers.0.mlp.down_proj", object()) == MXFP4_PARAMS
    wide = _mlp_plan(method=QuantMethod.AFFINE, group_size=64)
    with pytest.raises(PlanningError, match=r"AXQ-070.*4-bit group 64"):
        build_quant_predicate(wide, q_mode="mxfp4")


def test_plan_selected_mxfp4_ignores_mxfp8_q_mode() -> None:
    # Storage follows allocation bits; --q-mode no longer selects a format.
    predicate = build_quant_predicate(_mlp_plan(), q_mode="mxfp8")
    assert predicate("model.layers.0.mlp.down_proj", object()) == MXFP4_PARAMS


def test_plan_selected_mxfp4_requires_4bit_group32() -> None:
    # The predicate enforces the mxfp4 constraint even for hand-edited plans
    # whose method was swapped after schema validation.
    with pytest.raises(PlanningError, match=r"AXQ-070.*4-bit group 64"):
        build_quant_predicate(_mlp_plan(group_size=64))
    with pytest.raises(PlanningError, match=r"AXQ-070.*mxfp4 at 8-bit"):
        build_quant_predicate(_mlp_plan(bits=8))
