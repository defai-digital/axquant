from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from safetensors.numpy import save_file

from axquant import cuda_mix
from axquant.cli import main
from axquant.errors import ArtifactError, PlanningError
from axquant.schema.cuda import CudaQuantizationPlan
from axquant.schema.cuda_mix import (
    CudaMixPackManifest,
    CudaMixQuantizationPlan,
    CudaMixTensorAllocation,
)
from axquant.serde import file_sha256, load_model, read_data, stable_sha256, write_data

_BLOCK = 16


@pytest.fixture
def mix_source(tmp_path: Path) -> Path:
    source = tmp_path / "Qwen3-Mix"
    source.mkdir()
    write_data(source / "config.json", {"model_type": "qwen3", "num_hidden_layers": 1})
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "visual.patch_embed.proj.weight",
        "model.norm.weight",
    ]
    tensors = {name: np.arange(256, dtype=np.float32).reshape(8, 32) / 20 for name in names}
    tensors[names[-1]] = np.ones(32, dtype=np.float32)
    save_file(tensors, source / "model.safetensors")
    save_file(
        {"mtp.projection.weight": np.ones((8, 32), dtype=np.float32)}, source / "mtp.safetensors"
    )
    write_data(source / "tokenizer_config.json", {"model_max_length": 1024})
    write_data(source / "credentials.json", {"test_only": "never copy this file"})
    return source


def build_plan(source: Path, **kwargs: Any) -> CudaMixQuantizationPlan:
    return cuda_mix.plan_cuda_mix(source, allow_unmeasured=True, **kwargs)


def _allocation(name: str, method: str = "nvfp4", rows: int = 8, columns: int = 32):
    return CudaMixTensorAllocation(
        tensor_name=name,
        source_file="model.safetensors",
        shape=(rows, columns),
        dtype="BF16",
        role="mlp",
        method=method,
        reason="test",
    )


def _unit_allocations() -> list[CudaMixTensorAllocation]:
    prefix = "model.layers.0."
    return [
        _allocation(prefix + "mlp.down_proj.weight"),
        _allocation(prefix + "mlp.gate_proj.weight"),
        _allocation(prefix + "mlp.up_proj.weight"),
        _allocation(prefix + "self_attn.o_proj.weight"),
        _allocation(prefix + "self_attn.q_proj.weight"),
        _allocation(prefix + "self_attn.k_proj.weight"),
        _allocation(prefix + "self_attn.v_proj.weight"),
    ]


def _mix_payload() -> dict[str, Any]:
    def entry(name: str, method: str, role: str = "mlp") -> dict[str, Any]:
        return {
            "tensor_name": name,
            "source_file": "model.safetensors",
            "shape": [8, 32],
            "dtype": "BF16",
            "role": role,
            "method": method,
            "reason": "test",
        }

    experts = [
        entry(f"model.layers.0.mlp.experts.{expert}.{projection}.weight", "nvfp4", "expert")
        for expert in (0, 1)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    return {
        "target_bpw": 6.0,
        "trunk_bpw": 6.0,
        "model_id": "organization/model",
        "model_type": "qwen3",
        "config_sha256": "0" * 64,
        "source_files": [{"path": "model.safetensors", "sha256": "0" * 64, "size_bytes": 1}],
        "allocations": [
            entry("model.layers.0.self_attn.q_proj.weight", "nvfp4", "attention"),
            entry("model.layers.0.self_attn.k_proj.weight", "nvfp4", "attention"),
            entry("model.layers.0.self_attn.v_proj.weight", "nvfp4", "attention"),
            entry("model.layers.0.self_attn.o_proj.weight", "fp8", "attention"),
            *experts,
            entry("lm_head.weight", "preserve", "lm_head"),
        ],
        "estimated_weight_bytes": 4096,
    }


def _set_method(payload: dict[str, Any], name: str, method: str) -> None:
    allocation = next(item for item in payload["allocations"] if item["tensor_name"] == name)
    allocation["method"] = method
    allocation.pop("scale_group", None)


# --- planning -----------------------------------------------------------------


def test_plan_mixes_both_families_and_preserves_protected(mix_source: Path) -> None:
    plan = build_plan(mix_source)
    methods = {item.method for item in plan.allocations}
    assert methods == {"nvfp4", "fp8", "preserve"}
    assert plan.format == "nvfp4+fp8"
    assert plan.algorithm == "rtn"
    assert plan.activation_precision == 16
    assert plan.evidence_kind == "unmeasured"
    assert plan.target_bpw == 6.0
    assert plan.trunk_bpw >= plan.target_bpw
    assert plan.estimated_weight_bytes > 0
    assert plan.model_id == "Qwen3-Mix"
    preserved = {item.tensor_name for item in plan.allocations if item.method == "preserve"}
    assert {
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "visual.patch_embed.proj.weight",
        "mtp.projection.weight",
    } <= preserved
    serialized = plan.model_dump_json()
    assert str(mix_source.parent) not in serialized
    assert "credentials.json" not in serialized


def test_mix_plan_selects_both_families_on_a_realistic_grid(mix_source: Path) -> None:
    plan = build_plan(mix_source)
    nvfp4 = {item.tensor_name for item in plan.allocations if item.method == "nvfp4"}
    fp8 = {item.tensor_name for item in plan.allocations if item.method == "fp8"}
    assert nvfp4 and fp8
    assert nvfp4.isdisjoint(fp8)
    assert "model.layers.0.self_attn.q_proj.weight" in nvfp4
    assert "model.layers.0.mlp.gate_proj.weight" in fp8


def test_plan_and_convert_require_unmeasured_acknowledgement(mix_source: Path) -> None:
    with pytest.raises(PlanningError, match="allow-unmeasured"):
        cuda_mix.plan_cuda_mix(mix_source)
    with pytest.raises(PlanningError, match="allow-unmeasured"):
        cuda_mix.convert_cuda_mix(mix_source, build_plan(mix_source), mix_source.parent / "pack")


@pytest.mark.parametrize("target", [0.0, -1.0, float("inf"), float("nan")])
def test_target_bpw_must_be_positive_and_finite(mix_source: Path, target: float) -> None:
    with pytest.raises(PlanningError, match="positive and finite"):
        build_plan(mix_source, target_bpw=target)


@pytest.mark.parametrize("identity", ["/private/model", "~/model", "file:model", "C:\\model"])
def test_private_model_id_fails(mix_source: Path, identity: str) -> None:
    with pytest.raises(ValueError, match="filesystem path"):
        build_plan(mix_source, model_id=identity)


def test_keep_policy_preserves_selected_matrices(mix_source: Path) -> None:
    plan = build_plan(mix_source, keep_patterns=["*.self_attn.*"])
    methods = {
        item.method
        for item in plan.allocations
        if item.tensor_name.startswith("model.layers.0.self_attn.")
    }
    assert methods == {"preserve"}
    assert any(item.method == "fp8" for item in plan.allocations)


# --- promotion helper ---------------------------------------------------------


def test_trunk_bpw_counts_only_quantized_trunk_matrices() -> None:
    allocations = _unit_allocations()
    assert cuda_mix.trunk_bpw(allocations) == pytest.approx(4.625)
    assert cuda_mix.trunk_bpw([]) == 0.0
    assert cuda_mix.trunk_bpw([_allocation("lm_head.weight", method="preserve")]) == 0.0


def test_promotion_is_ascending_unit_size_and_deterministic() -> None:
    allocations = _unit_allocations()
    promoted = cuda_mix.promote_units_to_target(allocations, 4.7)
    fp8 = {item.tensor_name for item in promoted if item.method == "fp8"}
    assert fp8 == {"model.layers.0.mlp.down_proj.weight"}
    assert cuda_mix.trunk_bpw(promoted) == pytest.approx(5.25)
    assert [item.method for item in cuda_mix.promote_units_to_target(allocations, 4.7)] == [
        item.method for item in promoted
    ]


def test_promotion_stops_once_the_target_is_reached() -> None:
    allocations = _unit_allocations()
    promoted = cuda_mix.promote_units_to_target(allocations, 6.0)
    fp8 = {item.tensor_name for item in promoted if item.method == "fp8"}
    nvfp4 = {item.tensor_name for item in promoted if item.method == "nvfp4"}
    assert fp8 == {
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    }
    assert nvfp4 == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    }
    assert cuda_mix.trunk_bpw(promoted) >= 6.0
    assert all(item.scale_group is None for item in promoted if item.method == "fp8")
    # The whole attention block moves together because vLLM cannot match it by name.
    forced = cuda_mix.promote_units_to_target(_unit_allocations(), 8.0)
    attention = {item.method for item in forced if ".self_attn." in item.tensor_name}
    assert attention == {"fp8"}


def test_promotion_never_touches_preserved_tensors() -> None:
    allocations = [*_unit_allocations(), _allocation("lm_head.weight", method="preserve")]
    promoted = cuda_mix.promote_units_to_target(allocations, 6.0)
    preserved = [item for item in promoted if item.method == "preserve"]
    assert [item.tensor_name for item in preserved] == ["lm_head.weight"]


def test_fused_unit_name_is_the_promotion_unit() -> None:
    assert (
        cuda_mix.allocation_unit("model.layers.0.mlp.gate_proj.weight")
        == "model.layers.0.mlp.gate_up_proj"
    )
    assert cuda_mix.allocation_unit("model.layers.0.mlp.down_proj.weight") is None
    assert cuda_mix.allocation_unit("model.mtp.layers.0.experts.3.up_proj.weight") == (
        "model.mtp.layers.0.experts"
    )


# --- schema -------------------------------------------------------------------


def test_mix_payload_is_valid() -> None:
    assert CudaMixQuantizationPlan.model_validate(_mix_payload()).format == "nvfp4+fp8"


@pytest.mark.parametrize("degenerate", ["all-nvfp4", "all-fp8"])
def test_degenerate_mix_is_rejected(degenerate: str) -> None:
    payload = _mix_payload()
    for item in payload["allocations"]:
        if item["method"] == "preserve":
            continue
        item["method"] = "nvfp4" if degenerate == "all-nvfp4" else "fp8"
        item.pop("scale_group", None)
    with pytest.raises(ValidationError, match="at least one NVFP4 and one FP8"):
        CudaMixQuantizationPlan.model_validate(payload)


@pytest.mark.parametrize(
    "flipped",
    ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.experts.0.down_proj.weight"],
)
def test_fused_and_expert_units_must_stay_uniform(flipped: str) -> None:
    payload = _mix_payload()
    _set_method(payload, flipped, "fp8")
    with pytest.raises(ValidationError, match="uniform"):
        CudaMixQuantizationPlan.model_validate(payload)


def test_duplicate_tensors_and_unbound_files_are_rejected() -> None:
    payload = _mix_payload()
    payload["allocations"].append(dict(payload["allocations"][0]))
    with pytest.raises(ValidationError, match="duplicate files or tensors"):
        CudaMixQuantizationPlan.model_validate(payload)
    payload = _mix_payload()
    payload["allocations"][0]["source_file"] = "other.safetensors"
    with pytest.raises(ValidationError, match="unbound source file"):
        CudaMixQuantizationPlan.model_validate(payload)


@pytest.mark.parametrize(
    "override",
    [
        {"method": "nvfp4", "role": "lm_head"},
        {"method": "nvfp4", "shape": [8, 17]},
        {"method": "fp8", "shape": [8, 17]},
        {"method": "fp8", "role": "embedding"},
        {"method": "fp8", "shape": [8, 32, 2]},
        {"method": "nvfp4", "dtype": "I64"},
        {"method": "preserve", "scale_group": "unit"},
        {"method": "fp8", "scale_group": "unit"},
    ],
)
def test_eligibility_and_scale_group_rules_fail_closed(override: dict[str, Any]) -> None:
    fields: dict[str, Any] = {
        "tensor_name": "model.layers.0.mlp.down_proj.weight",
        "source_file": "model.safetensors",
        "shape": (8, 32),
        "dtype": "BF16",
        "role": "mlp",
        "method": "nvfp4",
        "reason": "test",
    }
    fields.update(override)
    with pytest.raises(ValidationError):
        CudaMixTensorAllocation(**fields)


def test_block_size_alignment_is_required(mix_source: Path) -> None:
    payload = _mix_payload()
    for item in payload["allocations"]:
        assert item["shape"][1] % _BLOCK == 0
    payload["allocations"][0]["shape"] = [8, 17]
    with pytest.raises(ValidationError, match="eligible floating text"):
        CudaMixQuantizationPlan.model_validate(payload)


# --- compressed-tensors config ------------------------------------------------


def test_config_uses_attention_class_target_and_regex_groups(mix_source: Path) -> None:
    config = cuda_mix._mix_quantization_config(build_plan(mix_source))
    assert config["quant_method"] == "compressed-tensors"
    assert config["quantization_status"] == "compressed"
    assert "format" not in config
    groups = config["config_groups"]
    assert set(groups) == {"nvfp4", "fp8"}
    # vLLM builds self-attention projections without a module prefix, so exactly
    # one group (the attention method) uses the "Linear" class target and the
    # other group uses anchored regex name targets.
    class_groups = [group for group in groups.values() if group["targets"] == ["Linear"]]
    assert len(class_groups) == 1
    for group in groups.values():
        assert group["output_activations"] is None
        if group["targets"] != ["Linear"]:
            assert group["targets"] and all(t.startswith("re:") for t in group["targets"])
    nvfp4, fp8 = groups["nvfp4"], groups["fp8"]
    assert nvfp4["format"] == "nvfp4-pack-quantized"
    assert nvfp4["weights"]["scale_dtype"] == "float8_e4m3fn"
    assert nvfp4["weights"]["group_size"] == 16
    assert nvfp4["input_activations"] is None
    assert fp8["format"] == "float-quantized"
    assert fp8["weights"]["strategy"] == "channel"
    assert fp8["input_activations"]["strategy"] == "token"
    assert fp8["input_activations"]["dynamic"] is True
    regex_group = nvfp4 if nvfp4["targets"] != ["Linear"] else fp8
    assert any(
        name in regex_group["targets"][0]
        for name in (r"model\.layers\.0\.mlp\.down_proj", r"model\.layers\.0\.mlp\.gate_up_proj")
    )
    assert all(entry.startswith("re:") for entry in config["ignore"])
    assert r"lm_head" in config["ignore"][0]
    assert r"model\.norm" in config["ignore"][0]
    assert config["ignore"][-2:] == [
        r"re:(?:model\.)?mtp(?:\..*)?",
        r"re:.*(?:vision|visual|sam_model|projector|view_sep|image_newline|audio).*",
    ]


def test_mtp_protection_overlap_fails_closed(mix_source: Path) -> None:
    plan = build_plan(mix_source)
    mtp = next(item for item in plan.allocations if item.tensor_name.startswith("mtp."))
    invalid = plan.model_copy(
        update={
            "allocations": [
                item.model_copy(update={"method": "fp8", "scale_group": None})
                if item == mtp
                else item
                for item in plan.allocations
            ]
        }
    )
    with pytest.raises(PlanningError, match="MTP protection overlaps"):
        cuda_mix._mix_quantization_config(invalid)


def test_vision_protection_overlap_fails_closed(mix_source: Path) -> None:
    plan = build_plan(mix_source)
    vision = next(
        item for item in plan.allocations if item.tensor_name == "visual.patch_embed.proj.weight"
    )
    invalid = plan.model_copy(
        update={
            "allocations": [
                item.model_copy(update={"method": "fp8", "scale_group": None})
                if item == vision
                else item
                for item in plan.allocations
            ]
        }
    )
    with pytest.raises(PlanningError, match="protection pattern overlaps"):
        cuda_mix._mix_quantization_config(invalid)


# --- converter fail-closed paths (no torch required) --------------------------


def test_existing_output_and_source_overlap_are_rejected(mix_source: Path) -> None:
    plan = build_plan(mix_source)
    with pytest.raises(ArtifactError, match="already exists"):
        cuda_mix.convert_cuda_mix(mix_source, plan, mix_source, allow_unmeasured=True)
    with pytest.raises(ArtifactError, match="overlap"):
        cuda_mix.convert_cuda_mix(mix_source, plan, mix_source / "pack", allow_unmeasured=True)


@pytest.mark.parametrize("device", ["mps", "cuda:abc", "cuda:-1"])
def test_invalid_device_fails_without_output(mix_source: Path, device: str) -> None:
    output = mix_source.parent / "pack"
    with pytest.raises(PlanningError, match="device must"):
        cuda_mix.convert_cuda_mix(
            mix_source, build_plan(mix_source), output, device=device, allow_unmeasured=True
        )
    assert not output.exists()


def test_replan_drift_fails_closed(mix_source: Path) -> None:
    plan = build_plan(mix_source)
    forged = plan.model_copy(update={"target_bpw": 5.0})
    output = mix_source.parent / "pack"
    with pytest.raises(PlanningError, match="protection policy"):
        cuda_mix.convert_cuda_mix(mix_source, forged, output, device="cpu", allow_unmeasured=True)
    assert not output.exists()


def test_source_change_fails_closed(mix_source: Path) -> None:
    plan = build_plan(mix_source)
    write_data(mix_source / "config.json", {"model_type": "qwen3", "num_hidden_layers": 2})
    with pytest.raises(ArtifactError, match="content changed"):
        cuda_mix.convert_cuda_mix(
            mix_source, plan, mix_source.parent / "pack", device="cpu", allow_unmeasured=True
        )


def test_same_size_weight_mutation_fails_exact_content_binding(mix_source: Path) -> None:
    plan = build_plan(mix_source)
    path = mix_source / "model.safetensors"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ArtifactError, match="content changed"):
        cuda_mix.convert_cuda_mix(
            mix_source, plan, mix_source.parent / "pack", device="cpu", allow_unmeasured=True
        )


# --- CLI ----------------------------------------------------------------------


def test_cli_plan_mix6(mix_source: Path) -> None:
    path = mix_source.parent / "plan.json"
    common = ["plan-cuda", str(mix_source), "--q-mode", "mix6", "--output", str(path)]
    assert main(common) == 2
    assert not path.exists()
    assert main([*common, "--allow-unmeasured", "--target-bpw", "6.0"]) == 0
    plan = load_model(path, CudaMixQuantizationPlan)
    assert plan.schema_version == "axquant.cuda-mix-plan.v1"
    assert plan.target_bpw == 6.0
    assert plan.trunk_bpw >= 6.0


def test_cli_default_q_mode_stays_nvfp4(mix_source: Path) -> None:
    path = mix_source.parent / "plan.json"
    assert main(["plan-cuda", str(mix_source), "--output", str(path), "--allow-unmeasured"]) == 0
    assert load_model(path, CudaQuantizationPlan).format == "nvfp4"


@pytest.mark.parametrize(
    "extra",
    [
        ["--embedding-protection"],
        ["--activation-bits", "4"],
    ],
)
def test_cli_mix6_rejects_unsupported_options(mix_source: Path, extra: list[str]) -> None:
    path = mix_source.parent / "plan.json"
    assert (
        main(
            [
                "plan-cuda",
                str(mix_source),
                "--q-mode",
                "mix6",
                "--output",
                str(path),
                "--allow-unmeasured",
                *extra,
            ]
        )
        == 2
    )
    assert not path.exists()


def test_cli_convert_reads_the_mix_plan_schema(mix_source: Path) -> None:
    pytest.importorskip("torch")
    path = mix_source.parent / "plan.json"
    output = mix_source.parent / "pack"
    assert (
        main(
            [
                "plan-cuda",
                str(mix_source),
                "--q-mode",
                "mix6",
                "--output",
                str(path),
                "--allow-unmeasured",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "convert-cuda",
                str(mix_source),
                "--plan",
                str(path),
                "--device",
                "cpu",
                "--allow-unmeasured",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert load_model(output / cuda_mix.MANIFEST_NAME, CudaMixPackManifest).format == "nvfp4+fp8"


# --- CPU export (torch required) ----------------------------------------------


def test_cpu_export_writes_both_families_and_a_two_group_config(mix_source: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file as load_torch
    from safetensors.torch import save_file as save_torch

    main_weights = {
        name: tensor.to(torch.bfloat16)
        for name, tensor in load_torch(str(mix_source / "model.safetensors")).items()
    }
    save_torch(main_weights, str(mix_source / "model.safetensors"))
    plan = build_plan(mix_source)
    output = mix_source.parent / "pack"
    manifest = cuda_mix.convert_cuda_mix(
        mix_source, plan, output, device="cpu", allow_unmeasured=True
    )
    assert manifest.backend == "numpy-reference"
    assert not manifest.runtime_verified and not manifest.quality_certified
    assert manifest.plan_sha256 == stable_sha256(plan)
    assert manifest.quantized_tensors == sorted(
        item.tensor_name for item in plan.allocations if item.method != "preserve"
    )
    assert not (output / "credentials.json").exists()
    saved = load_torch(str(output / "model.safetensors"))
    for item in plan.allocations:
        if item.source_file != "model.safetensors":
            continue
        prefix = item.tensor_name.removesuffix(".weight")
        if item.method == "preserve":
            assert torch.equal(saved[item.tensor_name], main_weights[item.tensor_name])
        elif item.method == "nvfp4":
            assert item.tensor_name not in saved
            assert saved[prefix + ".weight_packed"].dtype == torch.uint8
            assert saved[prefix + ".weight_scale"].dtype == torch.float8_e4m3fn
        else:
            encoded = saved[item.tensor_name]
            scales = saved[prefix + ".weight_scale"]
            assert encoded.dtype == torch.float8_e4m3fn
            assert scales.dtype == torch.float32 and scales.shape == (item.shape[0], 1)
            original = main_weights[item.tensor_name].float()
            decoded = encoded.float() * scales
            assert torch.linalg.norm(decoded - original) < 0.1 * torch.linalg.norm(original)
    index = read_data(output / "model.safetensors.index.json")
    assert set(index["weight_map"]) == set(saved)
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in saved.values()
    )
    for member in manifest.files:
        assert member.sha256 == file_sha256(output / member.path)
    config = read_data(output / "config.json")
    assert config["torch_dtype"] == plan.activation_dtype
    assert set(config["quantization_config"]["config_groups"]) == {"nvfp4", "fp8"}


def test_export_failure_rolls_back(mix_source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    output = mix_source.parent / "pack"

    def fail_quantizer(*_args: Any, **_kwargs: Any) -> Any:
        raise ArtifactError("fp8 quantizer failure")

    monkeypatch.setattr(cuda_mix, "quantize_fp8_channel", fail_quantizer)
    with pytest.raises(ArtifactError, match="fp8 quantizer failure"):
        cuda_mix.convert_cuda_mix(
            mix_source, build_plan(mix_source), output, device="cpu", allow_unmeasured=True
        )
    assert not output.exists()
    assert not list(output.parent.glob(".pack.mix-*"))


def test_mutation_during_conversion_rolls_back(
    mix_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    output = mix_source.parent / "pack"
    original = cuda_mix.quantize_nvfp4_reference

    def mutate_then_quantize(weight: Any, **kwargs: Any) -> Any:
        write_data(mix_source / "tokenizer_config.json", {"model_max_length": 2048})
        return original(weight, **kwargs)

    monkeypatch.setattr(cuda_mix, "quantize_nvfp4_reference", mutate_then_quantize)
    with pytest.raises(ArtifactError, match="content changed"):
        cuda_mix.convert_cuda_mix(
            mix_source, build_plan(mix_source), output, device="cpu", allow_unmeasured=True
        )
    assert not output.exists()
    assert not list(output.parent.glob(".pack.mix-*"))


def test_untouched_safetensors_are_copied_byte_for_byte(mix_source: Path) -> None:
    pytest.importorskip("torch")
    output = mix_source.parent / "pack"
    cuda_mix.convert_cuda_mix(
        mix_source, build_plan(mix_source), output, device="cpu", allow_unmeasured=True
    )
    assert file_sha256(output / "mtp.safetensors") == file_sha256(mix_source / "mtp.safetensors")


def test_math_helpers_agree_on_encoding_sizes() -> None:
    allocation = _allocation("model.layers.0.mlp.down_proj.weight", rows=8, columns=32)
    assert cuda_mix._quantized_bytes("nvfp4", 256, 8) == 148
    assert cuda_mix._quantized_bytes("fp8", 256, 8) == 288
    assert cuda_mix._estimated_weight_bytes([allocation]) == 148
    preserved = _allocation("lm_head.weight", method="preserve")
    assert cuda_mix._estimated_weight_bytes([preserved]) == 256 * 16 // 8
    assert math.isclose(cuda_mix.trunk_bpw([allocation]), 4.625)
