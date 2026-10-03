from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from axquant.cli._parser import _build_parser
from axquant.converter import _verify_converted_weights, convert_model
from axquant.errors import ArtifactError, PlanningError
from axquant.inspector import inspect_model
from axquant.manual import manual_quantization_plan
from axquant.predicate import allocation_quant_params, build_quant_predicate
from axquant.schema import ManualPlanRecipe, QuantMethod, TensorRole


@pytest.fixture
def mxfp8_source_and_plan(qwen36_model_dir):
    rng = np.random.default_rng(8)
    save_file(
        {
            "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight": rng.normal(
                size=(8, 64)
            ).astype(np.float32),
            "language_model.model.layers.0.mlp.down_proj.weight": rng.normal(size=(8, 64)).astype(
                np.float32
            ),
            "language_model.lm_head.weight": np.ones((16, 64), dtype=np.float32),
            "visual.patch_embed.proj.weight": np.ones((8, 64), dtype=np.float32),
        },
        qwen36_model_dir / "model.safetensors",
    )
    inventory = inspect_model(qwen36_model_dir, model_id="Qwen/Qwen3.6-27B", revision="a" * 40)
    plan = manual_quantization_plan(
        inventory, ManualPlanRecipe(default_bits=8, group_size=32, target_bpw=16.0)
    )
    return qwen36_model_dir, inventory, plan


def test_native_modes_are_explicit_and_mxfp6_is_not_accepted_by_mlx_cli():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "convert",
            "--model",
            "source",
            "--plan",
            "plan.json",
            "--output",
            "out",
            "--q-mode",
            "mxfp8",
        ]
    )
    assert args.q_mode == "mxfp8"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "convert",
                "--model",
                "source",
                "--plan",
                "plan.json",
                "--output",
                "out",
                "--q-mode",
                "mxfp6",
            ]
        )


def test_mxfp8_predicate_retains_bf16_and_rejects_invalid_geometry(mxfp8_source_and_plan):
    _, _, plan = mxfp8_source_and_plan
    predicate = build_quant_predicate(plan, execute_refinement=False, q_mode="mxfp8")
    for allocation in plan.assignments:
        config = predicate(allocation.module_path, object())
        if allocation.bits == 16:
            assert config is False
        else:
            assert config == {"bits": 8, "group_size": 32, "mode": "mxfp8"}
    assert not predicate.unmatched_quantized_modules()
    allocation = next(item for item in plan.assignments if item.bits == 8)
    with pytest.raises(PlanningError, match="group_size 32"):
        allocation_quant_params(allocation.model_copy(update={"group_size": 64}), "mxfp8")
    with pytest.raises(PlanningError, match="unrefined"):
        allocation_quant_params(allocation.model_copy(update={"method": QuantMethod.DWQ}), "mxfp8")
    with pytest.raises(PlanningError, match="unsupported allocation"):
        allocation_quant_params(
            allocation.model_copy(update={"strategy_metadata": {"physical_mode": "mxfp6"}})
        )


def test_mxfp8_does_not_remap_six_bit_or_plan_selected_mxfp4(mxfp8_source_and_plan):
    _, _, plan = mxfp8_source_and_plan
    allocation = next(item for item in plan.assignments if item.bits == 8)
    six = allocation.model_copy(update={"bits": 6})
    assert allocation_quant_params(six, "mxfp8")["mode"] == "affine"
    four = allocation.model_copy(update={"bits": 4, "method": QuantMethod.MXFP4})
    assert allocation_quant_params(four, "mxfp8")["mode"] == "mxfp4"
    with pytest.raises(PlanningError, match="8-bit allocation"):
        allocation_quant_params(
            six.model_copy(update={"strategy_metadata": {"physical_mode": "mxfp8"}})
        )


def test_mxfp8_repacking_requires_explicit_unmeasured_opt_in(mxfp8_source_and_plan, tmp_path):
    source, _, plan = mxfp8_source_and_plan
    with pytest.raises(PlanningError, match="affine sensitivity does not measure MXFP8"):
        convert_model(model=source, plan=plan, output=tmp_path / "output", q_mode="mxfp8")
    assert not (tmp_path / "output").exists()


def _write_quantized(source, plan, output, mode):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    predicate = build_quant_predicate(plan, execute_refinement=False, q_mode=mode)
    output.mkdir()
    arrays = {}
    quantization = {"bits": 8, "group_size": 32, "mode": mode}
    assignments = {item.tensor: item for item in plan.assignments}
    with safe_open(source / "model.safetensors", framework="numpy") as reader:
        keys = reader.keys()
        for name in keys:
            allocation = assignments[name]
            original = reader.get_tensor(name)
            if allocation.bits == 16:
                arrays[name] = original
                continue
            linear = nn.Linear(original.shape[1], original.shape[0], bias=False)
            linear.weight = mx.array(original)
            params = predicate(allocation.module_path, linear)
            assert isinstance(params, dict)
            quantized = linear.to_quantized(**params)
            mx.eval(quantized.weight, quantized.scales)
            arrays[name] = np.array(quantized.weight)
            arrays[allocation.module_path + ".scales"] = np.array(quantized.scales)
            if getattr(quantized, "biases", None) is not None:
                arrays[allocation.module_path + ".biases"] = np.array(quantized.biases)
            quantization[allocation.module_path] = params
            restored = mx.dequantize(quantized.weight, quantized.scales, quantized.biases, **params)
            if mode == "affine":
                np.testing.assert_allclose(
                    np.array(restored.astype(mx.float32)), original, rtol=0.03, atol=0.03
                )
                continue
            # Independently decode the E4M3 payload and E8M0 block scales.
            codes = arrays[name].astype("<u4").view(np.uint8).reshape(original.shape)
            exponent = ((codes >> 3) & 15).astype(np.int32)
            mantissa = (codes & 7).astype(np.float64) / 8
            values = np.ldexp(
                np.where(exponent == 0, mantissa, 1 + mantissa), np.maximum(exponent, 1) - 7
            )
            values = np.copysign(values, np.where(codes & 128, -1, 1))
            scales = arrays[allocation.module_path + ".scales"]
            expected = np.ldexp(values, np.repeat(scales.astype(np.int32) - 127, 32, axis=-1))
            np.testing.assert_array_equal(np.array(restored.astype(mx.float32)), expected)
            assert np.all(np.isfinite(expected))
            inputs = mx.ones((2, original.shape[1]), dtype=mx.float32)
            actual_output = quantized(inputs)
            reference_output = mx.matmul(inputs, restored.astype(mx.float32).T)
            np.testing.assert_allclose(
                np.array(actual_output.astype(mx.float32)),
                np.array(reference_output),
                rtol=1e-4,
                atol=1e-4,
            )
    config = json.loads((source / "config.json").read_text())
    config["quantization"] = quantization
    (output / "config.json").write_text(json.dumps(config))
    save_file(arrays, output / "model.safetensors")
    shutil.copyfile(source / "mtp.safetensors", output / "mtp.safetensors")
    return arrays


def test_real_mlx_mxfp8_save_inspect_and_conversion_verification(mxfp8_source_and_plan, tmp_path):
    source, inventory, plan = mxfp8_source_and_plan
    output = tmp_path / "output"
    arrays = _write_quantized(source, plan, output, "mxfp8")
    assert not any(name.endswith(".biases") for name in arrays)
    result = _verify_converted_weights(
        output, plan, source_tensors={item.name: item for item in inventory.tensors}, q_mode="mxfp8"
    )
    assert result[0] == sum(item.parameters for item in inventory.tensors)
    converted = inspect_model(output, model_id=plan.source_model.model_id, allow_quantized=True)
    weight = next(
        item
        for item in converted.tensors
        if item.role is TensorRole.MLP and not item.quantization_metadata
    )
    assert weight.current_bits == 8 and weight.current_group_size == 32
    assert weight.current_method is None  # Physical MXFP8 is not affine8 evidence.


def test_affine_payload_cannot_pass_as_mxfp8(mxfp8_source_and_plan, tmp_path):
    source, inventory, plan = mxfp8_source_and_plan
    output = tmp_path / "output"
    _write_quantized(source, plan, output, "affine")
    with pytest.raises(ArtifactError, match="does not declare MXFP8"):
        _verify_converted_weights(
            output,
            plan,
            source_tensors={item.name: item for item in inventory.tensors},
            q_mode="mxfp8",
        )


def test_forged_mxfp8_config_cannot_hide_affine_scales(mxfp8_source_and_plan, tmp_path):
    source, inventory, plan = mxfp8_source_and_plan
    output = tmp_path / "output"
    _write_quantized(source, plan, output, "affine")
    config = json.loads((output / "config.json").read_text())
    params = config["quantization"]
    params["mode"] = "mxfp8"
    for value in params.values():
        if isinstance(value, dict):
            value["mode"] = "mxfp8"
    (output / "config.json").write_text(json.dumps(config))
    with pytest.raises(ArtifactError, match="invalid MXFP8 scales"):
        _verify_converted_weights(
            output,
            plan,
            source_tensors={item.name: item for item in inventory.tensors},
            q_mode="mxfp8",
        )
