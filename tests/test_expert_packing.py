"""End-to-end microscaling expert-packing evidence on real MLX hardware.

These tests execute the exact AXQuant convert path for fused/packed expert
stacks: ``quantize_model`` with per-module ``allocation_quant_params`` dicts,
then safetensors save, fresh-module load, and ``gather_qmm`` forward. They
are the validation evidence behind fused-stack MXFP4-method and MXFP8
packing (predicate ``FUSED_STACK_METHODS``); the unit suite locks the
predicate contract without MLX.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.utils import quantize_model
from safetensors.numpy import load_file, save_file

from axquant.predicate import allocation_quant_params
from axquant.schema import Allocation, QuantMethod, TensorRole

pytestmark = pytest.mark.integration

_EXPERTS, _IN, _OUT = 4, 128, 64
_OUTER_GROUP_SIZE = 64  # plan.group_size on the real convert path


class _TinyMoELayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mx.random.seed(7)
        self.gate_proj = SwitchLinear(_IN, _OUT, _EXPERTS)


def _allocation(method: QuantMethod, bits: int) -> Allocation:
    return Allocation(
        module_path="model.layers.0.mlp.experts.gate_up_proj",
        tensor="model.layers.0.mlp.experts.gate_up_proj.weight",
        bits=bits,
        method=method,
        group_size=32,
        strategy_metadata={},
        role=TensorRole.EXPERT,
        parameters=_EXPERTS * _IN * _OUT,
        predicted_loss=0.0,
        metrics={},
        reason="expert packing evidence",
    )


def _quantize_save_load_forward(
    tmp_path: Path, label: str, allocation: Allocation, q_mode: str
) -> None:
    params = allocation_quant_params(allocation, q_mode)
    assert set(params) == {"group_size", "bits", "mode"}

    model = _TinyMoELayer()
    tokens = mx.random.normal((2, _EXPERTS, _IN))
    indices = mx.array([[0, 1], [2, 3]])
    mx.eval(tokens, model.parameters())

    model, _ = quantize_model(
        model,
        {},
        _OUTER_GROUP_SIZE,
        allocation.bits,
        mode="affine",
        quant_predicate=lambda path, module: params,
    )
    quantized = model.gate_proj
    assert isinstance(quantized, QuantizedSwitchLinear)
    assert (quantized.bits, quantized.group_size, quantized.mode) == (
        params["bits"],
        params["group_size"],
        params["mode"],
    )

    before = quantized(tokens, indices)
    mx.eval(before)
    flat = {}
    for key, array in tree_flatten(quantized.parameters()):
        mx.eval(array)
        flat[key] = np.asarray(array)
    snapshot = tmp_path / f"switch-{label}.safetensors"
    save_file(flat, str(snapshot))
    restored = load_file(str(snapshot))

    fresh = _TinyMoELayer()
    fresh, _ = quantize_model(
        fresh,
        {},
        _OUTER_GROUP_SIZE,
        allocation.bits,
        mode="affine",
        quant_predicate=lambda path, module: params,
    )
    fresh_quantized = fresh.gate_proj
    assert isinstance(fresh_quantized, QuantizedSwitchLinear)
    fresh_quantized.update({key: mx.array(value) for key, value in restored.items()})
    after = fresh_quantized(tokens, indices)
    mx.eval(after)
    assert float(mx.max(mx.abs(before - after))) == 0.0


def test_mxfp4_method_packed_expert_round_trip(tmp_path: Path) -> None:
    _quantize_save_load_forward(tmp_path, "mxfp4", _allocation(QuantMethod.MXFP4, 4), "affine")


def test_mxfp8_remap_packed_expert_round_trip(tmp_path: Path) -> None:
    _quantize_save_load_forward(tmp_path, "mxfp8", _allocation(QuantMethod.AFFINE, 8), "mxfp8")
