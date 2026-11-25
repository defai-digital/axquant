"""AXQ-070 product storage rules (``axquant.product_formats``)."""

from __future__ import annotations

import pytest

from axquant.errors import PlanningError
from axquant.product_formats import (
    MLX_GROUP_SIZE,
    MLX_STORAGE_MODES,
    mlx_candidate_allowed,
    mlx_default_quantization,
    mlx_storage_mode,
    mlx_storage_violation,
    require_mlx_product_storage,
)
from axquant.schema import Allocation, MetricVector, QuantMethod, TensorRole


def _allocation(
    module_path: str = "model.layers.0.mlp.down_proj",
    *,
    bits: int = 4,
    method: QuantMethod = QuantMethod.MXFP4,
    group_size: int | None = 32,
    parameters: int = 1024,
) -> Allocation:
    allocation = Allocation(
        tensor=f"{module_path}.weight",
        module_path=module_path,
        role=TensorRole.MLP,
        parameters=parameters,
        bits=16,
        method=QuantMethod.BF16,
        group_size=None,
        predicted_loss=0.0,
        metrics=MetricVector(),
        reason="product-format fixture",
    )
    # model_copy skips validation so non-product allocations can be expressed.
    return allocation.model_copy(update={"bits": bits, "method": method, "group_size": group_size})


def test_product_constants() -> None:
    assert MLX_GROUP_SIZE == 32
    assert MLX_STORAGE_MODES == {4: "mxfp4", 8: "mxfp8"}


@pytest.mark.parametrize(
    ("bits", "method", "group_size"),
    [
        (4, QuantMethod.MXFP4, 32),
        (4, QuantMethod.AFFINE, 32),
        (8, QuantMethod.AFFINE, 32),
        (16, QuantMethod.BF16, None),
        (16, QuantMethod.AFFINE, 64),
    ],
)
def test_product_storage_has_no_violation(
    bits: int, method: QuantMethod, group_size: int | None
) -> None:
    assert mlx_storage_violation(bits, method, group_size) is None


@pytest.mark.parametrize(
    ("bits", "method", "group_size", "reason"),
    [
        (2, QuantMethod.AFFINE, 32, "2-bit storage"),
        (3, QuantMethod.AFFINE, 32, "3-bit storage"),
        (6, QuantMethod.AFFINE, 64, "6-bit storage"),
        (4, QuantMethod.AFFINE, 64, "4-bit group 64"),
        (8, QuantMethod.AFFINE, 64, "8-bit group 64"),
        (8, QuantMethod.AFFINE, 128, "8-bit group 128"),
        (8, QuantMethod.MXFP4, 32, "mxfp4 at 8-bit"),
        (4, QuantMethod.AWQ, 32, "awq refinement"),
        (4, QuantMethod.GPTQ, 32, "gptq refinement"),
        (4, QuantMethod.GPTQ_ACT, 32, "gptq-act refinement"),
        (8, QuantMethod.DWQ, 32, "dwq refinement"),
    ],
)
def test_non_product_storage_violations(
    bits: int, method: QuantMethod, group_size: int, reason: str
) -> None:
    assert mlx_storage_violation(bits, method, group_size) == reason
    with pytest.raises(PlanningError, match="AXQ-070") as excinfo:
        mlx_storage_mode(_allocation(bits=bits, method=method, group_size=group_size))
    assert reason in str(excinfo.value)


@pytest.mark.parametrize(
    ("bits", "method", "expected"),
    [
        (4, QuantMethod.MXFP4, "mxfp4"),
        (4, QuantMethod.AFFINE, "mxfp4"),
        (8, QuantMethod.AFFINE, "mxfp8"),
    ],
)
def test_storage_mode_follows_bits(bits: int, method: QuantMethod, expected: str) -> None:
    assert mlx_storage_mode(_allocation(bits=bits, method=method)) == expected


def test_storage_mode_refuses_bf16_allocation() -> None:
    with pytest.raises(PlanningError, match=r"AXQ-070.*no quantization"):
        mlx_storage_mode(_allocation(bits=16, method=QuantMethod.BF16, group_size=None))


@pytest.mark.parametrize(
    ("bits", "method", "group_size", "allowed"),
    [
        (4, QuantMethod.MXFP4, 32, True),
        (4, QuantMethod.AFFINE, 32, False),
        (4, QuantMethod.AWQ, 32, False),
        (8, QuantMethod.AFFINE, 32, True),
        (8, QuantMethod.AFFINE, 64, False),
        (8, QuantMethod.DWQ, 32, False),
        (6, QuantMethod.AFFINE, 32, False),
        (16, QuantMethod.BF16, None, True),
    ],
)
def test_planner_candidates(
    bits: int, method: QuantMethod, group_size: int | None, allowed: bool
) -> None:
    assert mlx_candidate_allowed(bits, method, group_size) is allowed


def test_require_product_storage_accepts_product_plan() -> None:
    require_mlx_product_storage(
        [
            _allocation("a", bits=4, method=QuantMethod.MXFP4),
            _allocation("b", bits=4, method=QuantMethod.AFFINE),
            _allocation("c", bits=8, method=QuantMethod.AFFINE),
            _allocation("d", bits=16, method=QuantMethod.BF16, group_size=None),
        ]
    )


def test_require_product_storage_groups_and_truncates_violations() -> None:
    allocations = [
        _allocation(f"six.{index}", bits=6, method=QuantMethod.AFFINE, group_size=64)
        for index in range(7)
    ]
    allocations.append(_allocation("refined", bits=4, method=QuantMethod.AWQ))
    allocations.append(_allocation("floor", bits=8, method=QuantMethod.AFFINE, group_size=64))
    with pytest.raises(PlanningError, match="AXQ-070") as excinfo:
        require_mlx_product_storage(allocations)
    message = str(excinfo.value)
    assert "6-bit storage" in message
    assert "and 2 more" in message
    assert "awq refinement: ['refined']" in message
    assert "8-bit group 64: ['floor']" in message


def test_default_quantization_dominant_four_bit_is_mxfp4() -> None:
    allocations = [
        _allocation("trunk", bits=4, method=QuantMethod.MXFP4, parameters=4096),
        _allocation("attn", bits=8, method=QuantMethod.AFFINE, parameters=1024),
        _allocation("head", bits=16, method=QuantMethod.BF16, group_size=None, parameters=8192),
    ]
    assert mlx_default_quantization(allocations) == {
        "bits": 4,
        "group_size": 32,
        "mode": "mxfp4",
    }


def test_default_quantization_dominant_eight_bit_is_mxfp8() -> None:
    allocations = [
        _allocation("trunk", bits=8, method=QuantMethod.AFFINE, parameters=4096),
        _allocation("mlp", bits=4, method=QuantMethod.AFFINE, parameters=1024),
    ]
    assert mlx_default_quantization(allocations) == {
        "bits": 8,
        "group_size": 32,
        "mode": "mxfp8",
    }


def test_default_quantization_bits_always_agree_with_mode() -> None:
    for bits, mode in MLX_STORAGE_MODES.items():
        default = mlx_default_quantization([_allocation(bits=bits, method=QuantMethod.AFFINE)])
        assert default is not None
        assert default["bits"] == bits
        assert default["mode"] == mode
        assert default["group_size"] == MLX_GROUP_SIZE


def test_default_quantization_tie_is_deterministic_lower_width() -> None:
    allocations = [
        _allocation("four", bits=4, method=QuantMethod.MXFP4, parameters=1024),
        _allocation("eight", bits=8, method=QuantMethod.AFFINE, parameters=1024),
    ]
    default = mlx_default_quantization(allocations)
    assert default is not None
    assert default["mode"] == "mxfp4"


def test_default_quantization_is_none_without_quantized_lanes() -> None:
    assert mlx_default_quantization([]) is None
    assert (
        mlx_default_quantization([_allocation(bits=16, method=QuantMethod.BF16, group_size=None)])
        is None
    )
