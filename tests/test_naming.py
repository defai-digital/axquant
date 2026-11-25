import pytest

from axquant.errors import ArtifactError, PlanningError
from axquant.naming import (
    CUDA_LANES,
    MLX_PACK_FORMATS,
    RESERVED_EMPTY_MTP_LEAVES,
    assert_manifest_mtp_files_agree,
    cuda_pack_name,
    distinct_4bit_sibling_allowed,
    mlx_pack_format,
    mlx_pack_format_for_bpw,
    model_name,
    packaged_mtp_present,
    require_mtp_suffix_matches_packaging,
    target_class_for_bpw,
)
from axquant.schema import Allocation, MetricVector, QuantMethod, TensorRole


def test_default_name_uses_mlx_and_manifest_mtp() -> None:
    assert model_name("Qwen/Qwen3.6-27B") == "AX-Qwen3.6-27B-MLX-AXQ-4bit"
    assert model_name("Qwen/Qwen3.6-27B", quant_brand="AXQuant") == (
        "AX-Qwen3.6-27B-MLX-AXQuant-4bit"
    )


def test_mtp_suffix_remains_opt_in() -> None:
    assert model_name("Qwen/Qwen3.6-27B", mtp=True).endswith("-MTP")


def test_artifact_edition_precedes_mtp_suffix() -> None:
    assert (
        model_name(
            "Qwen/Qwen3.6-27B",
            target_class="6bit",
            artifact_edition=2,
            mtp=True,
        )
        == "AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP"
    )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_artifact_edition_must_be_a_positive_integer(value: object) -> None:
    with pytest.raises(ArtifactError, match="positive integer"):
        model_name("Qwen/Qwen3.6-27B", artifact_edition=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_target_class_rejects_nonfinite_budgets(value: float) -> None:
    with pytest.raises(PlanningError, match="finite"):
        target_class_for_bpw(value)


def test_model_name_rejects_unsafe_custom_components() -> None:
    with pytest.raises(ArtifactError, match="unsafe"):
        model_name("Qwen/Qwen3.6-27B", quant_brand="../outside")


def test_low_bit_product_classes_for_deepseek_fleet() -> None:
    assert target_class_for_bpw(2.0) == "2bit"
    assert target_class_for_bpw(3.0) == "3bit"
    assert (
        model_name("DeepSeek-V4-Flash", target_class="2bit") == "AX-DeepSeek-V4-Flash-MLX-AXQ-2bit"
    )
    assert (
        model_name("DeepSeek-V4-Flash", target_class="3bit") == "AX-DeepSeek-V4-Flash-MLX-AXQ-3bit"
    )


def test_development_card_name_regex_accepts_low_bit_classes() -> None:
    from axquant.model_card import _AXQ_NAME

    for name, product_class in (
        ("AX-DeepSeek-V4-Flash-MLX-AXQ-2bit", "2bit"),
        ("AX-DeepSeek-V4-Flash-MLX-AXQ-2bit-experimental", "2bit-experimental"),
        ("AX-DeepSeek-V4-Flash-MLX-AXQ-3bit", "3bit"),
        ("AX-DeepSeek-V4-Flash-MLX-AXQ-3bit-experimental", "3bit-experimental"),
        ("AX-DeepSeek-V4-Flash-MLX-AXQ-4bit", "4bit"),
        ("AX-DeepSeek-V4-Flash-MLX-AXQ-6bit", "6bit"),
    ):
        match = _AXQ_NAME.fullmatch(name)
        assert match is not None
        assert match.group("product_class") == product_class


def test_packaged_mtp_present_is_file_based_only() -> None:
    assert packaged_mtp_present(filenames=["mtp.safetensors"]) is True
    assert packaged_mtp_present(filenames=["mtp_head.safetensors"]) is True
    assert packaged_mtp_present(filenames=["axquant_mtp_sidecar_manifest.json"]) is True
    assert (
        packaged_mtp_present(filenames=["ax_gemma4_assistant_mtp.json", "assistant/config.json"])
        is True
    )
    assert packaged_mtp_present(filenames=["mtplx_runtime.json"]) is False
    assert packaged_mtp_present(filenames=["optiq/mtp.safetensors"]) is False
    assert packaged_mtp_present(filenames=["ax_gemma4_assistant_mtp.json"]) is False
    assert packaged_mtp_present(filenames=["assistant/config.json"]) is False
    assert packaged_mtp_present(filenames=["README.md", "model.safetensors"]) is False


def test_require_mtp_suffix_native_sidecar_needs_mtp_leaf() -> None:
    require_mtp_suffix_matches_packaging(
        "AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit-MTP",
        filenames=["mtp.safetensors", "axquant_mtp_sidecar_manifest.json"],
    )
    with pytest.raises(ArtifactError, match="must end with -MTP"):
        require_mtp_suffix_matches_packaging(
            "AX-Qwen3.8-2.4T-A95B-MLX-AXQ-2bit",
            filenames=["mtp.safetensors"],
        )


def test_require_mtp_suffix_rejects_mtp_leaf_without_packaged_files() -> None:
    with pytest.raises(ArtifactError, match="has no usable MTP artifact"):
        require_mtp_suffix_matches_packaging(
            "AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
            filenames=["mtplx_runtime.json", "README.md"],
        )


def test_require_mtp_suffix_accepts_gemma_assistant_with_false_manifest() -> None:
    names = ["ax_gemma4_assistant_mtp.json", "assistant/model.safetensors"]
    require_mtp_suffix_matches_packaging(
        "AX-gemma-4-12b-MLX-AXQ-4bit-MTP",
        filenames=names,
    )
    assert_manifest_mtp_files_agree(filenames=names, manifest_mtp_present=False)


def test_assert_manifest_mtp_files_agree_is_corrupt_pack_not_packaged() -> None:
    with pytest.raises(ArtifactError, match="corrupt pack"):
        assert_manifest_mtp_files_agree(
            filenames=["README.md"],
            manifest_mtp_present=True,
        )
    assert_manifest_mtp_files_agree(filenames=["README.md"], manifest_mtp_present=None)
    assert_manifest_mtp_files_agree(filenames=["README.md"], manifest_mtp_present=False)


def test_reserved_empty_mtp_leaf_requires_operator_flag() -> None:
    leaf = next(iter(RESERVED_EMPTY_MTP_LEAVES))
    with pytest.raises(ArtifactError, match="has no usable MTP artifact"):
        require_mtp_suffix_matches_packaging(leaf, filenames=[".gitattributes", "README.md"])
    require_mtp_suffix_matches_packaging(
        leaf,
        filenames=[".gitattributes", "README.md"],
        allow_reserved_empty_mtp=True,
    )
    with pytest.raises(ArtifactError, match="has no usable MTP artifact"):
        require_mtp_suffix_matches_packaging(
            "AX-Some-Other-MLX-AXQ-4bit-MTP",
            filenames=[".gitattributes", "README.md"],
            allow_reserved_empty_mtp=True,
        )
    with pytest.raises(ArtifactError, match="has no usable MTP artifact"):
        require_mtp_suffix_matches_packaging(
            leaf,
            filenames=[".gitattributes", "model.safetensors"],
            allow_reserved_empty_mtp=True,
        )


def test_distinct_4bit_sibling_requires_five_percent_complete_weight_savings() -> None:
    assert distinct_4bit_sibling_allowed(95, 100) is True
    assert distinct_4bit_sibling_allowed(94, 100) is True
    assert distinct_4bit_sibling_allowed(96, 100) is False
    with pytest.raises(ArtifactError, match="positive integers"):
        distinct_4bit_sibling_allowed(0, 100)


def test_cuda_pack_name_matches_published_lanes() -> None:
    assert frozenset({"NVFP4-W4A16", "NVFP4-W4A4"}) == CUDA_LANES
    assert cuda_pack_name("baidu/Unlimited-OCR", lane="NVFP4-W4A16") == (
        "AX-Unlimited-OCR-CUDA-AXQ-NVFP4-W4A16"
    )
    assert cuda_pack_name("Unlimited-OCR-3B-MoE", lane="NVFP4-W4A4") == (
        "AX-Unlimited-OCR-3B-MoE-CUDA-AXQ-NVFP4-W4A4"
    )
    assert cuda_pack_name("deepseek-ai/DeepSeek-OCR-2", lane="NVFP4-W4A16", mtp=True).endswith(
        "-CUDA-AXQ-NVFP4-W4A16-MTP"
    )


def test_cuda_pack_name_rejects_the_retired_fp8_mixed_lane() -> None:
    # AXQ-070 retires FP8 and the NVFP4-FP8-6bit mixed budget lane.
    with pytest.raises(ArtifactError, match="unsupported CUDA quantization lane"):
        cuda_pack_name("Some-Base", lane="NVFP4-FP8-6bit")


def _allocation(name: str, role: TensorRole, bits: int, parameters: int = 1024) -> Allocation:
    if bits == 16:
        method, group_size = QuantMethod.BF16, None
    elif bits == 4:
        method, group_size = QuantMethod.MXFP4, 32
    else:
        method, group_size = QuantMethod.AFFINE, 32
    return Allocation(
        tensor=f"{name}.weight",
        module_path=name,
        role=role,
        parameters=parameters,
        bits=bits,
        method=method,
        group_size=group_size,
        predicted_loss=0.0,
        metrics=MetricVector(),
        reason="test",
    )


def test_mlx_pack_formats_are_the_axq070_names() -> None:
    assert MLX_PACK_FORMATS == ("MXFP4", "MXFP4-attn8", "MXFP8")


@pytest.mark.parametrize(
    ("target_bpw", "expected"),
    [
        (2.0, "MXFP4"),
        (4.8, "MXFP4"),
        (6.0, "MXFP4"),
        (7.6, "MXFP4"),
        (7.65, "MXFP8"),
        (8.8, "MXFP8"),
        (16.0, "MXFP8"),
    ],
)
def test_mlx_pack_format_for_bpw(target_bpw: float, expected: str) -> None:
    assert mlx_pack_format_for_bpw(target_bpw) == expected


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_mlx_pack_format_for_bpw_rejects_invalid_budgets(value: float) -> None:
    with pytest.raises(PlanningError, match="positive and finite"):
        mlx_pack_format_for_bpw(value)


def test_mlx_pack_format_names_the_trunk_storage() -> None:
    floors = [
        _allocation("embed", TensorRole.EMBEDDING, 8, parameters=100_000),
        _allocation("lm_head", TensorRole.LM_HEAD, 16, parameters=100_000),
    ]
    mxfp4 = [
        _allocation("attn.q", TensorRole.ATTENTION, 4),
        _allocation("mlp.up", TensorRole.MLP, 4, parameters=4096),
        *floors,
    ]
    # Large 8-bit floors outside the trunk do not change the trunk format.
    assert mlx_pack_format(mxfp4) == "MXFP4"

    attn8 = [
        _allocation("attn.q", TensorRole.ATTENTION, 8),
        _allocation("attn.k", TensorRole.ATTENTION, 8),
        _allocation("mlp.up", TensorRole.MLP, 4, parameters=4096),
        _allocation("experts.up", TensorRole.EXPERT, 4, parameters=4096),
        *floors,
    ]
    assert mlx_pack_format(attn8) == "MXFP4-attn8"

    # A partial attention raise is not MXFP4-attn8.
    partial = [
        _allocation("attn.q", TensorRole.ATTENTION, 8),
        _allocation("attn.k", TensorRole.ATTENTION, 4),
        _allocation("mlp.up", TensorRole.MLP, 4, parameters=4096),
    ]
    assert mlx_pack_format(partial) == "MXFP4"

    mxfp8 = [
        _allocation("attn.q", TensorRole.ATTENTION, 8),
        _allocation("mlp.up", TensorRole.MLP, 8, parameters=4096),
        _allocation("experts.up", TensorRole.EXPERT, 4, parameters=1024),
    ]
    assert mlx_pack_format(mxfp8) == "MXFP8"


def test_mlx_pack_format_refuses_retired_widths_and_empty_trunks() -> None:
    six = [_allocation("attn.q", TensorRole.ATTENTION, 4)]
    six[0] = six[0].model_copy(update={"bits": 6, "method": QuantMethod.AFFINE})
    with pytest.raises(PlanningError, match="not 6-bit"):
        mlx_pack_format(six)
    with pytest.raises(PlanningError, match="no quantized trunk"):
        mlx_pack_format([_allocation("embed", TensorRole.EMBEDDING, 8)])
    with pytest.raises(PlanningError, match="no quantized trunk"):
        mlx_pack_format([_allocation("lm_head", TensorRole.LM_HEAD, 16)])


@pytest.mark.parametrize(
    "lane", ["MIX6", "MIXED-6bit", "NVFP4-FP8", "NVFP4-FP8-6bit", "6bit", "int6", "FP8"]
)
def test_cuda_pack_name_rejects_ad_hoc_tokens(lane: str) -> None:
    with pytest.raises(ArtifactError, match="unsupported CUDA quantization lane"):
        cuda_pack_name("Some-Base", lane=lane)
