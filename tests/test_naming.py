import pytest

from axquant.errors import ArtifactError, PlanningError
from axquant.naming import (
    RESERVED_EMPTY_MTP_LEAVES,
    assert_manifest_mtp_files_agree,
    distinct_4bit_sibling_allowed,
    model_name,
    packaged_mtp_present,
    require_mtp_suffix_matches_packaging,
    target_class_for_bpw,
)


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
