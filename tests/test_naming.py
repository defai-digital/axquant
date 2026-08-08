import pytest

from axquant.errors import ArtifactError, PlanningError
from axquant.naming import model_name, target_class_for_bpw


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
