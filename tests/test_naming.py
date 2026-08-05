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
