from __future__ import annotations

from pathlib import Path

from axquant.schema import ManualPlanRecipe, TensorRole
from axquant.serde import load_model

_ROOT = Path(__file__).resolve().parents[1]
_RECIPES = (
    "examples/ornith-15-9b-axq-mxfp4-v0.1.yaml",
    "examples/ornith-15-35b-axq-mxfp4-v0.1.yaml",
)


def test_ornith15_recipes_parse() -> None:
    for relative in _RECIPES:
        path = _ROOT / relative
        recipe = load_model(path, ManualPlanRecipe)
        roles = {role for rule in recipe.rules for role in rule.roles}
        assert TensorRole.VISION in roles
        assert TensorRole.ATTENTION in roles
        if "9b" in relative:
            assert TensorRole.EXPERT not in roles
            assert TensorRole.MLP in roles
        else:
            assert TensorRole.EXPERT in roles
            assert TensorRole.ROUTER in roles
        # AXQ-070: MXFP4 trunk, MXFP8 floors, BF16, all at group 32.
        assert recipe.default_bits == 4
        assert recipe.group_size == 32
        for rule in recipe.rules:
            assert (rule.bits, rule.method.value, rule.group_size) in {
                (4, "mxfp4", 32),
                (8, "affine", 32),
                (16, "bf16", None),
            }


def test_factory_catalog_has_no_withdrawn_397b_recipe_or_entry() -> None:
    from scripts.run_ornith_15_axq import MODELS

    assert set(MODELS) == {"9b", "35b"}
    assert not list((_ROOT / "examples").glob("ornith-15-397b-*.yaml"))
