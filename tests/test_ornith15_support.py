from __future__ import annotations

from pathlib import Path

from axquant.schema import ManualPlanRecipe, TensorRole
from axquant.serde import load_model

_ROOT = Path(__file__).resolve().parents[1]
_RECIPES = (
    "examples/ornith-15-9b-axq-mxfp4-v0.1.yaml",
    "examples/ornith-15-9b-axq6-v0.1.yaml",
    "examples/ornith-15-35b-axq-mxfp4-v0.1.yaml",
    "examples/ornith-15-35b-axq6-v0.1.yaml",
    "examples/ornith-15-397b-axq-mxfp4-v0.1.yaml",
    "examples/ornith-15-397b-axq6-v0.1.yaml",
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
        if "mxfp4" in relative:
            assert recipe.group_size == 32
        else:
            assert recipe.default_bits == 6
            assert recipe.group_size == 64
