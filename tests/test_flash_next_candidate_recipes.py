from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from axquant.schema import ManualPlanRecipe

ROOT = Path(__file__).resolve().parents[1]

# candidate -> (baseline, attention group size). AX Engine admits 8-bit
# protected projections at group 32 or 64 beside affine 6-bit/group64 experts,
# but only at group 32 beside MXFP4 experts.
CANDIDATES = {
    "qwen38-flash-next-axq6-attn8-v0.1.yaml": ("qwen38-flash-next-axq6-v0.1.yaml", 64),
    "qwen38-flash-next-axq-mxfp4-attn8-v0.1.yaml": ("qwen38-flash-next-axq-mxfp4-v0.1.yaml", 32),
}


def _load(name: str) -> ManualPlanRecipe:
    text = (ROOT / "examples" / name).read_text(encoding="utf-8")
    return ManualPlanRecipe.model_validate(yaml.safe_load(text))


@pytest.mark.parametrize(
    ("candidate", "baseline", "group_size"),
    [(name, base, group) for name, (base, group) in sorted(CANDIDATES.items())],
)
def test_candidate_only_raises_language_attention_to_8bit_affine(
    candidate: str, baseline: str, group_size: int
) -> None:
    recipe = _load(candidate)
    base = _load(baseline)

    attention = [rule for rule in recipe.rules if "attention" in rule.roles]
    assert len(attention) == 1
    assert attention[0].bits == 8
    assert attention[0].method.value == "affine"
    assert attention[0].group_size == group_size

    def without_attention(value: ManualPlanRecipe) -> list[tuple[object, ...]]:
        return [
            (tuple(rule.roles), rule.bits, rule.method, rule.group_size)
            for rule in value.rules
            if "attention" not in rule.roles
        ]

    assert without_attention(recipe) == without_attention(base)
    assert recipe.mtp == base.mtp
    assert recipe.lm_head_min_bits == base.lm_head_min_bits == 8
    assert (recipe.default_bits, recipe.group_size) == (base.default_bits, base.group_size)
