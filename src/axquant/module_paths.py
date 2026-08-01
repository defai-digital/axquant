from __future__ import annotations

import re

_EXPERT_MEMBER = re.compile(
    r"^(?P<prefix>.*\.mlp)\.experts\.(?P<index>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)$"
)


def fused_expert_module(module_path: str) -> str | None:
    """Map a per-expert checkpoint module to its fused MLX-LM switch module.

    Qwen-style MoE checkpoints store one tensor per expert
    (``...mlp.experts.<i>.gate_proj``); MLX-LM stacks every expert of a layer
    into one fused module (``...mlp.switch_mlp.gate_proj``) that is quantized
    as a single unit. Returns ``None`` for non-expert paths.
    """
    match = _EXPERT_MEMBER.match(module_path)
    if match is None:
        return None
    return f"{match.group('prefix')}.switch_mlp.{match.group('proj')}"


def _packed_expert_aliases(module_path: str) -> tuple[str, ...]:
    """Aliases for pre-fused MoE expert tensors (public MLX-LM sanitize contract).

    ``qwen3_5_moe`` checkpoints ship one packed 3-D tensor per layer:
    ``...mlp.experts.gate_up_proj`` (which MLX-LM splits into the fused
    ``switch_mlp.gate_proj`` and ``switch_mlp.up_proj`` modules) and
    ``...mlp.experts.down_proj`` (which becomes ``switch_mlp.down_proj``).
    One plan allocation therefore covers each packed tensor's MLX modules,
    which share its single precision by construction.
    """
    if module_path.endswith(".mlp.experts.gate_up_proj"):
        prefix = module_path.removesuffix(".experts.gate_up_proj")
        return (f"{prefix}.switch_mlp.gate_proj", f"{prefix}.switch_mlp.up_proj")
    if module_path.endswith(".mlp.experts.down_proj"):
        prefix = module_path.removesuffix(".experts.down_proj")
        return (f"{prefix}.switch_mlp.down_proj",)
    return ()


def mlx_module_aliases(module_path: str) -> tuple[str, ...]:
    """Return public MLX-LM module-path aliases for a checkpoint tensor path.

    Qwen 3.6 source checkpoints store the language backbone below
    ``model.language_model`` while MLX-LM exposes it below
    ``language_model.model``. The output head is similarly nested below the
    MLX-LM language-model wrapper. Packed MoE expert tensors additionally
    alias their fused ``switch_mlp`` modules. Keeping this mapping in one
    place makes preflight, conversion, and isolated probing use the same
    identity rule.
    """
    base = {module_path}
    base.update(_packed_expert_aliases(module_path))
    aliases = set()
    checkpoint_prefix = "model.language_model."
    mlx_prefix = "language_model.model."
    for candidate in base:
        aliases.add(candidate)
        if candidate.startswith(checkpoint_prefix):
            aliases.add(f"{mlx_prefix}{candidate.removeprefix(checkpoint_prefix)}")
        if candidate.startswith(mlx_prefix):
            aliases.add(f"{checkpoint_prefix}{candidate.removeprefix(mlx_prefix)}")
    if module_path == "lm_head":
        aliases.add("language_model.lm_head")
    elif module_path == "language_model.lm_head":
        aliases.add("lm_head")
    return tuple(sorted(aliases))
