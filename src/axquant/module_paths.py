from __future__ import annotations

import re

_EXPERT_MEMBER = re.compile(
    r"^(?P<prefix>.*\.mlp)\.experts\.(?P<index>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)$"
)
# Nemotron-H (Nemotron 3 Nano/Super/Ultra): experts live under mixer and fuse
# into SwitchMLP modules named fc1/fc2 (public mlx_lm.models.nemotron_h sanitize).
_NEMOTRON_EXPERT_MEMBER = re.compile(
    r"^(?P<prefix>.*\.mixer)\.experts\.(?P<index>\d+)\.(?P<proj>up_proj|down_proj)$"
)
_NEMOTRON_PROJ_TO_SWITCH = {
    "up_proj": "fc1",
    "down_proj": "fc2",
}
_QWEN_PACKED_EXPERT_TENSOR = re.compile(
    r"^(?P<prefix>(?:model\.language_model|language_model\.model)\..*\.mlp)"
    r"\.experts\.(?P<projection>gate_up_proj|down_proj)"
    r"(?P<suffix>\.(?:weight|scales|biases))?$"
)


def fused_expert_module(module_path: str) -> str | None:
    """Map a per-expert checkpoint module to its fused MLX-LM switch module.

    Qwen-style MoE checkpoints store one tensor per expert
    (``...mlp.experts.<i>.gate_proj``); MLX-LM stacks every expert of a layer
    into one fused module (``...mlp.switch_mlp.gate_proj``) that is quantized
    as a single unit.

    Nemotron-H MoE experts use ``...mixer.experts.<i>.{up,down}_proj`` and fuse
    into ``...mixer.switch_mlp.{fc1,fc2}``.

    Returns ``None`` for non-expert paths.
    """
    match = _EXPERT_MEMBER.match(module_path)
    if match is not None:
        return f"{match.group('prefix')}.switch_mlp.{match.group('proj')}"
    nemo = _NEMOTRON_EXPERT_MEMBER.match(module_path)
    if nemo is not None:
        switch = _NEMOTRON_PROJ_TO_SWITCH[nemo.group("proj")]
        return f"{nemo.group('prefix')}.switch_mlp.{switch}"
    return None


def fused_expert_tensor_target(tensor_path: str) -> tuple[str, int] | None:
    """Return the exact MLX-LM stack target and expert index for one source weight.

    Public MLX-LM sanitizers stack indexed Qwen-style and Nemotron-H expert
    weights along a new leading expert axis. The index is returned separately
    so converted-checkpoint verification can require a complete contiguous
    source membership set before accepting that many-to-one transform.
    """

    if not tensor_path.endswith(".weight"):
        return None
    module_path = tensor_path.removesuffix(".weight")
    match = _EXPERT_MEMBER.match(module_path)
    if match is not None:
        target = f"{match.group('prefix')}.switch_mlp.{match.group('proj')}.weight"
        return target, int(match.group("index"))
    nemo = _NEMOTRON_EXPERT_MEMBER.match(module_path)
    if nemo is not None:
        switch = _NEMOTRON_PROJ_TO_SWITCH[nemo.group("proj")]
        target = f"{nemo.group('prefix')}.switch_mlp.{switch}.weight"
        return target, int(nemo.group("index"))
    return None


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
    # Nemotron-H packed forms (if a future export packs experts the same way).
    if module_path.endswith(".mixer.experts.up_proj"):
        prefix = module_path.removesuffix(".experts.up_proj")
        return (f"{prefix}.switch_mlp.fc1",)
    if module_path.endswith(".mixer.experts.down_proj"):
        prefix = module_path.removesuffix(".experts.down_proj")
        return (f"{prefix}.switch_mlp.fc2",)
    return ()


def packed_expert_runtime_modules(module_path: str) -> tuple[str, ...]:
    """Return every MLX runtime module represented by one packed expert tensor.

    Some source checkpoints store gate/up projections in one tensor while
    MLX-LM exposes two independently visited ``SwitchLinear`` modules.  A
    conversion is only complete after every returned module was visited by
    the quantization predicate.
    """
    return _packed_expert_aliases(module_path)


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
    fused = fused_expert_module(module_path)
    if fused is not None:
        base.add(fused)
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


def _mlx_wrapper_tensor_aliases(tensor_path: str) -> tuple[str, ...]:
    """Return aliases for the Qwen wrapper-only tensor-path rewrite."""

    aliases = {tensor_path}
    checkpoint_prefix = "model.language_model."
    mlx_prefix = "language_model.model."
    if tensor_path.startswith(checkpoint_prefix):
        aliases.add(f"{mlx_prefix}{tensor_path.removeprefix(checkpoint_prefix)}")
    if tensor_path.startswith(mlx_prefix):
        aliases.add(f"{checkpoint_prefix}{tensor_path.removeprefix(mlx_prefix)}")
    checkpoint_head = "lm_head."
    mlx_head = "language_model.lm_head."
    if tensor_path.startswith(checkpoint_head):
        aliases.add(f"{mlx_head}{tensor_path.removeprefix(checkpoint_head)}")
    if tensor_path.startswith(mlx_head):
        aliases.add(f"{checkpoint_head}{tensor_path.removeprefix(mlx_head)}")
    return tuple(sorted(aliases))


def mlx_tensor_binding_groups(tensor_path: str) -> tuple[tuple[str, ...], ...]:
    """Return every required output component and its accepted path aliases.

    Most tensors bind one-to-one after MLX-LM's Qwen wrapper rename. Indexed
    expert weights bind many-to-one to a fused leading expert axis, while Qwen
    3.5/3.6 packed ``gate_up_proj`` tensors bind one-to-many to separate
    ``switch_mlp.gate_proj.weight`` and ``switch_mlp.up_proj.weight`` tensors.
    Each inner tuple is an alias set for exactly one required output component;
    callers separately prove complete membership for a shared fusion target.
    """

    fused = fused_expert_tensor_target(tensor_path)
    if fused is not None:
        return (_mlx_wrapper_tensor_aliases(fused[0]),)

    packed = _QWEN_PACKED_EXPERT_TENSOR.match(tensor_path)
    if packed is None:
        return (_mlx_wrapper_tensor_aliases(tensor_path),)

    suffix = packed.group("suffix") or ".weight"
    projections = (
        ("gate_proj", "up_proj") if packed.group("projection") == "gate_up_proj" else ("down_proj",)
    )
    return tuple(
        _mlx_wrapper_tensor_aliases(f"{packed.group('prefix')}.switch_mlp.{projection}{suffix}")
        for projection in projections
    )


def mlx_tensor_aliases(tensor_path: str) -> tuple[str, ...]:
    """Flatten the strict converted-tensor binding groups for lookup callers."""

    return tuple(
        sorted({alias for group in mlx_tensor_binding_groups(tensor_path) for alias in group})
    )
