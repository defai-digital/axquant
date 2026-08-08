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
# DeepSeek V4 Flash/Pro: per-expert w1/w2/w3 stack into switch_mlp projections;
# FusedSwitchGLU then concatenates gate+up into gate_proj (public deepseek_v4 sanitize).
_DEEPSEEK_EXPERT_MEMBER = re.compile(
    r"^(?P<prefix>.*\.ffn)\.experts\.(?P<index>\d+)\.(?P<proj>w1|w2|w3)$"
)
_DEEPSEEK_PROJ_TO_SWITCH = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}
_PACKED_TENSOR_SUFFIXES = (".weight", ".scales", ".biases")


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
    deepseek = _DEEPSEEK_EXPERT_MEMBER.match(module_path)
    if deepseek is not None:
        switch = _DEEPSEEK_PROJ_TO_SWITCH[deepseek.group("proj")]
        # After sanitize fuse, both w1 and w3 land on switch_mlp.gate_proj.
        if switch == "up_proj":
            switch = "gate_proj"
        return f"{deepseek.group('prefix')}.switch_mlp.{switch}"
    return None


def fused_expert_tensor_target(tensor_path: str) -> tuple[str, int] | None:
    """Return the exact MLX-LM stack target and expert index for one source weight.

    Public MLX-LM sanitizers stack indexed Qwen-style and Nemotron-H expert
    weights along a new leading expert axis. The index is returned separately
    so converted-checkpoint verification can require a complete contiguous
    source membership set before accepting that many-to-one transform.

    Integrated MTP heads (``mtp.*``) are byte-preserved into ``mtp.safetensors``
    without MLX expert fusion, so they must bind one-to-one to the source names.
    """

    if not tensor_path.endswith(".weight"):
        return None
    # deepseek_v4.sanitize drops ``mtp.*``; axquant re-emits them unfused.
    if tensor_path.startswith("mtp.") or tensor_path.startswith("model.mtp."):
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
    deepseek = _DEEPSEEK_EXPERT_MEMBER.match(module_path)
    if deepseek is not None:
        switch = _DEEPSEEK_PROJ_TO_SWITCH[deepseek.group("proj")]
        if switch == "up_proj":
            switch = "gate_proj"
        target = f"{deepseek.group('prefix')}.switch_mlp.{switch}.weight"
        return target, int(deepseek.group("index"))
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
        # DeepSeek FusedSwitchGLU also exposes the fused module under model.
        if not fused.startswith("model."):
            base.add(f"model.{fused}")
    # DeepSeek V4 export uses bare embed/head and layers.* (no model. prefix).
    if module_path in {"embed", "embed.weight", "embed_tokens"}:
        base.update({"embed", "model.embed_tokens", "embed_tokens"})
    if module_path in {"head", "head.weight", "lm_head"}:
        base.update({"head", "lm_head"})
    if module_path.startswith("layers.") and not module_path.startswith("model.layers."):
        base.add(f"model.{module_path}")
    if module_path.startswith("model.layers."):
        base.add(module_path.removeprefix("model."))
    # DeepSeek shared-expert w1/w2/w3 remaps (not stacked) in sanitize.
    for old, new in (
        (".shared_experts.w1", ".shared_experts.gate_proj"),
        (".shared_experts.w2", ".shared_experts.down_proj"),
        (".shared_experts.w3", ".shared_experts.up_proj"),
    ):
        if old in module_path:
            remapped = module_path.replace(old, new)
            base.add(remapped)
            if not remapped.startswith("model."):
                base.add(f"model.{remapped}")
    # MTP sidecars use mtp.* prefixes in the HF export.
    if module_path.startswith("mtp."):
        base.add(f"model.{module_path}")
    aliases = set()
    checkpoint_prefix = "model.language_model."
    mlx_prefix = "language_model.model."
    for candidate in base:
        aliases.add(candidate)
        if candidate.startswith(checkpoint_prefix):
            aliases.add(f"{mlx_prefix}{candidate.removeprefix(checkpoint_prefix)}")
        if candidate.startswith(mlx_prefix):
            aliases.add(f"{checkpoint_prefix}{candidate.removeprefix(mlx_prefix)}")
        if candidate.startswith("model.visual."):
            aliases.add(f"vision_tower.{candidate.removeprefix('model.visual.')}")
        if candidate.startswith("vision_tower."):
            aliases.add(f"model.visual.{candidate.removeprefix('vision_tower.')}")
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
    if tensor_path.startswith("model.visual."):
        aliases.add(f"vision_tower.{tensor_path.removeprefix('model.visual.')}")
    if tensor_path.startswith("vision_tower."):
        aliases.add(f"model.visual.{tensor_path.removeprefix('vision_tower.')}")
    checkpoint_head = "lm_head."
    mlx_head = "language_model.lm_head."
    if tensor_path.startswith(checkpoint_head):
        aliases.add(f"{mlx_head}{tensor_path.removeprefix(checkpoint_head)}")
    if tensor_path.startswith(mlx_head):
        aliases.add(f"{checkpoint_head}{tensor_path.removeprefix(mlx_head)}")
    # DeepSeek V4 export ↔ MLX runtime renames (public deepseek_v4.sanitize).
    # Expand to a fixed point so independent rewrites compose — e.g.
    # ``layers.0.hc_attn_base`` → ``model.layers.0.attn_hc.base``.
    if tensor_path in {"embed.weight", "embed_tokens.weight"}:
        aliases.update({"embed.weight", "model.embed_tokens.weight", "embed_tokens.weight"})
    if tensor_path in {"head.weight", "lm_head.weight"}:
        aliases.update({"head.weight", "lm_head.weight"})
    if tensor_path in {"norm.weight", "model.norm.weight"}:
        aliases.update({"norm.weight", "model.norm.weight"})
    shared_maps = (
        (".shared_experts.w1.", ".shared_experts.gate_proj."),
        (".shared_experts.w2.", ".shared_experts.down_proj."),
        (".shared_experts.w3.", ".shared_experts.up_proj."),
    )
    hc_param_maps = tuple(
        (f".hc_{sub}_{param}", f".{sub}_hc.{param}")
        for sub in ("attn", "ffn")
        for param in ("fn", "base", "scale")
    )
    # deepseek_v4.sanitize renames MoE router bias for e_score correction.
    gate_bias_maps = (
        (".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"),
    )
    changed = True
    while changed:
        changed = False
        expanded = set(aliases)
        for candidate in list(aliases):
            if candidate.startswith("layers.") and not candidate.startswith("model.layers."):
                expanded.add(f"model.{candidate}")
            if candidate.startswith("model.layers."):
                expanded.add(candidate.removeprefix("model."))
            if candidate.startswith("mtp.") and not candidate.startswith("model.mtp."):
                expanded.add(f"model.{candidate}")
            if candidate.startswith("model.mtp."):
                expanded.add(candidate.removeprefix("model."))
            for param in ("base", "fn", "scale"):
                if candidate == f"hc_head_{param}":
                    expanded.add(f"model.hc_head.{param}")
                if candidate == f"model.hc_head.{param}":
                    expanded.add(f"hc_head_{param}")
            for old, new in hc_param_maps:
                if old in candidate:
                    expanded.add(candidate.replace(old, new))
                if new in candidate:
                    expanded.add(candidate.replace(new, old))
            for old, new in shared_maps:
                if old in candidate:
                    expanded.add(candidate.replace(old, new))
                if new in candidate:
                    expanded.add(candidate.replace(new, old))
            for old, new in gate_bias_maps:
                if old in candidate:
                    expanded.add(candidate.replace(old, new))
                if new in candidate:
                    expanded.add(candidate.replace(new, old))
            # Singular source `.scale` becomes MLX affine `.scales` after re-pack.
            if candidate.endswith(".scale"):
                expanded.add(f"{candidate}s")
            if candidate.endswith(".scales"):
                expanded.add(candidate.removesuffix("s"))
        if expanded != aliases:
            changed = True
            aliases = expanded
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

    # Packed expert tensors must use the same alias rule as preflight and the
    # quantization predicate (`_packed_expert_aliases`): a prefix-restricted
    # rule here would accept a plan whose converted output can never bind.
    packed_suffix = ".weight"
    packed_module = tensor_path
    for storage_suffix in _PACKED_TENSOR_SUFFIXES:
        if tensor_path.endswith(storage_suffix):
            packed_suffix = storage_suffix
            packed_module = tensor_path.removesuffix(storage_suffix)
            break
    packed_aliases = _packed_expert_aliases(packed_module)
    if not packed_aliases:
        return (_mlx_wrapper_tensor_aliases(tensor_path),)
    return tuple(_mlx_wrapper_tensor_aliases(f"{alias}{packed_suffix}") for alias in packed_aliases)


def mlx_tensor_aliases(tensor_path: str) -> tuple[str, ...]:
    """Flatten the strict converted-tensor binding groups for lookup callers."""

    return tuple(
        sorted({alias for group in mlx_tensor_binding_groups(tensor_path) for alias in group})
    )
