from __future__ import annotations


def mlx_module_aliases(module_path: str) -> tuple[str, ...]:
    """Return public MLX-LM module-path aliases for a checkpoint tensor path.

    Qwen 3.6 source checkpoints store the language backbone below
    ``model.language_model`` while MLX-LM exposes it below
    ``language_model.model``. The output head is similarly nested below the
    MLX-LM language-model wrapper. Keeping this mapping in one place makes
    preflight, conversion, and isolated probing use the same identity rule.
    """
    aliases = {module_path}
    checkpoint_prefix = "model.language_model."
    mlx_prefix = "language_model.model."
    if module_path.startswith(checkpoint_prefix):
        aliases.add(f"{mlx_prefix}{module_path.removeprefix(checkpoint_prefix)}")
    if module_path.startswith(mlx_prefix):
        aliases.add(f"{checkpoint_prefix}{module_path.removeprefix(mlx_prefix)}")
    if module_path == "lm_head":
        aliases.add("language_model.lm_head")
    elif module_path == "language_model.lm_head":
        aliases.add("lm_head")
    return tuple(sorted(aliases))
