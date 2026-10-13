"""MLX model implementations for architectures mlx_lm does not ship.

Modules here mirror ``mlx_lm.models`` layout and are registered into
``sys.modules`` under ``mlx_lm.models.<model_type>`` on the MLX load path.
They import MLX at module level, so callers must import them lazily only
after the MLX backend is known to be installed.
"""
