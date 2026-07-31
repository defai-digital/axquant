from __future__ import annotations

import re

from axquant.errors import ArtifactError

_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_QUANT_SUFFIX = re.compile(
    r"-(?:MLX-)?(?:OptiQ|AXQuant|AWQ|GPTQ|DWQ)-[^/]+$",
    re.IGNORECASE,
)


def model_name(
    base_model: str,
    *,
    target_class: str = "4bit",
    mtp: bool = False,
    prefix: str = "AX-",
    include_mlx: bool = True,
) -> str:
    base = base_model.rstrip("/").split("/")[-1]
    base = _QUANT_SUFFIX.sub("", base)
    if not _VALID_NAME.fullmatch(base):
        raise ArtifactError(f"cannot derive a safe model name from {base_model}")
    parts = [f"{prefix}{base}"]
    if include_mlx:
        parts.append("MLX")
    parts.extend(["AXQuant", target_class])
    if mtp:
        parts.append("MTP")
    return "-".join(parts)
