from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from axquant.errors import ArtifactError, PlanningError

_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_QUANT_SUFFIX = re.compile(
    r"-(?:MLX-)?(?:OptiQ|AXQ|AXQuant|AWQ|GPTQ|DWQ)-[^/]+$",
    re.IGNORECASE,
)

# Hub / filesystem brand for converted checkpoints (short). The toolkit remains
# "AXQuant"; published model ids use AXQ so users can search MLX-AXQ packs.
_DEFAULT_QUANT_BRAND = "AXQ"


def distinct_4bit_sibling_allowed(four_bit_bytes: int, six_bit_bytes: int) -> bool:
    """Return whether a 4-bit SKU saves at least 5% of complete weight bytes."""

    if (
        type(four_bit_bytes) is not int
        or type(six_bit_bytes) is not int
        or four_bit_bytes <= 0
        or six_bit_bytes <= 0
    ):
        raise ArtifactError("4-bit and 6-bit complete weight bytes must be positive integers")
    return four_bit_bytes * 100 <= six_bit_bytes * 95


def target_class_for_bpw(target_bpw: float) -> str:
    """Return the product precision class for a requested BPW budget.

    The class describes the requested product target, not the lowest precision
    present in the candidate grid.  A mixed 6.0-BPW plan may legitimately use
    some 4-bit tensors and must still be labelled ``6bit``.
    """
    if not math.isfinite(target_bpw) or target_bpw <= 0:
        raise PlanningError("target_bpw must be positive and finite")
    centers = (("2bit", 2.0), ("3bit", 3.0), ("4bit", 4.0), ("6bit", 6.0), ("8bit", 8.0))
    for label, center in centers:
        if abs(target_bpw - center) <= 0.35:
            return label
    if 4.4 <= target_bpw <= 5.2:
        return "4bit"
    return f"{target_bpw:.1f}bpw".replace(".", "p")


def model_name(
    base_model: str,
    *,
    target_class: str = "4bit",
    mtp: bool = False,
    artifact_edition: int | None = None,
    prefix: str = "AX-",
    include_mlx: bool = True,
    quant_brand: str = _DEFAULT_QUANT_BRAND,
) -> str:
    base = base_model.rstrip("/").split("/")[-1]
    base = _QUANT_SUFFIX.sub("", base)
    if not _VALID_NAME.fullmatch(base):
        raise ArtifactError(f"cannot derive a safe model name from {base_model}")
    parts = [f"{prefix}{base}"]
    if include_mlx:
        parts.append("MLX")
    parts.extend([quant_brand, target_class])
    if artifact_edition is not None:
        if type(artifact_edition) is not int or artifact_edition < 1:
            raise ArtifactError("artifact edition must be a positive integer")
        parts.append(f"v{artifact_edition}")
    if mtp:
        parts.append("MTP")
    result = "-".join(parts)
    if not _VALID_NAME.fullmatch(result):
        raise ArtifactError("model name components produce an unsafe model name")
    return result


def certified_mixed_precision_name(
    base_model: str,
    measured_main_bpw: float,
    *,
    mtp: bool,
    prefix: str = "AX-",
    include_mlx: bool = True,
    quant_brand: str = _DEFAULT_QUANT_BRAND,
) -> str:
    """Return the claim-safe name for a certified mixed-precision checkpoint.

    The public label is rounded with decimal ROUND_HALF_UP, not binary float
    formatting, and always preserves two decimal places.
    """

    encoded_bpw = format_measured_bpw(measured_main_bpw).replace(".", "p")
    base = base_model.rstrip("/").split("/")[-1]
    base = _QUANT_SUFFIX.sub("", base)
    if not _VALID_NAME.fullmatch(base):
        raise ArtifactError(f"cannot derive a safe model name from {base_model}")
    parts = [f"{prefix}{base}"]
    if include_mlx:
        parts.append("MLX")
    parts.extend([quant_brand, "MP", f"{encoded_bpw}bpw"])
    if mtp:
        parts.append("MTP")
    result = "-".join(parts)
    if not _VALID_NAME.fullmatch(result):
        raise ArtifactError("certified model name components produce an unsafe model name")
    return result


def format_measured_bpw(measured_main_bpw: float) -> str:
    """Format measured BPW with normative decimal half-up rounding."""

    if not math.isfinite(measured_main_bpw) or not 0 < measured_main_bpw <= 16:
        raise ArtifactError("certified measured main BPW must be finite and in (0, 16]")
    try:
        rounded = Decimal(str(measured_main_bpw)).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    except InvalidOperation as exc:
        raise ArtifactError("certified measured main BPW cannot be rounded") from exc
    return format(rounded, ".2f")
