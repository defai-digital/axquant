from __future__ import annotations

import math
import re
from collections.abc import Collection
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

# Reserved empty -MTP Hub leaf (D2). Publish still fails closed; only the
# operator Hub-audit path may pass allow_reserved_empty_mtp=True.
RESERVED_EMPTY_MTP_LEAVES: frozenset[str] = frozenset(
    {
        "AX-DeepSeek-V4-Flash-0731-MLX-AXQ-4bit-MTP",
    }
)
_NATIVE_MTP_ROOT_FILES = frozenset(
    {
        "mtp.safetensors",
        "mtp_head.safetensors",
        "axquant_mtp_sidecar_manifest.json",
    }
)
_ASSISTANT_CONTRACT = "ax_gemma4_assistant_mtp.json"


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


def _posix_relative(name: str) -> str:
    return name.replace("\\", "/")


def _filename_set(filenames: Collection[str]) -> set[str]:
    return {_posix_relative(name) for name in filenames}


def _has_assistant_tree(names: set[str]) -> bool:
    return any(name == "assistant" or name.startswith("assistant/") for name in names)


def packaged_mtp_present(*, filenames: Collection[str]) -> bool:
    """Return True iff the published tree contains a usable MTP artifact.

    Native sidecar files count only at the repository root. Nested
    ``optiq/mtp.safetensors`` does not. Assistant-MTP requires the root
    contract file and an ``assistant/`` path. Leftover ``mtplx_runtime.json``
    and ``manifest.mtp_present`` never count here.
    """

    names = _filename_set(filenames)
    if names & _NATIVE_MTP_ROOT_FILES:
        return True
    return _ASSISTANT_CONTRACT in names and _has_assistant_tree(names)


def _has_reserved_empty_weights(filenames: Collection[str]) -> bool:
    names = _filename_set(filenames)
    if names & _NATIVE_MTP_ROOT_FILES:
        return True
    if any(name.endswith(".safetensors") for name in names):
        return True
    return _ASSISTANT_CONTRACT in names and _has_assistant_tree(names)


def assert_manifest_mtp_files_agree(
    *,
    filenames: Collection[str],
    manifest_mtp_present: bool | None,
) -> None:
    """Raise if the manifest claims MTP while the tree has no usable MTP files.

    ``False`` or ``None`` is a no-op, including Gemma assistant-MTP packs whose
    ``mtp_present`` is False.
    """

    if manifest_mtp_present is not True:
        return
    if packaged_mtp_present(filenames=filenames):
        return
    raise ArtifactError(
        "corrupt pack: axquant_manifest.json sets mtp_present true without a usable MTP artifact"
    )


def require_mtp_suffix_matches_packaging(
    repo_leaf: str,
    *,
    filenames: Collection[str],
    allow_reserved_empty_mtp: bool = False,
) -> None:
    """Fail closed: ``repo_leaf.endswith('-MTP')`` iff packaged MTP is present.

    Does not compare against ``has_mtp`` / ``manifest.mtp_present``. The D2
    reserved-empty exception applies only when the leaf is in
    ``RESERVED_EMPTY_MTP_LEAVES``, ``allow_reserved_empty_mtp`` is True, and
    the tree has no weights.
    """

    if type(allow_reserved_empty_mtp) is not bool:
        raise ArtifactError("allow_reserved_empty_mtp must be a boolean")
    packaged = packaged_mtp_present(filenames=filenames)
    named_mtp = repo_leaf.endswith("-MTP")
    if packaged == named_mtp:
        return
    if (
        named_mtp
        and not packaged
        and allow_reserved_empty_mtp
        and repo_leaf in RESERVED_EMPTY_MTP_LEAVES
        and not _has_reserved_empty_weights(filenames)
    ):
        return
    if packaged and not named_mtp:
        raise ArtifactError(
            f"repository name {repo_leaf!r} must end with -MTP because the "
            "published tree contains a usable MTP artifact"
        )
    raise ArtifactError(
        f"repository name {repo_leaf!r} ends with -MTP but the published tree "
        "has no usable MTP artifact"
    )
