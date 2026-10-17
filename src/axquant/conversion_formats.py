"""User-facing MLX presets over the existing development planning pipeline."""

from dataclasses import replace
from typing import Literal

from axquant.errors import PlanningError
from axquant.ladders import ConvertLadder, get_ladder
from axquant.schema import ConvertLadderName, QuantMethod

MLX_FORMATS = ("2bit", "3bit", "6bit", "mxfp4", "mxfp8")


def format_ladder(name: str) -> ConvertLadder:
    """Constrain base precision while retaining higher protection floors."""
    if name not in MLX_FORMATS:
        raise PlanningError(f"unknown MLX format {name!r}; choose {', '.join(MLX_FORMATS)}")
    bits = {"2bit": 2, "3bit": 3, "6bit": 6, "mxfp4": 4, "mxfp8": 8}[name]
    candidates = tuple(sorted({bits, 8, 16} | ({6} if bits < 4 else set())))
    methods: tuple[QuantMethod, ...] = (QuantMethod.AFFINE, QuantMethod.BF16)
    if name == "mxfp4":
        methods += (QuantMethod.MXFP4,)
    return replace(
        get_ladder(ConvertLadderName.PRIOR),
        candidate_bits=candidates,
        candidate_group_sizes=(32,),
        candidate_methods=methods,
        default_target_bpw=bits + 0.8,
    )


def format_q_mode(name: str | None) -> Literal["affine", "mxfp4", "mxfp8"]:
    """MXFP8 repacks unrefined affine8/group32; it is not measured sensitivity."""
    return "mxfp8" if name == "mxfp8" else "affine"


def validate_format_options(
    name: str | None,
    *,
    recipe: object,
    ladder: ConvertLadderName | str,
    allow_legacy_4bit: bool,
) -> None:
    if name is None:
        return
    format_ladder(name)
    if recipe is not None or ladder != ConvertLadderName.PRIOR:
        raise PlanningError(
            "--format requires the prior ladder and cannot be combined with --recipe"
        )
    if allow_legacy_4bit:
        raise PlanningError("--format cannot be combined with --allow-legacy-4bit")
