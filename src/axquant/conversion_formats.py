"""User-facing MLX presets over the existing development planning pipeline."""

from dataclasses import dataclass, replace

from axquant.errors import PlanningError
from axquant.ladders import ConvertLadder, get_ladder
from axquant.schema import ConvertLadderName, QuantMethod


@dataclass(frozen=True)
class MlxFormatPreset:
    """One source for CLI discovery and prior-planner defaults."""

    name: str
    bits: int
    limitation: str

    @property
    def target_bpw(self) -> float:
        return self.bits + 0.8


# AXQ-070: AXQuant writes only MLX MXFP4, MXFP8 and BF16.
_PRESETS = (
    MlxFormatPreset("mxfp4", 4, "MXFP4 trunk; MXFP8 and BF16 protection floors"),
    MlxFormatPreset("mxfp8", 8, "MXFP8 trunk and floors; BF16 where policy requires"),
)
MLX_FORMATS = tuple(preset.name for preset in _PRESETS)


def format_ladder(name: str) -> ConvertLadder:
    """Constrain base precision while retaining higher protection floors."""
    preset = next((item for item in _PRESETS if item.name == name), None)
    if preset is None:
        raise PlanningError(f"unknown MLX format {name!r}; choose {', '.join(MLX_FORMATS)}")
    bits = preset.bits
    candidates = tuple(sorted({bits, 8, 16}))
    methods: tuple[QuantMethod, ...] = (QuantMethod.AFFINE, QuantMethod.BF16)
    if name == "mxfp4":
        methods += (QuantMethod.MXFP4,)
    return replace(
        get_ladder(ConvertLadderName.PRIOR),
        candidate_bits=candidates,
        candidate_group_sizes=(32,),
        candidate_methods=methods,
        default_target_bpw=preset.target_bpw,
    )


def formats_markdown(backend: str = "all") -> str:
    """Describe conversion choices without importing or probing a runtime."""
    if backend not in {"all", "mlx", "cuda"}:
        raise PlanningError("format backend must be all, mlx or cuda")
    lines = [
        "# AXQuant conversion formats",
        "",
        "AXQuant is the MXFP / NVFP toolkit. It writes only these storage formats:",
        "MLX MXFP4, MXFP8 and BF16; CUDA NVFP4 and BF16. No affine integer storage.",
        "",
        "Development conversion choices, not runtime compatibility certificates.",
        "",
    ]
    if backend in {"all", "mlx"}:
        lines += [
            "## MLX (Apple Silicon)",
            "",
            "Install: python -m pip install 'axquant[mlx]'",
            "",
            "| --format | Default target BPW | Limits |",
            "| --- | --- | --- |",
            *[f"| {p.name} | {p.target_bpw:.1f} | {p.limitation} |" for p in _PRESETS],
            "",
            "Example: axquant quantize /path/to/model-bf16 --format mxfp4",
            "",
            "Storage follows allocation bits: 4-bit is MXFP4, 8-bit is MXFP8 (both",
            "group 32), 16-bit is BF16. Protected tensors (embeddings, routers, LM head,",
            "norms, MTP, vision and audio towers) stay MXFP8 or BF16; policy floors may",
            "raise the target BPW. Without --format, the prior planner targets 4.8 BPW",
            "over the same three formats. Recipes and measured ladders cannot be",
            "combined with --format.",
            "",
            "Runtime targets: MLX-LM, MTPLX, oMLX and AX Engine; exact model, format",
            "and requested features need separate load/generation validation.",
            "",
        ]
    if backend in {"all", "cuda"}:
        lines += [
            "## CUDA (NVIDIA)",
            "",
            "Install: python -m pip install 'axquant[cuda]'",
            "",
            "Primary format: NVFP4 W4A16 (--q-mode nvfp4). Requires a compatible",
            "CUDA-enabled PyTorch build and GPU. vLLM is the primary runtime target.",
            "",
            "Example: axquant quantize-cuda /path/to/model-bf16 --q-mode nvfp4 "
            "--device cuda:0 --allow-unmeasured --output ./AXQ-NVFP4",
            "",
            "Advanced calibrated W4A4 (NVFP4 activations): axquant quantize-cuda --help",
            "",
        ]
    lines += [
        "Use original, supported Safetensors checkpoints. No quality or speed claim",
        "follows from successful conversion. This catalog does not check this host.",
        "",
    ]
    return "\n".join(lines)


def validate_format_options(
    name: str | None,
    *,
    recipe: object,
    ladder: ConvertLadderName | str,
) -> None:
    if name is None:
        return
    format_ladder(name)
    if recipe is not None or ladder != ConvertLadderName.PRIOR:
        raise PlanningError(
            "--format requires the prior ladder and cannot be combined with --recipe"
        )
