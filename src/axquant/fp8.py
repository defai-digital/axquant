"""Native RTN E4M3FN channel quantization with lazy PyTorch execution."""

from __future__ import annotations

from typing import Any

from axquant.errors import ArtifactError, PlanningError


def quantize_fp8_channel(
    weight: Any, *, device: str = "cuda", rows_per_chunk: int = 256
) -> tuple[Any, Any]:
    """Return CPU FP8 weights and FP32 dequantization scales, one per row."""
    try:
        import torch
    except ImportError as exc:
        raise ArtifactError("FP8 serialization requires axquant[cuda]") from exc
    if rows_per_chunk < 1:
        raise PlanningError("rows_per_chunk must be positive")
    if device not in {"cpu", "cuda"} and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise PlanningError("device must be cpu, cuda or cuda:<index>")
    if device != "cpu" and not torch.cuda.is_available():
        raise ArtifactError("FP8 CUDA execution requires an available CUDA device")
    if weight.ndim != 2 or min(weight.shape) < 1 or not weight.is_floating_point():
        raise ArtifactError("FP8 encoding requires a nonempty floating matrix")
    encoded = torch.empty(weight.shape, dtype=torch.float8_e4m3fn, device="cpu")
    scales = torch.empty((weight.shape[0], 1), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for start in range(0, weight.shape[0], rows_per_chunk):
            chunk = weight[start : start + rows_per_chunk].to(device=device, dtype=torch.float32)
            if not bool(torch.isfinite(chunk).all().item()):
                raise ArtifactError("FP8 source weights must be finite")
            maximum = chunk.abs().amax(dim=1, keepdim=True)
            # CUDA scalar division may multiply an approximate FP32 reciprocal.
            # Compute in FP64 and explicitly round to FP32 on both backends.
            scale = torch.where(
                maximum == 0, torch.ones_like(maximum), (maximum.double() / 448.0).float()
            )
            if not bool((torch.isfinite(scale) & (scale > 0)).all().item()):
                raise ArtifactError("FP8 weight scale is not representable as positive FP32")
            normalized = (chunk.double() / scale.double()).float()
            quantized = normalized.clamp(-448, 448).to(torch.float8_e4m3fn)
            encoded[start : start + len(chunk)] = quantized.cpu()
            scales[start : start + len(chunk)] = scale.cpu()
    return encoded, scales
