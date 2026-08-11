"""Lazy MLX-Audio and MLX-VLM conversion backends.

These adapters use only the public runtime conversion APIs. They intentionally
remain separate from the generic MLX-LM path so modality towers can stay at
reference precision while :class:`PlanPredicate` controls the language decoder.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Literal

from axquant.errors import ArtifactError, BackendUnavailableError, PlanningError
from axquant.predicate import PlanPredicate
from axquant.schema import QuantizationPlan

ConversionBackend = Literal["mlx-lm", "mlx-audio", "mlx-vlm"]

# Dense 8B Instruct + thin 30B-A3B Instruct MoE both convert through MLX-VLM.
_QWEN3_VL_ADAPTERS = frozenset({"qwen3-vl-v1", "qwen3-vl-moe-v1"})


def conversion_backend(plan: QuantizationPlan) -> ConversionBackend:
    adapter_id = plan.architecture_profile.adapter_id
    if adapter_id == "qwen3-asr-v1":
        return "mlx-audio"
    if adapter_id in _QWEN3_VL_ADAPTERS:
        return "mlx-vlm"
    return "mlx-lm"


def _import(module: str, *, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise BackendUnavailableError(
            f"{module} is required for this architecture; install {extra}"
        ) from exc


def _audio_model(source: Path) -> tuple[Any, dict[str, Any], Any]:
    audio_convert = _import("mlx_audio.convert", extra="mlx-audio")
    config = audio_convert.load_config(source)
    # MLX-Audio's general path heuristic can match the shorter TTS family name
    # ``qwen3`` before the exact ``qwen3_asr`` config. This backend is selected
    # only by the fail-closed Qwen3-ASR adapter, so bind STT explicitly.
    domain = audio_convert.Domain.STT
    model_type = audio_convert.get_model_type(config, source, domain)
    if getattr(domain, "value", None) != "stt" or model_type != "qwen3_asr":
        raise ArtifactError(
            "Qwen3-ASR conversion requires MLX-Audio to resolve domain=stt and "
            f"model_type=qwen3_asr; found {getattr(domain, 'value', domain)!r}/{model_type!r}"
        )
    model_class = audio_convert.get_model_class(model_type, domain)
    model_config = (
        model_class.ModelConfig.from_dict(config) if hasattr(model_class, "ModelConfig") else config
    )
    if hasattr(model_config, "model_path"):
        model_config.model_path = source
    return model_class.Model(model_config), config, audio_convert


def _visit_modules(model: Any, predicate: PlanPredicate, *, backend: str) -> None:
    try:
        named_modules = model.named_modules()
    except AttributeError as exc:
        raise ArtifactError(f"{backend} model does not expose named_modules()") from exc
    for path, module in named_modules:
        if predicate.lookup(path) is not None:
            predicate(path, module)
    unmatched = predicate.unmatched_quantized_modules()
    if unmatched:
        preview = sorted(unmatched)[:10]
        suffix = "" if len(unmatched) <= 10 else f" and {len(unmatched) - 10} more"
        raise PlanningError(f"plan modules do not match the {backend} model: {preview}{suffix}")


def preflight_multimodal(
    source: Path,
    plan: QuantizationPlan,
    predicate: PlanPredicate,
) -> None:
    backend = conversion_backend(plan)
    if backend == "mlx-audio":
        model, _, _ = _audio_model(source)
        _visit_modules(model, predicate, backend="MLX-Audio")
        del model
        return
    if backend == "mlx-vlm":
        vlm_utils = _import("mlx_vlm.utils", extra="mlx-vlm")
        try:
            model = vlm_utils.load_model(source, lazy=True)
        except Exception as exc:
            raise ArtifactError(
                f"cannot load MLX-VLM model structure for preflight: {exc}"
            ) from exc
        _visit_modules(model, predicate, backend="MLX-VLM")
        del model
        return
    raise PlanningError("multimodal preflight called for the MLX-LM backend")


def _convert_audio(
    source: Path,
    destination: Path,
    plan: QuantizationPlan,
    predicate: PlanPredicate,
    default_bits: int,
) -> None:
    model, config, audio_convert = _audio_model(source)
    mlx = _import("mlx.core", extra="mlx")
    mlx_utils = _import("mlx.utils", extra="mlx")
    mlx_lm_utils = _import("mlx_lm.utils", extra="mlx-lm")
    try:
        weights = audio_convert.load_weights(source)
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        weights = dict(mlx_utils.tree_flatten(model.parameters()))
        target_dtype = config.get("torch_dtype")
        if isinstance(target_dtype, str) and target_dtype in audio_convert.MODEL_CONVERSION_DTYPES:
            dtype = getattr(mlx, target_dtype)
            weights = {name: value.astype(dtype) for name, value in weights.items()}
        model.load_weights(list(weights.items()))
        _, converted_config = mlx_lm_utils.quantize_model(
            model,
            config,
            plan.group_size,
            default_bits,
            mode="affine",
            quant_predicate=predicate,
        )
        destination.mkdir(parents=True, exist_ok=False)
        audio_convert.copy_model_files(source, destination)
        mlx_lm_utils.save_model(destination, model, donate_model=True)
        converted_config["model_type"] = "qwen3_asr"
        mlx_lm_utils.save_config(
            converted_config,
            config_path=destination / "config.json",
        )
    finally:
        del model


def _convert_vlm(
    source: Path,
    destination: Path,
    plan: QuantizationPlan,
    predicate: PlanPredicate,
    default_bits: int,
) -> None:
    vlm_convert = _import("mlx_vlm.convert", extra="mlx-vlm")
    vlm_convert.convert(
        str(source),
        mlx_path=destination,
        quantize=True,
        q_group_size=plan.group_size,
        q_bits=default_bits,
        q_mode="affine",
        quant_method="rtn",
        quant_predicate=predicate,
    )


def convert_multimodal(
    source: Path,
    destination: Path,
    plan: QuantizationPlan,
    predicate: PlanPredicate,
    default_bits: int,
) -> None:
    backend = conversion_backend(plan)
    try:
        if backend == "mlx-audio":
            _convert_audio(source, destination, plan, predicate, default_bits)
        elif backend == "mlx-vlm":
            _convert_vlm(source, destination, plan, predicate, default_bits)
        else:
            raise PlanningError("multimodal conversion called for the MLX-LM backend")
    except (ArtifactError, BackendUnavailableError, PlanningError):
        raise
    except Exception as exc:
        raise ArtifactError(f"{backend} conversion failed: {exc}") from exc
