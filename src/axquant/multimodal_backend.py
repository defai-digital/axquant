"""Lazy MLX-Audio and MLX-VLM conversion backends.

These adapters use only the public runtime conversion APIs. They intentionally
remain separate from the generic MLX-LM path so modality towers can stay at
reference precision while :class:`PlanPredicate` controls the language decoder.
"""

from __future__ import annotations

import copy
import importlib
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from axquant.errors import ArtifactError, BackendUnavailableError, PlanningError
from axquant.predicate import PlanPredicate, allocation_quant_params
from axquant.schema import QuantizationPlan

ConversionBackend = Literal["mlx-lm", "mlx-audio", "mlx-vlm"]

# Dense 8B Instruct + thin 30B-A3B Instruct MoE both convert through MLX-VLM.
_QWEN3_VL_ADAPTERS = frozenset({"qwen3-vl-v1", "qwen3-vl-moe-v1"})
# OCR / document VL families that share the public MLX-VLM convert entrypoint.
_MLX_VLM_ADAPTERS = _QWEN3_VL_ADAPTERS | frozenset(
    {"deepseek-ocr2-v1", "muse-glimmer-v1", "qwen4-exp-v1"}
)


def _force_cpu_if_requested(*, default: bool = False) -> None:
    flag = os.environ.get("AXQUANT_FORCE_CPU", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return
    if flag not in {"1", "true", "yes", "on"} and not (default and flag == ""):
        return
    mlx = _import("mlx.core", extra="mlx")
    mlx.set_default_device(mlx.cpu)


def conversion_backend(plan: QuantizationPlan) -> ConversionBackend:
    adapter_id = plan.architecture_profile.adapter_id
    if adapter_id == "qwen3-asr-v1":
        return "mlx-audio"
    if adapter_id in _MLX_VLM_ADAPTERS:
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


def _weight_last_dim(module: Any) -> int | None:
    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) < 1:
        return None
    return int(shape[-1])


def _quant_params_for_runtime_module(
    path: str,
    module: Any,
    predicate: PlanPredicate,
    *,
    default_group_size: int,
    default_bits: int,
    default_mode: str,
) -> bool | dict[str, Any]:
    """Return ``nn.quantize`` class_predicate output for one runtime module.

    ``mlx_vlm.quant_utils.quantize_model`` (and the mlx-lm twin) skip any
    module whose last dimension is not divisible by the *global* group size
    before the plan predicate runs. Qwen4-exp PLE shards are 160-wide and are
    planned at gs32 while the trunk stays gs64; that global skip would leave
    them dense. This helper uses the planned per-module group size and fails
    closed when the last dimension is not divisible.
    """

    if not hasattr(module, "to_quantized"):
        return False
    allocation = predicate.lookup(path)
    if allocation is None:
        return False
    if allocation.bits < 16:
        planned = allocation_quant_params(allocation, default_mode)
        group_size = planned.get("group_size")
        last_dim = _weight_last_dim(module)
        if last_dim is not None and group_size is not None and last_dim % int(group_size) != 0:
            raise ArtifactError(
                f"{path} last dimension {last_dim} is not divisible by planned "
                f"group_size {group_size}"
            )
    result = predicate(path, module)
    if result is False:
        return False
    if result is True:
        return {
            "group_size": default_group_size,
            "bits": default_bits,
            "mode": default_mode,
        }
    return dict(result)


def _require_quantized_plan_coverage(predicate: PlanPredicate, *, backend: str) -> None:
    unmatched = predicate.unmatched_quantized_modules()
    if unmatched:
        preview = sorted(unmatched)[:10]
        suffix = "" if len(unmatched) <= 10 else f" and {len(unmatched) - 10} more"
        raise PlanningError(f"{backend} did not quantize planned modules: {preview}{suffix}")


def _quantize_with_plan_predicate(
    model: Any,
    config: dict[str, Any],
    plan: QuantizationPlan,
    predicate: PlanPredicate,
    default_bits: int,
    q_mode: str,
) -> tuple[Any, dict[str, Any]]:
    mlx_nn = _import("mlx.nn", extra="mlx")
    quantized_config = copy.deepcopy(config)
    default_mode = str(q_mode or "affine")
    quant_params = {
        "group_size": plan.group_size,
        "bits": default_bits,
        "mode": default_mode,
    }
    if "quantization" not in quantized_config:
        quantized_config["quantization"] = dict(quant_params)

    def class_predicate(path: str, module: Any) -> bool | dict[str, Any]:
        params = _quant_params_for_runtime_module(
            path,
            module,
            predicate,
            default_group_size=plan.group_size,
            default_bits=default_bits,
            default_mode=default_mode,
        )
        if params is False:
            return False
        quantized_config["quantization"][path] = params
        return params

    mlx_nn.quantize(
        model,
        group_size=plan.group_size,
        bits=default_bits,
        mode=default_mode,
        class_predicate=class_predicate,
    )
    quantized_config["quantization_config"] = quantized_config["quantization"]
    return model, quantized_config


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


def _convert_deepseek_ocr2(
    source: Path,
    destination: Path,
    plan: QuantizationPlan,
    predicate: PlanPredicate,
    default_bits: int,
    q_mode: str = "affine",
) -> None:
    """Convert DeepSeek-OCR-2 without Hugging Face AutoProcessor remote-code.

    Official / mlx-community snapshots ship ``auto_map`` + ``modeling_*.py`` that
    require torch. MLX-VLM provides ``DeepseekOCR2Processor``; use it directly
    and quantize through the public ``quantize_model`` helper.
    """
    import glob
    import shutil

    vlm_utils = _import("mlx_vlm.utils", extra="mlx-vlm")
    quant_utils = _import("mlx_vlm.quant_utils", extra="mlx-vlm")
    processor_mod = _import(
        "mlx_vlm.models.deepseekocr_2.processing_deepseekocr",
        extra="mlx-vlm",
    )
    try:
        model = vlm_utils.load_model(source, lazy=True)
        config = vlm_utils.load_config(source)
        processor = processor_mod.DeepseekOCR2Processor.from_pretrained(str(source))
    except Exception as exc:
        raise ArtifactError(f"cannot load DeepSeek-OCR-2 for MLX-VLM convert: {exc}") from exc
    try:
        config.setdefault("vision_config", {})
        model, config = quant_utils.quantize_model(
            model,
            config,
            plan.group_size,
            default_bits,
            mode=q_mode,
            quant_predicate=predicate,
        )
        # MoEGate routers are not Linear modules; quantize_model skips them.
        # Visit remaining plan modules so fail-closed coverage matches preflight
        # (routers stay dense BF16 — still above the 8-bit floor).
        _visit_modules(model, predicate, backend="MLX-VLM")
        destination.mkdir(parents=True, exist_ok=False)
        vlm_utils.save_weights(destination, model, donate_weights=True)
        for pattern in ("*.py", "*.json", "*.jinja", "*.txt"):
            for file in glob.glob(str(source / pattern)):
                name = Path(file).name
                if name == "model.safetensors.index.json":
                    continue
                shutil.copy(file, destination / name)
        if hasattr(processor, "save_pretrained"):
            with suppress(Exception):
                processor.save_pretrained(destination)
                # Processor files were already copied from source if this fails.
        vlm_utils.save_config(config, config_path=destination / "config.json")
    finally:
        del model


def _convert_muse_glimmer(
    source: Path,
    destination: Path,
    plan: QuantizationPlan,
    predicate: PlanPredicate,
    default_bits: int,
    q_mode: str = "affine",
) -> None:
    """Convert Muse-Glimmer via MLX-VLM using MuseGlimmerProcessor directly."""
    import glob
    import shutil

    vlm_utils = _import("mlx_vlm.utils", extra="mlx-vlm")
    quant_utils = _import("mlx_vlm.quant_utils", extra="mlx-vlm")
    processor_mod = _import(
        "mlx_vlm.models.muse_glimmer.processing_muse_glimmer",
        extra="mlx-vlm",
    )
    try:
        model = vlm_utils.load_model(source, lazy=True)
        config = vlm_utils.load_config(source)
        processor = processor_mod.MuseGlimmerProcessor.from_pretrained(str(source))
    except Exception as exc:
        raise ArtifactError(f"cannot load Muse-Glimmer for MLX-VLM convert: {exc}") from exc
    try:
        config.setdefault("vision_config", {})
        model, config = quant_utils.quantize_model(
            model,
            config,
            plan.group_size,
            default_bits,
            mode=q_mode,
            quant_predicate=predicate,
        )
        _visit_modules(model, predicate, backend="MLX-VLM")
        destination.mkdir(parents=True, exist_ok=False)
        vlm_utils.save_weights(destination, model, donate_weights=True)
        for pattern in ("*.py", "*.json", "*.jinja", "*.txt", "*.md"):
            for file in glob.glob(str(source / pattern)):
                name = Path(file).name
                if name == "model.safetensors.index.json":
                    continue
                shutil.copy(file, destination / name)
        if hasattr(processor, "save_pretrained"):
            with suppress(Exception):
                processor.save_pretrained(destination)
        vlm_utils.save_config(config, config_path=destination / "config.json")
    finally:
        del model


def _convert_qwen4_exp(
    source: Path,
    destination: Path,
    plan: QuantizationPlan,
    predicate: PlanPredicate,
    default_bits: int,
    q_mode: str = "affine",
) -> None:
    """Convert Qwen3.8-Flash-Next via MLX-VLM + a plan-aware quant predicate.

    ``mlx_vlm.convert.convert`` applies a single group size / mode and hits GPU
    timeouts on this 180B checkpoint. PLE shards are 160-wide and need gs32,
    which ``mlx_vlm.quant_utils.quantize_model`` would skip against a gs64 trunk.
    """
    import glob
    import shutil

    _force_cpu_if_requested(default=True)
    vlm_utils = _import("mlx_vlm.utils", extra="mlx-vlm")
    processor_mod = _import(
        "mlx_vlm.models.qwen3_vl.processing_qwen3_vl",
        extra="mlx-vlm",
    )
    try:
        model = vlm_utils.load_model(source, lazy=True)
        config = vlm_utils.load_config(source)
        processor = processor_mod.Qwen3VLProcessor.from_pretrained(str(source))
    except Exception as exc:
        raise ArtifactError(f"cannot load Qwen4-exp for MLX-VLM convert: {exc}") from exc
    try:
        config.setdefault("vision_config", {})
        model, config = _quantize_with_plan_predicate(
            model,
            config,
            plan,
            predicate,
            default_bits,
            q_mode,
        )
        _require_quantized_plan_coverage(predicate, backend="MLX-VLM")
        destination.mkdir(parents=True, exist_ok=False)
        vlm_utils.save_weights(destination, model, donate_weights=True)
        for pattern in ("*.py", "*.json", "*.jinja", "*.txt", "*.md"):
            for file in glob.glob(str(source / pattern)):
                name = Path(file).name
                if name == "model.safetensors.index.json":
                    continue
                shutil.copy(file, destination / name)
        license_src = source / "LICENSE"
        if license_src.is_file():
            shutil.copy(license_src, destination / "LICENSE")
        if hasattr(processor, "save_pretrained"):
            with suppress(Exception):
                processor.save_pretrained(destination)
        vlm_utils.save_config(config, config_path=destination / "config.json")
    finally:
        del model


def _convert_vlm(
    source: Path,
    destination: Path,
    plan: QuantizationPlan,
    predicate: PlanPredicate,
    default_bits: int,
    q_mode: str = "affine",
) -> None:
    if plan.architecture_profile.adapter_id == "deepseek-ocr2-v1":
        _convert_deepseek_ocr2(source, destination, plan, predicate, default_bits, q_mode=q_mode)
        return
    if plan.architecture_profile.adapter_id == "muse-glimmer-v1":
        _convert_muse_glimmer(source, destination, plan, predicate, default_bits, q_mode=q_mode)
        return
    if plan.architecture_profile.adapter_id == "qwen4-exp-v1":
        _convert_qwen4_exp(source, destination, plan, predicate, default_bits, q_mode=q_mode)
        return
    vlm_convert = _import("mlx_vlm.convert", extra="mlx-vlm")
    vlm_convert.convert(
        str(source),
        mlx_path=destination,
        quantize=True,
        q_group_size=plan.group_size,
        q_bits=default_bits,
        q_mode=q_mode,
        quant_method="rtn",
        quant_predicate=predicate,
    )


def convert_multimodal(
    source: Path,
    destination: Path,
    plan: QuantizationPlan,
    predicate: PlanPredicate,
    default_bits: int,
    q_mode: str = "affine",
) -> None:
    backend = conversion_backend(plan)
    _force_cpu_if_requested()
    try:
        if backend == "mlx-audio":
            _convert_audio(source, destination, plan, predicate, default_bits)
        elif backend == "mlx-vlm":
            _convert_vlm(source, destination, plan, predicate, default_bits, q_mode=q_mode)
        else:
            raise PlanningError("multimodal conversion called for the MLX-LM backend")
    except (ArtifactError, BackendUnavailableError, PlanningError):
        raise
    except Exception as exc:
        raise ArtifactError(f"{backend} conversion failed: {exc}") from exc
