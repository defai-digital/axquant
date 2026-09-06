"""Layer-wise MLX-LM convert that never holds the full quantized model.

Stock ``mlx_lm.convert`` lazy-loads, then ``quantize_model`` builds the full
quantized graph and ``save_model`` flattens every parameter before writing.
Ornith-1.5-397B MXFP4 is ~216 GiB resident — above a 192 GB Studio — so convert
SIGKILLs while writing shard 2.

This path quantizes one leaf module, ``mx.eval``s it, appends it to the current
shard, donates the module weights, and flushes the shard at 5 GiB. Peak RAM is
one expert stack plus the open shard, not the whole pack.
"""

from __future__ import annotations

import copy
import gc
import glob
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

_LOG = structlog.get_logger()

STREAMING_CONVERT_ENV = "AXQUANT_STREAMING_CONVERT"
_TRUTHY = frozenset({"1", "true", "yes", "on", "always"})
_FALSY = frozenset({"0", "false", "no", "off", "never"})
_MODEL_CONVERSION_DTYPES = ("float16", "bfloat16", "float32")
# Match mlx_lm.utils.MAX_FILE_SIZE_GB.
_MAX_SHARD_BYTES = 5 << 30


def _env_flag(name: str) -> str:
    return os.environ.get(name, "").strip().lower()


def physical_memory_bytes() -> int:
    """Return installed unified/physical memory, or 0 when unknown."""

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0
    if pages <= 0 or page <= 0:
        return 0
    return int(pages) * int(page)


def source_weight_bytes(model_ref: str | Path) -> int:
    """Sum ``model*.safetensors`` payload bytes, excluding the MTP sidecar file."""

    path = Path(model_ref).expanduser()
    if not path.is_dir():
        return 0
    total = 0
    for shard in path.glob("model*.safetensors"):
        if shard.name == "model-mtp.safetensors":
            continue
        try:
            total += shard.stat().st_size
        except OSError:
            continue
    return total


def streaming_convert_enabled(model_ref: str | Path | None = None) -> bool:
    """Whether convert should stream instead of calling stock ``mlx_lm.convert``.

    ``AXQUANT_STREAMING_CONVERT=1`` forces on, ``=0`` forces off. Unset is auto:
    stream when the BF16 payload is larger than installed RAM (397B on 192 GB).
    """

    flag = _env_flag(STREAMING_CONVERT_ENV)
    if flag in _TRUTHY:
        return True
    if flag in _FALSY:
        return False
    if model_ref is None:
        return False
    source_bytes = source_weight_bytes(model_ref)
    memory = physical_memory_bytes()
    if source_bytes <= 0 or memory <= 0:
        return False
    return source_bytes > memory


def _clear_mlx_cache(mx: Any) -> None:
    clearer = getattr(mx, "clear_cache", None)
    if callable(clearer):
        clearer()


def _empty_tree(mx: Any, tree_map: Callable[..., Any], parameters: Any) -> Any:
    empty = mx.array([])
    return tree_map(lambda _: empty, parameters)


def _logical_parameters(module: Any, tree_flatten: Callable[..., Any]) -> int:
    if hasattr(module, "bits") and hasattr(module, "weight"):
        n = int(module.bias.size) if hasattr(module, "bias") else 0
        return n + int(module.weight.size) * 32 // int(module.bits)
    return sum(int(value.size) for _, value in tree_flatten(module.parameters()))


def _defaults_for_mode(mode: str, group_size: int | None, bits: int | None) -> tuple[int, int]:
    mode_defaults = {
        "affine": (64, 4),
        "mxfp4": (32, 4),
        "nvfp4": (16, 4),
        "mxfp8": (32, 8),
    }
    default_group_size, default_bits = mode_defaults.get(mode, (64, 4))
    return group_size or default_group_size, bits or default_bits


def _shard_name(index: int, count: int) -> str:
    if count <= 1:
        return "model.safetensors"
    return f"model-{index:05d}-of-{count:05d}.safetensors"


def quantize_and_save_streaming(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    dst_path: str | Path,
    src_path: str | Path,
    *,
    quant_predicate: Callable[..., Any] | None,
    q_group_size: int | None,
    q_bits: int | None,
    q_mode: str = "affine",
    hf_repo: str | None = None,
) -> None:
    """Quantize ``model`` one leaf at a time and write MLX shards under ``dst_path``."""

    mx = __import__("mlx.core", fromlist=["core"])
    nn = __import__("mlx.nn", fromlist=["nn"])
    mlx_utils = __import__("mlx.utils", fromlist=["utils"])
    mlx_lm_utils = __import__("mlx_lm.utils", fromlist=["utils"])
    tree_flatten = mlx_utils.tree_flatten
    tree_map = mlx_utils.tree_map
    tree_map_with_path = mlx_utils.tree_map_with_path

    dst = Path(dst_path)
    dst.mkdir(parents=True, exist_ok=True)
    src = Path(src_path)

    dtype_name = config.get("torch_dtype")
    if dtype_name is None:
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            dtype_name = text_config.get("dtype")
    target_dtype = getattr(mx, dtype_name) if dtype_name in _MODEL_CONVERSION_DTYPES else None
    if target_dtype is not None:
        print(f"[INFO] Using dtype: {dtype_name}", flush=True)
    cast_predicate = getattr(model, "cast_predicate", lambda _: True)

    quantized_config = copy.deepcopy(config)
    group_size, bits = _defaults_for_mode(q_mode, q_group_size, q_bits)
    quant_params = {"group_size": group_size, "bits": bits, "mode": q_mode}
    fine_grained = "quantization" in quantized_config
    if not fine_grained:
        quantized_config["quantization"] = dict(quant_params)

    def wrapped_predicate(path: str, module: Any) -> bool | dict[str, Any]:
        if not hasattr(module, "to_quantized"):
            return False
        last_dim = int(module.weight.shape[-1])
        if last_dim % group_size != 0:
            return False
        bool_or_params: bool | dict[str, Any] = True
        if quant_predicate is not None:
            bool_or_params = quant_predicate(path, module)
        if isinstance(bool_or_params, dict):
            quantized_config["quantization"][path] = bool_or_params
        elif fine_grained and bool_or_params:
            quantized_config["quantization"][path] = quant_params
        return bool_or_params

    leaves = list(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    print(f"[INFO] Streaming quantize {len(leaves)} leaf modules", flush=True)

    shard: dict[str, Any] = {}
    shard_size = 0
    written: list[tuple[Path, list[str], int]] = []
    total_nbytes = 0
    total_params = 0

    def flush_shard() -> None:
        nonlocal shard, shard_size
        if not shard:
            return
        index = len(written) + 1
        tmp_path = dst / f".shard-{index:05d}.safetensors"
        mx.save_safetensors(str(tmp_path), shard, metadata={"format": "mlx"})
        names = list(shard.keys())
        size = shard_size
        written.append((tmp_path, names, size))
        print(
            f"[INFO] Wrote streaming shard {index} "
            f"({size / (1 << 30):.2f} GiB, {len(names)} tensors)",
            flush=True,
        )
        shard = {}
        shard_size = 0
        gc.collect()
        _clear_mlx_cache(mx)

    for index, (path, module) in enumerate(leaves, start=1):
        if target_dtype is not None:

            def set_dtype(key: str, value: Any, *, _path: str = path) -> Any:
                full = f"{_path}.{key}" if _path else key
                if cast_predicate(full) and mx.issubdtype(value.dtype, mx.floating):
                    if value.dtype == target_dtype:
                        return value
                    return value.astype(target_dtype)
                return value

            module.update(tree_map_with_path(set_dtype, module.parameters()))

        decision = wrapped_predicate(path, module)
        working = module
        if decision:
            kwargs = (
                dict(decision)
                if isinstance(decision, dict)
                else {"group_size": group_size, "bits": bits, "mode": q_mode}
            )
            working = module.to_quantized(**kwargs)

        parameters = list(tree_flatten(working.parameters()))
        arrays = [value for _, value in parameters]
        if arrays:
            mx.eval(*arrays)
        total_params += _logical_parameters(working, tree_flatten)

        for name, value in parameters:
            key = f"{path}.{name}" if path else name
            nbytes = int(value.nbytes)
            if shard and shard_size + nbytes > _MAX_SHARD_BYTES:
                flush_shard()
            shard[key] = value
            shard_size += nbytes
            total_nbytes += nbytes

        module.update(_empty_tree(mx, tree_map, module.parameters()))
        if working is not module:
            working.update(_empty_tree(mx, tree_map, working.parameters()))
            del working
        del parameters, arrays
        if index == 1 or index == len(leaves) or index % 25 == 0:
            _LOG.info(
                "streaming_convert_progress",
                module=path,
                index=index,
                total=len(leaves),
                shards=len(written) + (1 if shard else 0),
            )
            print(f"[INFO] Streaming {index}/{len(leaves)} {path}", flush=True)
        if index % 5 == 0:
            gc.collect()
            _clear_mlx_cache(mx)

    flush_shard()
    del model
    gc.collect()
    _clear_mlx_cache(mx)

    count = len(written)
    weight_map: dict[str, str] = {}
    for index, (tmp_path, names, _size) in enumerate(written, start=1):
        final_name = _shard_name(index, count)
        tmp_path.rename(dst / final_name)
        for name in names:
            weight_map[name] = final_name

    quantized_config["quantization_config"] = quantized_config["quantization"]
    index_data = {
        "metadata": {
            "total_size": total_nbytes,
            "total_parameters": total_params,
        },
        "weight_map": {key: weight_map[key] for key in sorted(weight_map)},
    }
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(index_data, indent=4) + "\n",
        encoding="utf-8",
    )
    mlx_lm_utils.save_config(quantized_config, config_path=dst / "config.json")
    tokenizer.save_pretrained(dst)
    for pattern in ("*.py", "generation_config.json"):
        for file in glob.glob(str(src / pattern)):
            shutil.copy(file, dst)
    mlx_lm_utils.create_model_card(dst, hf_repo)
    bpw = (8.0 * total_nbytes / total_params) if total_params else 0.0
    print(f"[INFO] Quantized model with {bpw:.3f} bits per weight.", flush=True)
    _LOG.info(
        "streaming_convert_completed",
        shards=count,
        bytes=total_nbytes,
        parameters=total_params,
        bpw=bpw,
    )


def streaming_mlx_convert(
    model_ref: str,
    *,
    mlx_path: str,
    quantize: bool,
    q_group_size: int,
    q_bits: int,
    quant_predicate: Any,
    revision: str | None,
    q_mode: str = "affine",
) -> None:
    """Drop-in for ``mlx_lm.convert`` that streams quantized shards to disk."""

    if not quantize:
        raise ValueError("streaming convert requires quantize=True")
    mlx_path_obj = Path(mlx_path)
    if mlx_path_obj.exists():
        raise ValueError(
            f"Cannot save to the path {mlx_path} as it already exists. "
            "Please delete the file/directory or specify a new path to save to."
        )
    _LOG.info("streaming_convert_started", model=model_ref, output=mlx_path)
    print("[INFO] Loading (streaming convert)", flush=True)
    mlx_lm_utils = __import__("mlx_lm.utils", fromlist=["utils"])
    load = mlx_lm_utils.load
    model, tokenizer, config = load(
        model_ref,
        revision=revision,
        return_config=True,
        tokenizer_config={"trust_remote_code": False},
        lazy=True,
    )
    src_path = Path(model_ref)
    hf_repo = None
    if not src_path.exists():
        hf_repo = model_ref
        src_path = mlx_lm_utils.hf_repo_to_path(hf_repo)
    quantize_and_save_streaming(
        model,
        tokenizer,
        config,
        mlx_path_obj,
        src_path,
        quant_predicate=quant_predicate,
        q_group_size=q_group_size,
        q_bits=q_bits,
        q_mode=q_mode,
        hf_repo=hf_repo,
    )
    del model, tokenizer
    gc.collect()
    mx = __import__("mlx.core", fromlist=["core"])
    _clear_mlx_cache(mx)
