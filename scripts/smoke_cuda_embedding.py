#!/usr/bin/env python3
"""Verify CUDA embedding semantics, calibration-disjoint retrieval and native NVFP4 kernels."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any

from smoke_cuda_ocr import (
    file_sha256,
    inspect_native_fp4,
    require_kernel_coverage,
    write_result,
)


def embedding_policy(source: Path) -> dict[str, Any]:
    config = json.loads((source / "config.json").read_text())
    model_type = config.get("model_type")
    if model_type not in {"qwen3", "ministral3"}:
        raise ValueError("This development runner supports Qwen3 and Nemotron-3-Embed")
    pooling = json.loads((source / "1_Pooling/config.json").read_text())
    prompts = json.loads((source / "config_sentence_transformers.json").read_text())["prompts"]
    mean = model_type == "ministral3"
    if mean and config.get("is_causal") is not False:
        raise ValueError("Nemotron embedding requires original bidirectional attention")
    enabled = [
        name for name, value in pooling.items() if name.startswith("pooling_mode_") and value
    ]
    expected = "pooling_mode_mean_tokens" if mean else "pooling_mode_lasttoken"
    if enabled != [expected] or pooling.get("include_prompt") is not True:
        raise ValueError("Embedding pooling metadata differs from the supported source")
    if pooling["word_embedding_dimension"] != config["hidden_size"]:
        raise ValueError("Embedding pooling dimensions differ from the source")
    modules = json.loads((source / "modules.json").read_text())
    if [(item["path"], item["type"]) for item in modules] != [
        ("", "sentence_transformers.models.Transformer"),
        ("1_Pooling", "sentence_transformers.models.Pooling"),
        ("2_Normalize", "sentence_transformers.models.Normalize"),
    ]:
        raise ValueError("Unsupported embedding modules or missing normalization")
    return {
        "pooling": "MEAN" if mean else "LAST",
        "attention": "encoder_only" if mean else "decoder",
        "dimensions": config["hidden_size"],
        "layers": config["num_hidden_layers"],
        "query_prefix": prompts["query"],
        "document_prefix": prompts["document"],
        "append_eos": not mean,
        "model_type": model_type,
    }


def retrieval_prompts(corpus: dict[str, Any], policy: dict[str, Any], tokenizer: Any) -> list[Any]:
    queries = corpus["queries"]
    documents = corpus["documents"]
    if not queries or len(queries) != len(documents):
        raise ValueError("Retrieval smoke requires one matching document for every query")
    texts = [policy["query_prefix"] + text for text in queries] + [
        policy["document_prefix"] + text for text in documents
    ]
    result = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if policy["append_eos"]:
            eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if eos is None or eos == getattr(tokenizer, "unk_token_id", None):
                raise ValueError("Qwen embedding tokenizer requires its end-of-document token")
            if not tokens or tokens[-1] != eos:
                tokens.append(eos)
        result.append({"prompt_token_ids": tokens})
    return result


def check_embeddings(vectors: list[list[float]], dimensions: int, pairs: int) -> dict[str, Any]:
    if len(vectors) != 2 * pairs or any(len(row) != dimensions for row in vectors):
        raise RuntimeError("Embedding count or dimensions differ from the retrieval corpus")
    norms = []
    for row in vectors:
        if not all(math.isfinite(value) for value in row):
            raise RuntimeError("Embedding contains non-finite values")
        norm = math.sqrt(sum(value * value for value in row))
        if abs(norm - 1) > 0.02:
            raise RuntimeError("Embedding is not L2-normalized")
        norms.append(norm)
    scores = [
        [sum(a * b for a, b in zip(query, document, strict=True)) for document in vectors[pairs:]]
        for query in vectors[:pairs]
    ]
    ranking = [max(range(pairs), key=lambda index: row[index]) for row in scores]
    if ranking != list(range(pairs)):
        raise RuntimeError(f"Paired retrieval smoke failed: {ranking}")
    return {"norms": norms, "similarity_scores": scores, "matched_top1": ranking}


def compare_reference(
    result: dict[str, Any], reference: dict[str, Any], weight_plan: dict[str, Any]
) -> dict[str, Any]:
    source_config_sha256 = next(
        item["sha256"] for item in weight_plan["source_files"] if item["path"] == "config.json"
    )
    if reference["config_sha256"] != source_config_sha256:
        raise ValueError("Reference embedding config is not the bound original source")
    if reference["corpus_sha256"] != result["corpus_sha256"]:
        raise ValueError("Reference retrieval corpus differs from the quantized run")
    if any(
        reference["policy"][key] != result["policy"][key]
        for key in ("pooling", "attention", "dimensions")
    ):
        raise ValueError("Reference embedding semantics differ from the quantized run")
    pairs = len(result["matched_top1"])
    check_embeddings(result["embeddings"], result["policy"]["dimensions"], pairs)
    check_embeddings(reference["embeddings"], result["policy"]["dimensions"], pairs)
    cosines = []
    for actual, original in zip(result["embeddings"], reference["embeddings"], strict=True):
        denominator = math.sqrt(sum(x * x for x in actual) * sum(x * x for x in original))
        cosines.append(sum(x * y for x, y in zip(actual, original, strict=True)) / denominator)
    mean = sum(cosines) / len(cosines)
    if min(cosines) < 0.90 or mean < 0.95:
        raise RuntimeError("Quantized embedding cosine drift exceeds development smoke limits")
    return {
        "cosines": cosines,
        "minimum_cosine": min(cosines),
        "mean_cosine": mean,
        "reference_config_sha256": reference["config_sha256"],
        "scope": "Eight vectors, four paired queries; not a broad quality qualification",
    }


class EmbeddingWorkerExtension:
    def axquant_embedding_runtime(self: Any, native: bool) -> dict[str, Any]:
        model = self.get_model()
        attention_layers = [
            module
            for _, module in model.named_modules()
            if type(module).__name__ in {"Attention", "EncoderOnlyAttention"}
        ]
        attention = sorted(
            {
                str(getattr(module.attn_type, "value", module.attn_type))
                for module in attention_layers
            }
        )
        return {
            "attention_types": attention,
            "attention_layers": len(attention_layers),
            "kernels": inspect_native_fp4(model) if native else {},
        }


def create_engine(
    source: Path,
    policy: dict[str, Any],
    memory_fraction: float,
    native: bool,
    worker_extension: str,
) -> Any:
    for name in ("VLLM_BUILD_COMMIT", "VLLM_BUILD_PIPELINE", "VLLM_BUILD_URL", "VLLM_IMAGE_TAG"):
        os.environ.pop(name, None)
    from vllm import LLM

    return LLM(
        model=str(source),
        model_impl="vllm",
        runner="pooling",
        convert="embed",
        dtype="bfloat16",
        trust_remote_code=False,
        pooler_config={"pooling_type": policy["pooling"], "use_activation": True},
        worker_extension_cls=worker_extension,
        hf_overrides={"architectures": ["Ministral3ForCausalLM"]}
        if policy["model_type"] == "ministral3"
        else {},
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        gpu_memory_utilization=memory_fraction,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        attention_config={"backend": "TRITON_ATTN"},
        compilation_config={"custom_ops": ["none"]},
        kernel_config={"linear_backend": "cutlass"} if native else {},
    )


def run_retrieval(engine: Any, source: Path, corpus_path: Path, native: bool) -> dict[str, Any]:
    import torch

    policy = embedding_policy(source)
    corpus = json.loads(corpus_path.read_text())
    prompts = retrieval_prompts(corpus, policy, engine.get_tokenizer())
    runtime = engine.collective_rpc("axquant_embedding_runtime", args=(native,))
    if (
        len(runtime) != 1
        or runtime[0]["attention_types"] != [policy["attention"]]
        or runtime[0]["attention_layers"] != policy["layers"]
    ):
        raise RuntimeError(f"Runtime attention differs from source embedding semantics: {runtime}")
    if native:
        plan = json.loads((source / "axquant_cuda_plan.json").read_text())
        require_kernel_coverage(plan, runtime[0]["kernels"])
    outputs = engine.embed(prompts, use_tqdm=False)
    vectors = [output.outputs.embedding for output in outputs]
    checks = check_embeddings(vectors, policy["dimensions"], len(corpus["queries"]))
    return {
        "status": "development-smoke",
        "quality_certified": False,
        "scope": "Calibration-disjoint retrieval smoke; no broad quality or speed certification",
        "runtime": "vllm",
        "runtime_version": importlib.metadata.version("vllm"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "config_sha256": file_sha256(source / "config.json"),
        "corpus_sha256": file_sha256(corpus_path),
        "native_fp4_required": native,
        "policy": policy,
        "runtime_verification": runtime[0],
        "embeddings": vectors,
        **checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    parser.add_argument("--require-native-fp4", action="store_true")
    parser.add_argument("--reference", help="Optional exact-source BF16 retrieval result JSON")
    args = parser.parse_args()
    source = Path(args.model)
    if args.require_native_fp4:
        config = json.loads((source / "config.json").read_text())
        activation = (
            (config.get("quantization_config") or {})
            .get("config_groups", {})
            .get("nvfp4", {})
            .get("input_activations")
        )
        if not activation or activation.get("num_bits") != 4:
            raise ValueError("Native FP4 execution requires a calibrated W4A4 checkpoint")
    engine = create_engine(
        source,
        embedding_policy(source),
        args.memory_fraction,
        args.require_native_fp4,
        f"{Path(__file__).stem}.EmbeddingWorkerExtension",
    )
    result = run_retrieval(engine, source, Path(args.corpus), args.require_native_fp4)
    if args.reference:
        if not args.require_native_fp4:
            raise ValueError("Reference comparison requires the bound native W4A4 plan")
        reference = json.loads(Path(args.reference).read_text())
        plan = json.loads((source / "axquant_cuda_plan.json").read_text())
        result["reference_comparison"] = compare_reference(result, reference, plan["weight_plan"])
    write_result(Path(args.output), result)
    print("Verified normalized embeddings and calibration-disjoint paired retrieval", flush=True)


if __name__ == "__main__":
    main()
