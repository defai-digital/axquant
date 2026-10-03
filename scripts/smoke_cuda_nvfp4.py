#!/usr/bin/env python3
"""Create a deterministic tiny Qwen3 control, convert it, and run real vLLM.

This exercises checkpoint/runtime interoperability, not task quality. Runtime
processes are isolated so BF16 and NVFP4 never retain each other's GPU state.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def create_source(destination: Path) -> None:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(destination, safe_serialization=True)
    vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}
    vocabulary.update({f"token{i}": i for i in range(3, 256)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    ).save_pretrained(destination)


def infer(model: Path, result: Path, memory_fraction: float) -> None:
    import torch
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=1,
        enforce_eager=True,
        gpu_memory_utilization=memory_fraction,
        kv_cache_memory_bytes=64 * 1024 * 1024,
        enable_prefix_caching=False,
        trust_remote_code=False,
    )
    output = engine.generate(
        [{"prompt_token_ids": [1, 17, 23, 31]}],
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, detokenize=False),
    )
    tokens = list(output[0].outputs[0].token_ids)
    if len(tokens) != 8 or any(token < 0 or token >= 256 for token in tokens):
        raise RuntimeError("vLLM did not produce eight valid token IDs")
    record: dict[str, Any] = {
        "runtime": "vllm",
        "vllm_version": importlib.metadata.version("vllm"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "generated_token_ids": tokens,
        "runtime_load_and_generation": "passed",
        "model_kind": "synthetic-qwen3",
        "quality_certified": False,
    }
    result.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-source", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.02)
    args = parser.parse_args()
    if args.create_source is not None:
        create_source(args.create_source)
        return 0
    if args.source is None or args.output is None:
        parser.error("--source and --output are required")
    if args.infer_only:
        infer(args.source, args.output, args.gpu_memory_fraction)
        return 0

    from axquant.cuda import convert_cuda_nvfp4, plan_cuda_nvfp4
    from axquant.serde import stable_sha256, write_data

    args.output.mkdir(parents=True, exist_ok=True)
    plan = plan_cuda_nvfp4(args.source, allow_unmeasured=True)
    pack = args.output / "nvfp4-pack"
    manifest = convert_cuda_nvfp4(
        args.source, plan, pack, device="cuda", rows_per_chunk=1, allow_unmeasured=True
    )
    environment = {**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    records = {}
    for role, model in (("bf16", args.source), ("nvfp4", pack)):
        result = args.output / f"{role}-runtime.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--infer-only",
            "--source",
            str(model),
            "--output",
            str(result),
            "--gpu-memory-fraction",
            str(args.gpu_memory_fraction),
        ]
        with (args.output / f"{role}-runtime.log").open("w") as log:
            subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        runtime_log = (args.output / f"{role}-runtime.log").read_text()
        if "weight global scale is different for parallel layers" in runtime_log:
            raise RuntimeError("runtime rescaled fused NVFP4 weights; global scales must match")
        records[role] = json.loads(result.read_text())
        records[role]["quantization_kernels"] = sorted(
            set(re.findall(r"Using (\w*NvFp4\w*) for NVFP4 GEMM", runtime_log))
        )
        if role == "nvfp4" and not records[role]["quantization_kernels"]:
            raise RuntimeError("runtime did not identify an NVFP4 GEMM kernel")
    write_data(
        args.output / "smoke.json",
        {
            "backend": manifest.backend,
            "plan_sha256": stable_sha256(plan),
            "quantized_tensors": len(manifest.quantized_tensors),
            "weight_member_sha256": {
                member.path: member.sha256
                for member in manifest.files
                if member.path.endswith(".safetensors")
            },
            "runtimes": records,
            "scope": "synthetic checkpoint conversion, load and generation; no task quality claim",
        },
    )
    print(json.dumps({"status": "passed", "backend": manifest.backend, "runtimes": records}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
