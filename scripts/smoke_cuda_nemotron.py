#!/usr/bin/env python3
"""Bounded real-checkpoint Nemotron backbone or MTP generation smoke in vLLM.

Run in an isolated GPU runtime. This produces development evidence only and
never edits the checkpoint or promotes an AXQuant manifest to a certificate.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import signal
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.85)
    args = parser.parse_args()
    if not 0 < args.gpu_memory_fraction < 1:
        parser.error("GPU memory fraction must lie strictly between zero and one")
    signal.alarm(1800)

    import torch
    from vllm import LLM, SamplingParams

    release = json.loads((args.model / "axquant_nemotron_release.json").read_text())
    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "nemotron_h" or release.get("format") != "nvfp4":
        raise RuntimeError("smoke requires an AXQuant native Nemotron NVFP4 checkpoint")
    options: dict[str, Any] = {
        "model": str(args.model),
        "dtype": "bfloat16",
        "max_model_len": 256,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 256,
        "enforce_eager": True,
        "gpu_memory_utilization": args.gpu_memory_fraction,
        "kv_cache_memory_bytes": 64 * 1024 * 1024,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "trust_remote_code": True,
        "kernel_config": {"linear_backend": "marlin", "moe_backend": "marlin"},
    }
    if args.mtp:
        options["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": 1,
            "moe_backend": "triton",
        }
    model = LLM(**options)
    configured_mtp = model.llm_engine.vllm_config.speculative_config is not None
    if args.mtp and not configured_mtp:
        raise RuntimeError("vLLM did not configure the requested MTP execution path")
    tokenizer = model.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    responses = model.generate(
        [prompt], SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    )
    tokens = list(responses[0].outputs[0].token_ids)
    if len(tokens) != 8 or any(token < 0 or token >= config["vocab_size"] for token in tokens):
        raise RuntimeError("Nemotron smoke did not return eight valid token IDs")
    record = {
        "status": "passed",
        "repo_id": release["repo"],
        "source_revision": release["source_revision"],
        "manifest_sha256": release["manifest_sha256"],
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "vllm_version": importlib.metadata.version("vllm"),
        "mtp_requested": args.mtp,
        "mtp_runtime_configured": configured_mtp,
        "generated_token_ids": tokens,
        "scope": "exact checkpoint load and generation smoke; no quality or speed certificate",
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
