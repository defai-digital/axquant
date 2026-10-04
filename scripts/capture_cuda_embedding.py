#!/usr/bin/env python3
"""Capture source-bound BF16 embedding inputs and independent retrieval vectors."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from capture_cuda_ocr import collect_statistics, install_hooks
from smoke_cuda_embedding import (
    EmbeddingWorkerExtension,
    create_engine,
    embedding_policy,
    retrieval_prompts,
    run_retrieval,
)

from axquant.cuda import _verify_source, plan_cuda_nvfp4_w4a4
from axquant.schema.cuda import CudaFileDigest, CudaQuantizationPlan
from axquant.schema.cuda_activation import CudaActivationCalibration
from axquant.serde import file_sha256, load_model, stable_sha256, write_data


class EmbeddingCalibrationWorkerExtension(EmbeddingWorkerExtension):
    def axquant_install(
        self: Any, source: str, allocations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return install_hooks(self.get_model(), source, allocations)

    def axquant_collect(self: Any) -> list[dict[str, Any]]:
        return collect_statistics(self.get_model())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--calibration-corpus", required=True)
    parser.add_argument("--retrieval-corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    args = parser.parse_args()
    source = Path(args.model).resolve()
    plan = load_model(args.plan, CudaQuantizationPlan)
    _verify_source(source, plan)
    calibration_path = Path(args.calibration_corpus)
    calibration_digest = file_sha256(calibration_path)
    retrieval_path = Path(args.retrieval_corpus)
    retrieval_digest = file_sha256(retrieval_path)
    calibration = json.loads(calibration_path.read_text())
    retrieval = json.loads(retrieval_path.read_text())
    calibration_texts = set(calibration["queries"] + calibration["documents"])
    if calibration_texts.intersection(retrieval["queries"] + retrieval["documents"]):
        raise ValueError("Calibration and retrieval smoke text must be separate")
    policy = embedding_policy(source)
    engine = create_engine(
        source,
        policy,
        args.memory_fraction,
        False,
        f"{Path(__file__).stem}.EmbeddingCalibrationWorkerExtension",
    )
    print(
        engine.collective_rpc(
            "axquant_install", args=(str(source), [item.model_dump() for item in plan.allocations])
        ),
        flush=True,
    )
    engine.embed(retrieval_prompts(calibration, policy, engine.get_tokenizer()), use_tqdm=False)
    workers = engine.collective_rpc("axquant_collect")
    if len(workers) != 1:
        raise ValueError("Development calibration requires one worker for complete coverage")
    capture = CudaActivationCalibration(
        weight_plan_sha256=stable_sha256(plan),
        input_files=[
            CudaFileDigest(
                path=calibration_path.name,
                sha256=calibration_digest,
                size_bytes=calibration_path.stat().st_size,
            )
        ],
        runtime="vllm",
        runtime_version=importlib.metadata.version("vllm"),
        source_precision=plan.activation_dtype,
        statistics=workers[0],
    )
    plan_cuda_nvfp4_w4a4(plan, capture)
    # Remove hooks before calibration-disjoint retrieval, so it cannot affect the calibration.
    baseline = run_retrieval(engine, source, retrieval_path, False)
    _verify_source(source, plan)
    if (
        file_sha256(calibration_path) != calibration_digest
        or file_sha256(retrieval_path) != retrieval_digest
    ):
        raise ValueError("Embedding corpora changed during capture")
    write_data(args.output, capture)
    write_data(args.baseline_output, baseline)
    print(
        f"Verified complete embedding calibration: {len(capture.statistics)} matrices", flush=True
    )


if __name__ == "__main__":
    main()
