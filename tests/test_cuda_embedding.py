from __future__ import annotations

import importlib
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file

from axquant.cuda import (
    _quantization_config,
    _verify_source,
    convert_cuda_nvfp4_w4a4,
    plan_cuda_nvfp4,
    plan_cuda_nvfp4_w4a4,
)
from axquant.errors import ArtifactError, PlanningError
from axquant.schema.cuda import CudaFileDigest
from axquant.schema.cuda_activation import CudaActivationCalibration
from axquant.serde import file_sha256, stable_sha256, write_data


@pytest.fixture
def embedding_runner(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("smoke_cuda_embedding")


@pytest.fixture(params=["qwen3", "ministral3"])
def embedding_source(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    mean = request.param == "ministral3"
    source = tmp_path / ("Nemotron-3-Embed-1B-BF16" if mean else "Qwen3-Embedding-0.6B")
    source.mkdir()
    write_data(
        source / "config.json",
        {
            "model_type": request.param,
            "num_hidden_layers": 1,
            "hidden_size": 32,
            "is_causal": not mean,
            "pooling": "avg" if mean else "last",
        },
    )
    save_file(
        {
            "model.layers.0.mlp.down_proj.weight": np.ones((32, 32), dtype=np.float32),
            "model.embed_tokens.weight": np.ones((32, 32), dtype=np.float32),
            "model.norm.weight": np.ones(32, dtype=np.float32),
        },
        source / "model.safetensors",
    )
    write_data(
        source / "modules.json",
        [
            {"path": "", "type": "sentence_transformers.models.Transformer"},
            {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
        ],
    )
    (source / "1_Pooling").mkdir()
    write_data(
        source / "1_Pooling/config.json",
        {
            "word_embedding_dimension": 32,
            "pooling_mode_mean_tokens": mean,
            "pooling_mode_lasttoken": not mean,
            "include_prompt": True,
        },
    )
    write_data(
        source / "config_sentence_transformers.json",
        {
            "prompts": {"query": "query: ", "document": "passage: "},
        },
    )
    write_data(source / "sentence_bert_config.json", {"max_seq_length": 512})
    return source


def test_embedding_metadata_is_checksum_bound(embedding_source: Path) -> None:
    plan = plan_cuda_nvfp4(embedding_source, allow_unmeasured=True)
    members = {item.path for item in plan.source_files}
    assert {
        "modules.json",
        "1_Pooling/config.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
    } <= members
    write_data(embedding_source / "1_Pooling/config.json", {"pooling_mode_cls_token": True})
    with pytest.raises(ArtifactError, match="changed"):
        _verify_source(embedding_source, plan)


def test_flat_embedding_keep_policy_covers_vllm_wrapper(embedding_source: Path) -> None:
    plan = plan_cuda_nvfp4(embedding_source, allow_unmeasured=True)
    ignored = _quantization_config(plan)["ignore"]
    assert "model.embed_tokens" in ignored
    assert "model.norm" in ignored


@pytest.mark.parametrize("prefix", ["", "model."])
def test_embedding_protection_profile_enforces_floors(
    embedding_source: Path, prefix: str, tmp_path: Path
) -> None:
    from axquant.cli import main
    from axquant.serde import read_data

    config = read_data(embedding_source / "config.json")
    config["num_hidden_layers"] = 8
    write_data(embedding_source / "config.json", config)
    save_file(
        {
            prefix + name: np.ones((32, 32), dtype=np.float32)
            for name in [
                "layers.0.mlp.down_proj.weight",
                "layers.2.mlp.down_proj.weight",
                "layers.7.mlp.down_proj.weight",
                "layers.2.self_attn.q_proj.weight",
            ]
        },
        embedding_source / "model.safetensors",
    )
    if config["model_type"] == "ministral3":
        with pytest.raises(PlanningError, match="Qwen3 embedding"):
            plan_cuda_nvfp4(embedding_source, embedding_protection=True, allow_unmeasured=True)
        return
    plan = plan_cuda_nvfp4(embedding_source, embedding_protection=True, allow_unmeasured=True)
    selected = [x.tensor_name for x in plan.allocations if x.method == "nvfp4"]
    assert selected == [prefix + "layers.2.mlp.down_proj.weight"]
    manual = plan_cuda_nvfp4(
        embedding_source, keep_patterns=plan.keep_patterns, allow_unmeasured=True
    )
    assert stable_sha256(manual) == stable_sha256(plan)
    output = tmp_path / "profile-plan.json"
    assert (
        main(
            [
                "plan-cuda",
                str(embedding_source),
                "--output",
                str(output),
                "--embedding-protection",
                "--allow-unmeasured",
            ]
        )
        == 0
    )
    assert read_data(output)["keep_patterns"] == plan.keep_patterns
    ignored = _quantization_config(plan)["ignore"]
    assert "model.layers.2.self_attn.qkv_proj" in ignored


def test_embedding_export_preserves_bound_pooling_metadata(
    embedding_source: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    plan = plan_cuda_nvfp4(embedding_source, allow_unmeasured=True)
    calibration = CudaActivationCalibration(
        weight_plan_sha256=stable_sha256(plan),
        input_files=[CudaFileDigest(path="corpus.json", sha256="a" * 64, size_bytes=10)],
        runtime="vllm",
        runtime_version="test",
        source_precision="bfloat16",
        statistics=[
            {
                "tensor_name": item.tensor_name,
                "input_columns": item.shape[1],
                "sample_count": 16,
                "absolute_maximum": 1,
            }
            for item in plan.allocations
            if item.method == "nvfp4"
        ],
    )
    output = tmp_path / "converted"
    manifest = convert_cuda_nvfp4_w4a4(
        embedding_source,
        plan_cuda_nvfp4_w4a4(plan, calibration),
        output,
        device="cpu",
        allow_unmeasured=True,
    )
    digests = {item.path: item.sha256 for item in manifest.files}
    for name in ["modules.json", "1_Pooling/config.json", "config_sentence_transformers.json"]:
        assert file_sha256(output / name) == file_sha256(embedding_source / name) == digests[name]


def test_embedding_runner_preserves_attention_and_pooling(
    embedding_source: Path, embedding_runner: Any
) -> None:
    policy = embedding_runner.embedding_policy(embedding_source)
    mean = policy["model_type"] == "ministral3"
    assert policy["pooling"] == ("MEAN" if mean else "LAST")
    assert policy["attention"] == ("encoder_only" if mean else "decoder")
    write_data(
        embedding_source / "1_Pooling/config.json",
        {
            "word_embedding_dimension": 32,
            "pooling_mode_cls_token": True,
            "include_prompt": True,
        },
    )
    with pytest.raises(ValueError, match="pooling metadata"):
        embedding_runner.embedding_policy(embedding_source)


def test_embedding_runner_rejects_causal_nemotron(
    embedding_source: Path, embedding_runner: Any
) -> None:
    if embedding_runner.embedding_policy(embedding_source)["model_type"] != "ministral3":
        return
    write_data(
        embedding_source / "config.json",
        {
            "model_type": "ministral3",
            "hidden_size": 32,
            "is_causal": True,
        },
    )
    with pytest.raises(ValueError, match="bidirectional"):
        embedding_runner.embedding_policy(embedding_source)


def test_qwen_prompt_appends_one_eos(embedding_runner: Any) -> None:
    class Tokenizer:
        eos_token_id = 10

        def convert_tokens_to_ids(self, token: str) -> int:
            assert token == "<|endoftext|>"
            return 9

        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens
            return [1, 9] if text.startswith("query") else [2]

    prompts = embedding_runner.retrieval_prompts(
        {"queries": ["test"], "documents": ["document"]},
        {"query_prefix": "query: ", "document_prefix": "", "append_eos": True},
        Tokenizer(),
    )
    assert prompts == [{"prompt_token_ids": [1, 9]}, {"prompt_token_ids": [2, 9]}]


def test_attention_verification_normalizes_runtime_enum(embedding_runner: Any) -> None:
    class AttentionType(Enum):
        DECODER = "decoder"

    layer_class = type("Attention", (), {})
    first = layer_class()
    first.attn_type = AttentionType.DECODER
    second = layer_class()
    second.attn_type = "decoder"
    model = SimpleNamespace(
        named_modules=lambda: [
            ("", SimpleNamespace(attn_type="unrelated_model_declaration")),
            ("attention", first),
            ("other_attention", second),
        ]
    )
    worker = SimpleNamespace(get_model=lambda: model)
    result = embedding_runner.EmbeddingWorkerExtension.axquant_embedding_runtime(worker, False)
    assert result["attention_types"] == ["decoder"]
    assert result["attention_layers"] == 2


@pytest.mark.parametrize("failure", ["nonfinite", "normalization", "ranking", "dimensions"])
def test_embedding_checks_reject_invalid_output(embedding_runner: Any, failure: str) -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    if failure == "nonfinite":
        vectors[0][0] = float("nan")
    elif failure == "normalization":
        vectors[0][0] = 2.0
    elif failure == "ranking":
        vectors[2], vectors[3] = vectors[3], vectors[2]
    else:
        vectors[0].append(0.0)
    with pytest.raises(RuntimeError):
        embedding_runner.check_embeddings(vectors, 2, 2)


@pytest.mark.parametrize("failure", ["source", "corpus", "semantics", "drift", "none"])
def test_reference_comparison_binds_source_and_rejects_drift(
    embedding_runner: Any, failure: str
) -> None:
    from copy import deepcopy

    reference = {
        "config_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "policy": {"pooling": "LAST", "attention": "decoder", "dimensions": 2},
        "embeddings": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        "matched_top1": [0, 1],
    }
    result = deepcopy(reference)
    if failure == "source":
        reference["config_sha256"] = "c" * 64
    elif failure == "corpus":
        result["corpus_sha256"] = "c" * 64
    elif failure == "semantics":
        result["policy"]["pooling"] = "MEAN"
    elif failure == "drift":
        result["embeddings"][0] = [0.8, 0.6]
    if failure == "none":
        assert (
            embedding_runner.compare_reference(
                result,
                reference,
                {
                    "config_sha256": "c" * 64,
                    "source_files": [{"path": "config.json", "sha256": "a" * 64}],
                },
            )["mean_cosine"]
            == 1
        )
    else:
        with pytest.raises((ValueError, RuntimeError)):
            embedding_runner.compare_reference(
                result, reference, {"source_files": [{"path": "config.json", "sha256": "a" * 64}]}
            )
