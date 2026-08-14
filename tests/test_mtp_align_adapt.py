"""Stage-1 adapt-fc, provenance, and compose on the real shipped path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axquant.grafted_mtp import compose_grafted_mtp_onto_pack
from axquant.mtp_align.adapt_fc import (
    TRAIN_KEYS,
    adapt_fc_norms_from_features,
    compose_adapted_onto_pack,
    write_adapted_mtp_bundle,
)
from axquant.mtp_align.dataset import read_samples, write_samples
from axquant.mtp_align.provenance import ADAPTED_GRAFT_KIND, write_adapted_graft_record
from axquant.mtp_align.qwen_mtp_head import QwenMtpHead, QwenMtpHeadConfig
from axquant.serde import file_sha256


def _tiny_head():
    mx = pytest.importorskip("mlx.core")
    cfg = QwenMtpHeadConfig(
        hidden_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
    )
    h, e, i = cfg.hidden_size, cfg.num_experts, cfg.moe_intermediate_size
    w = {
        "mtp.fc.weight": mx.random.normal((h, 2 * h)) * 0.02,
        "mtp.layers.0.input_layernorm.weight": mx.ones((h,)),
        "mtp.layers.0.mlp.experts.down_proj": mx.random.normal((e, h, i)) * 0.02,
        "mtp.layers.0.mlp.experts.gate_up_proj": mx.random.normal((e, 2 * i, h)) * 0.02,
        "mtp.layers.0.mlp.gate.weight": mx.random.normal((e, h)) * 0.02,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": mx.random.normal((h, i)) * 0.02,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": mx.random.normal((i, h)) * 0.02,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": mx.random.normal((i, h)) * 0.02,
        "mtp.layers.0.mlp.shared_expert_gate.weight": mx.random.normal((1, h)) * 0.02,
        "mtp.layers.0.post_attention_layernorm.weight": mx.ones((h,)),
        "mtp.layers.0.self_attn.k_norm.weight": mx.ones((cfg.head_dim,)),
        "mtp.layers.0.self_attn.k_proj.weight": mx.random.normal(
            (cfg.num_key_value_heads * cfg.head_dim, h)
        )
        * 0.02,
        "mtp.layers.0.self_attn.o_proj.weight": mx.random.normal(
            (h, cfg.num_attention_heads * cfg.head_dim)
        )
        * 0.02,
        "mtp.layers.0.self_attn.q_norm.weight": mx.ones((cfg.head_dim,)),
        "mtp.layers.0.self_attn.q_proj.weight": mx.random.normal(
            (cfg.num_attention_heads * cfg.head_dim * 2, h)
        )
        * 0.02,
        "mtp.layers.0.self_attn.v_proj.weight": mx.random.normal(
            (cfg.num_key_value_heads * cfg.head_dim, h)
        )
        * 0.02,
        "mtp.norm.weight": mx.ones((h,)),
        "mtp.pre_fc_norm_embedding.weight": mx.ones((h,)),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((h,)),
    }
    return QwenMtpHead.from_weight_dict(w, config=cfg), cfg, mx


def test_adapt_fc_from_features_writes_provenance_and_sidecar(tmp_path: Path) -> None:
    head, cfg, mx = _tiny_head()
    lm_head = mx.random.normal((cfg.vocab_size, cfg.hidden_size)) * 0.02
    features = [
        {
            "hidden": mx.random.normal((cfg.hidden_size,)),
            "prev_embed": mx.random.normal((cfg.hidden_size,)),
            "label_token": 3,
        },
        {
            "hidden": mx.random.normal((cfg.hidden_size,)),
            "prev_embed": mx.random.normal((cfg.hidden_size,)),
            "label_token": 7,
        },
    ]
    before = {k: mx.array(head.weights[k]) for k in TRAIN_KEYS}
    trained, history = adapt_fc_norms_from_features(
        head, features, lm_head, steps=5, learning_rate=1e-2
    )
    assert len(history) == 5
    # At least one trainable tensor must move (real SGD path).
    moved = any(
        float(mx.sum(mx.abs(trained.weights[k] - before[k])).item()) > 0.0 for k in TRAIN_KEYS
    )
    assert moved

    init_path = tmp_path / "init.safetensors"
    head_init, _, _ = _tiny_head()
    head_init.save_safetensors(init_path)
    from axquant.mtp_align.provenance import sidecar_sha256

    bundle_dir = tmp_path / "bundle"
    result = write_adapted_mtp_bundle(
        trained,
        bundle_dir,
        init_mtp_sha256=sidecar_sha256(init_path),
        train_summary={
            "stage": "fc_norms",
            "steps": 5,
            "loss_history": history,
            "trainable": list(TRAIN_KEYS),
        },
        trunk_model_id="Hcompany/Holo3-35B-A3B",
        trunk_revision="208d5ae3a03f99d561f32ab5e606f73397a390ea",
        donor_model_id="Qwen/Qwen3.5-35B-A3B",
        donor_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
    )
    assert Path(result["mtp"]).is_file()
    graft = json.loads(Path(result["graft_record"]).read_text(encoding="utf-8"))
    assert graft["graft_kind"] == ADAPTED_GRAFT_KIND
    assert "not full co-training" in graft["notes"][0].lower()
    assert "Acceleration claims still require Tier 2" in graft["notes"][1]


def test_compose_adapted_does_not_mutate_main_weights(tmp_path: Path) -> None:
    head, _cfg, mx = _tiny_head()
    mtp_dir = tmp_path / "mtp"
    mtp_dir.mkdir()
    head.save_safetensors(mtp_dir / "mtp.safetensors")
    (mtp_dir / "axquant_mtp_sidecar_manifest.json").write_text("{}", encoding="utf-8")
    write_adapted_graft_record(
        mtp_dir,
        trunk_model_id="Hcompany/Holo3-35B-A3B",
        trunk_revision="rev",
        donor_model_id="Qwen/Qwen3.5-35B-A3B",
        donor_revision="drev",
        init_mtp_sha256="a" * 64,
        output_mtp_sha256="b" * 64,
        train_summary={"stage": "fc_norms"},
    )

    pack = tmp_path / "pack"
    pack.mkdir()
    main = pack / "model.safetensors"
    main.write_bytes(b"trunk-main-weights-v1")
    (pack / "axquant_manifest.json").write_text(
        json.dumps({"mtp_present": False}), encoding="utf-8"
    )
    (pack / "axquant_runtime.json").write_text(
        json.dumps({"mtp": {"detected": False}}), encoding="utf-8"
    )
    before_sha = file_sha256(main)

    out = tmp_path / "composed"
    composed = compose_adapted_onto_pack(pack, mtp_dir, output_dir=out)
    assert composed == out.resolve()
    assert file_sha256(out / "model.safetensors") == before_sha
    assert (out / "mtp.safetensors").is_file()
    assert (out / "axquant_mtp_graft.json").is_file()
    # Source pack untouched.
    assert not (pack / "mtp.safetensors").exists()
    assert file_sha256(main) == before_sha
    # compose_grafted_mtp path is the same function.
    out2 = tmp_path / "composed2"
    compose_grafted_mtp_onto_pack(pack, mtp_dir, output_dir=out2)
    assert file_sha256(out2 / "model.safetensors") == before_sha


def test_dataset_write_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    write_samples(
        path,
        [
            {
                "schema_version": "axquant.mtp-align-sample.v1",
                "input_ids": [1, 2, 3],
                "prev_token": 3,
                "label_token": 4,
                "label_source": "trunk_greedy",
            }
        ],
    )
    samples = read_samples(path)
    assert len(samples) == 1
    assert samples[0]["label_token"] == 4
