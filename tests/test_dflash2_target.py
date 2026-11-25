"""AXQ-067 DF-1 / DF-1b: DFlash2 target identity, declaration and publication gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from safetensors.numpy import save_file

from axquant.cli._parser import _build_parser as build_parser
from axquant.dflash2_target import (
    DECLARATION_NAME,
    admitted_targets,
    build_declaration,
    classify_source,
    format_lane,
    is_hub_identity,
    matching_target,
    pack_format_lane,
    predicted_format_lane,
    require_admitted_lane,
    require_publication_declaration,
)
from axquant.errors import ArtifactError, PlanningError, ValidationGateError
from axquant.schema import QuantMethod
from axquant.schema.dflash2 import (
    Dflash2AdmittedTargets,
    Dflash2SharedTensor,
    Dflash2TargetDeclaration,
)
from axquant.serde import write_data

REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def _config_27b(**text: Any) -> dict[str, Any]:
    text_config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "vocab_size": 248320,
        **text,
    }
    return {"model_type": "qwen3_5", "text_config": text_config}


def _config_35b() -> dict[str, Any]:
    return {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "vocab_size": 248320,
            "num_experts": 256,
        },
    }


def _write_pack(
    root: Path,
    *,
    trunk_mode: str = "mxfp4",
    trunk_bits: int = 4,
    trunk_group: int = 32,
    with_embedding: bool = True,
) -> Path:
    root.mkdir(parents=True)
    quantization: dict[str, Any] = {
        "bits": 4,
        "group_size": 32,
        "mode": "affine",
        "language_model.model.embed_tokens": {"bits": 8, "group_size": 64, "mode": "affine"},
        "language_model.lm_head": {"bits": 8, "group_size": 64, "mode": "affine"},
        "language_model.model.layers.0.mlp.down_proj": {
            "bits": trunk_bits,
            "group_size": trunk_group,
            "mode": trunk_mode,
        },
    }
    config = {**_config_27b(), "quantization": quantization}
    (root / "config.json").write_text(json.dumps(config))
    rng = np.random.default_rng(0)
    tensors = {
        "language_model.lm_head.weight": rng.integers(0, 2**31, (4, 8), dtype=np.uint32),
        "language_model.lm_head.scales": np.ones((4, 2), dtype=np.float16),
        "language_model.lm_head.biases": np.zeros((4, 2), dtype=np.float16),
        "language_model.model.layers.0.mlp.down_proj.weight": rng.integers(
            0, 2**31, (16, 16), dtype=np.uint32
        ),
        "language_model.model.layers.0.mlp.down_proj.scales": np.ones((16, 2), dtype=np.uint8),
    }
    if with_embedding:
        tensors.update(
            {
                "language_model.model.embed_tokens.weight": rng.integers(
                    0, 2**31, (4, 8), dtype=np.uint32
                ),
                "language_model.model.embed_tokens.scales": np.ones((4, 2), dtype=np.float16),
                "language_model.model.embed_tokens.biases": np.zeros((4, 2), dtype=np.float16),
            }
        )
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(root / shard))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, shard)})
    )
    return root


def _bind(root: Path, *, model_id: str, revision: str | None) -> None:
    declaration = root / DECLARATION_NAME
    files = []
    if declaration.is_file():
        files.append(
            {
                "path": DECLARATION_NAME,
                "size_bytes": declaration.stat().st_size,
                "sha256": hashlib.sha256(declaration.read_bytes()).hexdigest(),
            }
        )
    manifest = {"source_model": {"model_id": model_id, "revision": revision}, "files": files}
    (root / "axquant_manifest.json").write_text(json.dumps(manifest))


def _declare(root: Path, *, model_id: str, revision: str | None, reason: str | None = None):
    classification = classify_source(
        _config_27b(), model_id=model_id, revision=revision, not_target_reason=reason
    )
    assert classification is not None
    declaration = build_declaration(root, classification, target_num_hidden_layers=64)
    write_data(root / DECLARATION_NAME, declaration)
    _bind(root, model_id=model_id, revision=revision)
    return declaration


def test_mirror_matches_ax_engine_admitted_drafters() -> None:
    mirror = admitted_targets()
    by_source = {target.source_model_id: target for target in mirror.targets}
    dense = by_source["Qwen/Qwen3.8-27B"]
    assert dense.drafter_repo == "z-lab/Qwen3.8-27B-DFlash2"
    assert dense.drafter_revision == "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
    assert dense.tap_layer_ids == [5, 19, 33, 47, 61]
    assert dense.admitted_format_lanes == ["mxfp4", "mxfp8", "affine6_g64"]
    moe = by_source["Qwen/Qwen3.6-35B-A3B"]
    assert moe.drafter_repo == "incoai/Qwen3.6-35B-A3B-DFlash2"
    assert moe.tap_layer_ids == [1, 6, 11, 16, 22, 27, 32, 37]
    assert mirror.ax_engine_commit == "929e1f135"


def test_mirror_schema_fails_closed() -> None:
    payload = admitted_targets().model_dump(mode="json")
    with pytest.raises(ValidationError):
        Dflash2AdmittedTargets.model_validate({**payload, "extra": 1})
    broken = json.loads(json.dumps(payload))
    broken["targets"][0]["tap_layer_ids"] = [19, 5]
    with pytest.raises(ValidationError):
        Dflash2AdmittedTargets.model_validate(broken)
    broken = json.loads(json.dumps(payload))
    broken["targets"][0]["tap_layer_ids"] = [5, 64]
    with pytest.raises(ValidationError):
        Dflash2AdmittedTargets.model_validate(broken)
    duplicate = json.loads(json.dumps(payload))
    duplicate["targets"].append(duplicate["targets"][0])
    with pytest.raises(ValidationError):
        Dflash2AdmittedTargets.model_validate(duplicate)
    collision = json.loads(json.dumps(payload))
    clone = json.loads(json.dumps(collision["targets"][0]))
    clone["source_model_id"] = "Other/Same-Shape"
    collision["targets"].append(clone)
    with pytest.raises(ValidationError, match="distinguishable"):
        Dflash2AdmittedTargets.model_validate(collision)


def test_signature_matching() -> None:
    assert matching_target(_config_27b()).source_model_id == "Qwen/Qwen3.8-27B"  # type: ignore[union-attr]
    assert matching_target(_config_35b()).source_model_id == "Qwen/Qwen3.6-35B-A3B"  # type: ignore[union-attr]
    assert matching_target(_config_27b(vocab_size=151936)) is None
    assert matching_target(_config_27b(num_experts=8)) is None
    flash_next = {"model_type": "qwen4_exp_text", "hidden_size": 5120, "num_hidden_layers": 48}
    assert matching_target(flash_next) is None


def test_hub_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert is_hub_identity("Qwen/Qwen3.8-27B")
    assert not is_hub_identity("/Volumes/models/Qwen3.8-27B")
    assert not is_hub_identity("~/models/Qwen3.8-27B")
    assert not is_hub_identity("Qwen3.8-27B")
    assert not is_hub_identity("Qwen/Qwen3.8-27B", local_path="Qwen/Qwen3.8-27B")
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "models" / "Qwen3.8-27B"
    source.mkdir(parents=True)
    # The typed relative directory being converted is a location, not a Hub ID.
    assert not is_hub_identity("models/Qwen3.8-27B", source_argument="models/Qwen3.8-27B")
    assert not is_hub_identity("models/Qwen3.8-27B", local_path=str(source))
    # An unrelated directory in the working directory does not change an explicit ID.
    assert is_hub_identity("models/Qwen3.8-27B", source_argument="./other")
    # A Hub download passes its ID as the argument; it is not a directory.
    assert is_hub_identity("org/fine-tune", source_argument="org/fine-tune")


def test_classification_fails_closed_without_identity() -> None:
    with pytest.raises(PlanningError, match="--not-dflash2-target"):
        classify_source(_config_27b(), model_id="/Volumes/models/Qwen3.8-27B-bf16", revision=None)
    with pytest.raises(PlanningError, match="revision"):
        classify_source(_config_27b(), model_id="Qwen/Qwen3.8-27B", revision=None)
    with pytest.raises(PlanningError, match="revision"):
        classify_source(_config_27b(), model_id="Qwen/Qwen3.8-27B", revision="main")


def test_classification_statuses() -> None:
    target = classify_source(_config_27b(), model_id="Qwen/Qwen3.8-27B", revision=REVISION.upper())
    assert target is not None and target.status == "target"
    assert target.source_revision == REVISION
    fine_tune = classify_source(
        _config_27b(), model_id="ukisai/Swift-1.5-Qwen3.8-27b", revision=None
    )
    assert fine_tune is not None and fine_tune.status == "non_target"
    declined = classify_source(
        _config_27b(), model_id="/tmp/local", revision=None, not_target_reason="local study"
    )
    assert declined is not None and declined.status == "declared_non_target"
    assert declined.source_model_id is None and declined.reason == "local study"
    assert classify_source({"model_type": "llama"}, model_id="/tmp/x", revision=None) is None


def test_admitted_id_case_variant_fails_closed() -> None:
    # AX Engine applies its 27B format policy by exact model ID.
    with pytest.raises(PlanningError, match="exact admitted model ID"):
        classify_source(_config_27b(), model_id="qwen/qwen3.8-27b", revision=REVISION)


def test_moe_target_requires_revision_and_has_no_lane_policy() -> None:
    with pytest.raises(PlanningError, match="revision"):
        classify_source(_config_35b(), model_id="Qwen/Qwen3.6-35B-A3B", revision=None)
    target = classify_source(_config_35b(), model_id="Qwen/Qwen3.6-35B-A3B", revision=REVISION)
    assert target is not None and target.status == "target"
    # AX Engine 929e1f135 applies no format policy to qwen3_5_moe.
    require_admitted_lane(target, "affine6_g32")


def test_moe_expert_count_is_read_from_the_root_config() -> None:
    config = _config_35b()
    experts = config["text_config"].pop("num_experts")
    config["num_experts"] = experts
    assert matching_target(config).source_model_id == "Qwen/Qwen3.6-35B-A3B"  # type: ignore[union-attr]


def test_decline_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The admitted ID cannot be declined even when a same-named directory exists.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Qwen" / "Qwen3.8-27B").mkdir(parents=True)
    with pytest.raises(PlanningError, match="admitted DFlash2 source"):
        classify_source(
            _config_27b(),
            model_id="Qwen/Qwen3.8-27B",
            revision=None,
            local_path=str(tmp_path / "Qwen" / "Qwen3.8-27B"),
            source_argument="Qwen/Qwen3.8-27B",
            not_target_reason="study",
        )
    for reason in ("copy of /Volumes/private/model", "see ~/models", "file:///x"):
        with pytest.raises(PlanningError, match="filesystem paths"):
            classify_source(
                _config_27b(), model_id="/tmp/x", revision=None, not_target_reason=reason
            )
    with pytest.raises(PlanningError, match="non-empty reason"):
        classify_source(_config_27b(), model_id="/tmp/x", revision=None, not_target_reason=" ")
    with pytest.raises(PlanningError, match="admitted DFlash2 source"):
        classify_source(
            _config_27b(),
            model_id="Qwen/Qwen3.8-27B",
            revision=REVISION,
            not_target_reason="experiment",
        )


def test_format_lanes() -> None:
    assert format_lane("mxfp4", 4, 32) == "mxfp4"
    assert format_lane("mxfp8", 8, 32) == "mxfp8"
    assert format_lane("affine", 6, 64) == "affine6_g64"
    assert format_lane("affine", 6, 32) == "affine6_g32"
    # MLX packs microscaling in fixed blocks; a recorded group does not rename the lane.
    assert format_lane("mxfp4", 4, 64) == "mxfp4"
    allocations = [
        SimpleNamespace(
            method=QuantMethod.MXFP4,
            bits=4,
            group_size=32,
            parameters=900,
            module_path="language_model.model.layers.0.mlp.down_proj",
            strategy_metadata={},
        ),
        SimpleNamespace(
            method=QuantMethod.AFFINE,
            bits=8,
            group_size=64,
            parameters=5000,
            module_path="language_model.lm_head",
            strategy_metadata={},
        ),
        SimpleNamespace(
            method=QuantMethod.AFFINE,
            bits=6,
            group_size=32,
            parameters=100,
            module_path="mtp.layers.0.mlp.down_proj",
            strategy_metadata={},
        ),
    ]
    assert predicted_format_lane(allocations, q_mode="mxfp4", group_size=32) == "mxfp4"
    bf16_trunk = SimpleNamespace(
        method=QuantMethod.AFFINE,
        bits=16,
        group_size=None,
        parameters=10_000,
        module_path="language_model.model.layers.1.mlp.up_proj",
        strategy_metadata={},
    )
    # A small quantized island cannot name the lane of a mostly unquantized trunk.
    assert predicted_format_lane([*allocations, bf16_trunk], q_mode="mxfp4", group_size=32) == (
        "unquantized"
    )


def test_target_lane_must_be_admitted() -> None:
    target = classify_source(_config_27b(), model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    assert target is not None
    require_admitted_lane(target, "mxfp4")
    with pytest.raises(PlanningError, match="affine6_g32"):
        require_admitted_lane(target, "affine6_g32")
    fine_tune = classify_source(_config_27b(), model_id="org/fine-tune", revision=None)
    assert fine_tune is not None
    require_admitted_lane(fine_tune, "affine6_g32")


def test_target_declaration_records_drafter_shared_tensors(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    declaration = _declare(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    assert declaration.status == "target"
    assert declaration.format_lane == "mxfp4"
    assert declaration.drafter is not None
    assert declaration.drafter.repo == "z-lab/Qwen3.8-27B-DFlash2"
    roles = {tensor.role: tensor for tensor in declaration.shared_tensors}
    assert set(roles) == {"token_embedding", "lm_head"}
    head = roles["lm_head"]
    assert (head.mode, head.bits, head.group_size) == ("affine", 8, 64)
    assert {part.name.rsplit(".", 1)[1] for part in head.parts} == {"weight", "scales", "biases"}
    assert pack_format_lane(pack) == "mxfp4"


def test_target_declaration_rejects_unadmitted_pack_lane(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack", trunk_mode="affine", trunk_bits=6, trunk_group=32)
    classification = classify_source(_config_27b(), model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    assert classification is not None
    with pytest.raises(PlanningError, match="affine6_g32"):
        build_declaration(pack, classification, target_num_hidden_layers=64)


def test_non_target_tolerates_missing_shared_tensors(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack", with_embedding=False)
    declaration = _declare(pack, model_id="org/fine-tune", revision=None)
    assert declaration.status == "non_target"
    assert [tensor.role for tensor in declaration.shared_tensors] == ["lm_head"]
    classification = classify_source(_config_27b(), model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    assert classification is not None
    with pytest.raises(Exception, match="embed_tokens"):
        build_declaration(pack, classification, target_num_hidden_layers=64)


def test_declaration_schema_rules() -> None:
    base = {
        "admitted_source_model_id": "Qwen/Qwen3.8-27B",
        "mirror": {"observed_at": "2026-11-24", "ax_engine_commit": "929e1f135"},
        "tap_layer_ids": [5, 19, 33, 47, 61],
        "target_num_hidden_layers": 64,
        "format_lane": "mxfp4",
    }
    with pytest.raises(ValidationError):
        Dflash2TargetDeclaration.model_validate(
            {**base, "status": "target", "source_model_id": "Qwen/Qwen3.8-27B"}
        )
    with pytest.raises(ValidationError):
        Dflash2TargetDeclaration.model_validate(
            {**base, "status": "non_target", "source_model_id": "Qwen/Qwen3.8-27B"}
        )
    with pytest.raises(ValidationError):
        Dflash2TargetDeclaration.model_validate({**base, "status": "declared_non_target"})
    with pytest.raises(ValidationError):
        Dflash2TargetDeclaration.model_validate(
            {**base, "status": "non_target", "source_model_id": "/local/path"}
        )
    for taps in ([-1, 5], [19, 5], [5, 5]):
        with pytest.raises(ValidationError):
            Dflash2TargetDeclaration.model_validate(
                {**base, "status": "non_target", "source_model_id": "org/x", "tap_layer_ids": taps}
            )
    with pytest.raises(ValidationError):
        Dflash2TargetDeclaration.model_validate(
            {**base, "status": "non_target", "source_model_id": "org/x", "reason": "stray"}
        )
    with pytest.raises(ValidationError):
        Dflash2TargetDeclaration.model_validate(
            {**base, "status": "declared_non_target", "reason": "r", "source_revision": REVISION}
        )
    part = {"name": "m.weight", "file": "../x.safetensors", "dtype": "U32", "shape": [1]}
    with pytest.raises(ValidationError):
        Dflash2SharedTensor.model_validate(
            {"role": "lm_head", "module": "m", "parts": [{**part, "sha256": "0" * 64}]}
        )
    with pytest.raises(ValidationError):
        Dflash2SharedTensor.model_validate(
            {
                "role": "lm_head",
                "module": "m",
                "parts": [
                    {**part, "file": "a.safetensors", "name": "other.weight", "sha256": "0" * 64}
                ],
            }
        )


def test_publication_gate(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    _bind(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    with pytest.raises(ValidationGateError, match=DECLARATION_NAME):
        require_publication_declaration(pack)
    _declare(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    declaration = require_publication_declaration(pack)
    assert declaration is not None and declaration.status == "target"


def test_publication_gate_rejects_unbound_declaration(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    _declare(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    manifest = json.loads((pack / "axquant_manifest.json").read_text())
    manifest["files"] = []
    (pack / "axquant_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValidationGateError, match="not bound"):
        require_publication_declaration(pack)


def test_publication_gate_rejects_identity_drift(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    _declare(pack, model_id="org/fine-tune", revision=None)
    _bind(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    with pytest.raises(ValidationGateError, match="disagrees"):
        require_publication_declaration(pack)


def test_publication_gate_rejects_changed_shared_tensor(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    _declare(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    shard = pack / "model-00001-of-00001.safetensors"
    data = bytearray(shard.read_bytes())
    header_length = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + header_length])
    begin = header["language_model.lm_head.weight"]["data_offsets"][0]
    data[8 + header_length + begin] ^= 0xFF
    shard.write_bytes(bytes(data))
    with pytest.raises(ValidationGateError, match="drafter-shared tensors"):
        require_publication_declaration(pack)


def test_publication_gate_rejects_lane_drift(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    _declare(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    config = json.loads((pack / "config.json").read_text())
    config["quantization"]["language_model.model.layers.0.mlp.down_proj"] = {
        "bits": 6,
        "group_size": 64,
        "mode": "affine",
    }
    (pack / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValidationGateError, match="format lane"):
        require_publication_declaration(pack)


def test_publication_gate_rejects_shared_quantization_drift(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    _declare(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    config = json.loads((pack / "config.json").read_text())
    config["quantization"]["language_model.lm_head"]["group_size"] = 32
    (pack / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValidationGateError, match="drafter-shared tensors"):
        require_publication_declaration(pack)


def test_publication_gate_rejects_stale_mirror_fields(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    _declare(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    payload = json.loads((pack / DECLARATION_NAME).read_text())
    payload["tap_layer_ids"] = [4, 19, 33, 47, 61]
    (pack / DECLARATION_NAME).write_text(json.dumps(payload))
    _bind(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    with pytest.raises(ValidationGateError, match="tap layers"):
        require_publication_declaration(pack)
    payload["tap_layer_ids"] = [5, 19, 33, 47, 61]
    payload["drafter"]["revision"] = "0" * 40
    (pack / DECLARATION_NAME).write_text(json.dumps(payload))
    _bind(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    with pytest.raises(ValidationGateError, match="drafter pin"):
        require_publication_declaration(pack)


def test_publication_gate_rejects_revision_drift_for_non_targets(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    _declare(pack, model_id="org/fine-tune", revision=REVISION)
    _bind(pack, model_id="org/fine-tune", revision="1" * 40)
    with pytest.raises(ValidationGateError, match="revision"):
        require_publication_declaration(pack)


def test_declared_non_target_from_relative_directory_publishes(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    _declare(pack, model_id="/tmp/local", revision=None, reason="local study")
    # Without --model-id the manifest keeps the typed relative directory.
    _bind(pack, model_id="models/Qwen3.8-27B", revision=None)
    declaration = require_publication_declaration(pack)
    assert declaration is not None and declaration.source_model_id is None
    _bind(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    with pytest.raises(ValidationGateError, match="disagrees"):
        require_publication_declaration(pack)


def test_publication_gate_wraps_unreadable_packs(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    _declare(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    (pack / "axquant_manifest.json").write_text("[]")
    with pytest.raises(ValidationGateError):
        require_publication_declaration(pack)
    _bind(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    (pack / "model.safetensors.index.json").write_text("[]")
    with pytest.raises(ValidationGateError, match="weight_map"):
        require_publication_declaration(pack)


def test_shard_paths_cannot_leave_the_pack(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    index = json.loads((pack / "model.safetensors.index.json").read_text())
    name = next(iter(index["weight_map"]))
    index["weight_map"][name] = "../outside.safetensors"
    (pack / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ArtifactError, match="plain file name"):
        pack_format_lane(pack)


def test_mlx_pack_with_only_quantization_config_is_gated(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    config = json.loads((pack / "config.json").read_text())
    config["quantization_config"] = config.pop("quantization")
    (pack / "config.json").write_text(json.dumps(config))
    _bind(pack, model_id="Qwen/Qwen3.8-27B", revision=REVISION)
    with pytest.raises(ValidationGateError, match=DECLARATION_NAME):
        require_publication_declaration(pack)


def test_publication_gate_skips_non_mlx_and_unmatched_packs(tmp_path: Path) -> None:
    cuda = tmp_path / "cuda"
    cuda.mkdir()
    (cuda / "config.json").write_text(
        json.dumps({**_config_27b(), "quantization_config": {"quant_method": "compressed-tensors"}})
    )
    assert require_publication_declaration(cuda) is None
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(
        json.dumps({"model_type": "llama", "quantization": {"bits": 4, "group_size": 32}})
    )
    assert require_publication_declaration(other) is None


def test_cli_exposes_decline_flag() -> None:
    parser = build_parser()
    convert = parser.parse_args(
        ["convert", "--model", "m", "--plan", "p", "--output", "o", "--not-dflash2-target", "why"]
    )
    assert convert.not_dflash2_target == "why"
    quantize = parser.parse_args(["quantize", "m", "--not-dflash2-target", "why"])
    assert quantize.not_dflash2_target == "why"


def _quantize_args(source: Path, output: Path, *extra: str) -> list[str]:
    return [
        "quantize",
        str(source),
        "--format",
        "mxfp4",
        "--output",
        str(output),
        "--ax-engine-manifest",
        "skip",
        *extra,
    ]


def _prepare_fixture(
    qwen36_model_dir: Path, monkeypatch: pytest.MonkeyPatch, *, embedding: bool = False
) -> None:
    from safetensors.numpy import load_file
    from test_quantize import _install_fake_mlx

    source = qwen36_model_dir / "model.safetensors"
    tensors = load_file(source)
    if embedding:
        tensors["language_model.model.embed_tokens.weight"] = np.zeros((16, 8), np.float32)
    for name, value in tensors.items():
        if value.ndim == 2:
            tensors[name] = np.zeros((32, 128), dtype=np.float32)
    save_file(tensors, source)
    _install_fake_mlx(monkeypatch, qwen36_model_dir)


def test_quantize_local_target_shape_fails_closed_before_output(
    qwen36_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from axquant.cli import main

    _prepare_fixture(qwen36_model_dir, monkeypatch)
    output = tmp_path / "pack"
    assert main(_quantize_args(qwen36_model_dir, output)) == 2
    assert not output.exists()


def test_quantize_declared_non_target_binds_declaration(
    qwen36_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from axquant.cli import main

    _prepare_fixture(qwen36_model_dir, monkeypatch)
    output = tmp_path / "pack"
    args = _quantize_args(qwen36_model_dir, output, "--not-dflash2-target", "local study")
    assert main(args) == 0
    raw = (output / DECLARATION_NAME).read_bytes()
    declaration = Dflash2TargetDeclaration.model_validate_json(raw)
    assert declaration.status == "declared_non_target"
    assert declaration.source_model_id is None
    assert declaration.reason == "local study"
    manifest = json.loads((output / "axquant_manifest.json").read_text())
    bound = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    assert bound[DECLARATION_NAME] == hashlib.sha256(raw).hexdigest()
    # A local source path never leaks into the declaration.
    assert str(qwen36_model_dir) not in raw.decode()


def test_quantize_admitted_target_records_drafter_pin(
    qwen36_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from axquant.cli import main

    # The drafter reuses the embedding; a target pack without one fails closed.
    _prepare_fixture(qwen36_model_dir, monkeypatch)
    args = _quantize_args(
        qwen36_model_dir, tmp_path / "bad", "--model-id", "Qwen/Qwen3.8-27B", "--revision", REVISION
    )
    assert main(args) == 2
    assert not (tmp_path / "bad").exists()
    _prepare_fixture(qwen36_model_dir, monkeypatch, embedding=True)
    output = tmp_path / "pack"
    args = _quantize_args(
        qwen36_model_dir, output, "--model-id", "Qwen/Qwen3.8-27B", "--revision", REVISION
    )
    assert main(args) == 0
    declaration = require_publication_declaration(output)
    assert declaration is not None and declaration.status == "target"
    assert declaration.source_revision == REVISION
    assert declaration.format_lane == "mxfp4"
    assert {tensor.role for tensor in declaration.shared_tensors} == {"token_embedding", "lm_head"}
