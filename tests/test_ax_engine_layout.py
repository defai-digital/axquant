"""AX Engine weight-layout declaration and artifact v3 envelope (AXQ-066)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from safetensors.numpy import save_file

from axquant.ax_engine_layout import declare_weight_layout, verified_weight_layout
from axquant.errors import ArtifactError
from axquant.schema import (
    ArtifactManifest,
    ArtifactManifestV3,
    AxEngineOptimizationMetadata,
    AxEngineWeightLayout,
    ModelIdentity,
    MtpPolicy,
    MtpRuntimeMetadata,
    OptimizationScope,
    PrecisionShare,
    ProfileName,
    RuntimeMetadata,
    RuntimeMetadataV2,
    RuntimeName,
    RuntimeProfile,
    RuntimeSupportLevel,
    SoftwareVersions,
)
from axquant.schema.loading import (
    load_artifact_manifest,
    load_runtime_metadata,
    validate_artifact_manifest,
)
from axquant.serde import read_data, write_data

_LAYOUT = AxEngineWeightLayout(norm_weights="raw_delta", conv1d_axes="mlx")


def _runtime() -> RuntimeMetadata:
    return RuntimeMetadata(
        primary_runtime=RuntimeProfile(
            name=RuntimeName.AX_ENGINE,
            compatibility_level="A",
            support_level=RuntimeSupportLevel.OPTIMIZED,
            standard_inference=True,
            mtp_support="native",
        ),
        compatible_runtimes=[],
        optimization_scope=OptimizationScope.TEXT_PATH,
        mtp=MtpRuntimeMetadata(detected=True, sidecar_file="mtp.safetensors"),
        ax_engine=AxEngineOptimizationMetadata(preferred_group_size=32),
    )


def _manifest_fields(runtime: RuntimeMetadata) -> dict[str, object]:
    return {
        "axquant_version": "2.3.0",
        "source_model": ModelIdentity(model_id="Qwen/Qwen3.8-Flash-Next", revision="a" * 40),
        "plan_sha256": "b" * 64,
        "profile": ProfileName.AGENT_CODING,
        "target_class": "mixed-6bit",
        "effective_bpw": 6.0,
        "logical_parameters": 8,
        "main_logical_parameters": 8,
        "weight_file_size_bytes": 8,
        "main_weight_file_size_bytes": 8,
        "mtp_weight_file_size_bytes": 0,
        "protected_weight_file_size_bytes": 0,
        "measured_total_bpw": 8.0,
        "measured_main_bpw": 8.0,
        "weight_distribution": {"6bit": PrecisionShare(parameters=8, fraction=1.0)},
        "mtp_distribution": {},
        "mtp_present": True,
        "mtp_policy": MtpPolicy(mode="disabled", candidate_bits=(16,), min_bits=16),
        "runtime": runtime,
        "software_versions": SoftwareVersions(
            axquant="2.3.0", python="3.13", safetensors="0.6", pydantic="2"
        ),
        "files": [],
    }


def _write_pack(
    directory: Path,
    *,
    conv_shape: tuple[int, ...] = (64, 4, 1),
    trunk_norm_layout: str | None = "raw_hf_delta",
) -> Path:
    directory.mkdir()
    tensors = {
        "language_model.model.layers.0.linear_attn.conv1d.weight": np.zeros(
            conv_shape, dtype=np.float16
        ),
        "language_model.model.layers.0.input_layernorm.weight": np.zeros((8,), dtype=np.float16),
    }
    save_file(tensors, str(directory / "model-00001-of-00001.safetensors"))
    save_file(
        {"mtp.layers.0.conv1d.weight": np.zeros((64, 1, 4), dtype=np.float16)},
        str(directory / "mtp.safetensors"),
    )
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    weight_map["mtp.layers.0.conv1d.weight"] = "mtp.safetensors"
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    contract: dict[str, object] = {"mtp_norm_layout": "raw_hf_delta"}
    if trunk_norm_layout is not None:
        contract["trunk_norm_layout"] = trunk_norm_layout
    (directory / "mtplx_runtime.json").write_text(json.dumps(contract), encoding="utf-8")
    return directory


def test_flash_next_pack_with_raw_delta_norms_and_mlx_conv_is_declared(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")

    assert verified_weight_layout(pack, "qwen4_exp") == _LAYOUT


def test_other_families_are_never_declared(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")

    assert verified_weight_layout(pack, "qwen3_5_moe") is None
    assert verified_weight_layout(pack, None) is None


@pytest.mark.parametrize("trunk_norm_layout", [None, "mlx_multiplier"])
def test_flash_next_without_raw_delta_contract_is_not_declared(
    tmp_path: Path, trunk_norm_layout: str | None
) -> None:
    pack = _write_pack(tmp_path / "pack", trunk_norm_layout=trunk_norm_layout)

    assert verified_weight_layout(pack, "qwen4_exp") is None


def test_hf_ordered_conv1d_fails_closed(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack", conv_shape=(64, 1, 4))

    with pytest.raises(ArtifactError, match="MLX"):
        verified_weight_layout(pack, "qwen4_exp")


def test_declaration_writes_the_fields_ax_engine_reads(tmp_path: Path) -> None:
    runtime = declare_weight_layout(_runtime(), _LAYOUT)
    manifest = ArtifactManifestV3(**_manifest_fields(runtime))  # type: ignore[arg-type]
    path = tmp_path / "axquant_manifest.json"
    write_data(path, manifest)

    payload = read_data(path)
    assert payload["schema_version"] == "axquant.artifact.v3"
    assert payload["quantizer"] == "axquant"
    assert payload["format"] == "mlx"
    assert payload["runtime"]["schema_version"] == "axquant.runtime.v2"
    assert payload["runtime"]["ax_engine"]["weight_sanitize"] == "hf_norm_only"
    assert payload["runtime"]["ax_engine"]["weight_layout"] == {
        "norm_weights": "raw_delta",
        "conv1d_axes": "mlx",
    }
    assert payload["runtime"]["ax_engine"]["preferred_group_size"] == 32

    loaded = load_artifact_manifest(path)
    assert type(loaded) is ArtifactManifestV3
    assert isinstance(loaded, ArtifactManifest)
    runtime_path = tmp_path / "axquant_runtime.json"
    write_data(runtime_path, runtime)
    assert type(load_runtime_metadata(runtime_path)) is RuntimeMetadataV2


def test_undeclared_manifest_keeps_frozen_v2_envelope(tmp_path: Path) -> None:
    manifest = ArtifactManifest(**_manifest_fields(_runtime()))  # type: ignore[arg-type]
    path = tmp_path / "axquant_manifest.json"
    write_data(path, manifest)

    payload = read_data(path)
    assert payload["schema_version"] == "axquant.artifact.v2"
    assert payload["runtime"]["schema_version"] == "axquant.runtime.v1"
    assert "weight_layout" not in payload["runtime"]["ax_engine"]
    assert type(load_artifact_manifest(path)) is ArtifactManifest


def test_envelope_pairs_cannot_be_mixed() -> None:
    declared = declare_weight_layout(_runtime(), _LAYOUT)
    with pytest.raises(ValidationError, match="runtime"):
        ArtifactManifest(**_manifest_fields(declared))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ArtifactManifestV3(**_manifest_fields(_runtime()))  # type: ignore[arg-type]
    with pytest.raises(ArtifactError, match="already"):
        declare_weight_layout(declared, _LAYOUT)


def test_v1_runtime_rejects_layout_fields() -> None:
    payload = _runtime().model_dump(mode="json")
    payload["ax_engine"]["weight_sanitize"] = "hf_norm_only"
    with pytest.raises(ValidationError):
        RuntimeMetadata.model_validate(payload)


def test_unknown_artifact_version_fails_closed() -> None:
    payload = ArtifactManifest(**_manifest_fields(_runtime())).model_dump(mode="json")  # type: ignore[arg-type]
    payload["schema_version"] = "axquant.artifact.v4"
    with pytest.raises(ArtifactError, match="unsupported schema_version"):
        validate_artifact_manifest(payload)
