from __future__ import annotations

from axquant.gemma4_vlm import GEMMA4_MLX_VLM_VISION_LAYOUT
from axquant.hub_mtp_audit import (
    MtpHubPackKind,
    MtpHubRepositorySnapshot,
    SafetensorsHeader,
    audit_mtp_hub_snapshot,
)


def _snapshot(
    *,
    repo_id: str,
    files: set[str],
    documents: dict[str, dict],
    headers: dict[str, SafetensorsHeader],
) -> MtpHubRepositorySnapshot:
    return MtpHubRepositorySnapshot(
        repo_id=repo_id,
        revision="a" * 40,
        files=frozenset(files),
        documents=documents,
        safetensors_headers=headers,
    )


def test_gemma_assistant_audit_accepts_public_mlx_vlm_layout() -> None:
    names = ("embed_vision.weight", "vision_tower.layer.weight")
    snapshot = _snapshot(
        repo_id="AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP",
        files={
            "config.json",
            "model.safetensors.index.json",
            "vision.safetensors",
            "assistant/config.json",
            "ax_gemma4_assistant_mtp.json",
            "ax_composite_pack_manifest.json",
        },
        documents={
            "config.json": {"model_type": "gemma4"},
            "model.safetensors.index.json": {
                "weight_map": {name: "vision.safetensors" for name in names}
            },
            "assistant/config.json": {"model_type": "gemma4_assistant"},
            "ax_gemma4_assistant_mtp.json": {
                "backend": "gemma4_assistant",
                "assistant_path": "assistant",
                "pairing": "exact",
                "target_model_id": "gemma-4-26b-a4b-it",
                "assistant_model_id": "gemma-4-26b-a4b-it-assistant",
            },
            "ax_composite_pack_manifest.json": {"schema_version": "test"},
        },
        headers={
            "vision.safetensors": SafetensorsHeader(
                tensor_names=names,
                metadata={"format": "mlx", "axquant_layout": GEMMA4_MLX_VLM_VISION_LAYOUT},
            )
        },
    )

    result = audit_mtp_hub_snapshot(snapshot)

    assert result.kind == MtpHubPackKind.GEMMA_ASSISTANT
    assert result.passed


def test_gemma_assistant_audit_rejects_old_unindexed_layout() -> None:
    snapshot = _snapshot(
        repo_id="AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP",
        files={
            "config.json",
            "model.safetensors.index.json",
            "vision.safetensors",
            "assistant/config.json",
            "ax_gemma4_assistant_mtp.json",
            "ax_composite_pack_manifest.json",
        },
        documents={
            "config.json": {"model_type": "gemma4"},
            "model.safetensors.index.json": {"weight_map": {}},
            "assistant/config.json": {"model_type": "gemma4_assistant"},
            "ax_gemma4_assistant_mtp.json": {
                "backend": "gemma4_assistant",
                "assistant_path": "assistant",
                "pairing": "exact",
                "target_model_id": "gemma-4-31b-it",
                "assistant_model_id": "gemma-4-31b-it-assistant",
            },
            "ax_composite_pack_manifest.json": {"schema_version": "test"},
        },
        headers={
            "vision.safetensors": SafetensorsHeader(
                tensor_names=("model.vision_tower.layer.weight",),
                metadata={"format": "mlx"},
            )
        },
    )

    result = audit_mtp_hub_snapshot(snapshot)

    assert not result.passed
    assert any("source-prefixed" in issue for issue in result.issues)
    assert any("layout marker" in issue for issue in result.issues)
    assert any("index" in issue for issue in result.issues)


def test_resident_qwen_audit_requires_canonical_arch_id_and_tensor_count() -> None:
    snapshot = _snapshot(
        repo_id="AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
        files={"config.json", "mtplx_runtime.json", "mtp.safetensors"},
        documents={
            "config.json": {"model_type": "qwen3_5"},
            "mtplx_runtime.json": {
                "arch_id": "qwen3-next-mtp",
                "mtp_depth_max": 1,
                "mtp_norm_layout": "native",
                "mtp_tensor_count": 2,
            },
        },
        headers={
            "mtp.safetensors": SafetensorsHeader(
                tensor_names=("mtp.a", "mtp.b"), metadata={"format": "mlx"}
            )
        },
    )

    result = audit_mtp_hub_snapshot(snapshot)

    assert result.kind == MtpHubPackKind.QWEN_RESIDENT
    assert result.passed


def test_qwen_expert_stream_and_deepseek_use_distinct_contracts() -> None:
    qwen = _snapshot(
        repo_id="AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-2bit-MTP",
        files={
            "config.json",
            "mtplx_runtime.json",
            "mtp.safetensors",
            "ax_expert_stream.json",
        },
        documents={
            "config.json": {"model_type": "qwen4_exp"},
            "mtplx_runtime.json": {"mtp_depth_max": 1, "mtp_norm_layout": "native"},
            "ax_expert_stream.json": {"schema_version": "test"},
        },
        headers={
            "mtp.safetensors": SafetensorsHeader(
                tensor_names=("mtp.layer",), metadata={"format": "mlx"}
            )
        },
    )
    deepseek = _snapshot(
        repo_id="AutomatosX/AX-DeepSeek-V4-Flash-MLX-AXQ-2bit-MTP",
        files={"config.json", "mtplx_runtime.json", "mtp.safetensors"},
        documents={
            "config.json": {"model_type": "deepseek_v4"},
            "mtplx_runtime.json": {"mtp_depth_max": 1, "mtp_norm_layout": "native"},
        },
        headers={
            "mtp.safetensors": SafetensorsHeader(
                tensor_names=("model.layers.0.nextn.weight",), metadata={"format": "mlx"}
            )
        },
    )

    qwen_result = audit_mtp_hub_snapshot(qwen)
    deepseek_result = audit_mtp_hub_snapshot(deepseek)

    assert qwen_result.kind == MtpHubPackKind.QWEN_EXPERT_STREAM
    assert qwen_result.passed
    assert deepseek_result.kind == MtpHubPackKind.DEEPSEEK_NEXTN
    assert deepseek_result.passed
