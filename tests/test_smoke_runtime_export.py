from __future__ import annotations

from scripts.smoke_runtime_export import served_model_id


def test_served_model_id_prefers_exact_match() -> None:
    reply = {"data": [{"id": "Pack-mtplx"}, {"id": "pack-mtplx"}]}
    assert served_model_id(reply, "Pack-mtplx") == "Pack-mtplx"


def test_served_model_id_accepts_lowercased_peer_id() -> None:
    reply = {"object": "list", "data": [{"id": "ax-qwen3.8-flash-next-mlx-axq-6bit-mtp-mtplx"}]}
    assert (
        served_model_id(reply, "AX-Qwen3.8-Flash-Next-MLX-AXQ-6bit-MTP-mtplx")
        == "ax-qwen3.8-flash-next-mlx-axq-6bit-mtp-mtplx"
    )


def test_served_model_id_rejects_missing_ambiguous_and_malformed_replies() -> None:
    assert served_model_id({"data": [{"id": "other"}]}, "Pack") is None
    assert served_model_id({"data": [{"id": "PACK"}, {"id": "pack"}]}, "Pack") is None
    assert served_model_id({"data": [{"name": "Pack"}, "Pack"]}, "Pack") is None
    assert served_model_id(["Pack"], "Pack") is None
