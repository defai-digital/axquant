from __future__ import annotations

from axquant.architectures.qwen36 import Qwen36Adapter


def test_qwen36_adapter_requires_explicit_product_identity() -> None:
    adapter = Qwen36Adapter()
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "vocab_size": 248320,
            "mtp_num_hidden_layers": 1,
        },
    }
    assert adapter.matches("Qwen/Qwen3.6-27B", config)
    assert not adapter.matches("Qwen/Qwen3.5-27B", config)
    assert not adapter.matches("/models/anonymous-checkpoint", config)


def test_qwen36_adapter_can_use_pinned_config_identity() -> None:
    adapter = Qwen36Adapter()
    config = {
        "model_type": "qwen3_5",
        "_name_or_path": "Qwen/Qwen3.6-27B",
    }
    assert adapter.matches("/models/revision-snapshot", config)
