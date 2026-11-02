"""Loader selection without loading a model or requiring real weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from scripts import clef_mlx


def test_omni_decision_load_selects_vlm_without_indexed_vision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx_vlm

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_omni_moe"}))
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    calls = []

    def load(path: str):
        calls.append(path)
        raise RuntimeError("Selected the public VLM loader")

    monkeypatch.setattr(mlx_vlm, "load", load)
    with pytest.raises(RuntimeError, match="Selected the public VLM loader"):
        clef_mlx.load(tmp_path)
    assert calls == [str(tmp_path)]


def test_omni_text_decisions_supply_positions_without_generation_cache() -> None:
    from types import SimpleNamespace

    import numpy as np

    positions = object()
    calls = []

    def text_model(ids, **kwargs):
        calls.append(kwargs)
        return np.zeros((1, 2, 4))

    language_model = SimpleNamespace(
        model=text_model, lm_head=object(), get_rope_index=lambda ids: (positions, None)
    )
    backbone = SimpleNamespace(language_model=language_model)
    model = clef_mlx.ClefMLX(backbone, object(), object(), omni_text=True)
    assert model._hidden(np.array([1, 2]), None).shape == (2, 4)
    assert calls == [{"position_ids": positions}]
