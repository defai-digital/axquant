from __future__ import annotations

import json
from pathlib import Path

import pytest

from axquant.streaming_convert import (
    STREAMING_CONVERT_ENV,
    physical_memory_bytes,
    source_weight_bytes,
    streaming_convert_enabled,
)


def test_streaming_env_on_forces_enable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(STREAMING_CONVERT_ENV, "1")
    assert streaming_convert_enabled(tmp_path) is True


def test_streaming_env_off_forces_disable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(STREAMING_CONVERT_ENV, "0")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"x" * 100)
    monkeypatch.setattr(
        "axquant.streaming_convert.physical_memory_bytes",
        lambda: 1,
    )
    assert streaming_convert_enabled(tmp_path) is False


def test_streaming_auto_when_source_exceeds_ram(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(STREAMING_CONVERT_ENV, raising=False)
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"x" * 200)
    (tmp_path / "model-mtp.safetensors").write_bytes(b"x" * 500)
    monkeypatch.setattr("axquant.streaming_convert.physical_memory_bytes", lambda: 100)
    assert source_weight_bytes(tmp_path) == 200
    assert streaming_convert_enabled(tmp_path) is True


def test_streaming_auto_off_when_source_fits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(STREAMING_CONVERT_ENV, raising=False)
    (tmp_path / "model.safetensors").write_bytes(b"x" * 50)
    monkeypatch.setattr("axquant.streaming_convert.physical_memory_bytes", lambda: 100)
    assert streaming_convert_enabled(tmp_path) is False


def test_physical_memory_bytes_positive_on_host() -> None:
    memory = physical_memory_bytes()
    assert memory > 1 << 30


def test_mlx_convert_hook_uses_streaming_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import axquant.converter as converter

    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "out"
    called: dict[str, object] = {}

    def fake_stream(model_ref: str, **kwargs: object) -> None:
        called["model"] = model_ref
        called["path"] = kwargs["mlx_path"]
        Path(str(kwargs["mlx_path"])).mkdir()

    def fake_convert(*args: object, **kwargs: object) -> None:
        raise AssertionError("stock mlx_lm.convert should not run")

    monkeypatch.setenv(STREAMING_CONVERT_ENV, "1")
    monkeypatch.setattr(converter, "_mlx_api", lambda: (fake_convert, lambda *a, **k: None))
    monkeypatch.setattr(
        "axquant.streaming_convert.streaming_mlx_convert",
        fake_stream,
    )
    converter._mlx_convert_with_optional_dequant(
        str(src),
        mlx_path=str(dest),
        quantize=True,
        q_group_size=32,
        q_bits=4,
        quant_predicate=lambda path, module: False,
        revision=None,
    )
    assert called["model"] == str(src)
    assert called["path"] == str(dest)


def test_quantize_and_save_streaming_matches_stock_keys(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx_lm.utils import quantize_model, save_model

    from axquant.streaming_convert import quantize_and_save_streaming

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(64, 64, bias=False)
            self.b = nn.Linear(64, 32, bias=False)

        def __call__(self, x: object) -> object:
            return self.b(self.a(x))

    mx.random.seed(0)
    model = Tiny()
    model.a.weight = mx.arange(64 * 64, dtype=mx.float32).reshape(64, 64)
    model.b.weight = mx.arange(32 * 64, dtype=mx.float32).reshape(32, 64)
    mx.eval(model.parameters())

    def predicate(path: str, module: object) -> dict[str, int | str]:
        del path, module
        return {"group_size": 32, "bits": 4, "mode": "affine"}

    class Tokenizer:
        def save_pretrained(self, path: str | Path) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    src = tmp_path / "src"
    src.mkdir()
    stream_dir = tmp_path / "stream"
    stock_dir = tmp_path / "stock"
    config = {"model_type": "tiny", "torch_dtype": "float32"}
    quantize_and_save_streaming(
        model,
        Tokenizer(),
        dict(config),
        stream_dir,
        src,
        quant_predicate=predicate,
        q_group_size=32,
        q_bits=4,
        q_mode="affine",
    )

    mx.random.seed(0)
    stock = Tiny()
    stock.a.weight = mx.arange(64 * 64, dtype=mx.float32).reshape(64, 64)
    stock.b.weight = mx.arange(32 * 64, dtype=mx.float32).reshape(32, 64)
    mx.eval(stock.parameters())
    stock, stock_config = quantize_model(
        stock, dict(config), 32, 4, mode="affine", quant_predicate=predicate
    )
    stock_dir.mkdir()
    save_model(stock_dir, stock, donate_model=True)
    from mlx_lm.utils import save_config

    save_config(stock_config, config_path=stock_dir / "config.json")

    stream_weights = mx.load(str(stream_dir / "model.safetensors"))
    stock_weights = mx.load(str(stock_dir / "model.safetensors"))
    assert set(stream_weights) == set(stock_weights)
    for key, value in stock_weights.items():
        assert mx.array_equal(stream_weights[key], value), key
    stream_cfg = json.loads((stream_dir / "config.json").read_text(encoding="utf-8"))
    assert "quantization" in stream_cfg
    assert "a" in stream_cfg["quantization"]
    assert stream_cfg["quantization"]["a"]["mode"] == "affine"


def test_quantize_and_save_streaming_switch_linear_mxfp4(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_lm.utils import quantize_model, save_config, save_model

    from axquant.streaming_convert import quantize_and_save_streaming

    class TinyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = SwitchLinear(32, 64, num_experts=4, bias=False)

        def __call__(self, x: object, indices: object) -> object:
            return self.experts(x, indices)

    mx.random.seed(1)
    model = TinyMoE()
    model.experts.weight = mx.arange(4 * 64 * 32, dtype=mx.float32).reshape(4, 64, 32)
    mx.eval(model.parameters())

    def predicate(path: str, module: object) -> dict[str, int | str]:
        del path, module
        return {"group_size": 32, "bits": 4, "mode": "mxfp4"}

    class Tokenizer:
        def save_pretrained(self, path: str | Path) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    src = tmp_path / "src"
    src.mkdir()
    stream_dir = tmp_path / "stream"
    stock_dir = tmp_path / "stock"
    config = {"model_type": "tiny-moe"}
    quantize_and_save_streaming(
        model,
        Tokenizer(),
        dict(config),
        stream_dir,
        src,
        quant_predicate=predicate,
        q_group_size=32,
        q_bits=4,
        q_mode="mxfp4",
    )

    mx.random.seed(1)
    stock = TinyMoE()
    stock.experts.weight = mx.arange(4 * 64 * 32, dtype=mx.float32).reshape(4, 64, 32)
    mx.eval(stock.parameters())
    stock, stock_config = quantize_model(
        stock, dict(config), 32, 4, mode="mxfp4", quant_predicate=predicate
    )
    stock_dir.mkdir()
    save_model(stock_dir, stock, donate_model=True)
    save_config(stock_config, config_path=stock_dir / "config.json")

    stream_weights = mx.load(str(stream_dir / "model.safetensors"))
    stock_weights = mx.load(str(stock_dir / "model.safetensors"))
    assert set(stream_weights) == set(stock_weights)
    for key, value in stock_weights.items():
        assert mx.array_equal(stream_weights[key], value), key
    stream_cfg = json.loads((stream_dir / "config.json").read_text(encoding="utf-8"))
    assert stream_cfg["quantization"]["experts"]["mode"] == "mxfp4"


def test_quantize_and_save_streaming_keeps_parent_module_arrays(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx_lm.utils import quantize_model, save_config, save_model

    from axquant.streaming_convert import quantize_and_save_streaming

    class Gated(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(32, 32, bias=False)
            self.A_log = mx.arange(8, dtype=mx.float32)
            self.dt_bias = mx.ones((8,), dtype=mx.float32)

        def __call__(self, x: object) -> object:
            return self.proj(x)

    mx.random.seed(2)
    model = Gated()
    model.proj.weight = mx.arange(32 * 32, dtype=mx.float32).reshape(32, 32)
    mx.eval(model.parameters())

    def predicate(path: str, module: object) -> dict[str, int | str]:
        del path, module
        return {"group_size": 32, "bits": 4, "mode": "affine"}

    class Tokenizer:
        def save_pretrained(self, path: str | Path) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    src = tmp_path / "src"
    src.mkdir()
    stream_dir = tmp_path / "stream"
    stock_dir = tmp_path / "stock"
    config = {"model_type": "tiny-gdn"}
    quantize_and_save_streaming(
        model,
        Tokenizer(),
        dict(config),
        stream_dir,
        src,
        quant_predicate=predicate,
        q_group_size=32,
        q_bits=4,
        q_mode="affine",
    )

    mx.random.seed(2)
    stock = Gated()
    stock.proj.weight = mx.arange(32 * 32, dtype=mx.float32).reshape(32, 32)
    mx.eval(stock.parameters())
    stock, stock_config = quantize_model(
        stock, dict(config), 32, 4, mode="affine", quant_predicate=predicate
    )
    stock_dir.mkdir()
    save_model(stock_dir, stock, donate_model=True)
    save_config(stock_config, config_path=stock_dir / "config.json")

    stream_weights = mx.load(str(stream_dir / "model.safetensors"))
    stock_weights = mx.load(str(stock_dir / "model.safetensors"))
    assert "A_log" in stream_weights
    assert "dt_bias" in stream_weights
    assert set(stream_weights) == set(stock_weights)
    for key, value in stock_weights.items():
        assert mx.array_equal(stream_weights[key], value), key
