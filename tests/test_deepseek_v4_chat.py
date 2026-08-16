# ruff: noqa: RUF001
from __future__ import annotations

from pathlib import Path

from axquant.deepseek_v4_chat import (
    DEEPSEEK_V4_CHAT_TEMPLATE,
    maybe_write_deepseek_v4_chat_template,
    render_deepseek_v4_user_prompt,
    write_deepseek_v4_chat_template,
)
from axquant.schema import ArchitectureProfile, ArchitectureSupportLevel


class _Plan:
    def __init__(self, family: str, model_type: str | None = None) -> None:
        self.architecture_profile = ArchitectureProfile(
            adapter_id="deepseek-v4-v1",
            product_family=family,
            config_model_type=model_type,
            support_level=ArchitectureSupportLevel.SUPPORTED,
        )


def test_render_matches_official_chat_encode_for_one_user_turn() -> None:
    rendered = render_deepseek_v4_user_prompt("Say hello.", thinking=False)
    assert rendered == (
        "<｜begin▁of▁sentence｜><｜User｜>Say hello.<｜Assistant｜></think>"
    )
    thinking = render_deepseek_v4_user_prompt("Say hello.", thinking=True)
    assert thinking.endswith("<｜Assistant｜><think>")
    assert "<think>" in thinking and thinking.count("</think>") == 0


def test_maybe_write_emits_jinja_only_for_deepseek_v4(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    assert maybe_write_deepseek_v4_chat_template(other, _Plan("qwen3.6")) is None
    assert not (other / "chat_template.jinja").exists()

    target = tmp_path / "flash"
    target.mkdir()
    path = maybe_write_deepseek_v4_chat_template(target, _Plan("deepseek-v4", "deepseek_v4"))
    assert path == target / "chat_template.jinja"
    text = path.read_text(encoding="utf-8")
    assert text == DEEPSEEK_V4_CHAT_TEMPLATE
    assert "<｜User｜>" in text
    assert "</think>" in text

    # Existing non-empty template is left in place.
    path.write_text("kept", encoding="utf-8")
    again = maybe_write_deepseek_v4_chat_template(target, _Plan("deepseek-v4"))
    assert again.read_text(encoding="utf-8") == "kept"


def test_write_deepseek_v4_chat_template_overwrites(tmp_path: Path) -> None:
    path = write_deepseek_v4_chat_template(tmp_path)
    assert "add_generation_prompt" in path.read_text(encoding="utf-8")
