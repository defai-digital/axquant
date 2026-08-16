# ruff: noqa: RUF001
"""DeepSeek-V4 chat rendering from the official source encoder.

The Flash-0731 checkpoint ships ``encoding/encoding_dsv4.py`` and no Jinja
file. mlx-lm convert then emits a tokenizer without ``chat_template``, so
factory generate saw a raw user string and continued it as a document.

This module implements the official single-turn *chat* (non-thinking) path
from that encoder:

    BOS + User + text + Assistant + ``</think>``

It is AXQ-owned and is not copied from mlx-optiq. Thinking mode is the
same tokens with ``<think>`` instead of ``</think>`` after Assistant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
THINK_START = "<think>"
THINK_END = "</think>"

DEEPSEEK_V4_CHAT_TEMPLATE = """\
{%- set bos = '<｜begin▁of▁sentence｜>' -%}
{%- set user_sp = '<｜User｜>' -%}
{%- set asst_sp = '<｜Assistant｜>' -%}
{%- if enable_thinking is not defined -%}
    {%- set enable_thinking = false -%}
{%- endif -%}
{{- bos -}}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- message['content'] -}}
    {%- elif message['role'] == 'user' -%}
        {{- user_sp + message['content'] -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- message['content'] + '<｜end▁of▁sentence｜>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- asst_sp -}}
    {%- if enable_thinking -%}
        {{- '<think>' -}}
    {%- else -%}
        {{- '</think>' -}}
    {%- endif -%}
{%- endif -%}
"""


def render_deepseek_v4_user_prompt(text: str, *, thinking: bool = False) -> str:
    """Encode one user turn the way ``encoding_dsv4.encode_messages(..., 'chat')`` does."""

    think = THINK_START if thinking else THINK_END
    return f"{BOS_TOKEN}{USER_TOKEN}{text}{ASSISTANT_TOKEN}{think}"


def is_deepseek_v4_plan(plan: Any) -> bool:
    profile = getattr(plan, "architecture_profile", None)
    if profile is None:
        return False
    family = str(getattr(profile, "product_family", "") or "")
    model_type = str(getattr(profile, "config_model_type", "") or "")
    return family == "deepseek-v4" or model_type == "deepseek_v4"


def write_deepseek_v4_chat_template(directory: str | Path) -> Path:
    """Write the official chat-mode Jinja template into a convert staging dir."""

    path = Path(directory) / "chat_template.jinja"
    path.write_text(DEEPSEEK_V4_CHAT_TEMPLATE, encoding="utf-8")
    return path


def maybe_write_deepseek_v4_chat_template(directory: str | Path, plan: Any) -> Path | None:
    """Emit the Jinja file for DeepSeek V4 packs that mlx-lm left without one."""

    if not is_deepseek_v4_plan(plan):
        return None
    path = Path(directory) / "chat_template.jinja"
    if path.is_file() and path.stat().st_size > 0:
        return path
    return write_deepseek_v4_chat_template(directory)
