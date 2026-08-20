"""Flash-0731 factory QA protocol (generation-viability, not BF16 retention).

``v64`` is the original 64-token user-only chat protocol. ``v256-strict``
raises the decode budget and adds suite system prompts so Flash-0731's
chat preamble no longer consumes the entire generation budget. Weights are
unchanged; this is an eval-protocol change, not a recipe change.
"""

from __future__ import annotations

from typing import Literal

QaProtocol = Literal["v64", "v256-strict"]

PROTOCOL_V64: QaProtocol = "v64"
PROTOCOL_V256_STRICT: QaProtocol = "v256-strict"
DEFAULT_PROTOCOL: QaProtocol = PROTOCOL_V256_STRICT

AGENT_CODING_SYSTEM = (
    "Reply with Python source only. No preamble, no markdown fences, no explanation."
)
GENERAL_SYSTEM = (
    "Follow the user instruction exactly. Output only the requested text. "
    "No markdown, no extra words, no quotes around the answer unless asked."
)

_MAX_TOKENS = {
    PROTOCOL_V64: 64,
    PROTOCOL_V256_STRICT: 256,
}


def normalize_qa_protocol(raw: str | None) -> QaProtocol:
    """Map env/CLI text onto a known Flash-0731 QA protocol."""

    value = (raw or DEFAULT_PROTOCOL).strip().lower()
    aliases = {
        "v64": PROTOCOL_V64,
        "64": PROTOCOL_V64,
        "legacy": PROTOCOL_V64,
        "v256-strict": PROTOCOL_V256_STRICT,
        "v256": PROTOCOL_V256_STRICT,
        "strict": PROTOCOL_V256_STRICT,
        "256": PROTOCOL_V256_STRICT,
    }
    protocol = aliases.get(value)
    if protocol is None:
        allowed = ", ".join(sorted({PROTOCOL_V64, PROTOCOL_V256_STRICT}))
        raise ValueError(f"unknown DSV4 QA protocol {raw!r}; use {allowed}")
    return protocol


def qa_max_tokens(protocol: QaProtocol, override: int | None = None) -> int:
    """Return the decode budget. *override* wins when positive."""

    if override is not None:
        if override < 1:
            raise ValueError("QA max_tokens override must be at least 1")
        return override
    return _MAX_TOKENS[protocol]


def qa_system_prompt(suite: str, protocol: QaProtocol) -> str | None:
    """Suite system prompt, or None for the legacy user-only protocol."""

    if protocol == PROTOCOL_V64:
        return None
    if suite == "agent-coding":
        return AGENT_CODING_SYSTEM
    if suite == "general":
        return GENERAL_SYSTEM
    raise ValueError(f"unknown QA suite {suite!r}")


def build_qa_messages(
    suite: str,
    prompt: str,
    protocol: QaProtocol,
) -> list[dict[str, str]]:
    """OpenAI-style chat messages for one factory QA item."""

    messages: list[dict[str, str]] = []
    system = qa_system_prompt(suite, protocol)
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def qa_protocol_record(protocol: QaProtocol, max_tokens: int) -> dict[str, object]:
    """JSON fragment to bind on eval/certificate evidence."""

    return {
        "id": protocol,
        "max_tokens_qa": max_tokens,
        "system_prompts": protocol != PROTOCOL_V64,
        "agent_coding_system": qa_system_prompt("agent-coding", protocol),
        "general_system": qa_system_prompt("general", protocol),
        "notes": (
            "Legacy 64-token user-only chat."
            if protocol == PROTOCOL_V64
            else "256-token decode with suite system prompts; not a weight change."
        ),
    }
