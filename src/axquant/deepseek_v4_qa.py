# ruff: noqa: RUF001
"""Flash-0731 factory QA protocol (generation-viability, not BF16 retention).

``v64`` is the original 64-token user-only chat protocol. ``v256-strict``
raises decode to 256 tokens with suite system prompts (still one budget for
both suites). ``v-extract`` is the current default: coding keeps 256 tokens,
general stays at 64, both send stop sequences, and scoring extracts fenced
Python instead of requiring the whole completion to be a fence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QaProtocol = Literal["v64", "v256-strict", "v-extract"]

PROTOCOL_V64: QaProtocol = "v64"
PROTOCOL_V256_STRICT: QaProtocol = "v256-strict"
PROTOCOL_V_EXTRACT: QaProtocol = "v-extract"
DEFAULT_PROTOCOL: QaProtocol = PROTOCOL_V_EXTRACT

AGENT_CODING_SYSTEM = (
    "Reply with Python source only. No preamble, no markdown fences, no explanation."
)
GENERAL_SYSTEM = (
    "Follow the user instruction exactly. Output only the requested text. "
    "No markdown, no extra words, no quotes around the answer unless asked."
)

# DSV4 / OpenAI-style turn endings. Truncate generation before the next turn.
_CHAT_STOP = (
    "<|eot|>",
    "</think>",
    "<｜User｜>",
    "<｜end▁of▁sentence｜>",
)


@dataclass(frozen=True, slots=True)
class SuiteConfig:
    max_tokens: int
    stop: tuple[str, ...]


_SUITE_CONFIGS: dict[QaProtocol, dict[str, SuiteConfig]] = {
    PROTOCOL_V64: {
        "agent-coding": SuiteConfig(64, ()),
        "general": SuiteConfig(64, ()),
    },
    PROTOCOL_V256_STRICT: {
        "agent-coding": SuiteConfig(256, ()),
        "general": SuiteConfig(256, ()),
    },
    PROTOCOL_V_EXTRACT: {
        "agent-coding": SuiteConfig(256, _CHAT_STOP),
        "general": SuiteConfig(64, ("\n\n", *_CHAT_STOP)),
    },
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
        "v-extract": PROTOCOL_V_EXTRACT,
        "extract": PROTOCOL_V_EXTRACT,
        "v2": PROTOCOL_V_EXTRACT,
        "suite": PROTOCOL_V_EXTRACT,
    }
    protocol = aliases.get(value)
    if protocol is None:
        allowed = ", ".join(sorted({PROTOCOL_V64, PROTOCOL_V256_STRICT, PROTOCOL_V_EXTRACT}))
        raise ValueError(f"unknown DSV4 QA protocol {raw!r}; use {allowed}")
    return protocol


def qa_suite_config(
    suite: str,
    protocol: QaProtocol,
    *,
    max_tokens_override: int | None = None,
) -> SuiteConfig:
    """Per-suite decode budget and stop list."""

    suites = _SUITE_CONFIGS[protocol]
    if suite not in suites:
        raise ValueError(f"unknown QA suite {suite!r}")
    config = suites[suite]
    if max_tokens_override is None:
        return config
    if max_tokens_override < 1:
        raise ValueError("QA max_tokens override must be at least 1")
    return SuiteConfig(max_tokens_override, config.stop)


def qa_max_tokens(
    protocol: QaProtocol,
    override: int | None = None,
    *,
    suite: str = "agent-coding",
) -> int:
    """Return the decode budget. *override* wins when positive."""

    return qa_suite_config(protocol=protocol, suite=suite, max_tokens_override=override).max_tokens


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


def qa_protocol_record(protocol: QaProtocol) -> dict[str, object]:
    """JSON fragment to bind on eval/certificate evidence."""

    per_suite = {
        suite: {"max_tokens": cfg.max_tokens, "stop": list(cfg.stop)}
        for suite, cfg in _SUITE_CONFIGS[protocol].items()
    }
    coding_tokens = _SUITE_CONFIGS[protocol]["agent-coding"].max_tokens
    return {
        "id": protocol,
        "max_tokens_qa": coding_tokens,
        "system_prompts": protocol != PROTOCOL_V64,
        "agent_coding_system": qa_system_prompt("agent-coding", protocol),
        "general_system": qa_system_prompt("general", protocol),
        "per_suite": per_suite,
        "notes": {
            PROTOCOL_V64: "Legacy 64-token user-only chat.",
            PROTOCOL_V256_STRICT: (
                "256-token decode with suite system prompts; single budget for both suites."
            ),
            PROTOCOL_V_EXTRACT: (
                "Fenced Python extract + per-suite budgets (coding 256, general 64) and stop "
                "sequences. Scoring change, not a weight change."
            ),
        }[protocol],
    }
