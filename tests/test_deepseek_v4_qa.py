from __future__ import annotations

import pytest

from axquant.deepseek_v4_qa import (
    AGENT_CODING_SYSTEM,
    GENERAL_SYSTEM,
    PROTOCOL_V64,
    PROTOCOL_V256_STRICT,
    build_qa_messages,
    normalize_qa_protocol,
    qa_max_tokens,
    qa_protocol_record,
    qa_system_prompt,
)


def test_normalize_qa_protocol_aliases() -> None:
    assert normalize_qa_protocol(None) == PROTOCOL_V256_STRICT
    assert normalize_qa_protocol("v64") == PROTOCOL_V64
    assert normalize_qa_protocol("legacy") == PROTOCOL_V64
    assert normalize_qa_protocol("v256-strict") == PROTOCOL_V256_STRICT
    assert normalize_qa_protocol("256") == PROTOCOL_V256_STRICT
    with pytest.raises(ValueError, match="unknown DSV4 QA protocol"):
        normalize_qa_protocol("v512")


def test_v64_is_user_only_64_tokens() -> None:
    assert qa_max_tokens(PROTOCOL_V64) == 64
    assert qa_system_prompt("agent-coding", PROTOCOL_V64) is None
    assert qa_system_prompt("general", PROTOCOL_V64) is None
    assert build_qa_messages("agent-coding", "Write fizzbuzz", PROTOCOL_V64) == [
        {"role": "user", "content": "Write fizzbuzz"}
    ]


def test_v256_strict_adds_suite_system_prompts() -> None:
    assert qa_max_tokens(PROTOCOL_V256_STRICT) == 256
    coding = build_qa_messages("agent-coding", "Write fizzbuzz", PROTOCOL_V256_STRICT)
    general = build_qa_messages("general", "Say cold", PROTOCOL_V256_STRICT)
    assert coding[0] == {"role": "system", "content": AGENT_CODING_SYSTEM}
    assert coding[1] == {"role": "user", "content": "Write fizzbuzz"}
    assert general[0] == {"role": "system", "content": GENERAL_SYSTEM}
    assert "markdown" in AGENT_CODING_SYSTEM.lower()
    assert "exactly" in GENERAL_SYSTEM.lower()


def test_max_tokens_override_and_record() -> None:
    assert qa_max_tokens(PROTOCOL_V64, override=256) == 256
    with pytest.raises(ValueError, match="at least 1"):
        qa_max_tokens(PROTOCOL_V256_STRICT, override=0)
    record = qa_protocol_record(PROTOCOL_V256_STRICT, 256)
    assert record["id"] == PROTOCOL_V256_STRICT
    assert record["system_prompts"] is True
    assert record["max_tokens_qa"] == 256
    assert record["agent_coding_system"] == AGENT_CODING_SYSTEM
