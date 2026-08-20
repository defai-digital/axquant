from __future__ import annotations

import pytest

from axquant.deepseek_v4_qa import (
    AGENT_CODING_SYSTEM,
    GENERAL_SYSTEM,
    PROTOCOL_V64,
    PROTOCOL_V256_STRICT,
    PROTOCOL_V_EXTRACT,
    build_qa_messages,
    normalize_qa_protocol,
    qa_max_tokens,
    qa_protocol_record,
    qa_suite_config,
    qa_system_prompt,
)


def test_normalize_qa_protocol_aliases() -> None:
    assert normalize_qa_protocol(None) == PROTOCOL_V_EXTRACT
    assert normalize_qa_protocol("v64") == PROTOCOL_V64
    assert normalize_qa_protocol("legacy") == PROTOCOL_V64
    assert normalize_qa_protocol("v256-strict") == PROTOCOL_V256_STRICT
    assert normalize_qa_protocol("extract") == PROTOCOL_V_EXTRACT
    with pytest.raises(ValueError, match="unknown DSV4 QA protocol"):
        normalize_qa_protocol("v512")


def test_v64_is_user_only_64_tokens() -> None:
    assert qa_max_tokens(PROTOCOL_V64) == 64
    assert qa_system_prompt("agent-coding", PROTOCOL_V64) is None
    assert qa_system_prompt("general", PROTOCOL_V64) is None
    assert build_qa_messages("agent-coding", "Write fizzbuzz", PROTOCOL_V64) == [
        {"role": "user", "content": "Write fizzbuzz"}
    ]


def test_v_extract_splits_budgets_and_stops() -> None:
    coding = qa_suite_config("agent-coding", PROTOCOL_V_EXTRACT)
    general = qa_suite_config("general", PROTOCOL_V_EXTRACT)
    assert coding.max_tokens == 256
    assert general.max_tokens == 64
    assert "</think>" in coding.stop
    assert "\n\n" in general.stop
    messages = build_qa_messages("agent-coding", "Write fizzbuzz", PROTOCOL_V_EXTRACT)
    assert messages[0] == {"role": "system", "content": AGENT_CODING_SYSTEM}
    assert "exactly" in GENERAL_SYSTEM.lower()


def test_v256_strict_single_budget() -> None:
    assert qa_max_tokens(PROTOCOL_V256_STRICT) == 256
    assert qa_suite_config("general", PROTOCOL_V256_STRICT).max_tokens == 256
    assert qa_suite_config("general", PROTOCOL_V256_STRICT).stop == ()


def test_max_tokens_override_and_record() -> None:
    assert qa_suite_config("agent-coding", PROTOCOL_V64, max_tokens_override=256).max_tokens == 256
    with pytest.raises(ValueError, match="at least 1"):
        qa_suite_config("agent-coding", PROTOCOL_V_EXTRACT, max_tokens_override=0)
    record = qa_protocol_record(PROTOCOL_V_EXTRACT)
    assert record["id"] == PROTOCOL_V_EXTRACT
    assert record["system_prompts"] is True
    assert record["per_suite"]["general"]["max_tokens"] == 64  # type: ignore[index]
