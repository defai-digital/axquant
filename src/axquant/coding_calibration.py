"""Clean-room agent traces for development calibration, never evaluation tasks."""

from __future__ import annotations

import json
from typing import Any

DOMAINS = ("coding", "json", "tool", "multilingual", "long-context")


def agent_coding_samples() -> list[dict[str, Any]]:
    """Build deterministic CC0 traces with source, failure, patch, and tool output.

    These small authored fixtures exercise calibration plumbing. They do not
    establish representative repository-task coverage or replace a private,
    independently held-out agent evaluation.
    """
    samples: list[dict[str, Any]] = []
    for index in range(32):
        limit = index + 2
        path = f"src/buffer_{index}.py"
        source = (
            "def append_record(records, value, capacity):\n"
            "    if len(records) > capacity:\n"
            '        raise ValueError("full")\n'
            "    return [*records, value]\n"
        )
        test = f"append_record(list(range({limit})), 99, {limit})"
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Fix the capacity boundary in {path}. Preserve the API."},
            {
                "role": "assistant",
                "content": "I will inspect the implementation and its failing test.",
                "tool_calls": [
                    {
                        "id": f"read-{index}",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": path})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"read-{index}", "content": source},
            {"role": "user", "content": f"Test failure: {test}: DID NOT RAISE ValueError."},
            {
                "role": "assistant",
                "content": (
                    f"*** Update File: {path}\n"
                    "-    if len(records) > capacity:\n"
                    "+    if len(records) >= capacity:\n"
                    "The equality case must reject a full buffer."
                ),
            },
            {"role": "user", "content": "The boundary test passes. Check the empty-input case."},
            {
                "role": "assistant",
                "content": (
                    "assert append_record([], 7, 1) == [7]\n"
                    "A positive-capacity empty buffer accepts one record; capacity zero rejects it."
                ),
            },
        ]
        # Three coding trace types: complete repair, repository inspection, and
        # test-driven follow-up. Tool and long-context coverage is also explicit.
        variants = [
            ("coding", messages),
            ("coding", [messages[0], messages[2] | {"role": "user"}, messages[4]]),
            ("coding", [messages[3], messages[4], messages[5], messages[6]]),
        ]
        for variant, (domain, trace) in enumerate(variants):
            # A user message cannot carry a tool-only identifier.
            trace = [
                {k: v for k, v in item.items() if k != "tool_call_id" or item["role"] == "tool"}
                for item in trace
            ]
            samples.append(_record(f"repair-{index:02d}-{variant}", domain, trace))
        if index < 16:
            samples.append(_record(f"tool-{index:02d}", "tool", messages[:3]))
            samples.append(
                _record(
                    f"json-{index:02d}",
                    "json",
                    [
                        {
                            "role": "user",
                            "content": f"Return only JSON for test batch {index}: passed, failed.",
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps({"passed": limit, "failed": 0}),
                        },
                    ],
                )
            )
            # Translate code comments while preserving signatures and identifiers.
            samples.append(
                _record(
                    f"multi-{index:02d}",
                    "multilingual",
                    [
                        {
                            "role": "user",
                            "content": (
                                f"Review {path}. The Spanish issue says: "
                                "rechazar cuando el bufer esta lleno. "
                                "The German note says: Bei voller Kapazitaet ablehnen. "
                                "Explain in English."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": "Both notes require rejecting an append at capacity.",
                        },
                    ],
                )
            )
            context = "\n".join(
                f"module_{part}: owns partition {part}; its writes must preserve insertion order. "
                f"Revision {part + index} changes no public buffer signatures."
                for part in range(96)
            )
            samples.append(
                _record(
                    f"long-{index:02d}",
                    "long-context",
                    [
                        {
                            "role": "user",
                            "content": context + f"\nApply this constraint to {path}:\n{source}",
                        },
                        {
                            "role": "assistant",
                            "content": (
                                "Keep insertion order and the public signature. "
                                "Change only > to >= "
                                "in the capacity check, then test full and empty buffers."
                            ),
                        },
                    ],
                )
            )
    return samples


def _record(sample_id: str, domain: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    # The same serialized trace is available to overlap checking; tokenization
    # renders messages through the pinned source tokenizer's chat template.
    return {
        "id": sample_id,
        "domain": domain,
        "source": "clean-room-authored",
        "license": "CC0-1.0",
        "redistributable": True,
        "messages": messages,
        "text": json.dumps(messages, ensure_ascii=False, sort_keys=True),
    }


def agent_coding_dataset_bytes() -> bytes:
    return (
        "\n".join(
            json.dumps(sample, ensure_ascii=False, sort_keys=True)
            for sample in agent_coding_samples()
        )
        + "\n"
    ).encode("utf-8")
