from __future__ import annotations

import json
from pathlib import Path

from axquant.schema import BenchmarkSuiteManifest, QualityCheck, QualityTask
from axquant.serde import file_sha256, write_data, write_text

_VERSION = "2026.07.31.1"


def _jsonl(tasks: list[QualityTask]) -> str:
    return "\n".join(task.model_dump_json() for task in tasks) + "\n"


def _coding_tasks() -> list[QualityTask]:
    specs = (
        ("add", "Return the sum of two numbers."),
        ("clamp", "Clamp a number between lower and upper bounds."),
        ("is_palindrome", "Return whether a string is a palindrome."),
        ("deduplicate", "Remove duplicates from a list while preserving order."),
        ("chunks", "Split a list into chunks of a positive size."),
        ("safe_divide", "Divide two numbers and return None when the divisor is zero."),
        ("flatten_once", "Flatten one nesting level from a list of lists."),
        ("word_counts", "Return a dictionary containing case-insensitive word counts."),
        ("parse_bool", "Parse true/false, yes/no, and 1/0 strings into booleans."),
        ("merge_sorted", "Merge two sorted lists without calling sorted."),
    )
    return [
        QualityTask(
            task_id=f"coding-{index:02d}",
            category="coding",
            prompt=(
                f"Write only valid Python defining a function named `{name}`. {instruction} "
                "Do not include prose outside the code."
            ),
            reference=f"def {name}(...): ...",
            perplexity_text=f"Python task: {instruction}",
            checks=[
                QualityCheck(kind="python-syntax"),
                QualityCheck(kind="contains", value=f"def {name}"),
            ],
        )
        for index, (name, instruction) in enumerate(specs)
    ]


def _tool_tasks() -> list[QualityTask]:
    specs = (
        ("search_web", "Find current MLX documentation", ["query"]),
        ("read_file", "Read /tmp/config.json", ["path"]),
        ("write_file", "Write hello to /tmp/hello.txt", ["path", "content"]),
        ("get_weather", "Get Toronto weather in Celsius", ["location", "units"]),
        ("query_database", "Select active users from the users table", ["query"]),
        ("execute_python", "Calculate the first ten Fibonacci numbers", ["code"]),
        ("create_calendar_event", "Schedule a review tomorrow at 10 AM", ["title", "time"]),
        ("translate_text", "Translate good morning to Japanese", ["text", "language"]),
        ("send_email", "Email the build report to ops@example.com", ["to", "subject"]),
        ("get_stock_price", "Look up AAPL", ["ticker"]),
    )
    return [
        QualityTask(
            task_id=f"tool-{index:02d}",
            category="tool",
            prompt=(
                "Return only one JSON object with keys `name` and `arguments`. "
                f"Choose the `{name}` tool and provide arguments for: {instruction}"
            ),
            reference=name,
            perplexity_text=f"Tool selection: {instruction}",
            checks=[
                QualityCheck(kind="json-valid"),
                QualityCheck(kind="json-keys", value=["name", "arguments"]),
                QualityCheck(kind="contains", value=name),
            ],
        )
        for index, (name, instruction, _keys) in enumerate(specs)
    ]


def _json_tasks() -> list[QualityTask]:
    specs = (
        ("person", ["name", "age"]),
        ("build", ["status", "duration_seconds"]),
        ("error", ["code", "message"]),
        ("location", ["city", "country"]),
        ("package", ["name", "version"]),
        ("test", ["passed", "failed"]),
        ("patch", ["file", "summary"]),
        ("metric", ["name", "value"]),
        ("translation", ["source", "target"]),
        ("tool_result", ["ok", "result"]),
    )
    return [
        QualityTask(
            task_id=f"json-{index:02d}",
            category="json",
            prompt=(
                f"Return only valid JSON describing a {topic}. The object must contain "
                f"exactly the required fields: {', '.join(keys)}."
            ),
            reference=json.dumps({key: "value" for key in keys}, sort_keys=True),
            perplexity_text=f"Structured output schema for {topic}: {', '.join(keys)}",
            checks=[
                QualityCheck(kind="json-valid"),
                QualityCheck(kind="json-keys", value=keys),
            ],
        )
        for index, (topic, keys) in enumerate(specs)
    ]


def _multilingual_tasks() -> list[QualityTask]:
    specs = (
        ("Japanese", "Reply with the Japanese greeting おはようございます.", "おはようございます"),
        (
            "Japanese",
            "Reply with the Japanese phrase ありがとうございます.",
            "ありがとうございます",
        ),
        ("Japanese", "Reply with the Japanese phrase 問題を修正しました.", "問題を修正しました"),
        (
            "Japanese",
            "Reply with the Japanese phrase テストは成功しました.",
            "テストは成功しました",
        ),
        ("Traditional Chinese", "請只回答: 建置已成功。", "建置已成功"),
        ("Traditional Chinese", "請只回答: 資料格式正確。", "資料格式正確"),
        ("Traditional Chinese", "請只回答: 工具呼叫完成。", "工具呼叫完成"),
        ("Traditional Chinese", "請只回答: 測試全部通過。", "測試全部通過"),
        ("Simplified Chinese", "请只回答: 构建已成功。", "构建已成功"),
        ("Simplified Chinese", "请只回答: 数据格式正确。", "数据格式正确"),
        ("Simplified Chinese", "请只回答: 工具调用完成。", "工具调用完成"),
        ("Simplified Chinese", "请只回答: 测试全部通过。", "测试全部通过"),
    )
    return [
        QualityTask(
            task_id=f"multilingual-{index:02d}",
            category="multilingual",
            prompt=f"Answer in {language}. {instruction}",
            reference=expected,
            perplexity_text=f"{instruction}\n{expected}",
            checks=[QualityCheck(kind="contains", value=expected)],
        )
        for index, (language, instruction, expected) in enumerate(specs)
    ]


def _long_context_tasks() -> list[QualityTask]:
    tasks: list[QualityTask] = []
    for index in range(10):
        marker = f"AXQ-{index:02d}-{(index * 7919 + 104729):06d}"
        marker_line = 17 + index * 9
        lines = [
            (
                f"Record {line:03d}: component=module_{(line * 17 + index) % 97:02d}; "
                f"state={'ready' if line % 3 else 'pending'}; checksum={(line * 65537):08x}."
            )
            for line in range(160)
        ]
        lines[marker_line] += f" RELEASE_MARKER={marker}."
        document = "\n".join(lines)
        tasks.append(
            QualityTask(
                task_id=f"long-context-{index:02d}",
                category="long_context",
                prompt=(
                    "Read the records below and return only the value following "
                    "`RELEASE_MARKER=`.\n\n"
                    f"{document}\n\nEnd of records. Answer with the marker value only:"
                ),
                reference=marker,
                perplexity_text=document,
                checks=[QualityCheck(kind="exact", value=marker)],
            )
        )
    return tasks


def _general_structured_tasks() -> list[QualityTask]:
    """Structured-output tasks required for dual-profile release validation.

    Release validation always compares ``json_valid_rate`` and ``syntax_valid_rate``.
    Agent-coding already supplies those metrics; the general profile must emit the same
    governed pairs so dual-profile certification is complete by construction.
    """
    json_specs = (
        ("status", ["ok", "message"]),
        ("summary", ["title", "count"]),
        ("result", ["id", "value"]),
    )
    json_tasks = [
        QualityTask(
            task_id=f"general-json-{index:02d}",
            category="json",
            prompt=(
                f"Return only valid JSON describing a {topic}. The object must contain "
                f"exactly the required fields: {', '.join(keys)}."
            ),
            reference=json.dumps({key: "value" for key in keys}, sort_keys=True),
            perplexity_text=f"General structured output for {topic}: {', '.join(keys)}",
            checks=[
                QualityCheck(kind="json-valid"),
                QualityCheck(kind="json-keys", value=keys),
            ],
        )
        for index, (topic, keys) in enumerate(json_specs)
    ]
    syntax_specs = (
        ("double", "Return twice an integer."),
        ("is_even", "Return whether an integer is even."),
        ("first_item", "Return the first item of a non-empty list."),
    )
    syntax_tasks = [
        QualityTask(
            task_id=f"general-syntax-{index:02d}",
            category="coding",
            prompt=(
                f"Write only valid Python defining a function named `{name}`. {instruction} "
                "Do not include prose outside the code."
            ),
            reference=f"def {name}(...): ...",
            perplexity_text=f"General Python task: {instruction}",
            checks=[
                QualityCheck(kind="python-syntax"),
                QualityCheck(kind="contains", value=f"def {name}"),
            ],
        )
        for index, (name, instruction) in enumerate(syntax_specs)
    ]
    return json_tasks + syntax_tasks


def _general_tasks() -> list[QualityTask]:
    specs = (
        ("reasoning", "If all A are B and no B are C, can any A be C?", "no"),
        ("reasoning", "A box has 3 red and 2 blue balls. How many balls total?", "5"),
        ("instruction", "Reply with exactly the word READY.", "READY"),
        ("instruction", "Write the lowercase form of QUANTIZATION.", "quantization"),
        ("factual", "What planet is known as the Red Planet?", "Mars"),
        ("factual", "What is the chemical symbol for water?", "H2O"),
        ("prose", "Complete: The opposite of hot is ___.", "cold"),
        ("prose", "Complete: A triangle has ___ sides.", "three"),
        ("multilingual", "Translate hello to Japanese.", "こんにちは"),
        ("multilingual", "Translate thank you to Traditional Chinese.", "謝謝"),
    )
    prose_tasks = [
        QualityTask(
            task_id=f"general-{index:02d}",
            category=category,
            prompt=f"{prompt} Return only the answer.",
            reference=answer,
            perplexity_text=f"{prompt}\n{answer}",
            checks=[QualityCheck(kind="exact", value=answer)],
        )
        for index, (category, prompt, answer) in enumerate(specs)
    ]
    return prose_tasks + _general_structured_tasks()


def build_benchmark_suites(
    output_dir: str | Path,
    *,
    random_seed: int = 20260728,
) -> BenchmarkSuiteManifest:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    quality_tasks = (
        _coding_tasks()
        + _tool_tasks()
        + _json_tasks()
        + _multilingual_tasks()
        + _long_context_tasks()
    )
    general_tasks = _general_tasks()
    agent_prompt_records = [
        {"text": task.prompt, "task_id": task.task_id, "category": task.category}
        for task in quality_tasks[::3]
    ]
    general_prompt_records = [
        {"text": task.prompt, "task_id": task.task_id, "category": task.category}
        for task in general_tasks
    ]
    files = {
        "agent_coding_quality": "agent-coding-quality-v2.jsonl",
        "general_quality": "general-quality-v2.jsonl",
        "agent_coding_ax_engine_prompts": "agent-coding-ax-engine-prompts-v3.jsonl",
        "general_ax_engine_prompts": "general-ax-engine-prompts-v3.jsonl",
    }
    write_text(directory / files["agent_coding_quality"], _jsonl(quality_tasks))
    write_text(directory / files["general_quality"], _jsonl(general_tasks))
    write_text(
        directory / files["agent_coding_ax_engine_prompts"],
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in agent_prompt_records
        )
        + "\n",
    )
    write_text(
        directory / files["general_ax_engine_prompts"],
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True)
            for record in general_prompt_records
        )
        + "\n",
    )
    manifest = BenchmarkSuiteManifest(
        suite_id="axquant-qwen36-v1",
        version=_VERSION,
        random_seed=random_seed,
        files=files,
        sha256={name: file_sha256(directory / path) for name, path in files.items()},
        samples={
            "agent_coding_quality": len(quality_tasks),
            "general_quality": len(general_tasks),
            "agent_coding_ax_engine_prompts": len(agent_prompt_records),
            "general_ax_engine_prompts": len(general_prompt_records),
        },
        notes=[
            "Evaluation prompts are authored for AXQuant and are not present in calibration.",
            "Long-context tasks use deterministic synthetic records and hidden retrieval markers.",
            "Agent-coding and general runtime prompts are disjoint by task ID and file digest.",
            "General quality includes structured JSON and Python-syntax tasks so dual-profile "
            "release validation always receives governed json_valid_rate and syntax_valid_rate.",
        ],
    )
    write_data(directory / "suite-manifest.json", manifest)
    return manifest
