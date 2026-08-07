from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from axquant.artifact_paths import artifact_member_path
from axquant.errors import ArtifactError, BenchmarkError
from axquant.schema import (
    CodingOverlapMatch,
    CodingOverlapReport,
    CodingScorer,
    CodingSuiteManifest,
    CodingTaskManifest,
    CodingTaskPayload,
    QualityTask,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data, write_text

CODING_SUITE_ID = "axquant-qwen3-next-coding-v2"
CODING_SUITE_VERSION = "2026.08.07.1"
# Shared with campaign-overlap (axquant-token-5gram-v2): ASCII word runs stay
# whole tokens; every other letter (CJK, accented Latin, Kana, Hangul) is a
# single-character token so unspaced scripts still produce shingles.
NORMALIZATION_ALGORITHM = "axquant-token-5gram-v2"
NEAR_DUPLICATE_THRESHOLD = 0.85
SANDBOX_POLICY_CONTRACT = {
    "id": "axquant-macos-seatbelt-v2",
    "default": "allow-runtime-read",
    "network": "deny-all",
    "home": "deny-read-write-except-allowlisted-toolchains",
    "home_metadata": "allow-ancestor-resolution-only",
    "devices": ["/dev/null", "/dev/urandom"],
    "task_fixture": "read-only",
    "task_output": "write-only-scope",
    "limits": ["cpu", "wall", "address-space", "process", "file-size", "open-files"],
}
SANDBOX_PROFILE_SHA256 = stable_sha256(SANDBOX_POLICY_CONTRACT)

_CATEGORY_COUNTS = {
    "python": 24,
    "javascript-typescript": 20,
    "rust": 16,
    "go": 16,
    "repository-context": 16,
    "json-tool": 16,
    "algorithm-reasoning": 12,
    "long-context": 8,
}
_DEFAULT_TOOLCHAINS = {
    "python": "python3",
    "node": "node",
    "typescript": "tsc",
    "rust": "rustc",
    "go": "go",
    "sandbox": "sandbox-exec",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _payload_jsonl(payloads: Iterable[CodingTaskPayload]) -> str:
    return "\n".join(payload.model_dump_json() for payload in payloads) + "\n"


def _python_payload(index: int) -> CodingTaskPayload:
    function_name = f"normalize_records_{index:02d}"
    factor = index % 7 + 2
    offset = index * 3 + 1
    prompt = (
        "Return only valid Python source. Implement "
        f"`{function_name}(records, modulus)` where records is an iterable of integer pairs. "
        "For each pair `(key, value)`, preserve the first occurrence of each key, transform the "
        f"value to `(value * {factor} + {offset}) % modulus`, and return the retained pairs sorted "
        "by key. Reject a non-positive modulus with ValueError and do not mutate the input. "
        "Include type hints and a concise docstring."
    )
    tests = f"""from candidate import {function_name}

assert {function_name}([(3, 4), (1, 2), (3, 99)], 17) == [
    (1, (2 * {factor} + {offset}) % 17),
    (3, (4 * {factor} + {offset}) % 17),
]
source = [(2, -3), (1, 8)]
snapshot = list(source)
assert {function_name}(source, 11) == sorted([
    (2, (-3 * {factor} + {offset}) % 11),
    (1, (8 * {factor} + {offset}) % 11),
])
assert source == snapshot
try:
    {function_name}([], 0)
except ValueError:
    pass
else:
    raise AssertionError("non-positive modulus must fail")
"""
    reference = f'''from collections.abc import Iterable


def {function_name}(
    records: Iterable[tuple[int, int]], modulus: int
) -> list[tuple[int, int]]:
    """Normalize the first value for every key and return pairs ordered by key."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    retained: dict[int, int] = {{}}
    for key, value in records:
        if key not in retained:
            retained[key] = (value * {factor} + {offset}) % modulus
    return sorted(retained.items())
'''
    return CodingTaskPayload(
        task_id=f"python-{index:03d}",
        category="python",
        language="python",
        scorer=CodingScorer.UNIT_TEST,
        prompt=prompt,
        reference=reference,
        candidate_path="candidate.py",
        test_path="test_candidate.py",
        fixture_files={"test_candidate.py": tests},
        target_tokens=320,
    )


def _javascript_typescript_payload(index: int) -> CodingTaskPayload:
    function_name = f"summarizeWindow{index:02d}"
    window = index % 5 + 2
    if index % 2 == 0:
        prompt = (
            "Return only CommonJS JavaScript source. Export a function named "
            f"`{function_name}(values)` that validates all inputs are finite numbers and returns "
            f"an object with `total`, `minimum`, `maximum`, and `windowSums`, where windowSums "
            f"contains every contiguous sum of width {window}. Empty input uses null min/max; "
            "too-short input has an empty windowSums array. Do not mutate the input."
        )
        tests = f"""const assert = require('node:assert/strict');
const candidate = require(process.env.AXQ_CANDIDATE_PATH);
const fn = candidate.{function_name};
assert.equal(typeof fn, 'function');
const source = [3, -1, 4, 2, 5, -2];
const snapshot = source.slice();
const got = fn(source);
assert.equal(got.total, 11);
assert.equal(got.minimum, -2);
assert.equal(got.maximum, 5);
const want = [];
for (let i = 0; i + {window} <= source.length; i += 1) {{
  want.push(source.slice(i, i + {window}).reduce((a, b) => a + b, 0));
}}
assert.deepEqual(got.windowSums, want);
assert.deepEqual(source, snapshot);
assert.deepEqual(fn([]), {{ total: 0, minimum: null, maximum: null, windowSums: [] }});
assert.throws(() => fn([1, Number.NaN]), TypeError);
"""
        reference = f""""use strict";

function {function_name}(values) {{
  if (!Array.isArray(values) || values.some((value) => !Number.isFinite(value))) {{
    throw new TypeError("values must be finite numbers");
  }}
  const total = values.reduce((sum, value) => sum + value, 0);
  const windowSums = [];
  for (let index = 0; index + {window} <= values.length; index += 1) {{
    windowSums.push(values.slice(index, index + {window}).reduce((sum, value) => sum + value, 0));
  }}
  return {{
    total,
    minimum: values.length ? Math.min(...values) : null,
    maximum: values.length ? Math.max(...values) : null,
    windowSums,
  }};
}}

module.exports = {{ {function_name} }};
"""
        return CodingTaskPayload(
            task_id=f"javascript-typescript-{index:03d}",
            category="javascript-typescript",
            language="javascript",
            scorer=CodingScorer.UNIT_TEST,
            prompt=prompt,
            reference=reference,
            candidate_path="candidate.js",
            test_path="test_candidate.js",
            fixture_files={"test_candidate.js": tests},
            target_tokens=768,
        )
    prompt = (
        "Return only strict TypeScript source. Export `type Measurement = { name: string; "
        "value: number }` and a function "
        f"`{function_name}(items: readonly Measurement[]): ReadonlyMap<string, number>`. "
        "The function must sum values by name without mutating input, reject non-finite values, "
        "and expose no `any` types."
    )
    typecheck = f"""import {{ {function_name}, Measurement }} from './candidate';
const values: readonly Measurement[] = [{{ name: 'a', value: 1 }}, {{ name: 'a', value: 2 }}];
const result: ReadonlyMap<string, number> = {function_name}(values);
const total: number | undefined = result.get('a');
void total;
"""
    reference = f"""export type Measurement = {{ name: string; value: number }};

export function {function_name}(
  items: readonly Measurement[],
): ReadonlyMap<string, number> {{
  const totals = new Map<string, number>();
  for (const item of items) {{
    if (!Number.isFinite(item.value)) {{
      throw new TypeError("measurement values must be finite");
    }}
    totals.set(item.name, (totals.get(item.name) ?? 0) + item.value);
  }}
  return totals;
}}
"""
    return CodingTaskPayload(
        task_id=f"javascript-typescript-{index:03d}",
        category="javascript-typescript",
        language="typescript",
        scorer=CodingScorer.COMPILE,
        prompt=prompt,
        reference=reference,
        candidate_path="candidate.ts",
        test_path=None,
        fixture_files={"typecheck.ts": typecheck},
        target_tokens=768,
    )


def _rust_payload(index: int) -> CodingTaskPayload:
    function_name = f"coalesce_ranges_{index:02d}"
    gap = index % 4
    prompt = (
        "Return only Rust 2021 source. Implement `pub fn "
        f"{function_name}(ranges: &[(i64, i64)]) -> Result<Vec<(i64, i64)>, String>`. "
        "Reject any range with start > end, sort a copy by start/end, and merge overlapping "
        f"ranges or ranges separated by at most {gap}. Do not mutate the input and use checked "
        "arithmetic when comparing the allowed gap."
    )
    tests = f"""
#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn merges_and_sorts_without_mutation() {{
        let source = vec![(8, 10), (1, 3), (4 + {gap}, 7), (20, 22)];
        let snapshot = source.clone();
        let got = {function_name}(&source).expect("valid ranges");
        assert_eq!(source, snapshot);
        let mut expected = vec![(1, 3), (4 + {gap}, 10), (20, 22)];
        if 4 + {gap} <= 3 + {gap} + 1 {{ expected = vec![(1, 10), (20, 22)]; }}
        assert_eq!(got, expected);
    }}

    #[test]
    fn rejects_inverted_range() {{
        assert!({function_name}(&[(2, 1)]).is_err());
    }}
}}
"""
    reference = f"""pub fn {function_name}(
    ranges: &[(i64, i64)],
) -> Result<Vec<(i64, i64)>, String> {{
    if ranges.iter().any(|(start, end)| start > end) {{
        return Err("range start exceeds end".to_string());
    }}
    let mut ordered = ranges.to_vec();
    ordered.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (start, end) in ordered {{
        if let Some(last) = merged.last_mut() {{
            let merge_limit = last.1.checked_add({gap} + 1).unwrap_or(i64::MAX);
            if start <= merge_limit {{
                last.1 = last.1.max(end);
                continue;
            }}
        }}
        merged.push((start, end));
    }}
    Ok(merged)
}}
"""
    return CodingTaskPayload(
        task_id=f"rust-{index:03d}",
        category="rust",
        language="rust",
        scorer=CodingScorer.UNIT_TEST,
        prompt=prompt,
        reference=reference,
        candidate_path="candidate.rs",
        test_path="tests.rs",
        fixture_files={"tests.rs": tests},
        target_tokens=768,
    )


def _go_payload(index: int) -> CodingTaskPayload:
    function_name = f"AggregateBuckets{index:02d}"
    divisor = index % 7 + 2
    prompt = (
        "Return only Go source for package `candidate`. Define `type Sample struct { Key string; "
        "Value int }` and implement "
        f"`func {function_name}(samples []Sample) (map[string]int, error)`. Reject empty keys, "
        f"sum values by key, divide each completed sum by {divisor} using Go integer division, "
        "and do not mutate the input. Return an empty non-nil map for empty input."
    )
    tests = f"""package candidate

import "testing"

func TestAggregate(t *testing.T) {{
    source := []Sample{{{{Key: "a", Value: 7}}, {{Key: "b", Value: -3}}, {{Key: "a", Value: 4}}}}
    got, err := {function_name}(source)
    if err != nil {{ t.Fatal(err) }}
    if got["a"] != 11/{divisor} || got["b"] != -3/{divisor} {{
        t.Fatalf("unexpected result: %#v", got)
    }}
    if source[0].Value != 7 {{ t.Fatal("input mutated") }}
    empty, err := {function_name}(nil)
    if err != nil || empty == nil || len(empty) != 0 {{
        t.Fatalf("bad empty result: %#v %v", empty, err)
    }}
}}

func TestRejectsEmptyKey(t *testing.T) {{
    if _, err := {function_name}([]Sample{{{{Key: "", Value: 1}}}}); err == nil {{
        t.Fatal("expected error")
    }}
}}
"""
    reference = f"""package candidate

import "errors"

type Sample struct {{
    Key string
    Value int
}}

func {function_name}(samples []Sample) (map[string]int, error) {{
    totals := make(map[string]int)
    for _, sample := range samples {{
        if sample.Key == "" {{
            return nil, errors.New("sample key must not be empty")
        }}
        totals[sample.Key] += sample.Value
    }}
    for key, total := range totals {{
        totals[key] = total / {divisor}
    }}
    return totals, nil
}}
"""
    return CodingTaskPayload(
        task_id=f"go-{index:03d}",
        category="go",
        language="go",
        scorer=CodingScorer.UNIT_TEST,
        prompt=prompt,
        reference=reference,
        candidate_path="candidate.go",
        test_path="candidate_test.go",
        fixture_files={"candidate_test.go": tests},
        target_tokens=336,
    )


def _repository_payload(index: int) -> CodingTaskPayload:
    limit = index % 6 + 3
    models = """from dataclasses import dataclass

@dataclass(frozen=True)
class Job:
    name: str
    priority: int
    enabled: bool = True
"""
    policy = f"""MAX_SELECTED = {limit}

def allowed_name(name: str) -> bool:
    return bool(name) and not name.startswith('_')
"""
    prompt = (
        "You are editing one file in a small Python repository. Return only the replacement "
        "source for `selector.py`. Read the supplied `models.py` and `policy.py` contracts. "
        "Implement `select_jobs(jobs)` to keep enabled jobs whose names are allowed, retain only "
        "the highest-priority job for duplicate names, sort by descending priority then name, "
        "and return at most `policy.MAX_SELECTED` new list entries. Do not mutate the input.\n\n"
        f"models.py:\n{models}\npolicy.py:\n{policy}"
    )
    tests = """from models import Job
from selector import select_jobs
import policy

source = [
    Job('build', 3), Job('test', 7), Job('build', 9), Job('_hidden', 100),
    Job('docs', 7), Job('off', 99, False), Job('lint', 2), Job('audit', 8),
    Job('pack', 6), Job('ship', 5), Job('scan', 4),
]
snapshot = list(source)
got = select_jobs(source)
assert source == snapshot
assert len(got) <= policy.MAX_SELECTED
assert [job.name for job in got] == [
    job.name for job in sorted(got, key=lambda job: (-job.priority, job.name))
]
assert next(job for job in got if job.name == 'build').priority == 9
assert all(job.enabled and not job.name.startswith('_') for job in got)
"""
    reference = """from collections.abc import Iterable

import policy
from models import Job


def select_jobs(jobs: Iterable[Job]) -> list[Job]:
    selected: dict[str, Job] = {}
    for job in jobs:
        if not job.enabled or not policy.allowed_name(job.name):
            continue
        current = selected.get(job.name)
        if current is None or job.priority > current.priority:
            selected[job.name] = job
    return sorted(selected.values(), key=lambda job: (-job.priority, job.name))[
        : policy.MAX_SELECTED
    ]
"""
    return CodingTaskPayload(
        task_id=f"repository-context-{index:03d}",
        category="repository-context",
        language="python",
        scorer=CodingScorer.UNIT_TEST,
        prompt=prompt,
        reference=reference,
        candidate_path="selector.py",
        test_path="test_selector.py",
        fixture_files={
            "models.py": models,
            "policy.py": policy,
            "test_selector.py": tests,
        },
        target_tokens=384,
    )


def _json_tool_payload(index: int) -> CodingTaskPayload:
    tool_names = ("read_file", "search_code", "run_tests", "apply_patch")
    tool = tool_names[index % len(tool_names)]
    arguments: dict[str, object]
    if tool == "read_file":
        arguments = {"path": f"src/module_{index:02d}.py", "line": index + 1}
    elif tool == "search_code":
        arguments = {"query": f"symbol_{index:02d}", "glob": "src/**/*.py"}
    elif tool == "run_tests":
        arguments = {"target": f"tests/test_{index:02d}.py", "fail_fast": True}
    else:
        arguments = {
            "path": f"src/fix_{index:02d}.py",
            "patch": f"replace sentinel_{index:02d}",
        }
    expected = {"name": tool, "arguments": arguments}
    prompt = (
        "Return exactly one JSON object and no Markdown. The object must contain only `name` "
        f"and `arguments`. Call `{tool}` with this exact arguments object: "
        f"{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
    )
    return CodingTaskPayload(
        task_id=f"json-tool-{index:03d}",
        category="json-tool",
        language="json",
        scorer=CodingScorer.TOOL_EXACT,
        prompt=prompt,
        reference=json.dumps(expected, ensure_ascii=False, sort_keys=True),
        candidate_path="response.json",
        expected_json=expected,
        target_tokens=96,
    )


def _algorithm_payload(index: int) -> CodingTaskPayload:
    function_name = f"shortest_path_{index:02d}"
    prompt = (
        "Return only valid Python source. Implement "
        f"`{function_name}(edges, start, goal)` for a directed graph whose edges are "
        "`(source, destination, nonnegative_cost)` triples. Return `(cost, path)` for the "
        "minimum-cost path, resolving equal costs by lexicographically smallest full path. "
        "Return `None` when unreachable; reject negative costs with ValueError. Nodes are strings. "
        "Use an algorithm that remains correct when a cheaper path is discovered later."
    )
    tests = f"""from candidate import {function_name}

edges = [
    ('s', 'b', 2), ('s', 'a', 2), ('a', 'c', 2), ('b', 'c', 2),
    ('c', 't', 1), ('a', 't', 9), ('s', 't', 20),
]
assert {function_name}(edges, 's', 't') == (5, ['s', 'a', 'c', 't'])
assert {function_name}(edges, 't', 's') is None
assert {function_name}(edges, 's', 's') == (0, ['s'])
try:
    {function_name}([('a', 'b', -1)], 'a', 'b')
except ValueError:
    pass
else:
    raise AssertionError('negative costs must fail')
"""
    reference = f"""import heapq
from collections.abc import Iterable


def {function_name}(
    edges: Iterable[tuple[str, str, int]], start: str, goal: str
) -> tuple[int, list[str]] | None:
    adjacency: dict[str, list[tuple[str, int]]] = {{}}
    for source, destination, cost in edges:
        if cost < 0:
            raise ValueError("edge cost must be nonnegative")
        adjacency.setdefault(source, []).append((destination, cost))
    queue: list[tuple[int, tuple[str, ...], str]] = [(0, (start,), start)]
    best: dict[str, tuple[int, tuple[str, ...]]] = {{start: (0, (start,))}}
    while queue:
        cost, path, node = heapq.heappop(queue)
        if best.get(node) != (cost, path):
            continue
        if node == goal:
            return cost, list(path)
        for destination, edge_cost in adjacency.get(node, []):
            candidate = (cost + edge_cost, (*path, destination))
            if destination not in best or candidate < best[destination]:
                best[destination] = candidate
                heapq.heappush(queue, (*candidate, destination))
    return None
"""
    return CodingTaskPayload(
        task_id=f"algorithm-reasoning-{index:03d}",
        category="algorithm-reasoning",
        language="python",
        scorer=CodingScorer.UNIT_TEST,
        prompt=prompt,
        reference=reference,
        candidate_path="candidate.py",
        test_path="test_candidate.py",
        fixture_files={"test_candidate.py": tests},
        target_tokens=768,
    )


def _long_context_payload(index: int) -> CodingTaskPayload:
    marker = f"QN2-{index:02d}-{(index * 104729 + 7919):08d}"
    target_file = 53 + index * 17
    records = []
    for record in range(220):
        suffix = f" release_guard={marker}" if record == target_file else ""
        status = "ready" if record % 4 else "hold"
        records.append(
            f"src/component_{record:03d}.rs owner=team_{(record * 13) % 29:02d} "
            f"checksum={(record * 65537 + index):08x} status={status}{suffix}"
        )
    prompt = (
        "Inspect the repository inventory below. Find the only record containing `release_guard=` "
        "and return exactly its guard value, with no prose.\n\n" + "\n".join(records)
    )
    return CodingTaskPayload(
        task_id=f"long-context-{index:03d}",
        category="long-context",
        language="text",
        scorer=CodingScorer.TEXT_EXACT,
        prompt=prompt,
        reference=marker,
        candidate_path="answer.txt",
        expected_text=marker,
        target_tokens=64,
    )


def reference_coding_payloads() -> list[CodingTaskPayload]:
    factories = {
        "python": _python_payload,
        "javascript-typescript": _javascript_typescript_payload,
        "rust": _rust_payload,
        "go": _go_payload,
        "repository-context": _repository_payload,
        "json-tool": _json_tool_payload,
        "algorithm-reasoning": _algorithm_payload,
        "long-context": _long_context_payload,
    }
    return [
        factories[category](index)
        for category, count in _CATEGORY_COUNTS.items()
        for index in range(count)
    ]


# ASCII word runs stay whole tokens; every other letter becomes a
# single-character token (same pattern as dataset_overlap.NORMALIZATION_ALGORITHM).
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[^\W\d_]")


def _normalized_tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.casefold())


def _fingerprint(value: str) -> set[tuple[str, ...]]:
    tokens = _normalized_tokens(value)
    if len(tokens) < 5:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + 5]) for index in range(len(tokens) - 4)}


def _similarity(left: str, right: str) -> float:
    left_fingerprint = _fingerprint(left)
    right_fingerprint = _fingerprint(right)
    if not left_fingerprint or not right_fingerprint:
        return float(left_fingerprint == right_fingerprint)
    return len(left_fingerprint & right_fingerprint) / len(left_fingerprint | right_fingerprint)


def _calibration_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                record_id = payload.get("id")
                text = payload.get("text")
                if not isinstance(record_id, str) or not isinstance(text, str):
                    raise BenchmarkError(
                        f"invalid calibration overlap record at {path}:{line_number}"
                    )
                records.append((record_id, text))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot load calibration overlap dataset {path}: {exc}") from exc
    if not records:
        raise BenchmarkError("calibration overlap dataset contains no records")
    return records


def build_overlap_report(
    *,
    payloads: list[CodingTaskPayload],
    suite_dataset_sha256: str,
    calibration_path: str | Path,
    similarity_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> CodingOverlapReport:
    calibration = Path(calibration_path).expanduser().resolve()
    matches: list[CodingOverlapMatch] = []
    for payload in payloads:
        task_text = f"{payload.prompt}\n{payload.reference or ''}"
        task_normalized = " ".join(_normalized_tokens(task_text))
        for calibration_id, calibration_text in _calibration_records(calibration):
            calibration_normalized = " ".join(_normalized_tokens(calibration_text))
            exact = task_normalized == calibration_normalized
            similarity = 1.0 if exact else _similarity(task_text, calibration_text)
            if exact or similarity >= similarity_threshold:
                matches.append(
                    CodingOverlapMatch(
                        task_id=payload.task_id,
                        calibration_id=calibration_id,
                        similarity=similarity,
                        exact=exact,
                    )
                )
    return CodingOverlapReport(
        suite_dataset_sha256=suite_dataset_sha256,
        calibration_dataset_sha256=file_sha256(calibration),
        similarity_threshold=similarity_threshold,
        matches=matches,
        passed=not matches,
    )


def build_general_overlap_report(
    *,
    general_dataset_path: str | Path,
    calibration_path: str | Path,
    similarity_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> CodingOverlapReport:
    """Compare the general holdout with calibration using the frozen 5-gram algorithm."""

    from axquant.quality import load_quality_tasks

    general_dataset = Path(general_dataset_path).expanduser().resolve()
    calibration = Path(calibration_path).expanduser().resolve()
    tasks = load_quality_tasks(general_dataset)
    matches: list[CodingOverlapMatch] = []
    for task in tasks:
        task_text = f"{task.prompt}\n{task.reference or ''}\n{task.perplexity_text or ''}"
        task_normalized = " ".join(_normalized_tokens(task_text))
        for calibration_id, calibration_text in _calibration_records(calibration):
            calibration_normalized = " ".join(_normalized_tokens(calibration_text))
            exact = task_normalized == calibration_normalized
            similarity = 1.0 if exact else _similarity(task_text, calibration_text)
            if exact or similarity >= similarity_threshold:
                matches.append(
                    CodingOverlapMatch(
                        task_id=task.task_id,
                        calibration_id=calibration_id,
                        similarity=similarity,
                        exact=exact,
                    )
                )
    return CodingOverlapReport(
        suite_dataset_sha256=file_sha256(general_dataset),
        calibration_dataset_sha256=file_sha256(calibration),
        similarity_threshold=similarity_threshold,
        matches=matches,
        passed=not matches,
    )


def coding_general_overlap_issues(
    *,
    coding_payloads: list[CodingTaskPayload],
    general_tasks: list[QualityTask],
    similarity_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[str]:
    """Return deterministic coding/general holdout overlap blockers."""

    issues: list[str] = []
    for coding_task in coding_payloads:
        coding_text = f"{coding_task.prompt}\n{coding_task.reference or ''}"
        coding_normalized = " ".join(_normalized_tokens(coding_text))
        for general_task in general_tasks:
            general_text = (
                f"{general_task.prompt}\n{general_task.reference or ''}\n"
                f"{general_task.perplexity_text or ''}"
            )
            general_normalized = " ".join(_normalized_tokens(general_text))
            exact = coding_normalized == general_normalized
            similarity = 1.0 if exact else _similarity(coding_text, general_text)
            if exact or similarity >= similarity_threshold:
                issues.append(
                    "coding/general holdout overlap: "
                    f"{coding_task.task_id} vs {general_task.task_id} "
                    f"(similarity={similarity:.6f}, exact={exact})"
                )
    return issues


def probe_toolchains(
    executables: dict[str, str] | None = None,
) -> dict[str, str]:
    configured = {**_DEFAULT_TOOLCHAINS, **(executables or {})}
    resolved_executables = {
        name: resolved
        for name, executable in configured.items()
        if (resolved := shutil.which(executable)) is not None
    }
    probe_environment = {
        **os.environ,
        "PATH": os.pathsep.join(
            [
                *(sorted({str(Path(path).parent) for path in resolved_executables.values()})),
                os.environ.get("PATH", ""),
            ]
        ),
    }
    identities: dict[str, str] = {}
    for name in configured:
        resolved = resolved_executables.get(name)
        if resolved is None:
            identities[name] = "unavailable"
            continue
        version_args = [resolved, "version" if name == "go" else "--version"]
        if name == "sandbox":
            identities[name] = f"{resolved} :: macOS Seatbelt"
            continue
        try:
            result = subprocess.run(
                version_args,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                env=probe_environment,
            )
        except (OSError, subprocess.TimeoutExpired):
            identities[name] = "unavailable"
            continue
        first_line = (result.stdout or result.stderr).strip().splitlines()
        identities[name] = (
            f"{resolved} :: {first_line[0]}"
            if result.returncode == 0 and first_line
            else "unavailable"
        )
    return identities


def build_coding_suite(
    output_dir: str | Path,
    *,
    calibration_path: str | Path,
    random_seed: int = 20260803,
    toolchain_executables: dict[str, str] | None = None,
) -> CodingSuiteManifest:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payloads = reference_coding_payloads()
    shards: dict[str, str] = {}
    task_manifests: list[CodingTaskManifest] = []
    for category in _CATEGORY_COUNTS:
        category_payloads = [payload for payload in payloads if payload.category == category]
        shard_name = f"tasks-{category}.jsonl"
        shard_path = directory / shard_name
        write_text(shard_path, _payload_jsonl(category_payloads))
        shards[shard_name] = file_sha256(shard_path)
        for payload in category_payloads:
            task_manifests.append(
                CodingTaskManifest(
                    task_id=payload.task_id,
                    category=payload.category,
                    language=payload.language,
                    prompt_sha256=_sha256_text(payload.prompt),
                    reference_sha256=(
                        _sha256_text(payload.reference) if payload.reference is not None else None
                    ),
                    payload_sha256=stable_sha256(payload),
                    scorer=payload.scorer,
                    license_id="CC0-1.0",
                    provenance="clean-room-authored by the AXQuant project",
                    target_tokens=payload.target_tokens,
                    timeout_seconds=20.0,
                    cpu_time_seconds=15,
                    memory_limit_bytes=1_073_741_824,
                    process_limit=32,
                    output_limit_bytes=1_048_576,
                    file_size_limit_bytes=268_435_456,
                    open_file_limit=256,
                    long_context=payload.category == "long-context",
                )
            )
    dataset_sha256 = stable_sha256(shards)
    overlap = build_overlap_report(
        payloads=payloads,
        suite_dataset_sha256=dataset_sha256,
        calibration_path=calibration_path,
    )
    overlap_path = directory / "coding-overlap-report.json"
    write_data(overlap_path, overlap)
    manifest = CodingSuiteManifest(
        suite_id=CODING_SUITE_ID,
        version=CODING_SUITE_VERSION,
        dataset_sha256=dataset_sha256,
        tasks=task_manifests,
        task_shards=shards,
        calibration_overlap_attested=overlap.passed,
        calibration_overlap_report=overlap_path.name,
        calibration_overlap_report_sha256=file_sha256(overlap_path),
        toolchains=probe_toolchains(toolchain_executables),
        sandbox_profile_sha256=SANDBOX_PROFILE_SHA256,
        near_duplicate_threshold=NEAR_DUPLICATE_THRESHOLD,
        random_seed=random_seed,
    )
    write_data(directory / "coding-suite-manifest.json", manifest)
    return manifest


def load_coding_payloads(
    manifest_path: str | Path,
    manifest: CodingSuiteManifest | None = None,
) -> list[CodingTaskPayload]:
    source_path = Path(manifest_path).expanduser()
    if source_path.is_symlink():
        raise ArtifactError("coding suite manifest cannot be a symbolic link")
    path = source_path.resolve()
    recorded_suite = load_model(path, CodingSuiteManifest)
    if manifest is not None and manifest != recorded_suite:
        raise ArtifactError("supplied coding suite manifest differs from its on-disk record")
    suite = manifest or recorded_suite
    if suite.dataset_sha256 != stable_sha256(suite.task_shards):
        raise ArtifactError("coding suite dataset digest does not bind its shard checksums")
    if suite.sandbox_profile_sha256 != SANDBOX_PROFILE_SHA256:
        raise ArtifactError("coding suite sandbox policy does not match this AXQuant version")
    payloads: list[CodingTaskPayload] = []
    for shard_name, expected_sha256 in suite.task_shards.items():
        try:
            shard_path = artifact_member_path(path.parent, shard_name)
        except ValueError as exc:
            raise ArtifactError(f"coding suite shard path is unsafe: {shard_name}") from exc
        if (
            shard_path.is_symlink()
            or not shard_path.is_file()
            or file_sha256(shard_path) != expected_sha256
        ):
            raise ArtifactError(f"coding suite shard is missing or stale: {shard_name}")
        try:
            with shard_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        payloads.append(CodingTaskPayload.model_validate_json(line))
                    except ValueError as exc:
                        raise ArtifactError(
                            f"invalid coding payload at {shard_path}:{line_number}: {exc}"
                        ) from exc
        except OSError as exc:
            raise ArtifactError(f"cannot read coding suite shard {shard_path}: {exc}") from exc
    manifests = {task.task_id: task for task in suite.tasks}
    by_id = {payload.task_id: payload for payload in payloads}
    if len(by_id) != len(payloads) or set(by_id) != set(manifests):
        raise ArtifactError("coding suite payload membership differs from its manifest")
    for task_id, task in manifests.items():
        payload = by_id[task_id]
        if (
            payload.category != task.category
            or payload.language != task.language
            or payload.scorer is not task.scorer
            or payload.target_tokens != task.target_tokens
            or _sha256_text(payload.prompt) != task.prompt_sha256
            or (_sha256_text(payload.reference) if payload.reference is not None else None)
            != task.reference_sha256
            or stable_sha256(payload) != task.payload_sha256
        ):
            raise ArtifactError(f"coding suite payload binding differs for {task_id}")
    return [by_id[task.task_id] for task in suite.tasks]
