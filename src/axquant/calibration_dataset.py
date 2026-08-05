from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REQUIRED_DOMAINS = {"coding", "json", "tool", "multilingual", "long-context"}
_MIN_SAMPLES = 128
_MIN_ESTIMATED_TOKENS = 8192
_MIN_LONG_CONTEXT_CHARS = 2000


def _text_length(sample: dict[str, Any]) -> int:
    """Character count of the sample text; null / non-string text counts as 0."""
    text = sample.get("text")
    return len(text) if isinstance(text, str) else 0


def validate_calibration_dataset(path: Path) -> list[str]:
    issues: list[str] = []
    if not path.exists():
        return [f"dataset file not found: {path}"]

    samples: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                issues.append(f"line {line_number}: not a JSON object")
                continue
            samples.append(obj)
    except json.JSONDecodeError as exc:
        return [f"invalid JSONL: {exc}"]

    if len(samples) < _MIN_SAMPLES:
        issues.append(f"sample count {len(samples)} < {_MIN_SAMPLES} minimum")

    # Accept str or int IDs, but normalize to str so mixed types never crash
    # sorted() and so 1 / "1" are treated as the same identity.
    ids: list[str] = [
        str(sample_id) for s in samples if isinstance((sample_id := s.get("id")), (str, int))
    ]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for sample_id in ids:
        if sample_id in seen:
            duplicates.add(sample_id)
        seen.add(sample_id)
    if duplicates:
        issues.append(f"duplicate IDs: {sorted(duplicates)[:5]}")

    missing_domain = [
        s.get("id", f"index-{i}") for i, s in enumerate(samples) if not s.get("domain")
    ]
    if missing_domain:
        issues.append(f"samples missing 'domain' field: {missing_domain[:5]}")

    domains_present: set[str] = {
        domain for s in samples if isinstance(domain := s.get("domain"), str) and domain
    }
    missing_required = _REQUIRED_DOMAINS - domains_present
    if missing_required:
        issues.append(f"missing required domains: {sorted(missing_required)}")

    missing_text = [
        s.get("id", f"index-{i}")
        for i, s in enumerate(samples)
        if not s.get("text") and not s.get("messages")
    ]
    if missing_text:
        issues.append(f"samples missing 'text' or 'messages': {missing_text[:5]}")

    total_chars: int = sum(_text_length(s) for s in samples)
    estimated_tokens = total_chars // 4
    if estimated_tokens < _MIN_ESTIMATED_TOKENS:
        issues.append(f"estimated tokens {estimated_tokens} < {_MIN_ESTIMATED_TOKENS} minimum")

    long_context = [s for s in samples if s.get("domain") == "long-context"]
    if long_context:
        shortest: int = min(_text_length(s) for s in long_context)
        if shortest < _MIN_LONG_CONTEXT_CHARS:
            issues.append(
                f"shortest long-context sample is {shortest} chars "
                f"(< {_MIN_LONG_CONTEXT_CHARS} minimum)"
            )
    elif "long-context" in domains_present:
        issues.append("long-context domain present but has no samples")

    return issues
