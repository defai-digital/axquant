from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError, ValidationGateError
from axquant.schema import CampaignOverlapMatch, CampaignOverlapReport
from axquant.serde import file_sha256

NORMALIZATION_ALGORITHM = "axquant-token-5gram-v2"
DEFAULT_TEXT_FIELDS = ("text", "prompt", "reference", "perplexity_text")
DEFAULT_MAX_COMPARISON_PAIRS = 10_000_000

# ASCII word runs stay whole tokens; every other letter (CJK, accented Latin,
# Kana, Hangul, ...) becomes a single-character token so unspaced scripts still
# produce shingles instead of failing the empty-normalization gate.
_TOKEN_RE = re.compile(r"[a-z0-9_]+|[^\W\d_]")


def _normalized_tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.casefold())


def _normalized_text(value: str) -> str:
    return " ".join(_normalized_tokens(value))


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


def _record_text(payload: dict[str, Any], text_fields: tuple[str, ...]) -> str:
    values = [
        value.strip()
        for field in text_fields
        if isinstance((value := payload.get(field)), str) and value.strip()
    ]
    if not values:
        raise ValidationGateError(
            f"dataset record contains none of the required text fields: {list(text_fields)}"
        )
    return "\n".join(values)


def _load_records(
    path: Path,
    *,
    id_field: str,
    text_fields: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"overlap dataset must be an existing non-symlink file: {path}")
    records: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValidationGateError(
                        f"dataset record is not an object at {path}:{line_number}"
                    )
                record_id = payload.get(id_field)
                if not isinstance(record_id, str) or not record_id.strip():
                    raise ValidationGateError(
                        f"dataset record has no string {id_field!r} at {path}:{line_number}"
                    )
                if record_id in seen_ids:
                    raise ValidationGateError(
                        f"dataset contains duplicate record id {record_id!r}: {path}"
                    )
                seen_ids.add(record_id)
                text = _record_text(payload, text_fields)
                normalized = _normalized_text(text)
                if not normalized:
                    raise ValidationGateError(
                        f"dataset record normalizes to empty text at {path}:{line_number}"
                    )
                record_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                records.append((record_id, text, record_sha))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot load overlap dataset {path}: {exc}") from exc
    if not records:
        raise ValidationGateError(f"overlap dataset contains no records: {path}")
    return records


def _safe_dataset_path(value: str | Path) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise ArtifactError(f"overlap dataset must not be a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise ArtifactError(f"overlap dataset does not exist: {resolved}")
    return resolved


def build_campaign_overlap_report(
    *,
    dataset_path: str | Path,
    compared_paths: list[str | Path],
    similarity_threshold: float = 0.9,
    id_field: str = "id",
    text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS,
    max_comparison_pairs: int = DEFAULT_MAX_COMPARISON_PAIRS,
) -> CampaignOverlapReport:
    if not math.isfinite(similarity_threshold) or not 0 < similarity_threshold <= 1:
        raise ValidationGateError("overlap similarity threshold must be finite and in (0, 1]")
    if not id_field.strip() or not text_fields or any(not field.strip() for field in text_fields):
        raise ValidationGateError("overlap id/text field names must be non-empty")
    if max_comparison_pairs <= 0:
        raise ValidationGateError("maximum comparison pairs must be positive")
    dataset = _safe_dataset_path(dataset_path)
    compared = [_safe_dataset_path(path) for path in compared_paths]
    if not compared:
        raise ValidationGateError("campaign overlap requires at least one comparison dataset")
    dataset_sha = file_sha256(dataset)
    compared_sha = [file_sha256(path) for path in compared]
    if dataset_sha in compared_sha:
        raise ValidationGateError("campaign overlap cannot compare a dataset with itself")
    if len(compared_sha) != len(set(compared_sha)):
        raise ValidationGateError("campaign overlap comparison datasets must be byte-distinct")

    source_records = _load_records(
        dataset,
        id_field=id_field,
        text_fields=text_fields,
    )
    compared_records = {
        digest: _load_records(path, id_field=id_field, text_fields=text_fields)
        for path, digest in zip(compared, compared_sha, strict=True)
    }
    comparison_pairs = len(source_records) * sum(
        len(records) for records in compared_records.values()
    )
    if comparison_pairs > max_comparison_pairs:
        raise ValidationGateError(
            f"campaign overlap would compare {comparison_pairs} record pairs, above "
            f"the configured maximum {max_comparison_pairs}"
        )

    matches: list[CampaignOverlapMatch] = []
    for _source_id, source_text, source_sha in source_records:
        source_normalized = _normalized_text(source_text)
        for compared_digest, records in compared_records.items():
            for _compared_id, compared_text, compared_record_sha in records:
                compared_normalized = _normalized_text(compared_text)
                exact = source_normalized == compared_normalized
                similarity = 1.0 if exact else _similarity(source_text, compared_text)
                if exact or similarity >= similarity_threshold:
                    matches.append(
                        CampaignOverlapMatch(
                            dataset_record_sha256=source_sha,
                            compared_dataset_sha256=compared_digest,
                            compared_record_sha256=compared_record_sha,
                            similarity=similarity,
                            exact=exact,
                        )
                    )
    matches.sort(
        key=lambda match: (
            match.dataset_record_sha256,
            match.compared_dataset_sha256,
            match.compared_record_sha256,
        )
    )
    exact_count = sum(match.exact for match in matches)
    near_count = len(matches) - exact_count
    return CampaignOverlapReport(
        dataset_sha256=dataset_sha,
        compared_dataset_sha256=sorted(compared_sha),
        dataset_record_count=len(source_records),
        compared_record_count_by_sha256={
            digest: len(records) for digest, records in compared_records.items()
        },
        comparison_pair_count=comparison_pairs,
        exact_match_count=exact_count,
        near_duplicate_count=near_count,
        near_duplicate_threshold=similarity_threshold,
        matches=matches,
        passed=not matches,
    )
