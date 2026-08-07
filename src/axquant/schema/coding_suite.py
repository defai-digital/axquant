from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.inventory import ModelIdentity

_SHA256 = r"^[0-9a-f]{64}$"
_TASK_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class CodingScorer(StrEnum):
    UNIT_TEST = "unit-test"
    COMPILE = "compile"
    AST = "ast"
    JSON_SCHEMA = "json-schema"
    TOOL_EXACT = "tool-exact"
    TEXT_EXACT = "text-exact"


class CodingTaskManifest(StrictModel):
    task_id: str = Field(pattern=_TASK_ID)
    category: str = Field(min_length=1)
    language: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=_SHA256)
    reference_sha256: str | None = Field(default=None, pattern=_SHA256)
    payload_sha256: str = Field(pattern=_SHA256)
    scorer: CodingScorer
    license_id: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    target_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0.0)
    cpu_time_seconds: int = Field(ge=1)
    memory_limit_bytes: int = Field(gt=0)
    process_limit: int = Field(ge=1)
    output_limit_bytes: int = Field(ge=1)
    file_size_limit_bytes: int = Field(ge=1)
    open_file_limit: int = Field(ge=16)
    long_context: bool


class CodingSuiteManifest(StrictModel):
    schema_version: Literal["axquant.coding-suite-manifest.v2"] = "axquant.coding-suite-manifest.v2"
    suite_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=_SHA256)
    tasks: list[CodingTaskManifest] = Field(min_length=1)
    task_shards: dict[str, str] = Field(min_length=1)
    calibration_overlap_attested: bool
    calibration_overlap_report: str = Field(min_length=1)
    calibration_overlap_report_sha256: str = Field(pattern=_SHA256)
    toolchains: dict[str, str] = Field(min_length=1)
    sandbox_profile_sha256: str = Field(pattern=_SHA256)
    normalization_algorithm: Literal["axquant-token-5gram-v2"] = "axquant-token-5gram-v2"
    near_duplicate_threshold: float = Field(gt=0.0, le=1.0)
    random_seed: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def tasks_are_unique(self) -> CodingSuiteManifest:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("coding-suite task IDs must be unique")
        if not self.calibration_overlap_attested:
            raise ValueError("coding suite requires calibration-overlap attestation")
        if not _safe_relative_path(self.calibration_overlap_report):
            raise ValueError("coding overlap report must use a safe relative path")
        if any(not _safe_relative_path(path) for path in self.task_shards):
            raise ValueError("coding suite shards must use safe relative paths")
        return self


def _safe_relative_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    return bool(
        value
        and not normalized.startswith(("/", "~/"))
        and all(part not in {"", ".", ".."} for part in parts)
        and "\\" not in value
    )


class CodingTaskPayload(StrictModel):
    schema_version: Literal["axquant.coding-task-payload.v1"] = "axquant.coding-task-payload.v1"
    task_id: str = Field(pattern=_TASK_ID)
    category: str = Field(min_length=1)
    language: str = Field(min_length=1)
    scorer: CodingScorer
    prompt: str = Field(min_length=1)
    reference: str | None = None
    candidate_path: str = Field(min_length=1)
    test_path: str | None = None
    fixture_files: dict[str, str] = Field(default_factory=dict)
    expected_json: JsonValue | None = None
    json_required_keys: list[str] = Field(default_factory=list)
    expected_text: str | None = None
    target_tokens: int = Field(ge=1)

    @field_validator("candidate_path", "test_path")
    @classmethod
    def paths_are_relative(cls, value: str | None) -> str | None:
        if value is not None and not _safe_relative_path(value):
            raise ValueError("coding task paths must be safe relative paths")
        return value

    @field_validator("fixture_files")
    @classmethod
    def fixture_paths_are_relative(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not _safe_relative_path(path) for path in value):
            raise ValueError("coding fixture paths must be safe relative paths")
        return value

    @model_validator(mode="after")
    def scorer_contract_is_complete(self) -> CodingTaskPayload:
        if self.scorer is CodingScorer.UNIT_TEST and self.test_path is None:
            raise ValueError("unit-test task requires a test_path")
        if self.test_path is not None and self.test_path not in self.fixture_files:
            raise ValueError("test_path must identify a fixture file")
        if self.scorer is CodingScorer.TOOL_EXACT and self.expected_json is None:
            raise ValueError("tool-exact task requires expected_json")
        if self.scorer is CodingScorer.JSON_SCHEMA and not self.json_required_keys:
            raise ValueError("json-schema task requires json_required_keys")
        if self.scorer is CodingScorer.TEXT_EXACT and self.expected_text is None:
            raise ValueError("text-exact task requires expected_text")
        return self


class CodingModelOutput(StrictModel):
    task_id: str = Field(pattern=_TASK_ID)
    output: str
    generated_tokens: int = Field(ge=0)
    perplexity_loss: float = Field(ge=0.0)
    perplexity_tokens: int = Field(ge=0)
    model_error: str | None = None


class CodingEvaluationState(StrictModel):
    schema_version: Literal["axquant.coding-evaluation-state.v1"] = (
        "axquant.coding-evaluation-state.v1"
    )
    suite_manifest_sha256: str = Field(pattern=_SHA256)
    model: ModelIdentity
    model_artifact_sha256: str = Field(pattern=_SHA256)
    tokenizer_sha256: str = Field(pattern=_SHA256)
    random_seed: int = Field(ge=0)
    max_sequence_length: int = Field(ge=1)
    outputs: list[CodingModelOutput] = Field(default_factory=list)
    completed: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def output_ids_are_unique(self) -> CodingEvaluationState:
        ids = [output.task_id for output in self.outputs]
        if len(ids) != len(set(ids)):
            raise ValueError("coding evaluation output IDs must be unique")
        return self


class CodingOverlapMatch(StrictModel):
    task_id: str = Field(pattern=_TASK_ID)
    calibration_id: str = Field(min_length=1)
    similarity: float = Field(ge=0.0, le=1.0)
    exact: bool


class CodingOverlapReport(StrictModel):
    schema_version: Literal["axquant.coding-overlap-report.v1"] = "axquant.coding-overlap-report.v1"
    suite_dataset_sha256: str = Field(pattern=_SHA256)
    calibration_dataset_sha256: str = Field(pattern=_SHA256)
    similarity_threshold: float = Field(gt=0.0, le=1.0)
    matches: list[CodingOverlapMatch] = Field(default_factory=list)
    passed: bool
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def status_is_consistent(self) -> CodingOverlapReport:
        if self.passed != (not self.matches):
            raise ValueError("coding overlap report status is inconsistent")
        return self
