"""Architecture-aware static audit for published AXQ MTP Hub repositories."""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, get_session, hf_hub_download, hf_hub_url

from axquant.errors import ArtifactError
from axquant.gemma4_assistant_compose import validate_known_gemma4_assistant_pair
from axquant.gemma4_vlm import (
    GEMMA4_MLX_VLM_VISION_LAYOUT,
    normalize_gemma4_vision_tensor_names,
)
from axquant.mtp_sidecar import QWEN_NEXT_MTP_ARCH_ID

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_JSON_DOCUMENTS = (
    "config.json",
    "model.safetensors.index.json",
    "mtplx_runtime.json",
    "ax_expert_stream.json",
    "ax_gemma4_assistant_mtp.json",
    "ax_composite_pack_manifest.json",
    "assistant/config.json",
)
RESERVED_MTP_REPOSITORIES = frozenset({"AutomatosX/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-4bit-MTP"})


class MtpHubPackKind(StrEnum):
    """Published MTP packaging contracts with distinct runtime semantics."""

    GEMMA_ASSISTANT = "gemma-assistant"
    QWEN_RESIDENT = "qwen-resident"
    QWEN_EXPERT_STREAM = "qwen-expert-stream"
    DEEPSEEK_NEXTN = "deepseek-nextn"
    RESERVED = "reserved"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SafetensorsHeader:
    """Header-only view of a remote Safetensors file."""

    tensor_names: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MtpHubRepositorySnapshot:
    """Small immutable inputs required to audit one Hub repository."""

    repo_id: str
    revision: str
    files: frozenset[str]
    documents: Mapping[str, Mapping[str, Any]]
    safetensors_headers: Mapping[str, SafetensorsHeader]


@dataclass(frozen=True, slots=True)
class MtpHubAuditResult:
    """One architecture-aware fleet audit result."""

    repo_id: str
    revision: str
    kind: MtpHubPackKind
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    def model_dump(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "kind": self.kind.value,
            "passed": self.passed,
            "issues": list(self.issues),
        }


def _document(
    snapshot: MtpHubRepositorySnapshot,
    name: str,
    issues: list[str],
) -> Mapping[str, Any]:
    value = snapshot.documents.get(name)
    if value is None:
        issues.append(f"missing or unreadable {name}")
        return {}
    return value


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _audit_gemma(
    snapshot: MtpHubRepositorySnapshot,
    issues: list[str],
) -> MtpHubPackKind:
    config = _document(snapshot, "config.json", issues)
    contract = _document(snapshot, "ax_gemma4_assistant_mtp.json", issues)
    _document(snapshot, "ax_composite_pack_manifest.json", issues)
    assistant_config = _document(snapshot, "assistant/config.json", issues)
    index = _document(snapshot, "model.safetensors.index.json", issues)

    if contract.get("backend") != "gemma4_assistant":
        issues.append("Gemma assistant contract backend must be gemma4_assistant")
    if contract.get("assistant_path") != "assistant":
        issues.append("Gemma assistant contract must bind assistant_path=assistant")
    if contract.get("pairing") != "exact":
        issues.append("Gemma assistant contract must bind pairing=exact")
    target_model_id = contract.get("target_model_id")
    assistant_model_id = contract.get("assistant_model_id")
    if isinstance(target_model_id, str) and isinstance(assistant_model_id, str):
        try:
            validate_known_gemma4_assistant_pair(target_model_id, assistant_model_id)
        except ArtifactError as exc:
            issues.append(str(exc))
    else:
        issues.append("Gemma assistant contract is missing target/assistant model ids")
    if assistant_config.get("model_type") != "gemma4_assistant":
        issues.append("assistant/config.json model_type must be gemma4_assistant")

    model_type = config.get("model_type")
    if model_type not in {"gemma4", "gemma4_unified"}:
        issues.append("Gemma target model_type must be gemma4 or gemma4_unified")
    header = snapshot.safetensors_headers.get("vision.safetensors")
    if header is None:
        issues.append("missing or unreadable vision.safetensors header")
        vision_names: tuple[str, ...] = ()
    else:
        vision_names = header.tensor_names
        try:
            normalized = normalize_gemma4_vision_tensor_names(vision_names)
        except ArtifactError as exc:
            issues.append(str(exc))
        else:
            if normalized != vision_names:
                issues.append("vision.safetensors uses source-prefixed Gemma tensor names")
        if header.metadata.get("format") != "mlx":
            issues.append("vision.safetensors metadata format must be mlx")
        if header.metadata.get("axquant_layout") != GEMMA4_MLX_VLM_VISION_LAYOUT:
            issues.append("vision.safetensors is missing the mlx-vlm-gemma4-v1 layout marker")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        issues.append("model.safetensors.index.json has no non-empty weight_map")
    else:
        indexed_vision = {
            name for name, shard in weight_map.items() if shard == "vision.safetensors"
        }
        if indexed_vision != set(vision_names):
            issues.append("vision.safetensors keys do not exactly match index entries")

    if model_type == "gemma4_unified":
        required_prefixes = ["vision_embedder.", "embed_vision."]
        if isinstance(config.get("audio_config"), dict) and config["audio_config"]:
            required_prefixes.append("embed_audio.")
        for prefix in required_prefixes:
            if not any(name.startswith(prefix) for name in vision_names):
                issues.append(f"Gemma unified sidecar is missing {prefix} tensors")
    return MtpHubPackKind.GEMMA_ASSISTANT


def _audit_qwen(
    snapshot: MtpHubRepositorySnapshot,
    issues: list[str],
    *,
    expert_stream: bool,
) -> MtpHubPackKind:
    runtime = _document(snapshot, "mtplx_runtime.json", issues)
    sidecar_name = next(
        (name for name in ("mtp.safetensors", "mtp_head.safetensors") if name in snapshot.files),
        None,
    )
    if sidecar_name is None:
        issues.append("Qwen MTP pack is missing a recognized Safetensors sidecar")
    elif sidecar_name not in snapshot.safetensors_headers:
        issues.append(f"missing or unreadable {sidecar_name} header")
    if not _positive_integer(runtime.get("mtp_depth_max")):
        issues.append("mtplx_runtime.json mtp_depth_max must be a positive integer")
    if not isinstance(runtime.get("mtp_norm_layout"), str):
        issues.append("mtplx_runtime.json mtp_norm_layout must be a string")

    if expert_stream:
        _document(snapshot, "ax_expert_stream.json", issues)
        arch_id = runtime.get("arch_id")
        if arch_id is not None and arch_id != QWEN_NEXT_MTP_ARCH_ID:
            issues.append("expert-stream Qwen pack has an invalid explicit MTP arch_id")
        return MtpHubPackKind.QWEN_EXPERT_STREAM

    if runtime.get("arch_id") != QWEN_NEXT_MTP_ARCH_ID:
        issues.append(f"resident Qwen MTP arch_id must be {QWEN_NEXT_MTP_ARCH_ID}")
    tensor_count = runtime.get("mtp_tensor_count")
    if not _positive_integer(tensor_count):
        issues.append("resident Qwen mtp_tensor_count must be a positive integer")
    elif sidecar_name is not None and sidecar_name in snapshot.safetensors_headers:
        actual_count = len(snapshot.safetensors_headers[sidecar_name].tensor_names)
        if tensor_count != actual_count:
            issues.append(
                f"resident Qwen mtp_tensor_count {tensor_count} does not match {actual_count}"
            )
    return MtpHubPackKind.QWEN_RESIDENT


def _audit_deepseek(
    snapshot: MtpHubRepositorySnapshot,
    issues: list[str],
) -> MtpHubPackKind:
    runtime = _document(snapshot, "mtplx_runtime.json", issues)
    if "mtp.safetensors" not in snapshot.files:
        issues.append("DeepSeek nextn pack is missing mtp.safetensors")
    elif "mtp.safetensors" not in snapshot.safetensors_headers:
        issues.append("missing or unreadable mtp.safetensors header")
    if not _positive_integer(runtime.get("mtp_depth_max")):
        issues.append("DeepSeek mtp_depth_max must be a positive integer")
    if not isinstance(runtime.get("mtp_norm_layout"), str):
        issues.append("DeepSeek mtp_norm_layout must be a string")
    if runtime.get("arch_id") == QWEN_NEXT_MTP_ARCH_ID:
        issues.append("DeepSeek nextn sidecar must not claim the Qwen MTP arch_id")
    if "ax_expert_stream.json" in snapshot.files:
        _document(snapshot, "ax_expert_stream.json", issues)
    return MtpHubPackKind.DEEPSEEK_NEXTN


def audit_mtp_hub_snapshot(snapshot: MtpHubRepositorySnapshot) -> MtpHubAuditResult:
    """Audit one pinned Hub snapshot under its architecture-specific MTP contract."""

    issues: list[str] = []
    if not _IMMUTABLE_REVISION.fullmatch(snapshot.revision):
        issues.append("repository revision is not an immutable 40-character Git SHA")
    if not snapshot.repo_id.rsplit("/", 1)[-1].casefold().endswith("-mtp"):
        issues.append("repository with packaged MTP must use the -MTP suffix")

    if snapshot.repo_id in RESERVED_MTP_REPOSITORIES:
        forbidden = {
            name
            for name in snapshot.files
            if name.endswith(".safetensors") or name == "ax_gemma4_assistant_mtp.json"
        }
        if forbidden:
            issues.append(
                f"reserved MTP repository unexpectedly contains assets: {sorted(forbidden)}"
            )
        kind = MtpHubPackKind.RESERVED
    else:
        config = snapshot.documents.get("config.json", {})
        model_type = config.get("model_type")
        has_gemma_contract = "ax_gemma4_assistant_mtp.json" in snapshot.files
        expert_stream = "ax_expert_stream.json" in snapshot.files
        if has_gemma_contract or model_type in {"gemma4", "gemma4_unified"}:
            kind = _audit_gemma(snapshot, issues)
        elif isinstance(model_type, str) and model_type.startswith("deepseek_v4"):
            kind = _audit_deepseek(snapshot, issues)
        elif isinstance(model_type, str) and model_type.startswith("qwen"):
            kind = _audit_qwen(snapshot, issues, expert_stream=expert_stream)
        else:
            kind = MtpHubPackKind.UNKNOWN
            issues.append(f"unsupported or missing MTP model_type: {model_type!r}")

    return MtpHubAuditResult(
        repo_id=snapshot.repo_id,
        revision=snapshot.revision,
        kind=kind,
        issues=tuple(dict.fromkeys(issues)),
    )


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON key in remote document: {key}")
        result[key] = value
    return result


def _read_json_document(repo_id: str, revision: str, name: str) -> Mapping[str, Any]:
    path = hf_hub_download(repo_id=repo_id, filename=name, revision=revision)
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_json_object_without_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise ArtifactError(f"remote {name} must contain a JSON object")
    return value


def _read_range(url: str, start: int, end: int) -> bytes:
    response = get_session().get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=60,
    )
    try:
        if response.status_code != 206:
            raise ArtifactError(f"Hub did not honor byte range for {url}")
        return bytes(response.content)
    finally:
        response.close()


def read_remote_safetensors_header(
    repo_id: str,
    revision: str,
    name: str,
) -> SafetensorsHeader:
    """Read only the header of a pinned remote Safetensors file."""

    url = hf_hub_url(repo_id=repo_id, filename=name, revision=revision)
    first = _read_range(url, 0, 7)
    if len(first) != 8:
        raise ArtifactError(f"truncated remote Safetensors length for {name}")
    header_size = struct.unpack("<Q", first)[0]
    if header_size <= 1 or header_size > _MAX_SAFETENSORS_HEADER_BYTES:
        raise ArtifactError(f"unsafe remote Safetensors header size for {name}")
    header_bytes = _read_range(url, 8, 7 + header_size)
    try:
        header = json.loads(
            header_bytes,
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"invalid remote Safetensors header for {name}: {exc}") from exc
    if not isinstance(header, dict):
        raise ArtifactError(f"remote Safetensors header for {name} must be an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ArtifactError(f"remote Safetensors metadata for {name} must be an object")
    tensor_names = tuple(sorted(str(key) for key in header))
    if not tensor_names:
        raise ArtifactError(f"remote Safetensors file has no tensors: {name}")
    return SafetensorsHeader(tensor_names=tensor_names, metadata=metadata)


def load_mtp_hub_snapshot(
    api: HfApi,
    repo_id: str,
    *,
    revision: str | None = None,
) -> MtpHubRepositorySnapshot:
    """Load the small pinned metadata surface required by the fleet audit."""

    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    if not isinstance(info.sha, str):
        raise ArtifactError(f"Hub returned no immutable revision for {repo_id}")
    files = frozenset(item.rfilename for item in (info.siblings or []))
    documents: dict[str, Mapping[str, Any]] = {}
    for name in _JSON_DOCUMENTS:
        if name in files:
            documents[name] = _read_json_document(repo_id, info.sha, name)
    headers: dict[str, SafetensorsHeader] = {}
    for name in ("vision.safetensors", "mtp.safetensors", "mtp_head.safetensors"):
        if name in files:
            headers[name] = read_remote_safetensors_header(repo_id, info.sha, name)
    return MtpHubRepositorySnapshot(
        repo_id=repo_id,
        revision=info.sha,
        files=files,
        documents=documents,
        safetensors_headers=headers,
    )


def discover_axq_mtp_repositories(api: HfApi, author: str) -> tuple[str, ...]:
    """Discover AXQ repositories whose product identity declares packaged MTP."""

    repos = {
        model.id
        for model in api.list_models(author=author, search="MTP")
        if "-mlx-axq-" in model.id.casefold()
        and model.id.rsplit("/", 1)[-1].casefold().endswith("-mtp")
    }
    return tuple(sorted(repos, key=str.casefold))


def audit_mtp_hub_repositories(
    api: HfApi,
    repo_ids: Sequence[str],
) -> tuple[MtpHubAuditResult, ...]:
    """Load and audit pinned snapshots for a repository sequence."""

    return tuple(audit_mtp_hub_snapshot(load_mtp_hub_snapshot(api, repo)) for repo in repo_ids)
