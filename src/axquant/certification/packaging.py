from __future__ import annotations

import shutil
from pathlib import Path

from axquant.certification.common import required_directory, required_file
from axquant.certification.policy import direct_policy
from axquant.errors import PublishingError
from axquant.schema import ArtifactManifest, Qwen3NextReleaseAuditRequest
from axquant.serde import file_sha256, load_model, write_data, write_text

_PACKAGE_INPUTS = {
    "coding_suite_manifest.json": "coding_suite_manifest",
    "coding_suite_self_test.json": "coding_suite_self_test",
    "benchmark_evidence_index.json": "benchmark_evidence_index",
    "release_validation_index.json": "release_validation_index",
    "refinement_measurements.json": "refinement_measurements",
    "hardware_profile_registry.json": "hardware_registry",
    "compatibility_matrix.json": "compatibility_matrix",
    "pareto_report.json": "pareto_report",
    "reproduction_verification.json": "reproduction_verification",
    "evidence_archive_index.json": "evidence_archive_index",
}


def _copy_exact(source: Path, target: Path) -> None:
    if target.is_file():
        if file_sha256(target) != file_sha256(source):
            raise PublishingError(f"certification package target differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _direct_model_card(
    request: Qwen3NextReleaseAuditRequest,
    manifest: ArtifactManifest,
    *,
    repo_id: str,
) -> str:
    scope = request.certification_scope
    hardware = ", ".join(scope.hardware_scope_ids)
    return f"""---
library_name: mlx
license: apache-2.0
base_model: {scope.source_model.model_id}
tags:
  - mlx
  - axquant
  - qwen3-next
  - non-mtp
---

# {repo_id}

This is an AXQuant exact-checkpoint **non-MTP direct-decode certification candidate**. It is
certified only when the bundled `certification/audit.json` exists, reports `release_ready: true`,
and passes N0-N8 for this exact artifact. No family-wide or MTP claim is made.

| Field | Value |
| --- | --- |
| Source | `{scope.source_model.model_id}` |
| Immutable source revision | `{scope.source_model.revision}` |
| Certification track | `{scope.track.value}` |
| Target class | `{scope.target_class.value}` |
| Measured BPW | `{manifest.measured_total_bpw:.8f}` |
| Weight bytes | `{manifest.weight_file_size_bytes}` |
| Hardware scope | `{hardware}` |
| MTP | Absent by source architecture; no MTP capability or performance claim |
| Policy | `{scope.policy_id}` / `{request.policy_sha256}` |

The portable weights are intended for stock MLX-LM compatibility. AX Engine is the primary
certification runtime and must pass native-manifest, doctor, generation, matched benchmark, and
zero-fallback gates in the bundled audit. Metrics and public claims are authoritative only from
that checksum-bound audit.
"""


def prepare_direct_publication(
    *,
    model_dir: str | Path,
    repo_id: str,
    request_path: str | Path,
) -> list[Path]:
    request_source = Path(request_path).expanduser().resolve()
    request = load_model(request_source, Qwen3NextReleaseAuditRequest)
    artifact = required_directory(request_source.parent, request.artifact_directory, "artifact")
    requested_artifact = Path(model_dir).expanduser().resolve()
    if artifact != requested_artifact:
        raise PublishingError("direct certification request targets another artifact")
    manifest_path = required_file(artifact, "axquant_manifest.json", "artifact manifest")
    manifest = load_model(manifest_path, ArtifactManifest)
    if file_sha256(manifest_path) != request.certification_scope.artifact_manifest_sha256:
        raise PublishingError("direct certification scope binds another artifact manifest")

    certification = artifact / "certification"
    certification.mkdir(parents=True, exist_ok=True)
    _copy_exact(request_source, certification / "request.json")
    for target_name, request_field in _PACKAGE_INPUTS.items():
        value = getattr(request, request_field)
        source = required_file(request_source.parent, value, request_field.replace("_", " "))
        _copy_exact(source, certification / target_name)
    write_data(certification / "policy.json", direct_policy())
    write_data(certification / "exact_checkpoint_scope.json", request.certification_scope)
    write_text(
        artifact / "README.md",
        _direct_model_card(request, manifest, repo_id=repo_id),
    )
    return sorted(path for path in artifact.rglob("*") if path.is_file())
