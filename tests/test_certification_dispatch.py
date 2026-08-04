from __future__ import annotations

from pathlib import Path

import pytest

from axquant.certification.dispatch import (
    build_certification_audit,
    load_certification_request,
)
from axquant.errors import ArtifactError
from axquant.schema import (
    ArchitectureFingerprint,
    ExactCertificationScope,
    ModelIdentity,
    Qwen3NextReleaseAuditRequest,
    ReleaseAuditRequest,
)
from axquant.serde import write_data

_SHA = "a" * 64
_REVISION = "b" * 40


def _scope() -> ExactCertificationScope:
    return ExactCertificationScope(
        source_model=ModelIdentity(
            model_id="Qwen/Qwen3-Coder-Next",
            revision=_REVISION,
            architecture="Qwen3NextForCausalLM",
        ),
        architecture=ArchitectureFingerprint(
            model_type="qwen3_next",
            architecture="Qwen3NextForCausalLM",
            text_layer_count=48,
            hidden_size=2048,
            full_attention_interval=4,
            expert_count=512,
            experts_per_token=10,
            expert_intermediate_size=512,
            mtp_declared=False,
            vision_present=False,
            config_sha256=_SHA,
            tokenizer_sha256=_SHA,
        ),
        target_class="4bit",
        artifact_manifest_sha256=_SHA,
        hardware_scope_ids=["m2-ultra-192gb"],
    )


def _direct_request() -> Qwen3NextReleaseAuditRequest:
    paths = {
        "artifact_directory": "artifact",
        "source_inventory": "inventory.json",
        "source_checkpoint_manifest": "source.json",
        "feasibility_report": "feasibility.json",
        "sensitivity_report": "sensitivity.json",
        "refinement_result": "refinement.json",
        "refinement_measurements": "measurements.json",
        "release_validation_index": "validation.json",
        "benchmark_evidence_index": "benchmark.json",
        "coding_suite_manifest": "coding.json",
        "coding_suite_self_test": "coding-self-test.json",
        "hardware_registry": "hardware.json",
        "pareto_report": "pareto.json",
        "compatibility_matrix": "compatibility.json",
        "compatibility_request": "compatibility-request.json",
        "reproduction_recipe": "recipe.json",
        "reproduction_verification": "reproduction.json",
        "ax_engine_manifest_check": "ax-manifest.json",
        "ax_engine_doctor_check": "ax-doctor.json",
        "ax_engine_runtime_check": "ax-runtime.json",
        "mlx_lm_runtime_check": "mlx-runtime.json",
        "evidence_archive_index": "archive.json",
        "toolkit_wheel": "axquant.whl",
    }
    return Qwen3NextReleaseAuditRequest(
        certification_scope=_scope(),
        required_toolkit_version="1.2.0",
        policy_sha256=_SHA,
        **paths,
    )


def test_request_dispatch_is_schema_only(tmp_path: Path) -> None:
    direct_path = tmp_path / "direct.json"
    write_data(direct_path, _direct_request())
    assert isinstance(load_certification_request(direct_path), Qwen3NextReleaseAuditRequest)

    mtp_path = tmp_path / "mtp.json"
    write_data(
        mtp_path,
        ReleaseAuditRequest(
            artifact_directory="artifact",
            feasibility_report="feasibility.json",
            sensitivity_report="sensitivity.json",
            refinement_result="refinement.json",
            release_validation_index="validation.json",
            hardware_registry="hardware.json",
            pareto_report="pareto.json",
            compatibility_matrix="compatibility.json",
            compatibility_request="compatibility-request.json",
            reproduction_recipe="recipe.json",
            reproduction_verification="reproduction.json",
            ax_engine_check="ax.json",
            mlx_lm_check="mlx.json",
            toolkit_wheel="axquant.whl",
        ),
    )
    assert isinstance(load_certification_request(mtp_path), ReleaseAuditRequest)


def test_unknown_request_schema_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    write_data(path, {"schema_version": "axquant.release-audit-request.v999"})
    with pytest.raises(ArtifactError, match="unsupported release-audit request schema"):
        load_certification_request(path)


def test_dispatch_does_not_infer_direct_track_from_model_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "mtp.json"
    request = ReleaseAuditRequest(
        artifact_directory="Qwen3-Coder-Next",
        feasibility_report="feasibility.json",
        sensitivity_report="sensitivity.json",
        refinement_result="refinement.json",
        release_validation_index="validation.json",
        hardware_registry="hardware.json",
        pareto_report="pareto.json",
        compatibility_matrix="compatibility.json",
        compatibility_request="compatibility-request.json",
        reproduction_recipe="recipe.json",
        reproduction_verification="reproduction.json",
        ax_engine_check="ax.json",
        mlx_lm_check="mlx.json",
        toolkit_wheel="axquant.whl",
    )
    write_data(path, request)
    marker = object()
    monkeypatch.setattr("axquant.release_audit.build_release_audit", lambda _path: marker)
    assert build_certification_audit(path) is marker


def test_direct_request_cannot_carry_release_exception() -> None:
    payload = _direct_request().model_dump(mode="json")
    payload["release_exceptions"] = ["waive-mtp"]
    with pytest.raises(ValueError, match="at most 0 items"):
        Qwen3NextReleaseAuditRequest.model_validate(payload)
