"""Source-plan binding and publication identity gates (AXQ-048)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.converter import _resolve_bound_conversion_source
from axquant.errors import ArtifactError, PlanningError, PublishingError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.publisher import publish_model
from axquant.schema import PlanRequest, ProfileName, QuantizationPlan
from axquant.serde import write_data
from axquant.source_binding import (
    BINDING_NAME,
    artifact_identity_issues,
    binding_path_beside,
    build_source_plan_binding,
    load_source_plan_binding,
    source_binding_issues,
    tree_identity_issues,
    write_source_plan_binding,
)


def _plan(model_dir: Path) -> QuantizationPlan:
    inventory = inspect_model(
        model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="a" * 40,
    )
    return plan_quantization(
        architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING),
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
        ),
    )


def test_source_binding_fingerprint_detects_a_different_checkpoint(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    plan = _plan(qwen36_model_dir)
    binding = build_source_plan_binding(plan, qwen36_model_dir)

    assert source_binding_issues(binding=binding, plan=plan, source_dir=qwen36_model_dir) == []

    # A different checkpoint with the same architecture still differs.
    other = tmp_path / "other"
    other.mkdir()
    for name in ("config.json", "mtp.safetensors", "model.safetensors"):
        (other / name).write_bytes((qwen36_model_dir / name).read_bytes())
    (other / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "num_hidden_layers": 4}),
        encoding="utf-8",
    )
    issues = source_binding_issues(binding=binding, plan=plan, source_dir=other)
    assert any("config differs" in issue for issue in issues)

    # A changed member size is caught too.
    grown = tmp_path / "grown"
    grown.mkdir()
    for name in ("config.json", "mtp.safetensors", "model.safetensors"):
        (grown / name).write_bytes((qwen36_model_dir / name).read_bytes())
    (grown / "model.safetensors").write_bytes(b"x" * 4096)
    issues = source_binding_issues(binding=binding, plan=plan, source_dir=grown)
    assert any("members differ" in issue for issue in issues)


def test_source_binding_binds_one_plan_and_round_trips(
    qwen36_model_dir: Path,
    tmp_path: Path,
) -> None:
    plan = _plan(qwen36_model_dir)
    target = write_source_plan_binding(tmp_path, plan, qwen36_model_dir)

    assert target == tmp_path / BINDING_NAME
    loaded = load_source_plan_binding(target)
    assert loaded.plan_sha256 == build_source_plan_binding(plan, qwen36_model_dir).plan_sha256
    assert loaded.source_model.local_path is None

    # Mis-pairing with another plan is rejected.
    other_plan = plan.model_copy(update={"target_bpw": 5.0})
    issues = source_binding_issues(binding=loaded, plan=other_plan, source_dir=qwen36_model_dir)
    assert any("another plan" in issue for issue in issues)


def test_source_binding_requires_a_checkpoint(qwen36_model_dir: Path, tmp_path: Path) -> None:
    plan = _plan(qwen36_model_dir)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ArtifactError, match="requires a checkpoint config"):
        build_source_plan_binding(plan, empty)

    no_weights = tmp_path / "no-weights"
    no_weights.mkdir()
    (no_weights / "config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="found no Safetensors members"):
        build_source_plan_binding(plan, no_weights)


def test_path_neutral_plan_requires_a_binding(qwen36_model_dir: Path, tmp_path: Path) -> None:
    """A plan without a local path must bind the source structurally."""

    plan = _plan(qwen36_model_dir)
    neutral = plan.model_copy(
        update={"source_model": plan.source_model.model_copy(update={"local_path": None})}
    )

    with pytest.raises(PlanningError, match="requires a plan-bound source binding"):
        _resolve_bound_conversion_source(str(qwen36_model_dir), neutral, None, None)

    binding = build_source_plan_binding(neutral, qwen36_model_dir)
    assert (
        _resolve_bound_conversion_source(str(qwen36_model_dir), neutral, None, binding)
        == qwen36_model_dir.resolve()
    )

    with pytest.raises(PlanningError, match="does not match the plan binding"):
        _resolve_bound_conversion_source(str(tmp_path), neutral, None, binding)


def test_publication_rejects_a_path_shaped_source_identity(tmp_path: Path) -> None:
    """A locally sourced artifact may not publish where the checkpoint lived."""

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    write_data(
        artifact / "axquant_plan.json",
        {
            "schema_version": "axquant.plan.v2",
            "source_model": {
                "model_id": "/Volumes/Ext16TR0/models/Qwen3.6-27B-bf16",
                "revision": "a" * 40,
            },
        },
    )
    issues = artifact_identity_issues(artifact)
    assert issues and "axquant_plan.json" in issues[0]
    assert "--model-id" in issues[0]

    with pytest.raises(PublishingError, match="identity is not publishable"):
        publish_model(
            model_dir=artifact,
            repo_id="AutomatosX/AX-test",
            validation_index_path=tmp_path / "validation.json",
            hardware_registry_path=tmp_path / "hardware.json",
            pareto_report_path=tmp_path / "pareto.json",
        )

    write_data(
        artifact / "axquant_plan.json",
        {
            "schema_version": "axquant.plan.v2",
            "source_model": {"model_id": "Qwen/Qwen3.6-27B", "revision": "a" * 40},
        },
    )
    assert artifact_identity_issues(artifact) == []

    # The packaged capture manifest is published too, and inherits its model id
    # from the tokenized cache, so a path there is denied for the same reason.
    write_data(
        artifact / "activation_capture_manifest.json",
        {"schema_version": "axquant.activation-capture.v1", "model": "/Users/operator/x"},
    )
    capture_issues = artifact_identity_issues(artifact)
    assert capture_issues and "activation_capture_manifest.json" in capture_issues[0]


def test_binding_path_sits_beside_the_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plans" / "plan-01.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("{}\n", encoding="utf-8")
    assert binding_path_beside(plan_path) == plan_path.parent / BINDING_NAME
    assert binding_path_beside(plan_path.parent) == plan_path.parent / BINDING_NAME


def test_tree_identity_issues_lists_path_shaped_identities(tmp_path: Path) -> None:
    """AXQ-048: the remaining evidence writers are reported, not denied yet."""

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    write_data(
        artifact / "axquant_runtime.json",
        {"model": "/Volumes/Ext16TR0/runtime-src"},
    )
    write_data(
        artifact / "certification" / "nested.json",
        {"candidate_model": {"model_id": "~/models/Qwen3.6-27B"}},
    )
    write_data(artifact / "axquant_plan.json", {"source_model": {"model_id": "Qwen/Qwen3.6-27B"}})
    (artifact / "README.md").write_text("plain prose, no identity\n", encoding="utf-8")

    reported = tree_identity_issues(artifact)

    assert set(reported) == {"axquant_runtime.json", "certification/nested.json"}
    assert "model is a filesystem path" in reported["axquant_runtime.json"][0]
    assert "model_id is a filesystem path" in reported["certification/nested.json"][0]
    # The plan and manifest are denied by artifact_identity_issues, not reported here.
    assert artifact_identity_issues(artifact) == []
