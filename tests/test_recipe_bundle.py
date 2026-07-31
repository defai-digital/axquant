from __future__ import annotations

from pathlib import Path

import pytest

from axquant.analyzer import architecture_prior_report
from axquant.errors import ArtifactError
from axquant.inspector import inspect_model
from axquant.planner import plan_quantization
from axquant.recipes import export_recipe_bundle, load_recipe_bundle, resolve_recipe_plan
from axquant.schema import (
    EvidenceKind,
    Inventory,
    ModelIdentity,
    PlanRequest,
    ProfileName,
    QuantizationPlan,
    RecipeBundle,
)
from axquant.serde import load_model, write_data


def _inventory(model_dir: Path) -> Inventory:
    return inspect_model(
        model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="source-revision",
    )


def _plan(inventory: Inventory) -> QuantizationPlan:
    report = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    return plan_quantization(
        report,
        PlanRequest(
            profile=ProfileName.AGENT_CODING,
            target_bpw=14.0,
            allow_unmeasured=True,
        ),
    )


def _exported_bundle(qwen36_model_dir: Path, tmp_path: Path) -> tuple[Inventory, Path]:
    inventory = _inventory(qwen36_model_dir)
    plan_path = tmp_path / "plan.json"
    write_data(plan_path, _plan(inventory))
    bundle_path = export_recipe_bundle(
        plan=plan_path,
        output_dir=tmp_path / "bundle",
        bundle_id="qwen36-27b-prior-r1",
        lineage={"sensitivity": "a" * 64},
        notes=["development prior bundle"],
    )
    return inventory, bundle_path


def test_bundle_round_trips_to_identical_plan(qwen36_model_dir: Path, tmp_path: Path) -> None:
    inventory, bundle_path = _exported_bundle(qwen36_model_dir, tmp_path)
    record, payload = load_recipe_bundle(bundle_path)
    assert record.bundle_id == "qwen36-27b-prior-r1"
    assert record.lineage == {"sensitivity": "a" * 64}
    resolved_record, resolved_plan = resolve_recipe_plan(bundle_path, inventory=inventory)
    assert resolved_record == record
    assert resolved_plan == load_model(payload, QuantizationPlan)
    assert resolved_plan.evidence_kind == record.evidence_kind


def test_bundle_directory_resolution(qwen36_model_dir: Path, tmp_path: Path) -> None:
    inventory, bundle_path = _exported_bundle(qwen36_model_dir, tmp_path)
    _, resolved_plan = resolve_recipe_plan(bundle_path.parent, inventory=inventory)
    assert resolved_plan.source_model.model_id == "Qwen/Qwen3.6-27B"


def test_bundle_detects_payload_tampering(qwen36_model_dir: Path, tmp_path: Path) -> None:
    _, bundle_path = _exported_bundle(qwen36_model_dir, tmp_path)
    payload = bundle_path.parent / "plan.json"
    payload.write_text(payload.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_recipe_bundle(bundle_path)


def test_bundle_rejects_model_identity_mismatch(qwen36_model_dir: Path, tmp_path: Path) -> None:
    _, bundle_path = _exported_bundle(qwen36_model_dir, tmp_path)
    other = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-Other",
        revision="source-revision",
    )
    with pytest.raises(ArtifactError, match=r"targets Qwen/Qwen3\.6-27B"):
        resolve_recipe_plan(bundle_path, inventory=other)


def test_bundle_rejects_revision_mismatch(qwen36_model_dir: Path, tmp_path: Path) -> None:
    _, bundle_path = _exported_bundle(qwen36_model_dir, tmp_path)
    pinned = inspect_model(
        qwen36_model_dir,
        model_id="Qwen/Qwen3.6-27B",
        revision="another-revision",
    )
    with pytest.raises(ArtifactError, match="pins revision"):
        resolve_recipe_plan(bundle_path, inventory=pinned)


def test_bundle_requires_pinned_revision() -> None:
    with pytest.raises(ValueError, match="pin the source model revision"):
        RecipeBundle(
            bundle_id="unpinned",
            source_model=ModelIdentity(model_id="Qwen/Qwen3.6-27B"),
            evidence_kind="architecture_prior",
            payload_kind="plan",
            payload_file="plan.json",
            payload_sha256="a" * 64,
            axquant_version="1.0.0",
        )


def test_bundle_rejects_evidence_kind_upgrade(qwen36_model_dir: Path, tmp_path: Path) -> None:
    inventory, bundle_path = _exported_bundle(qwen36_model_dir, tmp_path)
    record = load_model(bundle_path, RecipeBundle)
    upgraded = record.model_copy(update={"evidence_kind": EvidenceKind.MEASURED})
    write_data(bundle_path, upgraded)
    with pytest.raises(ArtifactError, match="declares measured evidence"):
        resolve_recipe_plan(bundle_path, inventory=inventory)


def test_export_requires_revision_pinned_plan(qwen36_model_dir: Path, tmp_path: Path) -> None:
    inventory = inspect_model(qwen36_model_dir, model_id="Qwen/Qwen3.6-27B")
    plan_path = tmp_path / "plan.json"
    write_data(plan_path, _plan(inventory))
    with pytest.raises(ArtifactError, match="revision-pinned"):
        export_recipe_bundle(
            plan=plan_path,
            output_dir=tmp_path / "bundle",
            bundle_id="unpinned",
        )


def _fake_hub(remote_root: Path, *, expected_revision: str):
    def fake_download(*, repo_id: str, filename: str, revision: str) -> str:
        assert repo_id == "AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit"
        assert revision == expected_revision
        local = remote_root / filename
        if not local.is_file():
            raise FileNotFoundError(filename)
        return str(local)

    return fake_download


def test_remote_bundle_resolution_requires_revision_pin() -> None:
    from axquant.recipes import _parse_remote_reference

    with pytest.raises(ArtifactError, match="pin a revision"):
        _parse_remote_reference("hf://AutomatosX/AX-Model")
    with pytest.raises(ArtifactError, match="pin a revision"):
        _parse_remote_reference("hf://AutomatosX/AX-Model@")
    with pytest.raises(ArtifactError, match=r"hf://OWNER/REPO"):
        _parse_remote_reference("hf://not-a-repo@rev")
    repo, revision, path = _parse_remote_reference("hf://AutomatosX/AX-Model@abc123")
    assert (repo, revision, path) == (
        "AutomatosX/AX-Model",
        "abc123",
        "axquant_recipe_bundle.json",
    )
    repo, revision, path = _parse_remote_reference(
        "hf://AutomatosX/AX-Model@abc123/recipe/axquant_recipe_bundle.json"
    )
    assert path == "recipe/axquant_recipe_bundle.json"


def test_remote_bundle_resolves_and_verifies(
    qwen36_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axquant.recipes as recipes

    inventory = _inventory(qwen36_model_dir)
    plan_path = tmp_path / "plan.json"
    write_data(plan_path, _plan(inventory))
    remote_root = tmp_path / "remote-repo"
    export_recipe_bundle(
        plan=plan_path,
        output_dir=remote_root / "recipe",
        bundle_id="qwen36-27b-remote-r1",
    )
    monkeypatch.setattr(
        recipes,
        "hf_hub_download",
        _fake_hub(remote_root, expected_revision="deadbeef"),
    )
    reference = (
        "hf://AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit@deadbeef/recipe/axquant_recipe_bundle.json"
    )
    record, resolved_plan = resolve_recipe_plan(reference, inventory=inventory)
    assert record.bundle_id == "qwen36-27b-remote-r1"
    assert resolved_plan.source_model.model_id == "Qwen/Qwen3.6-27B"


def test_remote_bundle_download_failure_is_artifact_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axquant.recipes as recipes

    monkeypatch.setattr(
        recipes,
        "hf_hub_download",
        _fake_hub(tmp_path / "empty", expected_revision="deadbeef"),
    )
    with pytest.raises(ArtifactError, match="download failed"):
        recipes.load_recipe_bundle("hf://AutomatosX/AX-Qwen3.6-27B-MLX-AXQuant-4bit@deadbeef")
