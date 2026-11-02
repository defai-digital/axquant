from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axquant import converter, publisher
from axquant.analyzer import architecture_prior_report
from axquant.cuda import _verify_source, plan_cuda_nvfp4
from axquant.errors import ArtifactError
from axquant.inspector import inspect_model
from axquant.manual import manual_quantization_plan
from axquant.model_card import prepare_development_model_card
from axquant.model_support import model_support_withdrawal_reason, require_model_supported
from axquant.planner import plan_quantization
from axquant.predicate import PlanPredicate
from axquant.runtime import assert_conversion_scope, check_mlx_lm_generation
from axquant.schema import ManualPlanRecipe, PlanRequest, ProfileName
from axquant.serde import load_model

_WITHDRAWN = "ornith-ai/Ornith-1.5-397B"


@pytest.mark.parametrize(
    "reference",
    [
        _WITHDRAWN,
        "AutomatosX/AX-Ornith-1.5-397B-MLX-AXQ-MXFP4-MTP",
        "AutomatosX/AX-Ornith-1.5-397B-MLX-AXQ-6bit-MTP",
        "/models/ornith-15-397b/src",
        "/models/AX-Ornith-1.5-397B-MLX-AXQ-6bit-MTP",
        "ax-ornith-1_5-397b",
    ],
)
def test_withdrawn_identity_aliases_cannot_reenable_support(reference: str) -> None:
    assert "withdrawn" in (model_support_withdrawal_reason(reference) or "")
    with pytest.raises(ArtifactError, match="withdrawn"):
        require_model_supported(reference)


@pytest.mark.parametrize(
    "reference",
    [
        "ornith-ai/Ornith-1.5-9B",
        "ornith-ai/Ornith-1.5-35B-A3B",
        "Qwen/Qwen3.5-397B-A17B",
        "/data/ornith-research/Qwen3.5-397B-A17B",
    ],
)
def test_other_models_keep_their_existing_support(reference: str) -> None:
    assert model_support_withdrawal_reason(reference) is None
    require_model_supported(reference)


@pytest.mark.parametrize(
    "metadata",
    [
        {"_name_or_path": _WITHDRAWN},
        {"text_config": {"_name_or_path": _WITHDRAWN}},
        {"source_model": {"model_id": _WITHDRAWN}},
    ],
)
def test_local_generic_path_does_not_hide_withdrawn_config_identity(metadata: dict) -> None:
    with pytest.raises(ArtifactError, match="withdrawn"):
        require_model_supported("local-model", config=metadata)


@pytest.mark.parametrize("q_mode", ["affine", "mxfp4", "mxfp8"])
def test_stale_convertible_plan_cannot_bypass_withdrawal(
    qwen36_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, q_mode: str
) -> None:
    inventory = inspect_model(qwen36_model_dir, model_id="Qwen/Qwen3.6-27B", revision="a" * 40)
    report = architecture_prior_report(inventory, profile=ProfileName.AGENT_CODING)
    request = PlanRequest(profile=ProfileName.AGENT_CODING, target_bpw=14, allow_unmeasured=True)
    plan = plan_quantization(report, request)
    identity = plan.source_model.model_copy(update={"model_id": _WITHDRAWN})
    stale = plan.model_copy(update={"source_model": identity})

    def forbidden_backend():
        pytest.fail("withdrawn models must be rejected before MLX or downloads")

    monkeypatch.setattr(converter, "_mlx_api", forbidden_backend)
    output = tmp_path / "output"
    with pytest.raises(ArtifactError, match="withdrawn"):
        converter.convert_model(
            model=str(qwen36_model_dir),
            plan=stale,
            output=output,
            allow_unmeasured=True,
            q_mode=q_mode,
        )
    assert not output.exists()
    with pytest.raises(ArtifactError, match="withdrawn"):
        PlanPredicate(stale)
    with pytest.raises(ArtifactError, match="withdrawn"):
        assert_conversion_scope(stale)
    with pytest.raises(ArtifactError, match="withdrawn"):
        plan_quantization(report.model_copy(update={"model": identity}), request)
    recipe = load_model(
        Path(__file__).resolve().parents[1] / "examples/ornith-15-9b-axq6-v0.1.yaml",
        ManualPlanRecipe,
    )
    with pytest.raises(ArtifactError, match="withdrawn"):
        manual_quantization_plan(inventory.model_copy(update={"model": identity}), recipe)


def test_cuda_and_runtime_reject_withdrawn_source_before_backend_execution(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="withdrawn"):
        plan_cuda_nvfp4(tmp_path, model_id=_WITHDRAWN, allow_unmeasured=True)
    with pytest.raises(ArtifactError, match="withdrawn"):
        _verify_source(tmp_path, SimpleNamespace(model_id=_WITHDRAWN, source_files=[]))
    (tmp_path / "config.json").write_text(json.dumps({"_name_or_path": _WITHDRAWN}))

    def forbidden_runner(command):
        pytest.fail("withdrawn models must not execute inference")

    with pytest.raises(ArtifactError, match="withdrawn"):
        check_mlx_lm_generation(tmp_path, runner=forbidden_runner)


@pytest.mark.parametrize("mask_target", [False, True])
def test_publication_cannot_bypass_withdrawal_by_renaming_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mask_target: bool
) -> None:
    manifest = {"source_model": {"model_id": _WITHDRAWN}}
    original = json.dumps(manifest)
    (tmp_path / "axquant_manifest.json").write_text(original)
    repo = (
        "AutomatosX/AX-Example-MLX-AXQ-6bit"
        if mask_target
        else "AutomatosX/AX-Ornith-1.5-397B-MLX-AXQ-6bit-MTP"
    )

    def forbidden_client(*args, **kwargs):
        pytest.fail("withdrawn models must not contact the Hub")

    monkeypatch.setattr(publisher, "HfApi", forbidden_client)
    with pytest.raises(ArtifactError, match="withdrawn"):
        publisher.publish_model(
            model_dir=tmp_path,
            repo_id=repo,
            validation_index_path="unused",
            hardware_registry_path="unused",
            pareto_report_path="unused",
            execute=True,
        )
    assert (tmp_path / "axquant_manifest.json").read_text() == original


def test_card_preparation_cannot_advertise_withdrawn_model(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="withdrawn"):
        prepare_development_model_card(
            artifact_dir=tmp_path,
            repo_id="AutomatosX/AX-Ornith-1.5-397B-MLX-AXQ-6bit-MTP",
        )
    assert not list(tmp_path.iterdir())


def test_runtime_export_rejects_masked_withdrawn_source_before_writing(tmp_path: Path) -> None:
    from axquant.runtime_export import export_runtime_pack

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"_name_or_path": _WITHDRAWN}))
    output = tmp_path / "variant"
    with pytest.raises(ArtifactError, match="withdrawn"):
        export_runtime_pack(source, output, target="omlx")
    assert not output.exists()


def test_publication_preparation_rejects_withdrawn_target_before_writing(tmp_path: Path) -> None:
    from axquant.reporting import prepare_publication

    with pytest.raises(ArtifactError, match="withdrawn"):
        prepare_publication(
            model_dir=tmp_path,
            repo_id="AutomatosX/AX-Ornith-1.5-397B-MLX-AXQ-6bit-MTP",
            validation_index_path="unused",
            hardware_registry_path="unused",
            pareto_report_path="unused",
        )
    assert not list(tmp_path.iterdir())
