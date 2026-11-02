from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from axquant.schema import ArtifactFile
from axquant.serde import file_sha256
from scripts import publish_mlx_affine6_catalog as campaign


@pytest.fixture
def job(tmp_path: Path) -> dict[str, Any]:
    return {
        "key": "example",
        "source_id": "owner/model",
        "source_revision": "a" * 40,
        "source_path": str(tmp_path / "source"),
        "output": str(tmp_path / "output"),
        "repo_id": "AutomatosX/AX-Example-MLX-AXQ-6bit",
        "mtp": False,
        "task": "embedding",
        "license": "apache-2.0",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"source_revision": "main"},
        {"key": "../escape"},
        {"task": "audio"},
        {"repo_id": "AutomatosX/AX-Example-MLX-AXQ-MXFP6"},
        {"repo_id": "AutomatosX/AX-Example-MLX-AXQ-6bit-MTP"},
        {"mtp": True},
    ],
)
def test_reject_ambiguous_or_false_publication_identity(job: dict[str, Any], change: dict) -> None:
    with pytest.raises(ValueError):
        campaign.validate_job({**job, **change})


def test_output_must_not_overlap_source(job: dict[str, Any]) -> None:
    job["output"] = str(Path(job["source_path"]) / "converted")
    with pytest.raises(ValueError, match="disjoint"):
        campaign.validate_job(job)


@pytest.mark.parametrize(
    "name",
    [
        "../secret",
        "/etc/passwd",
        "a/../../secret",
        "a\\secret",
        ".cache/token",
        "runtime_check.json",
    ],
)
def test_private_or_unsafe_member_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        campaign.safe_member(tmp_path, name)


def test_symlink_member_rejected(tmp_path: Path) -> None:
    (tmp_path / "real").write_text("payload")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(ValueError, match="Symlink"):
        campaign.safe_member(tmp_path, "link")


@pytest.mark.parametrize("change", [{"passed": False}, {"task": "text"}, {"payload_sha256": "old"}])
def test_failed_wrong_task_or_stale_smoke_blocks_publication(
    job: dict[str, Any], change: dict
) -> None:
    receipt = {"passed": True, "task": "embedding", "payload_sha256": "current"}
    with pytest.raises(ValueError):
        campaign.require_smoke(job, {**receipt, **change}, "current")


def test_task_bound_smoke_accepted(job: dict[str, Any]) -> None:
    campaign.require_smoke(
        job, {"passed": True, "task": "embedding", "payload_sha256": "current"}, "current"
    )


def test_metadata_only_edits_do_not_reuse_changed_runtime_payload(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"quantization":{"bits":6}}')
    (tmp_path / "README.md").write_text("first card")
    files = [
        ArtifactFile(
            path=name,
            size_bytes=(tmp_path / name).stat().st_size,
            sha256=file_sha256(tmp_path / name),
        )
        for name in ("config.json", "README.md")
    ]
    manifest = SimpleNamespace(files=files)
    first = campaign.payload_digest(tmp_path, manifest)
    (tmp_path / "README.md").write_text("updated card")
    assert campaign.payload_digest(tmp_path, manifest) == first
    (tmp_path / "config.json").write_text('{"quantization":{"bits":8}}')
    assert campaign.payload_digest(tmp_path, manifest) != first


@pytest.mark.parametrize(
    "distribution,physical,mtp",
    [
        ({"4bit": 1, "6bit": 1}, {"bits": 6}, False),
        ({"8bit": 1}, {"bits": 8}, False),
        ({"6bit": 1}, {"bits": 6, "mode": "mxfp4"}, False),
        ({"6bit": 1}, {"bits": 4, "module": {"bits": 6}}, False),
        ({"6bit": 1}, {"bits": 6}, True),
    ],
)
def test_reject_legacy_packing_and_false_mtp(
    tmp_path: Path,
    job: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    distribution: dict,
    physical: dict,
    mtp: bool,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"quantization": physical}))
    manifest = SimpleNamespace(
        source_model=SimpleNamespace(model_id=job["source_id"], revision=job["source_revision"]),
        weight_distribution={
            key: SimpleNamespace(parameters=value) for key, value in distribution.items()
        },
        mtp_present=mtp,
    )
    monkeypatch.setattr(campaign, "load_model", lambda *args: manifest)
    with pytest.raises(ValueError):
        campaign.validate_pack(job, tmp_path)


def test_unrelated_private_repository_cannot_be_overwritten(
    tmp_path: Path,
    job: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    class FakeApi:
        def whoami(self) -> dict:
            return {"name": "AutomatosX"}

        def model_info(self, repo: str) -> SimpleNamespace:
            return SimpleNamespace(private=True)

        def upload_large_folder(self, **kwargs: Any) -> None:
            pytest.fail("Existing unrelated private repo must never be uploaded")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(campaign, "validate_pack", lambda *args: SimpleNamespace())
    monkeypatch.setattr(campaign, "payload_digest", lambda *args: "current")
    monkeypatch.setattr(
        campaign,
        "verify_files",
        lambda *args: {
            "axquant_manifest.json": {"sha256": "manifest"},
        },
    )
    (tmp_path / "axquant_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="unrelated existing"):
        campaign.publish(
            job,
            tmp_path,
            tmp_path,
            {
                "passed": True,
                "task": "embedding",
                "payload_sha256": "current",
            },
        )


def test_protected_decision_head_counts_once_and_preserves_bytes(
    tmp_path: Path,
    job: dict[str, Any],
    qwen36_model_dir: Path,
) -> None:
    import struct

    from axquant.schema import PrecisionShare

    origin = Path(job["source_path"])
    origin.mkdir()
    output = Path(job["output"])
    output.mkdir()
    header = json.dumps(
        {"joint.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}}
    ).encode()
    source_bytes = struct.pack("<Q", len(header)) + header + bytes(12)
    (origin / "joint_head.safetensors").write_bytes(source_bytes)
    (origin / "joint_head_config.json").write_text('{"hidden_size":3}')
    from axquant.analyzer import architecture_prior_report
    from axquant.inspector import inspect_model
    from axquant.planner import plan_quantization
    from axquant.runtime import build_runtime_metadata
    from axquant.schema import ArtifactManifest, PlanRequest, ProfileName
    from axquant.serde import load_model, stable_sha256, write_data

    inventory = inspect_model(qwen36_model_dir)
    plan = plan_quantization(
        architecture_prior_report(inventory, profile=ProfileName.GENERAL),
        PlanRequest(profile=ProfileName.GENERAL, target_bpw=14, allow_unmeasured=True),
    )
    manifest = ArtifactManifest(
        axquant_version="2.3.0",
        source_model=plan.source_model,
        plan_sha256=stable_sha256(plan),
        profile=plan.profile,
        target_class="6bit",
        effective_bpw=7,
        logical_parameters=1000,
        main_logical_parameters=1000,
        weight_file_size_bytes=1000,
        main_weight_file_size_bytes=1000,
        mtp_weight_file_size_bytes=0,
        protected_weight_file_size_bytes=0,
        measured_total_bpw=8,
        measured_main_bpw=8,
        weight_distribution={
            "6bit": PrecisionShare(parameters=900, fraction=0.9),
            "bf16": PrecisionShare(parameters=100, fraction=0.1),
        },
        mtp_distribution={},
        mtp_present=False,
        mtp_policy=plan.mtp,
        runtime=build_runtime_metadata(plan, output),
        software_versions=plan.software_versions,
        files=[],
    )
    write_data(output / "axquant_manifest.json", manifest)
    campaign.attach_decision_head(job, output)
    campaign.attach_decision_head(job, output)
    manifest = load_model(output / "axquant_manifest.json", ArtifactManifest)
    assert manifest.logical_parameters == 1006
    assert manifest.weight_file_size_bytes == 1000 + len(source_bytes)
    assert manifest.protected_weight_file_size_bytes == len(source_bytes)
    assert manifest.weight_distribution["bf16"].parameters == 106
    assert sum(share.fraction for share in manifest.weight_distribution.values()) == pytest.approx(
        1
    )
    assert (output / "joint_head.safetensors").read_bytes() == source_bytes
    assert (origin / "joint_head.safetensors").read_bytes() == source_bytes


def test_collection_update_keeps_existing_members(
    tmp_path: Path,
    job: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    members = {slug: {"owner/unrelated-model"} for slug in campaign.BASE_COLLECTIONS}

    class FakeApi:
        def model_info(self, repo: str) -> SimpleNamespace:
            return SimpleNamespace(private=False)

        def get_collection(self, slug: str) -> SimpleNamespace:
            return SimpleNamespace(
                items=[
                    SimpleNamespace(item_id=repo)
                    for repo in members.setdefault(slug, {"owner/unrelated-model"})
                ]
            )

        def add_collection_item(self, slug: str, repo: str, *args: Any, **kwargs: Any) -> None:
            members[slug].add(repo)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    campaign.update_collections(job, tmp_path)
    assert all(repos == {job["repo_id"], "owner/unrelated-model"} for repos in members.values())


def test_git_blob_identity_matches_git_standard(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"abc")
    assert campaign.git_blob_sha1(path) == "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"


@pytest.mark.parametrize("dtype", ["U32", "I8", "F8_E4M3"])
def test_external_assistant_cannot_hide_low_precision_weights(tmp_path: Path, dtype: str) -> None:
    import struct

    header = json.dumps(
        {"head.weight": {"dtype": dtype, "shape": [2, 3], "data_offsets": [0, 12]}}
    ).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + bytes(12)
    )
    with pytest.raises(ValueError, match="floating protection"):
        campaign.assistant_weight_accounting(tmp_path)


def test_digest_cache_rejects_same_size_edits_even_when_mtime_is_restored(tmp_path: Path) -> None:
    import os

    path = tmp_path / "weights"
    path.write_bytes(b"first")
    before = path.stat()
    first = campaign.file_sha256(path)
    path.write_bytes(b"other")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert campaign.file_sha256(path) != first


@pytest.mark.parametrize(
    "message,quantized,retries",
    [
        ("GPU Timeout Error", False, 2),
        ("Unmatched tensor", False, 1),
        ("GPU Timeout Error", True, 1),
    ],
)
def test_cpu_conversion_retry_is_bounded_and_does_not_mask_other_failures(
    tmp_path: Path,
    job: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    quantized: bool,
    retries: int,
) -> None:
    import subprocess

    monkeypatch.delenv("AXQUANT_FORCE_CPU", raising=False)
    job["source_quantized"] = quantized
    state = tmp_path / "state"
    state.mkdir()
    calls = []

    def run(command, log, timeout, *, overrides=None):
        calls.append(overrides)
        if len(calls) == 1:
            log.write_text(message)
            raise subprocess.CalledProcessError(2, command)
        assert overrides["AXQUANT_FORCE_CPU"] == "1"
        Path(job["output"]).mkdir()

    monkeypatch.setattr(campaign, "run_logged", run)
    monkeypatch.setattr(campaign, "validate_pack", lambda *args: None)
    if retries == 1:
        with pytest.raises(subprocess.CalledProcessError):
            campaign.convert(job, state)
    else:
        assert campaign.convert(job, state) == Path(job["output"])
        assert (state / "conversion-metal-timeout.log").read_text() == message
    assert len(calls) == retries


@pytest.mark.parametrize(
    "output,expected",
    [
        ("to=self<|message|>The result is 42", None),
        ("to=self<|message|>42<|eom|><|start|>assistant to=user<|message|>43<|eot|>", "43"),
        ("to=self<|message|>Thinking<|eom|><|start|>assistant to=user<|message|>42<|eot|>", "42"),
        ("to=user<|message|>READY<|eot|>", "READY"),
    ],
)
def test_atem_checks_only_explicit_user_channel(output: str, expected: str | None) -> None:
    assert campaign.atem_user_text(output) == expected


def test_custom_license_retains_source_name_and_link(job: dict[str, Any]) -> None:
    source = Path(job["source_path"])
    source.mkdir()
    job["license"] = "other"
    (source / "README.md").write_text(
        "---\nlicense: other\nlicense_name: example-research-license\n"
        "license_link: https://example.org/license\n---\nSource card\n"
    )
    front = campaign.license_frontmatter(job)
    assert 'license_name: "example-research-license"' in front
    assert 'license_link: "https://example.org/license"' in front
    (source / "README.md").write_text("---\nlicense: other\n---\n")
    with pytest.raises(ValueError, match="original name and link"):
        campaign.license_frontmatter(job)


@pytest.mark.parametrize(
    "output,prompt,expected",
    [
        ("The answer is 42", "assistant<think>", None),
        ("<think>42", "assistant", None),
        ("42</think>\n43", "assistant<think>", "43"),
        ("Thoughts</think>\nAXQ 7319", "assistant<think>", "AXQ 7319"),
        ("READY", "assistant<think></think>", "READY"),
    ],
)
def test_qwen_compares_final_payload_and_rejects_unfinished_reasoning(
    output: str,
    prompt: str,
    expected: str | None,
) -> None:
    assert campaign.qwen_user_text(output, prompt) == expected


@pytest.mark.parametrize("failed,backend", [(False, "mlx"), (True, "mlx"), (False, "llama_cpp")])
def test_native_stream_smoke_uses_source_tokens_and_preserves_engine_error(
    job: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failed: bool, backend: str
) -> None:
    import sys
    from types import ModuleType

    output = Path(job["output"])
    output.mkdir()
    (output / "config.json").write_text('{"model_type":"qwen3_5_moe"}')
    job.update(task="text", runtime="ax-engine-stream")
    monkeypatch.setenv("AX_MLX_PIPELINE_GRANULARITY", "layer")
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return [10, 11] if len(text) > 10 else [12]

        def decode(self, tokens, **kwargs):
            return "42" if tokens == [42] else "READY"

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda path: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(campaign, "engine_cli_version", lambda: "7.9.0")
    monkeypatch.setattr(campaign.importlib.metadata, "version", lambda name: "test")

    def run(command, **kwargs):
        calls.append(command)
        assert "--prompt" not in command
        assert command[command.index("--tokens") + 1] == "10,11"
        assert kwargs["env"]["AX_STREAM_EXPERTS"] == "on"
        assert kwargs["env"]["AX_MLX_PIPELINE_GRANULARITY"] == "layer"
        assert "--mlx" in command
        return SimpleNamespace(
            returncode=1 if failed else 0,
            stderr="Native runtime rejected this layout" if failed else "",
            stdout=json.dumps(
                {
                    "output_tokens": [42 if len(calls) == 1 else 99],
                    "runtime": {"selected_backend": backend},
                }
            ),
        )

    monkeypatch.setattr(campaign.subprocess, "run", run)
    if failed or backend != "mlx":
        expected_error = (
            "Native runtime rejected this layout" if failed else "did not select the requested MLX"
        )
        with pytest.raises(RuntimeError, match=expected_error):
            campaign.smoke_worker(job)
    else:
        receipt = campaign.smoke_worker(job)
        assert receipt["passed"] is True
        assert [case["output"] for case in receipt["cases"]] == ["42", "READY"]
        assert receipt["mtp_execution_verified"] is False
        assert receipt["runtime_environment"]["AX_MLX_PIPELINE_GRANULARITY"] == "layer"


def test_withdrawn_source_blocks_cached_campaign_before_conversion_or_inference(job) -> None:
    from axquant.errors import ArtifactError

    job["source_id"] = "ornith-ai/Ornith-1.5-397B"
    with pytest.raises(ArtifactError, match="withdrawn"):
        campaign.validate_job(job)
    with pytest.raises(ArtifactError, match="withdrawn"):
        campaign.smoke_worker(job)
