from __future__ import annotations

from pathlib import Path

import pytest

from axquant import publisher
from axquant.errors import PublishingError
from axquant.publisher import publication_privacy_issues, require_publication_privacy
from axquant.serde import write_data


def test_publication_privacy_accepts_claim_safe_package(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "Run the model from /path/to/downloaded-checkpoint.\n",
        encoding="utf-8",
    )
    (tmp_path / "public-claim.json").write_text(
        '{"tokenizer_sha256":"abc","limitations":["internal reproduction"]}\n',
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_bytes(b"binary checkpoint")

    assert publication_privacy_issues(tmp_path) == []
    require_publication_privacy(tmp_path)


def test_publication_privacy_scans_large_tokenizer_but_rejects_large_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publisher, "_MAX_PUBLIC_TEXT_SCAN_BYTES", 32)
    (tmp_path / "tokenizer.json").write_text('{"vocab":"' + ("x" * 64) + '"}\n', encoding="utf-8")

    assert publication_privacy_issues(tmp_path) == []

    (tmp_path / "tokenizer.json").write_text(
        '{"vocab":"' + ("x" * 64) + ' /Users/operator/private"}\n',
        encoding="utf-8",
    )
    assert any("macOS home path" in issue for issue in publication_privacy_issues(tmp_path))

    (tmp_path / "tokenizer.json").write_text('{"vocab":"' + ("x" * 64) + '"}\n', encoding="utf-8")
    (tmp_path / "evidence.json").write_text('{"payload":"' + ("x" * 64) + '"}\n', encoding="utf-8")
    issues = publication_privacy_issues(tmp_path)
    assert any("exceeds privacy-scan limit" in issue for issue in issues)


def test_publication_privacy_rejects_paths_tokens_private_hosts_and_raw_holdout(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "producer=/Users/operator/models/candidate host=192.168.1.10\n",
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        '{"token":"hf_abcdefghijklmnopqrstuvwxyz123456","cache":"/tmp/private-run"}\n',
        encoding="utf-8",
    )
    raw = tmp_path / "formal" / "raw"
    raw.mkdir(parents=True)
    (raw / "task.json").write_text('{"prompt":"private holdout"}\n', encoding="utf-8")
    (tmp_path / "runtime.json").write_text(
        '{"artifact":"/Volumes/Models/campaign/candidate"}\n',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("API_KEY=abcdefghijk\n", encoding="utf-8")
    (tmp_path / ".env.production").write_text("SAFE=redacted\n", encoding="utf-8")
    (tmp_path / "credentials.yaml").write_text("token: redacted\n", encoding="utf-8")

    issues = publication_privacy_issues(tmp_path)

    assert any("macOS home path" in issue for issue in issues)
    assert any("macOS volume path" in issue for issue in issues)
    assert any("POSIX temporary path" in issue for issue in issues)
    assert any("private IPv4 address" in issue for issue in issues)
    assert any("Hugging Face token" in issue for issue in issues)
    assert any("formal raw evidence" in issue for issue in issues)
    sensitive_filenames = [issue for issue in issues if "sensitive publication filename" in issue]
    assert len(sensitive_filenames) == 3
    with pytest.raises(PublishingError, match="privacy scan failed"):
        require_publication_privacy(tmp_path)


def test_flagship_publication_rescans_files_added_after_request_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FlagshipAudit:
        pass

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    request = tmp_path / "request.json"
    write_data(
        request,
        {
            "schema_version": "axquant.flagship-release-audit-request.v1",
        },
    )

    monkeypatch.setattr(publisher, "FlagshipReleaseAudit", _FlagshipAudit)
    monkeypatch.setattr(
        publisher,
        "_require_release_audit",
        lambda **_kwargs: _FlagshipAudit(),
    )
    monkeypatch.setattr(
        publisher,
        "_require_flagship_request_inputs",
        lambda **_kwargs: None,
    )

    def rerun_with_leak(**_kwargs: object) -> None:
        leaked = artifact / "packaged-evidence.json"
        leaked.write_text('{"producer":"/Users/operator/private/run"}\n', encoding="utf-8")

    monkeypatch.setattr(publisher, "_rerun_release_audit", rerun_with_leak)
    monkeypatch.setattr(publisher, "_require_release_validation", lambda **_kwargs: None)
    monkeypatch.setattr(
        publisher,
        "prepare_publication",
        lambda **_kwargs: pytest.fail("flagship publication must skip legacy preparation"),
    )

    with pytest.raises(PublishingError, match="privacy scan failed"):
        publisher.publish_model(
            model_dir=artifact,
            repo_id="AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-MP-5p30bpw-MTP",
            validation_index_path=tmp_path / "validation.json",
            hardware_registry_path=tmp_path / "hardware.json",
            pareto_report_path=tmp_path / "pareto.json",
            release_audit_request_path=request,
        )
