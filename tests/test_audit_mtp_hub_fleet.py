from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from huggingface_hub.errors import HfHubHTTPError

from scripts import audit_mtp_hub_fleet


def _fake_response(status_code: int) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.headers.get.return_value = "test"
    return response


def _patch(monkeypatch, *, repo_loaders, name_discover=(), manifest_discover=()) -> None:
    monkeypatch.setattr(audit_mtp_hub_fleet, "HfApi", lambda: None)
    monkeypatch.setattr(
        audit_mtp_hub_fleet,
        "discover_axq_mtp_repositories",
        lambda api, author: tuple(name_discover),
    )
    monkeypatch.setattr(
        audit_mtp_hub_fleet,
        "discover_axq_mtp_artifact_repositories",
        lambda api, author: tuple(manifest_discover),
    )
    monkeypatch.setattr(
        audit_mtp_hub_fleet,
        "load_mtp_hub_snapshot",
        lambda api, repo_id: repo_loaders[repo_id](),
    )
    monkeypatch.setattr(
        audit_mtp_hub_fleet,
        "audit_mtp_hub_snapshot",
        lambda snapshot: snapshot,
    )


def _snapshot(repo_id: str) -> object:
    return audit_mtp_hub_fleet.MtpHubAuditResult(
        repo_id=repo_id,
        revision="0" * 40,
        kind=audit_mtp_hub_fleet.MtpHubPackKind.QWEN_RESIDENT,
        issues=(),
    )


def test_main_returns_0_when_every_repo_passes(monkeypatch, capsys, tmp_path: Path) -> None:
    _patch(
        monkeypatch,
        repo_loaders={
            "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP": lambda: _snapshot(
                "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP"
            )
        },
        name_discover=("AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP",),
        manifest_discover=(),
    )

    json_path = tmp_path / "fleet.json"
    code = audit_mtp_hub_fleet.main(["--author", "AutomatosX", "--json-output", str(json_path)])

    assert code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["passed_count"] == 1
    assert payload["failed_count"] == 0
    assert payload["audit_unavailable_count"] == 0
    assert payload["discovery"] == {"name": 1, "manifest": 0, "union": 1}
    assert "discovery: name=1 manifest=0 union=1" in capsys.readouterr().out


def test_main_returns_2_when_only_transient_load_failures(monkeypatch, capsys) -> None:
    def _raise_5xx() -> object:
        raise HfHubHTTPError("upstream 503", response=_fake_response(503))

    _patch(
        monkeypatch,
        repo_loaders={"AutomatosX/AX-Flash-Next-MLX-AXQ-MXFP4-MTP": _raise_5xx},
    )

    code = audit_mtp_hub_fleet.main(
        [
            "--author",
            "AutomatosX",
            "--repo",
            "AutomatosX/AX-Flash-Next-MLX-AXQ-MXFP4-MTP",
        ]
    )

    out = capsys.readouterr().out
    assert code == 2
    assert "SKIP" in out
    assert "[transient] audit could not load repository" in out


def test_main_returns_1_when_real_load_failure(monkeypatch) -> None:
    def _raise_404() -> object:
        raise HfHubHTTPError("not found", response=_fake_response(404))

    _patch(
        monkeypatch,
        repo_loaders={"AutomatosX/AX-Gone": _raise_404},
    )

    code = audit_mtp_hub_fleet.main(["--author", "AutomatosX", "--repo", "AutomatosX/AX-Gone"])

    assert code == 1


def test_main_returns_1_when_mixed_real_failure_and_transient(monkeypatch) -> None:
    def _raise_5xx() -> object:
        raise HfHubHTTPError("upstream 503", response=_fake_response(503))

    def _raise_404() -> object:
        raise HfHubHTTPError("not found", response=_fake_response(404))

    _patch(
        monkeypatch,
        repo_loaders={
            "AutomatosX/AX-Ok": lambda: _snapshot("AutomatosX/AX-Ok"),
            "AutomatosX/AX-5xx": _raise_5xx,
            "AutomatosX/AX-404": _raise_404,
        },
        name_discover=(
            "AutomatosX/AX-Ok",
            "AutomatosX/AX-5xx",
            "AutomatosX/AX-404",
        ),
    )

    code = audit_mtp_hub_fleet.main(["--author", "AutomatosX"])

    assert code == 1


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (HfHubHTTPError("hub 429", response=_fake_response(429)), "transient"),
        (HfHubHTTPError("hub 503", response=_fake_response(503)), "transient"),
        (HfHubHTTPError("hub 500", response=_fake_response(500)), "transient"),
        (HfHubHTTPError("hub 404", response=_fake_response(404)), "real"),
        (HfHubHTTPError("hub 400", response=_fake_response(400)), "real"),
        (ConnectionError("network unreachable"), "transient"),
        (TimeoutError("read timeout"), "transient"),
        (ValueError("unexpected schema"), "real"),
    ],
)
def test_classify_error_table(error: Exception, expected: str) -> None:
    assert audit_mtp_hub_fleet._classify_error(error) == expected


def test_discovery_counts_are_independent_of_union(monkeypatch) -> None:
    _patch(
        monkeypatch,
        repo_loaders={
            "AutomatosX/AX-Named": lambda: _snapshot("AutomatosX/AX-Named"),
            "AutomatosX/AX-Manifest-Only": lambda: _snapshot("AutomatosX/AX-Manifest-Only"),
        },
        name_discover=("AutomatosX/AX-Named",),
        manifest_discover=(
            "AutomatosX/AX-Named",
            "AutomatosX/AX-Manifest-Only",
        ),
    )

    json_path = Path("/tmp/ngram-review/fleet-discovery.json")
    code = audit_mtp_hub_fleet.main(["--author", "AutomatosX", "--json-output", str(json_path)])

    assert code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["discovery"] == {"name": 1, "manifest": 2, "union": 2}
    assert payload["repository_count"] == 2
