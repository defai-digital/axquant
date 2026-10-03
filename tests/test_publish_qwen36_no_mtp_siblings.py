from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import publish_qwen36_no_mtp_siblings as siblings


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mtp.safetensors", True),
        ("mtp_head.safetensors", True),
        ("ax_mtp_sidecar_manifest.json", True),
        ("axquant_mtp_graft.json", True),
        ("mtplx_runtime.json", True),
        ("axquant_omlx_compat.json", True),
        ("model.safetensors", False),
        ("model-00001.safetensors", False),
        ("config.json", False),
        ("README.md", False),
        ("axquant_manifest.json", False),
    ],
)
def test_is_mtp_path_table(name: str, expected: bool) -> None:
    assert siblings._is_mtp_path(name) is expected


def test_pinned_source_revisions_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN36_MTP_REVISIONS", raising=False)

    assert siblings.pinned_source_revisions() == {}


def test_pinned_source_revisions_accepts_immutable_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN36_MTP_REVISIONS", json.dumps({"AutomatosX/AX-Repo": "a" * 40}))

    assert siblings.pinned_source_revisions() == {"AutomatosX/AX-Repo": "a" * 40}


@pytest.mark.parametrize("raw", ["not-json", '{"repo": "main"}', "[1, 2]"])
def test_pinned_source_revisions_rejects_bad_maps(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("QWEN36_MTP_REVISIONS", raw)

    with pytest.raises(SystemExit, match="QWEN36_MTP_REVISIONS"):
        siblings.pinned_source_revisions()


def _write_snap(snap: Path) -> dict[str, bytes]:
    snap.mkdir(parents=True)
    (snap / "model-00001.safetensors").write_bytes(b"weights")
    (snap / "mtp.safetensors").write_bytes(b"mtp")
    (snap / "mtplx_runtime.json").write_text("{}", encoding="utf-8")
    (snap / "axquant_omlx_compat.json").write_text("{}", encoding="utf-8")
    (snap / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "num_mtp_layers": 1}), encoding="utf-8"
    )
    (snap / "axquant_plan.json").write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "role": "mtp_block",
                        "tensor": "mtp.fc.weight",
                        "module_path": "mtp.fc",
                        "bits": 16,
                        "parameters": 8,
                    },
                    {
                        "role": "mlp",
                        "tensor": "model.layers.0.mlp.down_proj.weight",
                        "module_path": "model.layers.0.mlp.down_proj",
                        "bits": 4,
                        "parameters": 16,
                    },
                ],
                "mtp": {"mode": "protected"},
                "mtp_distribution": {"mtp_block": 1.0},
                "effective_bpw": 8.0,
                "nominal_bpw": 8.0,
            }
        ),
        encoding="utf-8",
    )
    (snap / "axquant_manifest.json").write_text(
        json.dumps(
            {
                "mtp_present": True,
                "mtp_weight_file_size_bytes": 3,
                "mtp_acceptance_retention": 0.9,
                "mtp_measured_speedup": 1.2,
                "logical_parameters": 24,
                "main_logical_parameters": 16,
                "mtp_distribution": {"mtp_block": 1.0},
                "mtp_policy": {"mode": "protected"},
                "files": [{"path": "mtp.safetensors"}, {"path": "model-00001.safetensors"}],
            }
        ),
        encoding="utf-8",
    )
    return {path.name: path.read_bytes() for path in sorted(snap.iterdir()) if path.is_file()}


def test_materialize_strips_mtp_companions(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    _write_snap(snap)
    dest = tmp_path / "dest"

    siblings.materialize_no_mtp(snap, dest)

    assert not (dest / "mtp.safetensors").exists()
    assert not (dest / "mtplx_runtime.json").exists()
    assert not (dest / "axquant_omlx_compat.json").exists()
    assert (dest / "model-00001.safetensors").is_file()
    manifest = json.loads((dest / "axquant_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mtp_present"] is False
    assert manifest["mtp_weight_file_size_bytes"] == 0
    assert manifest["mtp_policy"]["mode"] == "disabled"
    assert manifest["files"] == [{"path": "model-00001.safetensors"}]
    config = json.loads((dest / "config.json").read_text(encoding="utf-8"))
    assert config["num_mtp_layers"] == 0
    plan = json.loads((dest / "axquant_plan.json").read_text(encoding="utf-8"))
    assert plan["mtp"]["mode"] == "disabled"
    assert plan["mtp_distribution"] == {}
    assert all("mtp" not in str(a.get("tensor", "")).lower() for a in plan["assignments"])


def test_materialize_leaves_reused_snapshot_byte_identical(tmp_path: Path) -> None:
    snap = tmp_path / "snap"
    before = _write_snap(snap)

    siblings.materialize_no_mtp(snap, tmp_path / "dest")

    after = {path.name: path.read_bytes() for path in sorted(snap.iterdir()) if path.is_file()}
    assert after == before
