from __future__ import annotations

from types import SimpleNamespace

from scripts import annotate_hub_omlx_mtp


class _FakeApi:
    def __init__(self, models: list[str], files: dict[str, list[str]]) -> None:
        self._models = models
        self._files = files

    def list_models(self, **kwargs: object) -> list[SimpleNamespace]:
        del kwargs
        return [SimpleNamespace(id=model_id) for model_id in self._models]

    def list_repo_files(self, repo_id: str) -> list[str]:
        return self._files[repo_id]


def test_repo_discovery_includes_unnamed_sidecar_packs() -> None:
    api = _FakeApi(
        models=[
            "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP",
            "AutomatosX/AX-Unnamed-Sidecar-Pack",
            "AutomatosX/AX-Plain-Embedding",
        ],
        files={
            "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP": [
                "config.json",
                "mtp.safetensors",
            ],
            "AutomatosX/AX-Unnamed-Sidecar-Pack": [
                "config.json",
                "mtp.safetensors",
            ],
            "AutomatosX/AX-Plain-Embedding": ["config.json"],
        },
    )

    assert annotate_hub_omlx_mtp._list_mtp_repos(api, "AutomatosX") == [  # type: ignore[arg-type]
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP",
        "AutomatosX/AX-Unnamed-Sidecar-Pack",
    ]
