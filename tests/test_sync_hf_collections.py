from __future__ import annotations

from dataclasses import replace

import pytest

from scripts import sync_hf_collections as collections


def test_collection_specs_are_internally_consistent() -> None:
    """Spec validation is the dry-run gate for every Hub apply."""
    assert collections._validate() is None


def test_method_collections_exist_with_items() -> None:
    titles = {spec.title: spec for spec in collections.COLLECTIONS}
    for title in ("MXFP4", "MXFP8", "NVFP4", "MTP"):
        assert titles[title].items, f"{title} collection must not be empty"
    for spec in collections.COLLECTIONS:
        for item in spec.items:
            assert item.repo.startswith("AutomatosX/")


def test_runtime_catalogs_cover_models_without_overlap() -> None:
    titles = {spec.title: spec for spec in collections.COLLECTIONS}
    mlx = {item.repo for item in titles["MLX"].items}
    cuda = {item.repo for item in titles["CUDA"].items}
    all_models = {item.repo for spec in collections.COLLECTIONS for item in spec.items}
    assert mlx.isdisjoint(cuda)
    assert all("-MLX-" in repo for repo in mlx)
    assert all("-CUDA-" in repo for repo in cuda)
    assert all_models == mlx | cuda
    embedding_cuda = {item.repo for item in titles["Embeddings"].items if "-CUDA-" in item.repo}
    assert len(embedding_cuda) == 5
    assert embedding_cuda <= cuda
    assert embedding_cuda <= {item.repo for item in titles["NVFP4"].items}


def test_runtime_catalog_overlap_aborts_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = next(spec for spec in collections.COLLECTIONS if spec.title == "CUDA")
    mlx = collections.COLLECTIONS[-1]
    invalid = replace(mlx, items=(*mlx.items, cuda.items[0]))
    monkeypatch.setattr(collections, "COLLECTIONS", (*collections.COLLECTIONS[:-1], invalid))
    with pytest.raises(SystemExit, match="must not overlap"):
        collections.sync(apply=False)
