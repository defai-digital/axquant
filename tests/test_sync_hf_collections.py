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
    reference = {item.repo for item in titles["AXQ Reference Storage"].items}
    all_models = {item.repo for spec in collections.COLLECTIONS for item in spec.items}
    assert mlx.isdisjoint(cuda)
    assert all("-MLX-" in repo for repo in mlx)
    assert all("-CUDA-" in repo for repo in cuda)
    assert reference.isdisjoint(mlx | cuda)
    assert all_models == mlx | cuda | reference
    embedding_cuda = {item.repo for item in titles["Embeddings"].items if "-CUDA-" in item.repo}
    assert len(embedding_cuda) == 6
    assert embedding_cuda <= cuda
    assert embedding_cuda <= {item.repo for item in titles["NVFP4"].items}


def test_runtime_catalog_overlap_aborts_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = next(spec for spec in collections.COLLECTIONS if spec.title == "CUDA")
    mlx = collections.COLLECTIONS[-1]
    invalid = replace(mlx, items=(*mlx.items, cuda.items[0]))
    monkeypatch.setattr(collections, "COLLECTIONS", (*collections.COLLECTIONS[:-1], invalid))
    with pytest.raises(SystemExit, match="must not overlap"):
        collections.sync(apply=False)


def test_reference_storage_stays_out_of_native_runtime_and_method_catalogs() -> None:
    titles = {spec.title: spec for spec in collections.COLLECTIONS}
    reference = {item.repo for item in titles["AXQ Reference Storage"].items}
    assert "AutomatosX/AX-Image-2.1-Turbo-AXQ-NVFP4" in reference
    assert len(reference) == 1
    for title in ("MLX", "CUDA", "NVFP4", "MXFP4", "MXFP8"):
        assert reference.isdisjoint(item.repo for item in titles[title].items)


def test_doclayout_collection_tracks_source_and_native_mlx_bundles() -> None:
    titles = {spec.title: spec for spec in collections.COLLECTIONS}
    doclayout = {item.repo for item in titles["AX DocLayout"].items}
    expected = {
        "AutomatosX/AXDocLayout-M-v1.2-CUDA-TF32",
        "AutomatosX/AXDocLayout-M-v1.2-MLX-AXQ-MXFP8",
        "AutomatosX/AXDocLayout-M-v1.3-CUDA-TF32",
        "AutomatosX/AXDocLayout-M-v1.3-MLX-AXQ-MXFP8",
    }
    assert doclayout == expected
    runtime_catalogs = {item.repo for title in ("CUDA", "MLX") for item in titles[title].items}
    assert expected <= runtime_catalogs
    assert all("-v1.0" not in repo for repo in doclayout)


def test_reference_runtime_overlap_aborts_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = next(spec for spec in collections.COLLECTIONS if spec.title == "CUDA")
    specs = tuple(
        replace(spec, items=(*spec.items, cuda.items[0]))
        if spec.title == "AXQ Reference Storage"
        else spec
        for spec in collections.COLLECTIONS
    )
    monkeypatch.setattr(collections, "COLLECTIONS", specs)
    with pytest.raises(SystemExit, match="must not overlap"):
        collections.sync(apply=False)
