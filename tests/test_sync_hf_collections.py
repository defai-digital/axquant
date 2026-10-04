from __future__ import annotations

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
