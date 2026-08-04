from __future__ import annotations

from axquant.revisions import is_immutable_revision


def test_immutable_revision_requires_full_lowercase_commit_sha() -> None:
    assert is_immutable_revision("a" * 40)
    assert not is_immutable_revision(None)
    assert not is_immutable_revision("")
    assert not is_immutable_revision("main")
    assert not is_immutable_revision("A" * 40)
    assert not is_immutable_revision("a" * 39)
    assert not is_immutable_revision("a" * 41)
