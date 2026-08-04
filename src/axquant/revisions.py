from __future__ import annotations

import re

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def is_immutable_revision(revision: str | None) -> bool:
    """Return whether ``revision`` is a full, lowercase Git commit SHA."""

    return revision is not None and _IMMUTABLE_REVISION.fullmatch(revision) is not None
