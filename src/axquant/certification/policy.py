from __future__ import annotations

from functools import lru_cache

from axquant.schema import Qwen3NextDirectPolicy
from axquant.serde import stable_sha256


@lru_cache(maxsize=1)
def direct_policy() -> Qwen3NextDirectPolicy:
    """Return the immutable, wheel-owned non-MTP certification policy."""

    return Qwen3NextDirectPolicy()


def direct_policy_sha256() -> str:
    return stable_sha256(direct_policy())
