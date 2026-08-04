from __future__ import annotations

from functools import lru_cache

from axquant.schema import Qwen3NextDirectPolicy
from axquant.serde import stable_sha256


@lru_cache(maxsize=1)
def _cached_direct_policy() -> Qwen3NextDirectPolicy:
    return Qwen3NextDirectPolicy()


def direct_policy() -> Qwen3NextDirectPolicy:
    """Return an isolated copy of the wheel-owned non-MTP policy.

    Pydantic models validate assignment but remain mutable. Returning the
    cached instance itself would let one caller lower process-global release
    thresholds for every later audit.
    """

    return _cached_direct_policy().model_copy(deep=True)


def direct_policy_sha256() -> str:
    return stable_sha256(direct_policy())
