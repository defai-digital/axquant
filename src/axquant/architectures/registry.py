from __future__ import annotations

from typing import Any

from axquant.architectures.qwen36 import Qwen36Adapter
from axquant.architectures.types import ArchitectureAdapter

_ADAPTERS: tuple[ArchitectureAdapter, ...] = (Qwen36Adapter(),)


def adapter_for(
    model_reference: str,
    config: dict[str, Any],
) -> ArchitectureAdapter | None:
    return next(
        (adapter for adapter in _ADAPTERS if adapter.matches(model_reference, config)),
        None,
    )
