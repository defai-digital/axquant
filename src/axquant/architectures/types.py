from __future__ import annotations

from typing import Any, Protocol

from axquant.schema import ArchitectureProfile, TensorRole


class ArchitectureAdapter(Protocol):
    adapter_id: str

    def matches(self, model_reference: str, config: dict[str, Any]) -> bool: ...

    def profile(self, model_reference: str, config: dict[str, Any]) -> ArchitectureProfile: ...

    def classify_tensor(self, name: str, source_file: str) -> TensorRole | None: ...
