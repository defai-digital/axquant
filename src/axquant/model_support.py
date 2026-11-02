"""Current model-specific withdrawals, independent of saved family profiles."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from axquant.errors import ArtifactError

_ORNITH_397B = re.compile(r"ornith[^/\\]*?397b(?=$|[^a-z0-9])", re.IGNORECASE)
_ORNITH_397B_REASON = (
    "Ornith 397B support was withdrawn by the user on 2026-10-24; "
    "inspection only. Conversion and publication are disabled."
)


def model_support_withdrawal_reason(
    *references: str | Path | None,
    config: Mapping[str, Any] | None = None,
) -> str | None:
    """Recognize source IDs, pack names and local/config identity aliases."""
    names = [str(reference) for reference in references if reference is not None]
    if config is not None:
        for metadata in (config, config.get("text_config"), config.get("source_model")):
            if isinstance(metadata, Mapping):
                for key in ("_name_or_path", "model_id"):
                    value = metadata.get(key)
                    if isinstance(value, str):
                        names.append(value)
    return _ORNITH_397B_REASON if any(_ORNITH_397B.search(name) for name in names) else None


def require_model_supported(
    *references: str | Path | None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Reject withdrawn models even when an older plan calls them convertible."""
    reason = model_support_withdrawal_reason(*references, config=config)
    if reason is not None:
        raise ArtifactError(reason)
