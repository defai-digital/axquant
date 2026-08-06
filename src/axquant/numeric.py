"""Shared numeric coercions for weight-refinement backends."""

from __future__ import annotations

from typing import Any

from axquant.errors import PlanningError
from axquant.package_data import message_template


def as_finite_float32_matrix(weight: Any, *, component: str) -> Any:
    """Coerce ``weight`` to a finite float32 array for AWQ/GPTQ-style backends.

    Message wording is preserved bit-for-bit via ``messages.yaml`` templates so
    existing tests that match on the component label keep passing.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise PlanningError(
            message_template("quantizer", "requires_numpy").format(component=component)
        ) from exc
    try:
        result = np.asarray(weight, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanningError(
            message_template("quantizer", "weight_not_numeric").format(
                component=component, error=exc
            )
        ) from exc
    if result.size == 0:
        raise PlanningError(
            message_template("quantizer", "weight_empty").format(component=component)
        )
    if not bool(np.all(np.isfinite(result))):
        raise PlanningError(
            message_template("quantizer", "weight_non_finite").format(component=component)
        )
    return result
