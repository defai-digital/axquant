from __future__ import annotations

from axquant.kv_exec import _execution_summary


def test_execution_summary_matches_when_every_layer_hits_its_planned_bits() -> None:
    summary = _execution_summary(
        [4, 4, 16],
        [4, 4, 16],
        quantized_active=2,
        ok=True,
        output_characters=2,
    )
    assert summary["per_layer_execution"] is True


def test_execution_summary_fails_when_one_planned_layer_silently_reverts() -> None:
    # Layer 1 was planned for 4-bit but executed at BF16 (e.g. a cache-type or
    # layer-index mismatch); layer 0 still quantized correctly. A single
    # quantized layer must not be enough to report full per-layer fidelity —
    # the `or quantized_active > 0` escape hatch this replaces would have
    # wrongly reported True here.
    summary = _execution_summary(
        [4, 4, 16],
        [4, 16, 16],
        quantized_active=1,
        ok=True,
        output_characters=2,
    )
    assert summary["per_layer_execution"] is False
    assert summary["quantized_layers_active"] == 1
