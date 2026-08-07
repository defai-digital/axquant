"""Measured kernel-latency evidence for latency-aware planning (ADR-0003).

A latency table is a *planning input*, never quality evidence: it re-ranks
candidates inside the quality-feasible set and its digest is recorded on the
plan for provenance. Tables are host-scoped like every performance artifact —
authorizing only on the formal host, informative elsewhere.
"""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from typing import Literal

from pydantic import Field, model_validator

from axquant.schema._base import SoftwareVersions, StrictModel, utc_now
from axquant.schema.enums import QuantMethod, RuntimeName


class KernelLatencyEntry(StrictModel):
    """Median timing for one executable (bits, group, method) configuration."""

    runtime: RuntimeName
    bits: int = Field(ge=2, le=16)
    group_size: int | None = Field(default=None, ge=1)
    method: QuantMethod
    hidden_size: int = Field(ge=1)
    decode_median_us: float = Field(gt=0.0)
    prefill_median_us: float = Field(gt=0.0)
    # Relative dispersion (IQR / median) across measured iterations; noisy
    # entries stay in the table but are flaggable by consumers.
    dispersion: float = Field(ge=0.0)
    iterations: int = Field(ge=1)

    @model_validator(mode="after")
    def bf16_has_no_group(self) -> KernelLatencyEntry:
        if self.bits == 16 and self.group_size is not None:
            raise ValueError("16-bit kernel entries must not declare a group size")
        if self.bits < 16 and self.group_size is None:
            raise ValueError("quantized kernel entries require a group size")
        return self


class KernelLatencyTable(StrictModel):
    """Host-scoped kernel latency evidence (``axquant.kernel-latency.v1``)."""

    schema_version: Literal["axquant.kernel-latency.v1"] = "axquant.kernel-latency.v1"
    # Hardware-registry host identifier the timings bind to (e.g. ``df-macbookpro-m5``).
    host_id: str = Field(min_length=1)
    chip: str = Field(min_length=1)
    os_version: str = Field(min_length=1)
    software_versions: SoftwareVersions
    warmup_iterations: int = Field(ge=0)
    entries: list[KernelLatencyEntry] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_configurations(self) -> KernelLatencyTable:
        keys = [
            (
                entry.runtime,
                entry.bits,
                entry.group_size,
                entry.method,
                entry.hidden_size,
            )
            for entry in self.entries
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("kernel latency entries must be unique per configuration")
        return self

    def decode_latency_us(
        self,
        *,
        runtime: RuntimeName,
        bits: int,
        group_size: int | None,
        method: QuantMethod,
        hidden_size: int,
    ) -> float | None:
        """Decode latency for a configuration, interpolated over hidden size.

        Exact (bits, group, method) matches only — a latency model must never
        guess across packing configurations. Between measured hidden sizes the
        lookup interpolates linearly; outside the measured range it clamps to
        the nearest measured size. Returns ``None`` when the configuration was
        not measured at all.
        """
        matching = sorted(
            (
                entry
                for entry in self.entries
                if entry.runtime == runtime
                and entry.bits == bits
                and entry.group_size == group_size
                and entry.method == method
            ),
            key=lambda entry: entry.hidden_size,
        )
        if not matching:
            return None
        if hidden_size <= matching[0].hidden_size:
            return matching[0].decode_median_us
        if hidden_size >= matching[-1].hidden_size:
            return matching[-1].decode_median_us
        for lower, upper in pairwise(matching):
            if lower.hidden_size <= hidden_size <= upper.hidden_size:
                span = upper.hidden_size - lower.hidden_size
                fraction = (hidden_size - lower.hidden_size) / span
                return (
                    lower.decode_median_us
                    + (upper.decode_median_us - lower.decode_median_us) * fraction
                )
        return matching[-1].decode_median_us
