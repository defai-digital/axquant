"""Development-only weight x KV interaction diagnostic (AXQuant 1.9.0b1).

This schema is evidence, never a certificate. ``certification_eligible`` is
frozen false so a diagnostic report cannot be mistaken for a release claim.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.enums import EvidenceKind, ProfileName
from axquant.schema.inventory import ModelIdentity
from axquant.schema.planning import AX_ENGINE_EXECUTABLE_BITS


class JointProxyScores(StrictModel):
    """Isolated-probe additive proxies. None means that side was not measured."""

    weight_output_kl: float | None = Field(default=None, ge=0.0)
    kv_output_kl: float | None = Field(default=None, ge=0.0)
    additive_output_kl: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def additive_matches_parts(self) -> JointProxyScores:
        if self.weight_output_kl is None or self.kv_output_kl is None:
            if self.additive_output_kl is not None:
                raise ValueError("additive proxy requires both weight and KV isolated scores")
            return self
        expected = self.weight_output_kl + self.kv_output_kl
        if self.additive_output_kl is None:
            raise ValueError("additive proxy is required when both isolated scores exist")
        if abs(self.additive_output_kl - expected) > 1e-9:
            raise ValueError("additive proxy does not equal weight + KV isolated scores")
        return self


class JointMeasuredDeltas(StrictModel):
    """One bound (W-only, KV-only, joint) quality-delta triple.

    ``interaction = joint_delta - weight_only_delta - kv_only_delta``.
    Positive interaction means isolated additivity understates the joint loss.
    """

    weight_only_delta: float = Field(ge=0.0)
    kv_only_delta: float = Field(ge=0.0)
    joint_delta: float = Field(ge=0.0)
    interaction: float
    threshold: float = Field(gt=0.0)
    material: bool
    weight_only_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kv_only_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    joint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def interaction_is_exact(self) -> JointMeasuredDeltas:
        expected = self.joint_delta - self.weight_only_delta - self.kv_only_delta
        if abs(self.interaction - expected) > 1e-9:
            raise ValueError("interaction does not match joint - weight - KV deltas")
        if self.material != (abs(self.interaction) >= self.threshold):
            raise ValueError("material flag does not match the interaction threshold")
        return self


class JointBudgetCandidate(StrictModel):
    """One (weight BPW, KV default bits, context) cell under a shared memory budget."""

    target_bpw: float = Field(gt=0.0, le=16.0)
    kv_default_bits: int
    context_length: int = Field(gt=0)
    weight_bytes: int = Field(ge=0)
    kv_bytes: int = Field(ge=0)
    reserve_bytes: int = Field(ge=0)
    limit_bytes: int = Field(gt=0)
    remainder_bytes: int
    feasible: bool
    estimated_main_bpw: float = Field(gt=0.0, le=16.0)
    proxy: JointProxyScores
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kv_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def accounting_is_exact(self) -> JointBudgetCandidate:
        if self.kv_default_bits not in AX_ENGINE_EXECUTABLE_BITS:
            raise ValueError(f"unsupported KV default bits {self.kv_default_bits}")
        expected = self.limit_bytes - (self.weight_bytes + self.kv_bytes + self.reserve_bytes)
        if self.remainder_bytes != expected:
            raise ValueError("candidate remainder does not match byte accounting")
        if self.feasible != (expected >= 0):
            raise ValueError("candidate feasibility does not match byte accounting")
        return self


class JointContextWinner(StrictModel):
    """Cheapest-proxy feasible pair at one context length, if any."""

    context_length: int = Field(gt=0)
    target_bpw: float | None = Field(default=None, gt=0.0, le=16.0)
    kv_default_bits: int | None = None
    feasible_count: int = Field(ge=0)
    proxy_score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def winner_is_consistent(self) -> JointContextWinner:
        has_winner = self.target_bpw is not None
        if has_winner != (self.kv_default_bits is not None):
            raise ValueError("context winner must set both target BPW and KV bits, or neither")
        if has_winner and self.feasible_count < 1:
            raise ValueError("a context winner requires at least one feasible candidate")
        if not has_winner and self.proxy_score is not None:
            raise ValueError("infeasible context cannot record a proxy winner")
        if (
            self.kv_default_bits is not None
            and self.kv_default_bits not in AX_ENGINE_EXECUTABLE_BITS
        ):
            raise ValueError(f"unsupported KV winner bits {self.kv_default_bits}")
        return self


class JointCrossoverSummary(StrictModel):
    winners: list[JointContextWinner] = Field(min_length=1)
    detected: bool

    @model_validator(mode="after")
    def detection_matches_winners(self) -> JointCrossoverSummary:
        pairs = {
            (winner.target_bpw, winner.kv_default_bits)
            for winner in self.winners
            if winner.target_bpw is not None
        }
        expected = len(pairs) > 1
        if self.detected != expected:
            raise ValueError("crossover detection does not match the distinct feasible winners")
        contexts = [winner.context_length for winner in self.winners]
        if len(contexts) != len(set(contexts)):
            raise ValueError("crossover winners must be unique per context length")
        return self


class JointInteractionReport(StrictModel):
    """Beta diagnostic: isolated-vs-joint interaction plus budget crossover."""

    schema_version: Literal["axquant.joint-interaction.v1"] = "axquant.joint-interaction.v1"
    experimental: Literal[True] = True
    certification_eligible: Literal[False] = False
    evidence_kind: EvidenceKind
    profile: ProfileName
    model: ModelIdentity
    limit_bytes: int = Field(gt=0)
    reserve_bytes: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    weight_bpws: tuple[float, ...] = Field(min_length=1)
    kv_default_bits: tuple[int, ...] = Field(min_length=1)
    contexts: tuple[int, ...] = Field(min_length=1)
    interaction: JointMeasuredDeltas | None = None
    candidates: list[JointBudgetCandidate] = Field(min_length=1)
    crossover: JointCrossoverSummary
    verdict: Literal[
        "insufficient-measured-interaction",
        "interaction-small",
        "interaction-material",
    ]
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def beta_contract_is_consistent(self) -> JointInteractionReport:
        if self.certification_eligible:
            raise ValueError("joint-interaction reports cannot be certification-eligible")
        if any(bits not in AX_ENGINE_EXECUTABLE_BITS for bits in self.kv_default_bits):
            raise ValueError("KV default bits must be AX Engine executable widths")
        if self.interaction is None:
            if self.verdict != "insufficient-measured-interaction":
                raise ValueError(
                    "missing measured triple must use insufficient-measured-interaction"
                )
        elif self.interaction.material:
            if self.verdict != "interaction-material":
                raise ValueError("material interaction must use verdict interaction-material")
        elif self.verdict != "interaction-small":
            raise ValueError("immaterial interaction must use verdict interaction-small")
        if self.evidence_kind is EvidenceKind.MEASURED:
            raise ValueError(
                "joint-interaction is development evidence; use measured_development, "
                "not release-quality measured"
            )
        return self
