"""Development-only weight x KV interaction diagnostic (AXQuant 1.9.0).

This schema is evidence, never a certificate. ``certification_eligible`` is
frozen false so a diagnostic report cannot be mistaken for a release claim.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from axquant.schema._base import StrictModel, utc_now
from axquant.schema.enums import ProfileName
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
    """One bound (baseline, W-only, KV-only, joint) quality-score quadruple.

    Scores are mean task scores in ``[0, 1]``. Deltas are signed:

    ``delta = baseline_score - treatment_score``
    ``interaction = joint_delta - weight_only_delta - kv_only_delta``

    Positive interaction means isolated additivity understates the joint loss.
    """

    baseline_score: float = Field(ge=0.0, le=1.0)
    weight_only_score: float = Field(ge=0.0, le=1.0)
    kv_only_score: float = Field(ge=0.0, le=1.0)
    joint_score: float = Field(ge=0.0, le=1.0)
    weight_only_delta: float
    kv_only_delta: float
    joint_delta: float
    interaction: float
    threshold: float = Field(gt=0.0)
    material: bool
    baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight_only_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kv_only_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    joint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def interaction_is_exact(self) -> JointMeasuredDeltas:
        if abs(self.weight_only_delta - (self.baseline_score - self.weight_only_score)) > 1e-9:
            raise ValueError("weight-only delta does not match baseline - weight-only score")
        if abs(self.kv_only_delta - (self.baseline_score - self.kv_only_score)) > 1e-9:
            raise ValueError("KV-only delta does not match baseline - KV-only score")
        if abs(self.joint_delta - (self.baseline_score - self.joint_score)) > 1e-9:
            raise ValueError("joint delta does not match baseline - joint score")
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
    ranking_available: bool
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
        if self.ranking_available and self.proxy.additive_output_kl is None:
            raise ValueError("rankable candidates require a complete additive proxy")
        if not self.ranking_available and self.proxy.additive_output_kl is not None:
            raise ValueError("complete additive proxy must be marked rankable")
        return self


class JointContextWinner(StrictModel):
    """Lowest-proxy feasible rankable pair at one context length, if any."""

    context_length: int = Field(gt=0)
    target_bpw: float | None = Field(default=None, gt=0.0, le=16.0)
    kv_default_bits: int | None = None
    feasible_count: int = Field(ge=0)
    rankable_count: int = Field(ge=0)
    proxy_score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def winner_is_consistent(self) -> JointContextWinner:
        has_winner = self.target_bpw is not None
        if has_winner != (self.kv_default_bits is not None):
            raise ValueError("context winner must set both target BPW and KV bits, or neither")
        if has_winner and self.rankable_count < 1:
            raise ValueError("a context winner requires at least one rankable candidate")
        if has_winner and self.feasible_count < 1:
            raise ValueError("a context winner requires at least one feasible candidate")
        if not has_winner and self.proxy_score is not None:
            raise ValueError("a context without a winner cannot record a proxy score")
        if self.rankable_count > self.feasible_count:
            raise ValueError("rankable count cannot exceed feasible count")
        if (
            self.kv_default_bits is not None
            and self.kv_default_bits not in AX_ENGINE_EXECUTABLE_BITS
        ):
            raise ValueError(f"unsupported KV winner bits {self.kv_default_bits}")
        return self


class JointCrossoverSummary(StrictModel):
    winners: list[JointContextWinner] = Field(min_length=1)
    detected: bool
    ranking_complete: bool

    @model_validator(mode="after")
    def detection_matches_winners(self) -> JointCrossoverSummary:
        pairs = {
            (winner.target_bpw, winner.kv_default_bits)
            for winner in self.winners
            if winner.target_bpw is not None
        }
        expected = len(pairs) > 1
        if self.detected != expected:
            raise ValueError("crossover detection does not match the distinct rankable winners")
        complete = all(winner.rankable_count == winner.feasible_count for winner in self.winners)
        if self.ranking_complete != complete:
            raise ValueError("ranking_complete does not match per-context rankable coverage")
        contexts = [winner.context_length for winner in self.winners]
        if len(contexts) != len(set(contexts)):
            raise ValueError("crossover winners must be unique per context length")
        return self


class JointInteractionReport(StrictModel):
    """Beta diagnostic: isolated-vs-joint interaction plus budget crossover."""

    schema_version: Literal["axquant.joint-interaction.v1"] = "axquant.joint-interaction.v1"
    experimental: Literal[True] = True
    certification_eligible: Literal[False] = False
    evidence_kind: Literal["architecture_prior", "measured_development"]
    profile: ProfileName
    model: ModelIdentity
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight_sensitivity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kv_sensitivity_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
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
                    "missing measured quadruple must use insufficient-measured-interaction"
                )
        elif self.interaction.material:
            if self.verdict != "interaction-material":
                raise ValueError("material interaction must use verdict interaction-material")
        elif self.verdict != "interaction-small":
            raise ValueError("immaterial interaction must use verdict interaction-small")
        return self


class JointScoredCell(StrictModel):
    """One feasible cell scored for joint selection."""

    target_bpw: float = Field(gt=0.0, le=16.0)
    kv_default_bits: int
    context_length: int = Field(gt=0)
    additive_loss: float = Field(ge=0.0)
    coupled_loss: float
    selected: bool

    @model_validator(mode="after")
    def bits_are_executable(self) -> JointScoredCell:
        if self.kv_default_bits not in AX_ENGINE_EXECUTABLE_BITS:
            raise ValueError(f"unsupported KV default bits {self.kv_default_bits}")
        return self


class JointSelectionReport(StrictModel):
    """I-gated choice of a convert-ready weight + KV plan."""

    schema_version: Literal["axquant.joint-selection.v1"] = "axquant.joint-selection.v1"
    experimental: Literal[True] = True
    certification_eligible: Literal[False] = False
    selection_basis: Literal["independent", "coupled-interaction"]
    context_length: int = Field(gt=0)
    independent_target_bpw: float = Field(gt=0.0, le=16.0)
    selected_target_bpw: float = Field(gt=0.0, le=16.0)
    selected_kv_bits: int
    differs_from_independent: bool
    interaction: float | None = None
    interaction_material: bool
    coupled_loss: float | None = None
    additive_loss: float | None = None
    scored_cells: list[JointScoredCell] = Field(default_factory=list)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kv_plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    diagnostic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def selection_is_consistent(self) -> JointSelectionReport:
        if self.certification_eligible:
            raise ValueError("joint-selection reports cannot be certification-eligible")
        if self.selected_kv_bits not in AX_ENGINE_EXECUTABLE_BITS:
            raise ValueError(f"unsupported selected KV bits {self.selected_kv_bits}")
        if self.selection_basis == "independent" and self.interaction_material:
            raise ValueError("material interaction cannot select the independent basis")
        if self.selection_basis == "coupled-interaction" and not self.interaction_material:
            raise ValueError("coupled selection requires a material interaction")
        selected = [cell for cell in self.scored_cells if cell.selected]
        if self.selection_basis == "coupled-interaction":
            if len(selected) != 1:
                raise ValueError("coupled selection must mark exactly one scored cell")
            cell = selected[0]
            if (
                cell.target_bpw != self.selected_target_bpw
                or cell.kv_default_bits != self.selected_kv_bits
                or cell.context_length != self.context_length
            ):
                raise ValueError("selected cell does not match the reported plan coordinates")
        elif selected:
            raise ValueError("independent selection cannot mark a coupled scored cell")
        return self
