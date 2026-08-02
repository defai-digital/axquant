"""Family support best practices encoded as product policy (AXQ-017).

Best practices (reviewed)
-------------------------
1. **Primary track first.** Certify and deepen Qwen 3.6 + AX Engine before
   matching OptiQ's full Mac catalog breadth.
2. **Thin secondary families.** Ship inspect/convert for high-ROI dense paths
   (Mistral/Devstral, Gemma-4, MiniCPM5, Qwen 3.5) without overselling cert.
3. **Nemotron is thin, not OptiQ-parity.** Support catalog **Nano-30B-A3B** MoE
   convert only. Super/Ultra (SSD stream / huge MoE product features) stay
   inspect-only until AX Engine commits a hybrid product path.
4. **Evidence never upgrades.** Architecture-prior / simple convert = development
   evidence. Certified claims need measured + release audit.
5. **Fail closed on unscoped catalogs.** Non-catalog Nemotron refs and unscoped
   Llama checkpoints do not become convertible by accident.
6. **Honest matrix.** Every family carries investment posture + notes so users
   do not confuse "adapter exists" with "OptiQ-level product support."

These rules live in code so ``support-matrix`` and adapters cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from axquant.schema import SupportTier


class InvestmentPosture(StrEnum):
    """How much product investment this family should receive."""

    PRIMARY = "primary"  # Flagship cert / AX Engine focus (Qwen 3.6)
    SECONDARY = "secondary"  # Convertible dense breadth (Devstral, Gemma, MiniCPM, …)
    THIN = "thin"  # Limited convert scope (e.g. Nemotron Nano only)
    DEFERRED = "deferred"  # Explicitly not a convert product path yet


@dataclass(frozen=True, slots=True)
class FamilySupportPolicy:
    product_family: str
    adapter_id: str
    investment_posture: InvestmentPosture
    # Lower number = higher priority for engineering and certification work.
    priority: int
    declared_tier: SupportTier
    cert_track: bool
    summary: str
    do: tuple[str, ...]
    do_not: tuple[str, ...]


# Single ordered source of truth for family investment (not convert eligibility).
FAMILY_POLICIES: tuple[FamilySupportPolicy, ...] = (
    FamilySupportPolicy(
        product_family="qwen3.6",
        adapter_id="qwen36-v1",
        investment_posture=InvestmentPosture.PRIMARY,
        priority=1,
        declared_tier=SupportTier.CONVERTIBLE,
        cert_track=True,
        summary="Flagship: dense 27B + MoE 35B-A3B, MTP, AX Engine certification track.",
        do=(
            "Finish formal release audit and MTP speed gate on AX Engine.",
            "Prefer measured ladders and dual-profile quality for public claims.",
        ),
        do_not=("Do not divert cert engineering to Super-class MoE stream features first.",),
    ),
    FamilySupportPolicy(
        product_family="qwen3.5",
        adapter_id="qwen35-dense-v1",
        investment_posture=InvestmentPosture.SECONDARY,
        priority=10,
        declared_tier=SupportTier.CONVERTIBLE,
        cert_track=False,
        summary="Secondary dense family with real convert evidence; development claims only.",
        do=("Keep dense convert path green; promote cert only after Qwen 3.6 cert.",),
        do_not=("Do not treat architecture-prior quants as certified releases.",),
    ),
    FamilySupportPolicy(
        product_family="qwen3-next",
        adapter_id="qwen3-next-v1",
        investment_posture=InvestmentPosture.SECONDARY,
        priority=11,
        declared_tier=SupportTier.CONVERTIBLE,
        cert_track=False,
        summary=(
            "Qwen3-Next hybrid MoE coding path (Coder-Next); fused-expert convert, "
            "development evidence only."
        ),
        do=(
            "Ship AXQ Coder-Next packs to close the OptiQ coding catalog gap.",
            "Keep MoE convert fail-closed for non-catalog qwen3_next refs without evidence.",
        ),
        do_not=(
            "Do not claim coding-bench scores from architecture-prior converts.",
            "Do not treat Next MoE as Qwen 3.6 cert track.",
        ),
    ),
    FamilySupportPolicy(
        product_family="qwen3",
        adapter_id="qwen3-dense-v1",
        investment_posture=InvestmentPosture.SECONDARY,
        priority=12,
        declared_tier=SupportTier.CONVERTIBLE,
        cert_track=False,
        summary=(
            "Base Qwen3 dense (model_type=qwen3), including Embedding-0.6B/4B/8B retrieval "
            "backbones."
        ),
        do=(
            "Publish AXQ embedding packs for catalog parity with uniform/OptiQ Hub rows.",
            "Label embedding packs as feature-extraction; use embedding evals for quality.",
        ),
        do_not=(
            "Do not claim generative chat quality or MTP metrics for embedding checkpoints.",
            "Do not match Qwen 3.5 / 3.6 / Next under this adapter.",
        ),
    ),
    FamilySupportPolicy(
        product_family="mistral-devstral",
        adapter_id="mistral-devstral-dense-v1",
        investment_posture=InvestmentPosture.SECONDARY,
        priority=15,
        declared_tier=SupportTier.CONVERTIBLE,
        cert_track=False,
        summary="High-ROI dense coding/agent family (MLX mistral→llama remap).",
        do=("Prefer Devstral/Mistral dense for breadth over Nemotron Super/Ultra.",),
        do_not=("Do not claim unrelated Llama checkpoints as Mistral support.",),
    ),
    FamilySupportPolicy(
        product_family="mistral3",
        adapter_id="mistral3-dense-v1",
        investment_posture=InvestmentPosture.SECONDARY,
        priority=16,
        declared_tier=SupportTier.CONVERTIBLE,
        cert_track=False,
        summary=(
            "Mistral3 multimodal shell (incl. Ministral-3): language path convertible; "
            "vision not optimized."
        ),
        do=("Optimize text path; preserve vision only when present as protected tensors.",),
        do_not=("Do not claim VLM optimization for Mistral3 vision towers.",),
    ),
    FamilySupportPolicy(
        product_family="gemma-4",
        adapter_id="gemma4-dense-v1",
        investment_posture=InvestmentPosture.SECONDARY,
        priority=20,
        declared_tier=SupportTier.CONVERTIBLE,
        cert_track=False,
        summary="Convertible via gemma4_unified→gemma4 text-path prep; multimodal sidecars.",
        do=("Keep source_prep + vision sidecar path covered by tests.",),
        do_not=("Do not claim native gemma4_unified MLX support without prep.",),
    ),
    FamilySupportPolicy(
        product_family="minicpm5",
        adapter_id="minicpm5-dense-v1",
        investment_posture=InvestmentPosture.SECONDARY,
        priority=25,
        declared_tier=SupportTier.CONVERTIBLE,
        cert_track=False,
        summary="Small dense family with real convert evidence; good fixture host.",
        do=("Use as low-cost convert/smoke fixture when possible.",),
        do_not=("Do not expand MiniCPM scope into unrelated Llama models.",),
    ),
    FamilySupportPolicy(
        product_family="nemotron3",
        adapter_id="nemotron3-v1",
        investment_posture=InvestmentPosture.THIN,
        priority=40,
        declared_tier=SupportTier.CONVERTIBLE,
        cert_track=False,
        summary=(
            "Thin support: catalog Nano-30B-A3B MoE convert only. "
            "Not OptiQ-parity (no SSD Super stream / hybrid KV product)."
        ),
        do=(
            "Allow convert for Nano-30B-A3B hybrid MoE via mlx_lm nemotron_h.",
            "Label all Nemotron outputs as development evidence until certified.",
            "Keep Super/Ultra inspect-only until AX Engine hybrid product path exists.",
        ),
        do_not=(
            "Do not market Super-120B / Ultra as AXQuant product targets.",
            "Do not invent SSD expert streaming or multi-Mac cluster to match OptiQ.",
            "Do not promote non-catalog Nemotron refs to convertible.",
        ),
    ),
)


_POLICY_BY_FAMILY = {policy.product_family: policy for policy in FAMILY_POLICIES}
_POLICY_BY_ADAPTER = {policy.adapter_id: policy for policy in FAMILY_POLICIES}


def policy_for_family(product_family: str) -> FamilySupportPolicy | None:
    return _POLICY_BY_FAMILY.get(product_family)


def policy_for_adapter(adapter_id: str) -> FamilySupportPolicy | None:
    return _POLICY_BY_ADAPTER.get(adapter_id)


def ordered_policies() -> list[FamilySupportPolicy]:
    return sorted(FAMILY_POLICIES, key=lambda item: (item.priority, item.adapter_id))


def support_policy_markdown() -> str:
    """Operator-facing family investment policy."""
    lines = [
        "# AXQuant family support policy",
        "",
        "Best practices for *which* families to invest in — distinct from the mechanical",
        "convert tier on a single checkpoint.",
        "",
        "| Priority | Family | Adapter | Posture | Declared tier | Cert track | Summary |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for policy in ordered_policies():
        lines.append(
            f"| {policy.priority} | `{policy.product_family}` | `{policy.adapter_id}` | "
            f"`{policy.investment_posture.value}` | `{policy.declared_tier.value}` | "
            f"{'yes' if policy.cert_track else 'no'} | {policy.summary} |"
        )
    lines.extend(["", "## Do / do not by family", ""])
    for policy in ordered_policies():
        lines.append(f"### `{policy.product_family}`")
        lines.append("")
        lines.append(policy.summary)
        lines.append("")
        lines.append("**Do**")
        for item in policy.do:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("**Do not**")
        for item in policy.do_not:
            lines.append(f"- {item}")
        lines.append("")
    lines.extend(
        [
            "## Global rules",
            "",
            "1. Two doors: simple convert = development; release = measured + audit.",
            "2. Primary cert track is Qwen 3.6 + AX Engine.",
            "3. Nemotron is thin (Nano only); Super/Ultra are not product convert targets.",
            "4. Prefer Mistral/Devstral dense breadth over hybrid Super-class MoE work.",
            "5. Never upgrade evidence_kind automatically.",
            "",
        ]
    )
    return "\n".join(lines)
