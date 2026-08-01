from __future__ import annotations

from enum import StrEnum


class TensorRole(StrEnum):
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "norm"
    LM_HEAD = "lm_head"
    ROUTER = "router"
    EXPERT = "expert"
    MTP_PROJECTION = "mtp_projection"
    MTP_BLOCK = "mtp_block"
    MTP_OUTPUT = "mtp_output"
    VISION = "vision"
    OTHER = "other"

    @property
    def is_mtp(self) -> bool:
        return self in {
            TensorRole.MTP_PROJECTION,
            TensorRole.MTP_BLOCK,
            TensorRole.MTP_OUTPUT,
        }


class QuantMethod(StrEnum):
    AFFINE = "affine"
    AWQ = "awq"
    DWQ = "dwq"
    GPTQ = "gptq"
    BF16 = "bf16"


class EvidenceKind(StrEnum):
    MEASURED = "measured"
    MEASURED_DEVELOPMENT = "measured_development"
    IMPORTED = "imported"
    ARCHITECTURE_PRIOR = "architecture_prior"

    @property
    def release_quality(self) -> bool:
        return self in {EvidenceKind.MEASURED, EvidenceKind.IMPORTED}


class ProfileName(StrEnum):
    GENERAL = "general"
    AGENT_CODING = "agent-coding"
    AGENT = "agent"
    CODING = "coding"
    TRANSLATION = "translation"
    REASONING = "reasoning"
    LONG_CONTEXT = "long-context"
    CJK = "cjk"
    RAG = "rag"
    OCR = "ocr"
    VLM = "vlm"


class RuntimeName(StrEnum):
    AX_ENGINE = "ax-engine"
    MLX_LM = "mlx-lm"


class RuntimeSupportLevel(StrEnum):
    OPTIMIZED = "optimized"
    STANDARD_INFERENCE = "standard-inference"


class BaselineKind(StrEnum):
    BF16_SOURCE = "bf16-source"
    UNIFORM_4BIT = "uniform-4bit"
    UNIFORM_6BIT = "uniform-6bit"
    MIXED_PRECISION = "mixed-precision"


class BenchmarkEvidenceKind(StrEnum):
    BF16 = "bf16"
    UNIFORM_4BIT = "uniform-4bit"
    UNIFORM_6BIT = "uniform-6bit"
    MIXED_PRECISION = "mixed-precision"
    AWQ = "awq"
    DWQ = "dwq"
    AXQUANT_MTP_OFF = "axquant-mtp-off"
    AXQUANT_MTP_ON = "axquant-mtp-on"


class ArchitectureSupportLevel(StrEnum):
    SUPPORTED = "supported"
    INVENTORY_ONLY = "inventory-only"
    UNSUPPORTED = "unsupported"


class SupportTier(StrEnum):
    """Evidence-backed permission tier for a model family (AXQ-017).

    ``support_level`` states what the adapter can do mechanically; the tier states
    what the recorded promotion evidence permits. Artifacts from versions that
    predate the tier field load as ``inspect-only`` and fail closed.
    """

    CERTIFIED = "certified"
    CONVERTIBLE = "convertible"
    INSPECT_ONLY = "inspect-only"


class OptimizationScope(StrEnum):
    TEXT_PATH = "text-path"
    FULL_MODEL = "full-model"
    INVENTORY_ONLY = "inventory-only"


class MtpSidecarLayout(StrEnum):
    BYTE_PRESERVED = "byte-preserved"
    AX_ENGINE_QWEN36_V1 = "ax-engine-qwen36-v1"
