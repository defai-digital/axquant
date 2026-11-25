"""Publication gate: real runtime smoke runs for generative MLX packs.

A static compatibility report does not prove that a pack loads. Before a
generative MLX pack is published, every runtime the owner targets must load it
and answer a short greedy prompt:

* MLX-LM, text, and AX Engine agreement with MLX-LM over a fixed prompt set:
  AX Engine's greedy tokens are teacher-forced through MLX-LM, and every token
  must be within a small log-probability deficit of MLX-LM's top choice. Exact
  token equality is not used: engines break exact ties differently, and one
  exact answer is brittle on a pack that sits on a near tie;
* MLX-VLM, text and image, when the pack ships vision weights;
* oMLX with MTP off and on, for Qwen 3.5-family packs with an MTP sidecar
  (the selected optional peer runtime).

The receipt binds the runtime payload digest, so any later change to weights,
config, index, tokenizer, or sidecars makes it stale. A pass is load and
generation evidence only, never quality, speed, or MTP acceptance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from axquant.artifact_paths import artifact_tree_files
from axquant.errors import ValidationGateError
from axquant.runtime_compatibility import QWEN35_MODEL_TYPES
from axquant.schema.runtime_export import PublicationSmokeCheckId, PublicationSmokeReceipt
from axquant.serde import file_sha256, stable_sha256

SMOKE_RECEIPT_FILENAME = "axquant_publication_smoke.json"
AX_ENGINE_AGREEMENT_CHECK: Final[PublicationSmokeCheckId] = "ax-engine:mlx-lm-agreement"
AGREEMENT_MAX_TOKENS = 16
AGREEMENT_MIN_RATIO = 0.9
AGREEMENT_PROMPTS = (
    "What is the capital of France? One word.",
    "Name the largest planet in our solar system.",
    "What is 12+30? Number only.",
    "Translate 'thank you' to French.",
    "What color is the sky on a clear day?",
    "Write the first three letters of the alphabet.",
    "What is 9*9? Number only.",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
    "Spell the word cat backwards.",
)
# Calibrated 2026-11-20 on four Qwen 3.5-family packs: healthy deficits were
# 0 or 0.125 nat (one BF16 logit step). A wrong runtime picks tokens nats away.
AGREEMENT_MAX_DEFICIT_NAT = 0.5
AGREEMENT_PROMPT_SET_SHA256 = stable_sha256(
    {
        "method": "teacher-forced-logprob-deficit",
        "prompts": list(AGREEMENT_PROMPTS),
        "max_tokens": AGREEMENT_MAX_TOKENS,
        "max_deficit_nat": AGREEMENT_MAX_DEFICIT_NAT,
    }
)
_GENERATIVE_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")
_VISION_PREFIXES = ("model.visual.", "visual.", "vision_tower.", "model.vision_tower.")
# Files that may change after the smoke without changing what runtimes load.
# AX Engine writes model-manifest.json on first load when it is absent; it is a
# regenerable cache derived from the bound files, not an input.
_UNBOUND_NAMES = frozenset({"README.md", ".gitattributes", "model-manifest.json"})

# oMLX classifies a directory holding either complete pair as a decision model
# and refuses chat requests (``omlx.model_discovery.decision_kind``). The MTP
# head belongs to the chat backbone, so the oMLX checks serve a private copy
# without the pair; the receipt must disclose that, and the gate enforces it.
DECISION_HEAD_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("joint_head.safetensors", "joint_head_config.json"),
    ("readout.safetensors", "decision_config.json"),
)
DECISION_HEAD_NOTE: Final = "decision head excluded so oMLX serves the chat backbone"


def decision_head_files(directory: str | Path) -> list[str]:
    """Return the files of every complete decision-head pair in ``directory``."""

    pack = Path(directory)
    return sorted(
        name
        for pair in DECISION_HEAD_PAIRS
        if all((pack / member).is_file() for member in pair)
        for name in pair
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationGateError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationGateError(f"{path.name} must be a JSON object")
    return value


def _model_types(config: dict[str, Any]) -> set[str]:
    types = {str(config.get("model_type") or "")}
    text = config.get("text_config")
    if isinstance(text, dict):
        types.add(str(text.get("model_type") or ""))
    types.discard("")
    return types


def smoke_gate_applies(directory: str | Path) -> bool:
    """True for MLX-quantized packs with a text-generation architecture."""

    config_path = Path(directory) / "config.json"
    if not config_path.is_file():
        return False
    config = _read_object(config_path)
    foreign = config.get("quantization_config")
    if isinstance(foreign, dict) and foreign.get("quant_method"):
        return False  # CUDA / non-MLX checkpoints do not load in the Apple runtimes
    if not isinstance(config.get("quantization"), dict):
        return False
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        return False
    return any(
        isinstance(name, str) and name.endswith(_GENERATIVE_SUFFIXES) for name in architectures
    )


def pack_has_vision(directory: str | Path) -> bool:
    pack = Path(directory)
    if (pack / "vision.safetensors").is_file():
        return True
    index_path = pack / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    weight_map = _read_object(index_path).get("weight_map")
    if not isinstance(weight_map, dict):
        return False
    return any(str(name).startswith(_VISION_PREFIXES) for name in weight_map)


def required_smoke_checks(directory: str | Path) -> list[PublicationSmokeCheckId]:
    pack = Path(directory)
    if not smoke_gate_applies(pack):
        return []
    required: list[PublicationSmokeCheckId] = [AX_ENGINE_AGREEMENT_CHECK, "mlx-lm:text"]
    if pack_has_vision(pack):
        required += ["mlx-vlm:text", "mlx-vlm:image"]
    config = _read_object(pack / "config.json")
    if (pack / "mtp.safetensors").is_file() and _model_types(config) & QWEN35_MODEL_TYPES:
        required += ["omlx:mtp-off", "omlx:mtp-on"]
    return required


def trim_at_stop(tokens: list[int], *, stop_ids: set[int], max_tokens: int) -> list[int]:
    """Generated tokens up to the first stop token, within the token budget."""

    kept: list[int] = []
    for token in tokens[:max_tokens]:
        if token in stop_ids:
            break
        kept.append(token)
    return kept


def agreed_prompts(deficits: list[float]) -> int:
    """Prompts whose worst per-token deficit stays within the tolerance."""

    return sum(1 for deficit in deficits if deficit <= AGREEMENT_MAX_DEFICIT_NAT)


def agreement_passes(agreed: int | None, total: int | None) -> bool:
    if agreed is None or total is None or total < len(AGREEMENT_PROMPTS) or agreed > total:
        return False
    return agreed / total >= AGREEMENT_MIN_RATIO


def runtime_payload_sha256(directory: str | Path) -> str:
    """Digest every file a runtime may read; AXQuant evidence files are excluded."""

    pack = Path(directory).resolve()
    try:
        members = artifact_tree_files(pack)
    except ValueError as exc:
        raise ValidationGateError(str(exc)) from exc
    entries = []
    for path in members:
        relative = path.relative_to(pack).as_posix()
        if relative in _UNBOUND_NAMES or path.name.startswith("axquant_"):
            continue
        # ``.cache/`` holds Hugging Face upload state, a local tool cache that no
        # runtime reads and the Hub never receives. Other hidden paths stay bound.
        if relative.startswith(".cache/"):
            continue
        entries.append({"path": relative, "sha256": file_sha256(path)})
    return stable_sha256(sorted(entries, key=lambda entry: entry["path"]))


def require_publication_smoke(directory: str | Path) -> PublicationSmokeReceipt | None:
    """Fail closed unless a current receipt passes every required runtime check."""

    pack = Path(directory)
    required = required_smoke_checks(pack)
    if not required:
        return None
    receipt_path = pack / SMOKE_RECEIPT_FILENAME
    if not receipt_path.is_file():
        raise ValidationGateError(
            f"publication requires {SMOKE_RECEIPT_FILENAME}: run "
            f"scripts/publication_smoke.py for {', '.join(required)}"
        )
    try:
        receipt = PublicationSmokeReceipt.model_validate(_read_object(receipt_path))
    except ValidationError as exc:
        raise ValidationGateError(f"invalid {SMOKE_RECEIPT_FILENAME}: {exc}") from exc
    if receipt.payload_sha256 != runtime_payload_sha256(pack):
        raise ValidationGateError(
            "publication smoke receipt is stale: the runtime payload changed after the smoke"
        )
    missing = [name for name in required if name not in receipt.required_checks]
    if missing:
        raise ValidationGateError(f"publication smoke receipt omits required checks: {missing}")
    results = {check.check: check for check in receipt.checks}
    failed = [name for name in required if name not in results or results[name].status != "pass"]
    agreement = results.get(AX_ENGINE_AGREEMENT_CHECK)
    deficits = agreement.logprob_deficits if agreement is not None else None
    # Recompute from the recorded counts; a "pass" label alone is not trusted.
    if (
        AX_ENGINE_AGREEMENT_CHECK in required
        and AX_ENGINE_AGREEMENT_CHECK not in failed
        and agreement is not None
        and (
            agreement.prompt_set_sha256 != AGREEMENT_PROMPT_SET_SHA256
            or deficits is None
            or len(deficits) != agreement.total
            or agreed_prompts(deficits) != agreement.agreed
            or not agreement_passes(agreement.agreed, agreement.total)
        )
    ):
        failed.append(AX_ENGINE_AGREEMENT_CHECK)
    excluded = decision_head_files(pack)
    if excluded:
        for name in required:
            check = results.get(name)
            if (
                name.startswith("omlx:")
                and name not in failed
                and (
                    check is None
                    or DECISION_HEAD_NOTE not in check.detail
                    or not all(f"'{member}'" in check.detail for member in excluded)
                )
            ):
                failed.append(name)
    if failed:
        raise ValidationGateError(f"publication smoke checks did not pass: {failed}")
    return receipt
