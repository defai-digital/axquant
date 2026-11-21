"""Publication gate: real runtime smoke receipts bound to the runtime payload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axquant.errors import ValidationGateError
from axquant.publication_smoke import (
    AGREEMENT_PROMPT_SET_SHA256,
    AX_ENGINE_AGREEMENT_CHECK,
    SMOKE_RECEIPT_FILENAME,
    agreed_prompts,
    agreement_passes,
    require_publication_smoke,
    required_smoke_checks,
    runtime_payload_sha256,
    trim_at_stop,
)
from axquant.schema.runtime_export import PublicationSmokeCheck, PublicationSmokeReceipt
from axquant.serde import write_data


def _pack(
    directory: Path,
    *,
    model_type: str = "qwen3_5",
    architectures: list[str] | None = None,
    quantization: dict[str, object] | None = None,
    vision: bool = False,
    mtp: bool = False,
    extra: dict[str, object] | None = None,
) -> Path:
    directory.mkdir()
    config: dict[str, object] = {
        "model_type": model_type,
        "architectures": architectures or ["Qwen3_5ForConditionalGeneration"],
        "quantization": quantization if quantization is not None else {"bits": 4},
        **(extra or {}),
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"trunk")
    if vision:
        (directory / "vision.safetensors").write_bytes(b"vision")
    if mtp:
        (directory / "mtp.safetensors").write_bytes(b"mtp")
    return directory


def _receipt(
    directory: Path,
    *,
    failing: tuple[str, ...] = (),
    deficits: list[float] | None = None,
    agreed: int | None = None,
    prompt_set: str = AGREEMENT_PROMPT_SET_SHA256,
) -> None:
    deficits = [0.0] * 10 if deficits is None else deficits
    required = required_smoke_checks(directory)

    def check(name: str) -> PublicationSmokeCheck:
        status = "fail" if name in failing else "pass"
        if name == AX_ENGINE_AGREEMENT_CHECK:
            return PublicationSmokeCheck(
                check=name,
                status=status,
                expected=">= 90% of 10",
                agreed=agreed_prompts(deficits) if agreed is None else agreed,
                total=10,
                logprob_deficits=deficits,
                prompt_set_sha256=prompt_set,
            )
        return PublicationSmokeCheck(check=name, status=status, expected="42", output="42")

    receipt = PublicationSmokeReceipt(
        payload_sha256=runtime_payload_sha256(directory),
        required_checks=required,
        checks=[check(name) for name in required],
        hardware="Apple M2 Ultra",
        created_at="2026-11-20T00:00:00+00:00",
    )
    write_data(directory / SMOKE_RECEIPT_FILENAME, receipt)


def test_required_checks_follow_vision_and_qwen_mtp(tmp_path: Path) -> None:
    assert required_smoke_checks(_pack(tmp_path / "text")) == [
        "ax-engine:mlx-lm-agreement",
        "mlx-lm:text",
    ]
    assert required_smoke_checks(_pack(tmp_path / "vl", vision=True, mtp=True)) == [
        "ax-engine:mlx-lm-agreement",
        "mlx-lm:text",
        "mlx-vlm:text",
        "mlx-vlm:image",
        "omlx:mtp-off",
        "omlx:mtp-on",
    ]
    # oMLX is the selected optional peer only for the Qwen 3.5-family MTP profile.
    nemotron = _pack(
        tmp_path / "nemotron",
        model_type="nemotron_h",
        architectures=["NemotronHForCausalLM"],
        mtp=True,
    )
    assert required_smoke_checks(nemotron) == ["ax-engine:mlx-lm-agreement", "mlx-lm:text"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"architectures": ["Gemma3TextModel"]},  # embedding pack
        {"quantization": {}, "extra": {"quantization_config": {"quant_method": "compressed"}}},
    ],
)
def test_gate_does_not_apply_to_embedding_or_cuda_packs(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    pack = _pack(tmp_path / "pack", **kwargs)  # type: ignore[arg-type]
    assert required_smoke_checks(pack) == []
    assert require_publication_smoke(pack) is None


def test_gate_requires_a_current_passing_receipt(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack", vision=True)
    with pytest.raises(ValidationGateError, match=r"requires axquant_publication_smoke\.json"):
        require_publication_smoke(pack)

    _receipt(pack, failing=("mlx-vlm:image",))
    with pytest.raises(ValidationGateError, match=r"did not pass: \['mlx-vlm:image'\]"):
        require_publication_smoke(pack)

    _receipt(pack)
    assert require_publication_smoke(pack) is not None
    # README and AXQuant evidence files are not runtime payload.
    (pack / "README.md").write_text("card", encoding="utf-8")
    (pack / "axquant_manifest.json").write_text("{}", encoding="utf-8")
    assert require_publication_smoke(pack) is not None

    (pack / "vision.safetensors").write_bytes(b"changed")
    with pytest.raises(ValidationGateError, match="stale"):
        require_publication_smoke(pack)


def test_gate_rejects_a_receipt_that_omits_a_required_check(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    _receipt(pack)
    (pack / "mtp.safetensors").write_bytes(b"mtp")  # now requires oMLX checks
    receipt = json.loads((pack / SMOKE_RECEIPT_FILENAME).read_text(encoding="utf-8"))
    receipt["payload_sha256"] = runtime_payload_sha256(pack)
    (pack / SMOKE_RECEIPT_FILENAME).write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValidationGateError, match="omits required checks"):
        require_publication_smoke(pack)


def test_agreement_tolerates_ties_but_not_wrong_tokens() -> None:
    assert trim_at_stop([1, 2, 99, 5], stop_ids={99}, max_tokens=16) == [1, 2]
    assert trim_at_stop([1, 2, 3], stop_ids={99}, max_tokens=2) == [1, 2]
    # Exact ties (0) and one BF16 logit step (0.125) agree; a 3-nat pick does not.
    assert agreed_prompts([0.0, 0.125, 0.5, 3.0]) == 3
    assert agreement_passes(9, 10) and agreement_passes(10, 10)
    assert not agreement_passes(8, 10)
    assert not agreement_passes(5, 5)  # fewer prompts than the fixed set
    assert not agreement_passes(None, 10)


@pytest.mark.parametrize(
    ("deficits", "agreed", "prompt_set"),
    [
        ([0.0] * 8 + [2.0, 3.0], None, AGREEMENT_PROMPT_SET_SHA256),  # 8/10
        ([0.0] * 8 + [2.0, 3.0], 10, AGREEMENT_PROMPT_SET_SHA256),  # inflated count
        ([0.0] * 10, None, "0" * 64),  # foreign prompt set
    ],
)
def test_gate_recomputes_agreement_instead_of_trusting_the_label(
    tmp_path: Path, deficits: list[float], agreed: int | None, prompt_set: str
) -> None:
    pack = _pack(tmp_path / "pack")
    _receipt(pack, deficits=deficits, agreed=agreed, prompt_set=prompt_set)  # "pass"
    with pytest.raises(ValidationGateError, match="ax-engine:mlx-lm-agreement"):
        require_publication_smoke(pack)
    _receipt(pack, deficits=[0.0] * 9 + [4.0])
    assert require_publication_smoke(pack) is not None
