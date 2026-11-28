#!/usr/bin/env python3
"""Run the publication runtime smoke checks and write the bound receipt.

Each check loads the pack in a separate process and answers one short greedy
prompt. The receipt (``axquant_publication_smoke.json``) binds the runtime
payload digest; ``publish`` refuses a pack whose receipt is missing, stale, or
has a failing required check. Evidence of load and generation only: never
quality, speed, or MTP acceptance.

oMLX checks import the MTP sidecar into a private copy of the pack (the
documented one-step ``import_mtplx_sidecar`` path) and serve it with MTP off
and on. Decision-head files (Clef joint head, Decision 3.0 readout) are left
out of that copy: oMLX would otherwise serve the pack only on its decision
endpoint, while the MTP head belongs to the chat backbone. The source pack is
never modified except for the receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from axquant.publication_smoke import (  # noqa: E402
    AGREEMENT_MAX_DEFICIT_NAT,
    AGREEMENT_MAX_TOKENS,
    AGREEMENT_MIN_RATIO,
    AGREEMENT_PROMPT_SET_SHA256,
    AGREEMENT_PROMPTS,
    AX_ENGINE_AGREEMENT_CHECK,
    AX_ENGINE_VLM_AGREEMENT_CHECK,
    DECISION_HEAD_NOTE,
    SMOKE_RECEIPT_FILENAME,
    agreed_prompts,
    agreement_passes,
    decision_head_files,
    required_smoke_checks,
    runtime_payload_sha256,
    trim_at_stop,
)
from axquant.schema.runtime_export import (  # noqa: E402
    PublicationSmokeCheck,
    PublicationSmokeCheckId,
    PublicationSmokeReceipt,
)
from axquant.serde import write_data  # noqa: E402

TEXT_PROMPT = "What is 6*7? Answer with the number only."
TEXT_EXPECTED = "42"
IMAGE_PROMPT = "What digit is in the image? Answer with the digit only."
IMAGE_EXPECTED = "7"

_MLX_LM = """
import json, sys
from mlx_lm import generate, load
model, tokenizer = load(sys.argv[1])
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": sys.argv[2]}],
    add_generation_prompt=True, tokenize=False, enable_thinking=False,
)
print("RESULT " + json.dumps(generate(model, tokenizer, prompt=prompt, max_tokens=16)))
"""

_MLX_VLM = """
import json, sys
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
path, question, image = sys.argv[1], sys.argv[2], sys.argv[3] or None
model, processor = load(path)
images = [image] if image else None
prompt = apply_chat_template(
    processor, load_config(path), question, num_images=1 if image else 0,
    enable_thinking=False,
)
result = generate(model, processor, prompt, images, max_tokens=16, verbose=False)
print("RESULT " + json.dumps(getattr(result, "text", result)))
"""

_MLX_LM_TOKENIZE = """
import json, sys
from mlx_lm import load
model, tokenizer = load(sys.argv[1])
rows = []
for question in json.loads(sys.argv[2]):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    rows.append(list(tokenizer.encode(text, add_special_tokens=False)))
stop = set(getattr(tokenizer, "eos_token_ids", None) or [])
stop |= set(getattr(tokenizer, "all_special_ids", None) or [])
if tokenizer.eos_token_id is not None:
    stop.add(tokenizer.eos_token_id)
print("RESULT " + json.dumps({"ids": rows, "stop_ids": sorted(int(t) for t in stop)}))
"""

# Teacher-force each candidate continuation through MLX-LM in one forward pass
# and report the worst gap between MLX-LM's top log-probability and the
# candidate token's log-probability.
_MLX_LM_SCORE = """
import json, sys
import mlx.core as mx
from mlx_lm import load
model, _ = load(sys.argv[1])
deficits = []
for row in json.load(open(sys.argv[2])):
    ids, cont = row["ids"], row["tokens"]
    if not cont:
        deficits.append(0.0)
        continue
    logits = model(mx.array([ids + cont]))[0].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    start = len(ids) - 1
    worst = 0.0
    for offset, token in enumerate(cont):
        step = logprobs[start + offset]
        worst = max(worst, (step.max() - step[token]).item())
    deficits.append(worst)
print("RESULT " + json.dumps(deficits))
"""

# MLX-VLM reference for model types MLX-LM cannot load. Tokenizing loads only
# the pack's tokenizer (the one MLX-VLM's processor wraps), so AX Engine never
# shares memory with a resident MLX-VLM copy. mlx_vlm.utils.load_processor alone
# cannot resolve a custom image processor such as qwen4_exp's.
_MLX_VLM_TOKENIZE = """
import json, sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
rows = []
for question in json.loads(sys.argv[2]):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    rows.append(list(tokenizer.encode(text, add_special_tokens=False)))
stop = set(getattr(tokenizer, "all_special_ids", None) or [])
if tokenizer.eos_token_id is not None:
    stop.add(tokenizer.eos_token_id)
print("RESULT " + json.dumps({"ids": rows, "stop_ids": sorted(int(t) for t in stop)}))
"""

_MLX_VLM_SCORE = """
import json, sys
import mlx.core as mx
from mlx_vlm import load
model, _ = load(sys.argv[1])
deficits = []
for row in json.load(open(sys.argv[2])):
    ids, cont = row["ids"], row["tokens"]
    if not cont:
        deficits.append(0.0)
        continue
    result = model.language_model(mx.array([ids + cont]))
    logits = getattr(result, "logits", result)[0].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    start = len(ids) - 1
    worst = 0.0
    for offset, token in enumerate(cont):
        step = logprobs[start + offset]
        worst = max(worst, (step.max() - step[token]).item())
    deficits.append(worst)
print("RESULT " + json.dumps(deficits))
"""

_OMLX_IMPORT = """
import json, sys
from omlx.oq import import_mtplx_sidecar
print("RESULT " + json.dumps(import_mtplx_sidecar(sys.argv[1]), default=str))
"""


def _env() -> dict[str, str]:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    # Pre-M5 GPUs can end long prefill command buffers with a Metal context-store
    # timeout; the flag relaxes preemption scheduling only.
    env.setdefault("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1")
    return env


def _python_result(python: str, code: str, *args: str, timeout: int) -> Any:
    completed = subprocess.run(
        [python, "-I", "-c", code, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_env(),
        check=False,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line.removeprefix("RESULT "))
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    raise RuntimeError(detail[-1] if detail else f"exit code {completed.returncode}")


def _package_version(python: str, package: str) -> str | None:
    code = (
        "import importlib.metadata as m, json\n"
        f"print('RESULT ' + json.dumps(m.version({package!r})))"
    )
    try:
        value = _python_result(python, code, timeout=60)
    except (RuntimeError, subprocess.TimeoutExpired):
        return None
    return str(value)


def _check(
    name: str,
    expected: str,
    run: Any,
    *,
    version: str | None,
) -> PublicationSmokeCheck:
    started = time.monotonic()
    output, detail = "", ""
    try:
        output = str(run()).strip()
    except Exception as exc:  # recorded as a failing check, never swallowed
        detail = f"{type(exc).__name__}: {exc}"[:2000]
    passed = not detail and expected in output.split("</think>")[-1]
    if not detail and not passed:
        detail = "answer does not contain the expected text"
    print(f"{name}: {'pass' if passed else 'FAIL'} {output!r} {detail}", flush=True)
    return PublicationSmokeCheck(
        check=name,  # type: ignore[arg-type]
        status="pass" if passed else "fail",
        runtime_version=version,
        expected=expected,
        output=output[:500],
        detail=detail,
        seconds=round(time.monotonic() - started, 3),
    )


def _ax_engine_generate(args: argparse.Namespace, ids: list[int]) -> list[int]:
    completed = subprocess.run(
        [
            args.ax_engine_bench,
            "generate",
            "--mlx",
            "--mlx-model-artifacts-dir",
            str(args.directory),
            "--tokens",
            ",".join(str(token) for token in ids),
            "--max-output-tokens",
            str(AGREEMENT_MAX_TOKENS),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=args.timeout,
        env=_env(),
        check=False,
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip()[-1000:])
    reply = json.loads(completed.stdout[completed.stdout.find("{") :])
    backend = reply.get("runtime", {}).get("selected_backend")
    if backend not in (None, "mlx"):
        raise RuntimeError(f"AX Engine selected backend {backend!r}, not MLX")
    if not isinstance(reply.get("output_tokens"), list):
        raise RuntimeError("AX Engine reply has no output_tokens list")
    return [int(token) for token in reply["output_tokens"]]


# Recorded for a prompt whose AX Engine continuation was empty (far above the
# agreement tolerance, so the gate's recount treats it as disagreement).
EMPTY_CONTINUATION_DEFICIT = 99.0


def _ax_engine_agreement(
    args: argparse.Namespace,
    version: str | None,
    name: PublicationSmokeCheckId = AX_ENGINE_AGREEMENT_CHECK,
) -> PublicationSmokeCheck:
    """AX Engine's greedy tokens must be near-top choices of the reference runtime."""

    if name == AX_ENGINE_VLM_AGREEMENT_CHECK:
        reference_name, tokenize, score = "MLX-VLM", _MLX_VLM_TOKENIZE, _MLX_VLM_SCORE
    else:
        reference_name, tokenize, score = "MLX-LM", _MLX_LM_TOKENIZE, _MLX_LM_SCORE
    started = time.monotonic()
    total = len(AGREEMENT_PROMPTS)
    detail, deficits = "", []
    try:
        reference = _python_result(
            args.python,
            tokenize,
            str(args.directory),
            json.dumps(list(AGREEMENT_PROMPTS)),
            timeout=args.timeout,
        )
        stop_ids = set(reference["stop_ids"])
        rows = [
            {
                "ids": ids,
                "tokens": trim_at_stop(
                    _ax_engine_generate(args, ids),
                    stop_ids=stop_ids,
                    max_tokens=AGREEMENT_MAX_TOKENS,
                ),
            }
            for ids in reference["ids"]
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(rows, handle)
        try:
            deficits = [
                round(float(value), 4)
                for value in _python_result(
                    args.python,
                    score,
                    str(args.directory),
                    handle.name,
                    timeout=args.timeout,
                )
            ]
        finally:
            os.unlink(handle.name)
        # An empty continuation scores a zero deficit trivially; it is not
        # agreement. Record it as a disagreeing prompt.
        empty = [index for index, row in enumerate(rows) if not row["tokens"]]
        for index in empty:
            deficits[index] = EMPTY_CONTINUATION_DEFICIT
        if empty:
            detail_note = f"AX Engine produced no tokens before a stop for prompts {empty}"
            print(detail_note, flush=True)
    except Exception as exc:  # recorded as a failing check, never swallowed
        detail = f"{type(exc).__name__}: {exc}"[:2000]
    agreed = agreed_prompts(deficits) if len(deficits) == total else 0
    passed = not detail and agreement_passes(agreed, total)
    if not detail and agreed < total:
        outliers = [i for i, d in enumerate(deficits) if d > AGREEMENT_MAX_DEFICIT_NAT]
        detail = f"prompts beyond {AGREEMENT_MAX_DEFICIT_NAT} nat: {outliers}"
    output = f"{agreed}/{total} prompts within {AGREEMENT_MAX_DEFICIT_NAT} nat of {reference_name}"
    print(f"{name}: {'pass' if passed else 'FAIL'} {output} {deficits}")
    return PublicationSmokeCheck(
        check=name,
        status="pass" if passed else "fail",
        runtime_version=version,
        expected=(
            f">= {AGREEMENT_MIN_RATIO:.0%} of {total} prompts with every token within "
            f"{AGREEMENT_MAX_DEFICIT_NAT} nat of {reference_name}'s top choice"
        ),
        output=output,
        detail=detail,
        seconds=round(time.monotonic() - started, 3),
        agreed=agreed,
        total=total,
        logprob_deficits=deficits or None,
        prompt_set_sha256=AGREEMENT_PROMPT_SET_SHA256,
    )


def _ax_engine_version(bench: str) -> str | None:
    """The bench CLI has no --version flag; its usage banner names the version."""

    completed = subprocess.run([bench], capture_output=True, text=True, check=False)
    match = re.search(r"AX Engine v(\S+)", completed.stdout + completed.stderr)
    return match.group(1) if match else None


def _hardware() -> str:
    completed = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() or platform.machine()


def _synthetic_digit(path: Path) -> Path:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(canvas).text((90, 60), IMAGE_EXPECTED, fill="black", font_size=120)
    canvas.save(path)
    return path


def _private_import_copy(source: Path, destination: Path, excluded: list[str]) -> Path:
    """Symlink weights, copy the small files the oMLX import rewrites."""

    destination.mkdir()
    for path in source.iterdir():
        target = destination / path.name
        if path.name in excluded:
            continue
        if path.is_dir() and path.name in {"__pycache__", ".cache"}:
            continue
        if path.is_dir():
            raise RuntimeError(f"oMLX private copy does not support subdirectories: {path.name}")
        if path.suffix == ".safetensors" and path.name != "mtp.safetensors":
            target.symlink_to(path)
        else:
            shutil.copy2(path, target)
    return destination


def _omlx_checks(args: argparse.Namespace, work: Path) -> list[PublicationSmokeCheck]:
    from smoke_runtime_export import run_case

    version = _package_version(args.omlx_python, "omlx")
    excluded = decision_head_files(args.directory)
    copy = _private_import_copy(args.directory, work / args.directory.name, excluded)
    try:
        _python_result(args.omlx_python, _OMLX_IMPORT, str(copy), timeout=900)
    except Exception as exc:
        detail = f"oMLX sidecar import failed: {exc}"[:2000]
        return [
            PublicationSmokeCheck(
                check=name,
                status="fail",
                runtime_version=version,
                expected=TEXT_EXPECTED,
                detail=detail,
            )
            for name in ("omlx:mtp-off", "omlx:mtp-on")
        ]
    checks = []
    executable = str(Path(args.omlx_python).parent / "omlx")
    for enabled in (False, True):
        name = "omlx:mtp-on" if enabled else "omlx:mtp-off"
        case_args = SimpleNamespace(
            directory=copy,
            output=work / f"{name.replace(':', '-')}.json",
            runtime="omlx",
            executable=executable,
            timeout=args.timeout,
            vision=False,
            ple_mode="auto",
            # Same prompt as the MLX-LM and MLX-VLM checks, so a wrong answer
            # separates a runtime fault from the model's own answer.
            text_prompt=TEXT_PROMPT,
        )

        def run(case_args: SimpleNamespace = case_args, enabled: bool = enabled) -> str:
            case = run_case(case_args, enabled)  # type: ignore[arg-type]
            if not case["passed"]:
                raise RuntimeError(case.get("error") or "oMLX case failed")
            message = case["requests"][0]["response"]["choices"][0]["message"]
            return str(message.get("content") or message.get("reasoning_content") or "")

        check = _check(name, TEXT_EXPECTED, run, version=version)
        if excluded:
            note = f"{DECISION_HEAD_NOTE}: {excluded}"
            check = check.model_copy(update={"detail": (f"{check.detail} {note}").strip()})
        checks.append(check)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable, help="Python with mlx-lm/mlx-vlm")
    parser.add_argument("--ax-engine-bench", default="ax-engine-bench")
    parser.add_argument("--omlx-python", help="Python of an oMLX install (required for MTP packs)")
    parser.add_argument("--image", type=Path, help="Image whose answer is the digit 7")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true", help="Print the required checks and exit")
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    required = required_smoke_checks(args.directory)
    print("required checks:", ", ".join(required) or "none (gate not applicable)")
    if args.dry_run or not required:
        return 0
    if any(name.startswith("omlx:") for name in required) and not args.omlx_python:
        parser.error("this pack requires oMLX checks: pass --omlx-python")
    payload = runtime_payload_sha256(args.directory)
    directory = str(args.directory)
    checks: list[PublicationSmokeCheck] = []
    with tempfile.TemporaryDirectory(prefix="axquant-publication-smoke-") as temporary:
        work = Path(temporary)
        image = args.image.resolve() if args.image else _synthetic_digit(work / "digit.png")
        mlx_lm_version = _package_version(args.python, "mlx-lm")
        mlx_vlm_version = _package_version(args.python, "mlx-vlm")
        engine = _ax_engine_version(args.ax_engine_bench)
        for name in required:
            if name in (AX_ENGINE_AGREEMENT_CHECK, AX_ENGINE_VLM_AGREEMENT_CHECK):
                checks.append(_ax_engine_agreement(args, engine, name))
            elif name == "mlx-lm:text":
                checks.append(
                    _check(
                        name,
                        TEXT_EXPECTED,
                        lambda: _python_result(
                            args.python, _MLX_LM, directory, TEXT_PROMPT, timeout=args.timeout
                        ),
                        version=mlx_lm_version,
                    )
                )
            elif name in {"mlx-vlm:text", "mlx-vlm:image"}:
                with_image = name == "mlx-vlm:image"
                checks.append(
                    _check(
                        name,
                        IMAGE_EXPECTED if with_image else TEXT_EXPECTED,
                        lambda with_image=with_image: _python_result(
                            args.python,
                            _MLX_VLM,
                            directory,
                            IMAGE_PROMPT if with_image else TEXT_PROMPT,
                            str(image) if with_image else "",
                            timeout=args.timeout,
                        ),
                        version=mlx_vlm_version,
                    )
                )
        if any(name.startswith("omlx:") for name in required):
            checks.extend(_omlx_checks(args, work))
    if runtime_payload_sha256(args.directory) != payload:
        raise SystemExit("runtime payload changed during the smoke; receipt not written")
    receipt = PublicationSmokeReceipt(
        payload_sha256=payload,
        required_checks=required,
        checks=checks,
        hardware=_hardware(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    write_data(args.directory / SMOKE_RECEIPT_FILENAME, receipt)
    failed = [check.check for check in checks if check.status != "pass"]
    print("receipt:", SMOKE_RECEIPT_FILENAME, "failed:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
