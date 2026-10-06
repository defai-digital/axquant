#!/usr/bin/env python3
"""Serial MTP off/on and vision generation checks against a loopback peer CLI.

Evidence only: this script never publishes, rewrites a pack, or grants a
certificate. Each server has isolated state, a deadline, and owned cleanup.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axquant.factory import require_factory_host  # noqa: E402
from axquant.runtime_compatibility import (  # noqa: E402
    MTPLX_QWEN4_BASE_ENV,
    inspect_runtime_export,
)
from axquant.serde import file_sha256, write_data  # noqa: E402

_PROFILE_RUNNER = """
import json, os, runpy, sys, threading
counts = {}
lock = threading.Lock()
def witness(frame, event, arg):
    names = {'mtp_forward', 'mtp_draft_logits', 'mtp_update_cache'}
    if event == 'call' and frame.f_code.co_name in names:
        module = frame.f_globals.get('__name__', '')
        key = module + ':' + frame.f_code.co_name
        with lock:
            counts[key] = counts.get(key, 0) + 1
            path = os.environ['AXQUANT_MTP_WITNESS']
            with open(path + '.tmp', 'w') as handle:
                json.dump(counts, handle)
            os.replace(path + '.tmp', path)
sys.setprofile(witness)
threading.setprofile(witness)
executable = sys.argv.pop(1)
sys.argv[0] = executable
runpy.run_path(executable, run_name='__main__')
"""


def _request(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _image_content() -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((100, 80), "7", fill="black", font_size=96)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode()
    return [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
        {"type": "text", "text": "Read the single digit in the image. Answer briefly."},
    ]


def run_case(args: argparse.Namespace, enabled: bool) -> dict[str, Any]:
    work = args.output.parent / (args.output.stem + ("-mtp-on" if enabled else "-mtp-off"))
    work.mkdir(parents=True, exist_ok=False)
    state = work / "state"
    state.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    model = args.directory.name
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if args.runtime == "omlx":
        env["OMLX_QWEN4_PLE_MODE"] = args.ple_mode
        models = work / "models"
        models.mkdir()
        (models / model).symlink_to(args.directory, target_is_directory=True)
        write_data(
            state / "model_settings.json",
            {
                "version": 1,
                "models": {
                    model: {
                        "mtp_enabled": enabled,
                        "mtp_num_draft_tokens": 1,
                        "qwen4_ple_ssd_offload": args.ple_mode == "mmap",
                    }
                },
            },
        )
        env["OMLX_BASE_PATH"] = str(state)
        command = [
            args.executable,
            "serve",
            "--model-dir",
            str(models),
            "--base-path",
            str(state),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-cache",
            "--no-hf-cache",
        ]
        env["AXQUANT_MTP_WITNESS"] = str(work / "mtp-witness.json")
        python = Path(args.executable).resolve().parent / "python"
        command = [str(python), "-c", _PROFILE_RUNNER, *command]
    else:
        # Keep the table on SSD and disable warmup/pre-read for this load smoke.
        env.update(MTPLX_QWEN4_BASE_ENV)
        command = [
            args.executable,
            "serve",
            "--model",
            str(args.directory),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-auth",
            "--profile",
            "stable",
            "--generation-mode",
            "mtp" if enabled else "ar",
            "--load-mtp" if enabled else "--no-load-mtp",
            "--depth",
            "1",
            "--ngram-prewarm",
            "off",
            "--warmup-tokens",
            "0",
            "--ssd-session-cache",
            "off",
            "--ssd-session-cache-dir",
            str(state / "ssd-session-cache"),
        ]
    result: dict[str, Any] = {
        "mtp_enabled": enabled,
        "command": command,
        "passed": False,
        "requests": [],
        "runtime_verified": False,
        "environment": {
            key: env[key]
            for key in (
                "OMLX_QWEN4_PLE_MODE",
                "MTPLX_NGRAM_RESIDENT",
                "MTPLX_FUSED_GATE_UP",
                "MTPLX_FUSED_GDN_OUT",
                "MTPLX_FUSED_GDN_INPROJ",
                "MTPLX_QWEN4_RELAXED_DRAFT_TIES",
                "MTPLX_QWEN4_COMPILED_MTP_PREPARE",
                "MTPLX_QWEN4_FIXED_M4_VERIFY",
                "MTPLX_QWEN4_M4_STAGE3",
                "MTPLX_QWEN4_ROUTE_KERNEL",
                "MTPLX_QWEN4_OPDIET",
                "MTPLX_QWEN4_DRAFT_K20_PRESCATTER",
                "MTPLX_QWEN4_BLOCK_VERIFY",
                "MTPLX_QWEN4_VERIFY_GLUE",
                "MTPLX_QWEN4_BATCHED_TARGET_DISTRIBUTIONS",
            )
            if key in env
        },
    }
    started = time.monotonic()
    with (work / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            url = f"http://127.0.0.1:{port}/v1"
            deadline = started + args.timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited with code {process.returncode}")
                try:
                    models_reply = _request(url + "/models", None, 3)
                    if any(item.get("id") == model for item in models_reply.get("data", [])):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("server readiness deadline")
            prompts: list[tuple[str, Any]] = [("text", "Calculate 17 + 25. Answer briefly.")]
            if args.vision:
                prompts.append(("vision", _image_content()))
            for kind, prompt in prompts:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64,
                    "temperature": 0,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                try:
                    reply = _request(
                        url + "/chat/completions", payload, max(1, deadline - time.monotonic())
                    )
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode(errors="replace")
                    if (
                        args.runtime == "mtplx"
                        and not enabled
                        and kind == "vision"
                        and exc.code == 400
                        and "image content requires MTP generation mode" in body
                    ):
                        result.setdefault("unsupported_requests", []).append(
                            {
                                "kind": kind,
                                "status": exc.code,
                                "reason": "vision requires MTP mode",
                            }
                        )
                        continue
                    raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
                if not reply.get("choices"):
                    raise RuntimeError("generation response has no choices")
                message = reply["choices"][0].get("message", {})
                if not message.get("content") and not message.get("reasoning_content"):
                    raise RuntimeError("generation returned no text")
                result["requests"].append({"kind": kind, "response": reply})
                print(
                    f"{args.runtime} MTP {'on' if enabled else 'off'} {kind}: generated", flush=True
                )
            if args.runtime == "mtplx":
                health = _request(url.removesuffix("/v1") + "/health", None, 10)
                result["engagement"] = {
                    key: health.get(key)
                    for key in (
                        "mtp_enabled",
                        "load_mtp",
                        "generation_mode",
                        "depth",
                        "vision",
                    )
                }
                if health.get("mtp_enabled") is not enabled:
                    raise RuntimeError("runtime MTP engagement differs from the requested mode")
            else:
                witness_path = work / "mtp-witness.json"
                counts = json.loads(witness_path.read_text()) if witness_path.is_file() else {}
                engaged = any(count > 0 for count in counts.values())
                result["engagement"] = {"head_calls": counts, "mtp_engaged": engaged}
                if engaged is not enabled:
                    raise RuntimeError("native MTP execution differs from the requested mode")
            result["passed"] = True
            result["runtime_verified"] = True
        except urllib.error.HTTPError as exc:
            result["error"] = f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            result["elapsed_seconds"] = time.monotonic() - started
            result["server_exit_code"] = process.returncode
            write_data(work / "receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--runtime", choices=("omlx", "mtplx"), required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--ple-mode", choices=("auto", "resident", "mmap"), default="mmap")
    args = parser.parse_args()
    require_factory_host(socket.gethostname())
    args.directory = args.directory.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("output already exists")
    verdict = inspect_runtime_export(args.directory)["targets"][args.runtime]
    if verdict["status"] != "static-compatible":
        parser.error(
            f"runtime export is not ready: {verdict['blockers'] + verdict['required_actions']}"
        )
    bindings = {
        name: file_sha256(args.directory / name)
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "mtplx_runtime.json",
            "axquant_runtime_export.json",
        )
        if (args.directory / name).is_file()
    }
    result = {
        "schema_version": "axquant.private-runtime-smoke.v1",
        "runtime": args.runtime,
        "input_bindings": bindings,
        "quality_certified": False,
        "host": socket.gethostname(),
        "cases": [],
    }
    for enabled in (False, True):
        case = run_case(args, enabled)
        result["cases"].append(case)
        write_data(args.output, result)
        if not case["passed"]:
            print(case.get("error"), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
