#!/usr/bin/env python3
"""Run an isolated oMLX VLM-MTP compatibility smoke for a Gemma composite."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axquant.gemma4_assistant_compose import validate_gemma4_assistant_composite
from axquant.serde import write_data

_STATS = re.compile(
    r"vlm_mtp stats: .*?rounds=(?P<rounds>\d+) "
    r"accepted=(?P<accepted>\d+)/(?P<proposed>\d+)"
)


def _request_json(
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("oMLX response must be a JSON object")
        return response.status, parsed


def _wait_for_health(port: int, *, api_key: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, _ = _request_json(
                f"http://127.0.0.1:{port}/health",
                api_key=api_key,
                timeout=2.0,
            )
            if status == 200:
                return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise TimeoutError(f"oMLX health check did not become ready: {type(last_error).__name__}")


def _parse_stats(log_text: str) -> dict[str, int] | None:
    matches = list(_STATS.finditer(log_text))
    if not matches:
        return None
    match = matches[-1]
    return {key: int(value) for key, value in match.groupdict().items()}


def _runtime_versions(omlx: Path) -> dict[str, str | None]:
    result: dict[str, str | None] = {"omlx": None, "mlx": None, "mlx_vlm": None}
    version = subprocess.run(
        [str(omlx), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if version.returncode == 0:
        result["omlx"] = version.stdout.strip().rsplit(" ", 1)[-1]
    python = omlx.parent / "python"
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as m, json, mlx.core as mx; "
                "print(json.dumps({'mlx': mx.__version__, "
                "'mlx_vlm': m.version('mlx-vlm')}))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode == 0:
        values = json.loads(probe.stdout)
        if isinstance(values, dict):
            result["mlx"] = str(values.get("mlx"))
            result["mlx_vlm"] = str(values.get("mlx_vlm"))
    return result


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("oMLX response has no completion choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("oMLX response has no text content")
    return str(message["content"]).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--omlx", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18089)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--expected-text", default="OMLX_GEMMA_MTP_OK")
    args = parser.parse_args(argv)

    artifact = args.artifact.expanduser().resolve()
    omlx = args.omlx.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    if not omlx.is_file():
        parser.error("--omlx must name an executable file")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.model_id):
        parser.error("--model-id must be a Hub-safe repository leaf")
    validate_gemma4_assistant_composite(artifact)
    versions = _runtime_versions(omlx)
    if versions["mlx"] != "0.32.2":
        parser.error(f"oMLX environment must use MLX 0.32.2, got {versions['mlx']!r}")

    work_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="gemma4-omlx-", dir=work_root))
    models = temporary / "models"
    base = temporary / "base"
    models.mkdir()
    base.mkdir()
    assistant_id = f"{args.model_id}-assistant"
    (models / args.model_id).symlink_to(artifact, target_is_directory=True)
    (models / assistant_id).symlink_to(artifact / "assistant", target_is_directory=True)
    write_data(
        base / "model_settings.json",
        {
            "version": 1,
            "models": {
                args.model_id: {
                    "mtp_enabled": False,
                    "vlm_mtp_enabled": True,
                    "vlm_mtp_draft_model": assistant_id,
                    "vlm_mtp_draft_block_size": 2,
                }
            },
        },
    )

    api_key = secrets.token_hex(32)
    stdout_path = temporary / "server.log"
    process: subprocess.Popen[str] | None = None
    evidence: dict[str, Any] = {
        "schema_version": "axquant.gemma4-omlx-smoke.v1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "host_id": platform.node().removesuffix(".local"),
        "model_id": args.model_id,
        "software_versions": versions,
        "mlx_required": "0.32.2",
        "oMLX_official_mlx_pin": "0.32.0",
        "passed": False,
    }
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout:
            process = subprocess.Popen(
                [
                    str(omlx),
                    "serve",
                    "--model-dir",
                    str(models),
                    "--base-path",
                    str(base),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--max-concurrent-requests",
                    "1",
                    "--no-cache",
                    "--api-key",
                    api_key,
                ],
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            _wait_for_health(args.port, api_key=api_key, timeout=args.startup_timeout)
            status, response = _request_json(
                f"http://127.0.0.1:{args.port}/v1/chat/completions",
                api_key=api_key,
                payload={
                    "model": args.model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Reply with exactly {args.expected_text}",
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 24,
                },
                timeout=args.request_timeout,
            )
        output_text = _response_text(response)
        log_text = stdout_path.read_text(encoding="utf-8", errors="replace")
        stats = _parse_stats(log_text)
        evidence.update(
            {
                "http_status": status,
                "response_exact": output_text == args.expected_text,
                "vlm_batched_engine_loaded": "VLMBatchedEngine loaded" in log_text,
                "assistant_auto_detected": "Auto-detected --draft-kind='mtp'" in log_text,
                "assistant_attached": "VLM MTP drafter attached" in log_text,
                "native_extension_abi_fallback": (
                    "native extension is present but failed to load" in log_text
                ),
                "mtp_stats": stats,
            }
        )
        evidence["passed"] = bool(
            status == 200
            and evidence["response_exact"]
            and evidence["vlm_batched_engine_loaded"]
            and evidence["assistant_auto_detected"]
            and evidence["assistant_attached"]
            and stats is not None
            and stats["rounds"] > 0
            and stats["accepted"] > 0
            and stats["proposed"] >= stats["accepted"]
        )
    except Exception as exc:
        evidence["failure_type"] = type(exc).__name__
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        write_data(args.output, evidence)
        shutil.rmtree(temporary)

    print(json.dumps(evidence, sort_keys=True))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
