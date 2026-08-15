#!/usr/bin/env python3
"""Practical head-to-head: Qwen3.8-27B vs Qwen3.6-27B AXQ 4-bit MTP.

Runs on an AX Engine OpenAI-compatible server. Designed for df-macstudio-m2.

  python scripts/run_qwen38_vs_qwen36_practical.py --phase all
  python scripts/run_qwen38_vs_qwen36_practical.py --phase quality --model qwen38
  python scripts/run_qwen38_vs_qwen36_practical.py --phase report
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUITE_DIR = ROOT / "data" / "eval" / "practical-qwen38-vs-qwen36"
DEFAULT_WORK = Path(
    os.environ.get(
        "QWEN38_VS36_WORK",
        "/Volumes/Ext4T/axquant/eval/qwen38-vs-qwen36-27b-axq4-mtp",
    )
)
ENGINE_BIN = Path(
    os.environ.get(
        "AX_ENGINE_SERVER",
        "/Users/devop/opt/ax-engine-6.16.1/bin/ax-engine-server",
    )
)
ENGINE_BENCH = Path(
    os.environ.get(
        "AX_ENGINE_BENCH",
        str(ENGINE_BIN.parent / "ax-engine-bench"),
    )
)

MODELS: dict[str, dict[str, str]] = {
    "qwen36": {
        "label": "Qwen3.6-27B AXQ 4-bit MTP",
        "short": "3.6",
        "path": os.environ.get(
            "QWEN36_PACK",
            "/Volumes/Ext4T/models/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP",
        ),
        "hub": "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP",
        "commit": "f44a9eeebec0c488d0f42201c8763db770a1c0a8",
    },
    "qwen38": {
        "label": "Qwen3.8-27B AXQ 4-bit MTP",
        "short": "3.8",
        "path": os.environ.get(
            "QWEN38_PACK",
            "/Volumes/Ext4T/models/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
        ),
        "hub": "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
        "commit": "32f448461caf4aedcc3c16a77a63b6a94bf0667c",
    },
}

MTP_ENV = {
    "AX_MLX_QWEN_LINEAR_MTP_EXACT": "1",
    "AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE": "1",
    "AX_MLX_MTP_BYPASS_MIN_SAMPLES": "1000",
    "AX_MLX_MTP_DRAFT_MIN_CONFIDENCE": "0",
    "AX_MLX_MTP_LINEAR_EXACT_REPLAY": "0",
    "AX_MLX_MTP_MIN_REMAINING_TOKENS": "0",
    "AX_MLX_QWEN_DENSE_FFN_GATE_UP_MATVEC_METAL": "0",
    "AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS": "0",
    "AX_MLX_SPECULATIVE_INVARIANT_PROJECTIONS": "all",
    "AX_MLX_SPECULATIVE_ROW_EXACT_POST_INPUT": "1",
    "AX_MLX_SPECULATIVE_SPLIT_FFN": "1",
    "AX_MLX_MTP_ASYNC_DRAFT": "1",
    "AX_MLX_MTP_VERIFY_SUBMIT_LAYERS": "8",
    "AX_MLX_PIPELINE_GRANULARITY": "layer",
}

SYSTEM_PROMPT = (
    "You are a precise assistant. After any brief reasoning, put the final "
    "answer alone on the last line in the form:\nFINAL: <answer>\n"
    "Do not add extra commentary after that line."
)
DECODE_SPEED_PROMPT = (
    "Continue listing incrementing integers, space-separated, forever. Start at 1.\n"
)
PREFILL_SPEED_PROMPT = "Reply with the single word READY after reading this block.\n\n"

FINAL_RE = re.compile(r"(?im)^\s*FINAL:\s*(.+?)\s*$")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for name in ("chatbot.jsonl", "coding.jsonl", "logic.jsonl", "vision.jsonl"):
        path = SUITE_DIR / name
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    return tasks


def extract_final(text: str) -> str:
    matches = list(FINAL_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip().strip("`").strip('"').strip()
    boxed = list(BOXED_RE.finditer(text))
    if boxed:
        return boxed[-1].group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_code(text: str) -> str:
    text = strip_think(text)
    fences = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fences:
        return fences[-1].strip()
    return text.strip()


def json_value(text: str) -> Any:
    text = strip_think(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object")
    return json.loads(text[start : end + 1])


def run_python_exec(output: str, spec: dict[str, Any]) -> bool:
    code = extract_code(output)
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    tests = spec.get("tests") or []
    script = code + "\n\n" + "\n".join(tests) + "\n"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0


def score_task(task: dict[str, Any], output: str) -> tuple[float, dict[str, float], str]:
    visible = strip_think(output)
    final = extract_final(visible)
    check_scores: dict[str, float] = {}
    for index, check in enumerate(task.get("checks") or []):
        kind = check["kind"]
        value = check.get("value")
        key = f"{kind}:{index}"
        ok = 0.0
        if kind == "final":
            ok = float(final.lower() == str(value).lower())
        elif kind == "contains":
            ok = float(str(value).lower() in visible.lower())
        elif kind == "regex":
            ok = float(re.search(str(value), visible, flags=re.IGNORECASE | re.DOTALL) is not None)
        elif kind == "numeric":
            nums = re.findall(r"-?\d+(?:\.\d+)?", final.replace(",", ""))
            if nums:
                ok = float(abs(float(nums[0]) - float(value)) <= 0.02 * max(1.0, abs(float(value))))
        elif kind == "json-keys":
            try:
                parsed = json_value(visible)
                ok = float(isinstance(parsed, dict) and all(k in parsed for k in value))
            except (ValueError, json.JSONDecodeError):
                ok = 0.0
        elif kind == "python-exec":
            ok = float(run_python_exec(visible, value if isinstance(value, dict) else {}))
        elif kind == "word-count":
            counted = visible.split("FINAL:")[-1] if "FINAL:" in visible else final
            words = re.findall(r"[A-Za-z0-9']+", counted)
            if not words:
                words = re.findall(r"[A-Za-z0-9']+", final)
            ok = float(len(words) == int(value))
        else:
            raise ValueError(f"unknown check {kind}")
        check_scores[key] = ok
    score = sum(check_scores.values()) / len(check_scores) if check_scores else 0.0
    return score, check_scores, final


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail[:800]}") from exc
    return json.loads(body) if body else {}


def loaded_model_id(base: str) -> str:
    data = http_json("GET", base + "/v1/models", timeout=30)
    items = data.get("data") or []
    if not items:
        raise RuntimeError("server advertised no models")
    return str(items[0]["id"])


def wait_health(base: str, timeout: int = 600) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (OSError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = str(exc)
        time.sleep(2)
    raise RuntimeError(f"server did not become healthy: {last}")


def encode_image(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def chat_complete(
    base: str,
    prompt: str,
    *,
    model_id: str,
    image: Path | None,
    max_tokens: int,
    temperature: float = 0.0,
    request_id: str = "",
) -> tuple[str, dict[str, Any], float]:
    content: Any
    if image is not None:
        content = [
            {"type": "image_url", "image_url": {"url": encode_image(image)}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    system = SYSTEM_PROMPT
    if request_id:
        system = f"{SYSTEM_PROMPT}\nRequest id: {request_id}."
    payload = {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
    }
    started = time.perf_counter()
    data = http_json("POST", base + "/v1/chat/completions", payload, timeout=900)
    elapsed = time.perf_counter() - started
    text = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage") or {}
    return text, usage, elapsed


def start_server(model_dir: Path, port: int, log_path: Path) -> subprocess.Popen[str]:
    if not ENGINE_BIN.is_file():
        raise SystemExit(f"ax-engine-server not found: {ENGINE_BIN}")
    env = os.environ.copy()
    env.update(MTP_ENV)
    env["PATH"] = f"{ENGINE_BIN.parent}:{env.get('PATH', '')}"
    cmd = [
        str(ENGINE_BIN),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--mlx",
        "--mlx-model-artifacts-dir",
        str(model_dir),
        "--support-tier",
        "mlx-certified",
        "--speculation-profile",
        "chatbot",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("w", encoding="utf-8")
    log("starting " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env)
    return proc


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def ensure_manifest(model_dir: Path) -> None:
    manifest = model_dir / "model-manifest.json"
    if manifest.is_file():
        return
    if not ENGINE_BENCH.is_file():
        log(f"no bench binary, skip generate-manifest for {model_dir}")
        return
    log(f"generate-manifest {model_dir}")
    completed = subprocess.run(
        [str(ENGINE_BENCH), "generate-manifest", "--force", "--", str(model_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        log(completed.stdout[-1000:])
        log(completed.stderr[-1000:])
        raise SystemExit("generate-manifest failed")


def host_record() -> dict[str, Any]:
    def sysctl(name: str) -> str | None:
        try:
            out = subprocess.check_output(["/usr/sbin/sysctl", "-n", name], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
        return out or None

    mem = sysctl("hw.memsize")
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "ncpu": sysctl("hw.ncpu"),
        "mem_bytes": int(mem) if mem and mem.isdigit() else mem,
        "engine": str(ENGINE_BIN),
        "engine_help": (
            subprocess.check_output([str(ENGINE_BIN), "--help"], text=True).splitlines()[0]
            if ENGINE_BIN.is_file()
            else None
        ),
        "collected_at": datetime.now(UTC).isoformat(),
    }


def result_path(work: Path, model_key: str) -> Path:
    return work / "runs" / f"{model_key}-quality.json"


def run_quality(
    model_key: str, work: Path, port: int, categories: set[str] | None
) -> dict[str, Any]:
    spec = MODELS[model_key]
    model_dir = Path(spec["path"])
    if not model_dir.is_dir():
        raise SystemExit(f"missing pack: {model_dir}")
    ensure_manifest(model_dir)
    dest = result_path(work, model_key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if dest.is_file():
        existing = json.loads(dest.read_text(encoding="utf-8"))
    done = {row["task_id"]: row for row in existing.get("tasks", [])}
    tasks = load_tasks()
    if categories:
        tasks = [t for t in tasks if t["category"] in categories]
    pending = [t for t in tasks if t["task_id"] not in done]
    log(f"{model_key}: {len(done)} done, {len(pending)} pending, {len(tasks)} selected")
    proc = None
    base = f"http://127.0.0.1:{port}"
    if pending:
        proc = start_server(model_dir, port, work / "logs" / f"{model_key}-server.log")
        try:
            wait_health(base, timeout=900)
            model_id = loaded_model_id(base)
            log(f"{model_key}: server healthy model_id={model_id}")
            for task in pending:
                image = None
                if task.get("image"):
                    image = SUITE_DIR / task["image"]
                    if not image.is_file():
                        raise SystemExit(f"missing image {image}")
                max_tokens = 1024 if task["category"] == "coding" else 640
                try:
                    # Unique prefix defeats prefix-cache reuse of a previous
                    # completion (observed as stray "Osaka" after chat-02).
                    prompt = f"[task {task['task_id']}]\n{task['prompt']}"
                    text, usage, elapsed = chat_complete(
                        base,
                        prompt,
                        model_id=model_id,
                        image=image,
                        max_tokens=max_tokens,
                        request_id=task["task_id"],
                    )
                    score, checks, final = score_task(task, text)
                    error = None
                except Exception as exc:
                    text, usage, elapsed, score, checks, final = "", {}, 0.0, 0.0, {}, ""
                    error = f"{type(exc).__name__}: {exc}"
                    log(f"FAIL {task['task_id']}: {error}")
                row = {
                    "task_id": task["task_id"],
                    "category": task["category"],
                    "subcategory": task.get("subcategory"),
                    "score": score,
                    "passed": score >= 0.999,
                    "final": final,
                    "check_scores": checks,
                    "usage": usage,
                    "elapsed_seconds": elapsed,
                    "output": text,
                    "error": error,
                }
                done[task["task_id"]] = row
                log(
                    f"{model_key} {task['task_id']} score={score:.2f} "
                    f"pass={row['passed']} {elapsed:.1f}s final={final!r:.80}"
                )
                payload = _quality_payload(model_key, spec, model_dir, list(done.values()))
                dest.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            run_speed_http(base, work / "runs" / f"{model_key}-speed.json", model_key, model_id)
        finally:
            if proc is not None:
                stop_server(proc)
    else:
        log(f"{model_key}: quality complete, speed only if missing")
        speed_path = work / "runs" / f"{model_key}-speed.json"
        if not speed_path.is_file():
            proc = start_server(model_dir, port, work / "logs" / f"{model_key}-server.log")
            try:
                wait_health(base, timeout=900)
                model_id = loaded_model_id(base)
                run_speed_http(base, speed_path, model_key, model_id)
            finally:
                stop_server(proc)
    payload = _quality_payload(model_key, spec, model_dir, list(done.values()))
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _quality_payload(
    model_key: str, spec: dict[str, str], model_dir: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: r["task_id"])
    by_cat: dict[str, list[float]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(float(row["score"]))
    return {
        "model_key": model_key,
        "label": spec["label"],
        "path": str(model_dir),
        "hub": spec["hub"],
        "commit": spec["commit"],
        "n_tasks": len(rows),
        "mean_score": (sum(r["score"] for r in rows) / len(rows)) if rows else 0.0,
        "pass_rate": (sum(1 for r in rows if r["passed"]) / len(rows)) if rows else 0.0,
        "by_category": {
            cat: {
                "n": len(vals),
                "mean_score": sum(vals) / len(vals),
                "pass_rate": sum(1 for v in vals if v >= 0.999) / len(vals),
            }
            for cat, vals in sorted(by_cat.items())
        },
        "tasks": rows,
    }


def repeat_prompt(target_chars: int) -> str:
    unit = (
        "The factory acceptance test records prefill throughput on a repeated "
        "technical paragraph so the tokenizer sees mixed English, digits 0123456789, "
        "and punctuation. Sequence marker {n:05d}. "
    )
    parts: list[str] = []
    n = 0
    while sum(len(p) for p in parts) < target_chars:
        parts.append(unit.format(n=n))
        n += 1
    return "".join(parts)[:target_chars]


def run_speed_http(base: str, dest: Path, model_key: str, model_id: str) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cases = [
        {"name": "prefill-512", "chars": 2000, "max_tokens": 8, "kind": "prefill"},
        {"name": "prefill-2048", "chars": 8000, "max_tokens": 8, "kind": "prefill"},
        {"name": "prefill-4096", "chars": 16000, "max_tokens": 8, "kind": "prefill"},
        {"name": "decode-128", "chars": 400, "max_tokens": 128, "kind": "decode"},
        {"name": "decode-256", "chars": 400, "max_tokens": 256, "kind": "decode"},
        {"name": "decode-512", "chars": 400, "max_tokens": 512, "kind": "decode"},
    ]
    results: list[dict[str, Any]] = []
    for case in cases:
        # Unique salt per trial so prefix-cache hits cannot inflate prefill.
        trials = []
        for trial in range(3):
            salt = f"SALT-{model_key}-{case['name']}-t{trial}-{time.time_ns()}\n"
            lead = DECODE_SPEED_PROMPT if case["kind"] == "decode" else PREFILL_SPEED_PROMPT
            prompt = lead + salt + repeat_prompt(case["chars"])
            payload = {
                "model": model_id,
                "temperature": 0.0,
                "max_tokens": case["max_tokens"],
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": False},
            }
            started = time.perf_counter()
            try:
                data = http_json("POST", base + "/v1/chat/completions", payload, timeout=900)
                elapsed = time.perf_counter() - started
                usage = data.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                err = None
            except Exception as exc:
                elapsed = time.perf_counter() - started
                usage, prompt_tokens, completion_tokens = {}, 0, 0
                err = f"{type(exc).__name__}: {exc}"
            row = {
                "trial": trial,
                "warmup": trial == 0,
                "elapsed_seconds": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "usage": usage,
                "error": err,
            }
            if trial > 0:
                trials.append(row)
            log(
                f"{model_key} speed {case['name']} trial={trial} "
                f"{elapsed:.2f}s in={prompt_tokens} out={completion_tokens} err={err}"
            )
        measured = [t for t in trials if not t.get("error")]
        prompt_tps = None
        decode_tps = None
        if measured:
            # For prefill-heavy cases most time is prompt eval; for decode cases
            # subtract a residual using the shortest prefill sibling if present.
            avg_in = sum(t["prompt_tokens"] for t in measured) / len(measured)
            avg_out = sum(t["completion_tokens"] for t in measured) / len(measured)
            avg_s = sum(t["elapsed_seconds"] for t in measured) / len(measured)
            if case["kind"] == "prefill" and avg_s > 0:
                prompt_tps = avg_in / avg_s
            if case["kind"] == "decode" and avg_s > 0 and avg_out > 0:
                decode_tps = avg_out / avg_s
        results.append(
            {
                **case,
                "trials": trials,
                "prompt_tps": prompt_tps,
                "decode_tps": decode_tps,
            }
        )
    payload = {
        "model_key": model_key,
        "collected_at": datetime.now(UTC).isoformat(),
        "cases": results,
    }
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def summarize(work: Path) -> dict[str, Any]:
    host = host_record()
    models = {}
    for key in MODELS:
        qpath = result_path(work, key)
        spath = work / "runs" / f"{key}-speed.json"
        models[key] = {
            "quality": json.loads(qpath.read_text(encoding="utf-8")) if qpath.is_file() else None,
            "speed": json.loads(spath.read_text(encoding="utf-8")) if spath.is_file() else None,
        }
    summary = {
        "host": host,
        "protocol": {
            "temperature": 0.0,
            "thinking": False,
            "mtp_profile": "qwen38-exact-async",
            "runtime": "AX Engine 6.16.1",
            "system_prompt": SYSTEM_PROMPT,
        },
        "models": models,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (work / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["quality", "speed", "report", "all"], default="all")
    parser.add_argument("--model", choices=["qwen36", "qwen38", "both"], default="both")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--port", type=int, default=31418)
    parser.add_argument(
        "--categories",
        default="chatbot,coding,logic,vision",
        help="comma-separated categories",
    )
    args = parser.parse_args()
    work: Path = args.work
    work.mkdir(parents=True, exist_ok=True)
    cats = {c.strip() for c in args.categories.split(",") if c.strip()}
    keys = ["qwen36", "qwen38"] if args.model == "both" else [args.model]
    if args.phase in {"quality", "all"}:
        for key in keys:
            run_quality(key, work, args.port, cats)
    if args.phase == "speed":
        for key in keys:
            spec = MODELS[key]
            model_dir = Path(spec["path"])
            ensure_manifest(model_dir)
            proc = start_server(model_dir, args.port, work / "logs" / f"{key}-server.log")
            try:
                wait_health(f"http://127.0.0.1:{args.port}", timeout=900)
                model_id = loaded_model_id(f"http://127.0.0.1:{args.port}")
                run_speed_http(
                    f"http://127.0.0.1:{args.port}",
                    work / "runs" / f"{key}-speed.json",
                    key,
                    model_id,
                )
            finally:
                stop_server(proc)
    summarize(work)
    log(f"wrote {work / 'summary.json'}")


if __name__ == "__main__":
    main()
