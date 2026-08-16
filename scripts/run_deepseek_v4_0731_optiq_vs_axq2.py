#!/usr/bin/env python3
"""QA + speed: mlx-community OptiQ-2bit vs AutomatosX AXQ-2bit (Flash-0731).

Intended host: df-macstudio-m2. Not a certification. Each pack uses its
native runtime (AXQ = resident mlx-lm; OptiQ = mlx-optiq expert streaming).

  PYTHONPATH=src .venv/bin/python scripts/run_deepseek_v4_0731_optiq_vs_axq2.py all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FACTORY_DATASETS = "/Volumes/Ext12T/axquant-certification/datasets"
FACTORY_HF_HOME = "/Volumes/Ext12T/huggingface"
FACTORY_HOST_ID = "df-macstudio-m2"
FACTORY_MODELS = "/Volumes/Ext12T/models"

SOURCE_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
SOURCE_REV = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
OPTIQ_ID = "mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit"
AXQ_ID = "local/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-v1.9.0"
AXQ_REV = "1.9.0"
SEED = 20260728
MAX_TOKENS_QA = 64
MAX_TOKENS_DECODE = 128
OPTIQ_VENV = Path(os.environ.get("OPTIQ_VENV", "/Volumes/Ext12T/venvs/mlx-optiq"))

PACKS: dict[str, dict[str, Any]] = {
    "axq2": {
        "label": os.environ.get(
            "DSV4_AXQ_LABEL", "DeepSeek V4 Flash-0731 AXQ 2-bit (v1.9.0)"
        ),
        "hub": AXQ_ID,
        "commit": AXQ_REV,
        "runtime": "mlx-lm-resident",
        "path": Path(
            os.environ.get(
                "DSV4_AXQ2",
                f"{FACTORY_MODELS}/AX-DeepSeek-V4-Flash-0731-MLX-AXQ-2bit-v1.9.0",
            )
        ),
    },
    "optiq2": {
        "label": "DeepSeek V4 Flash-0731 OptiQ 2-bit",
        "hub": OPTIQ_ID,
        "commit": os.environ.get("DSV4_OPTIQ_REV", "main"),
        "runtime": "mlx-optiq-stream",
        "path": Path(
            os.environ.get(
                "DSV4_OPTIQ2",
                f"{FACTORY_MODELS}/DeepSeek-V4-Flash-0731-OptiQ-2bit",
            )
        ),
    },
}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def _require_factory_host() -> None:
    host = socket.gethostname().split(".", 1)[0]
    if host not in {"df-macstudio-m2", "devopsmacstudio"}:
        raise SystemExit(f"factory eval must run on df-macstudio-m2; observed {host}")


def _quality_lib():
    sys.path.insert(0, str(ROOT / "src"))
    from axquant.quality import load_quality_tasks, score_quality_task_output

    return load_quality_tasks, score_quality_task_output


def _load_task_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no tasks in {path}")
    return rows


def work_dir() -> Path:
    return Path(
        os.environ.get(
            "DSV4_OPTIQ_VS_AXQ_WORK",
            "/Volumes/Ext12T/axquant-certification/deepseek-v4-0731-optiq-vs-axq2-v190",
        )
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def hf_env() -> dict[str, str]:
    env = {
        **os.environ,
        "HF_HOME": os.environ.get("HF_HOME", FACTORY_HF_HOME),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", f"{FACTORY_HF_HOME}/hub"),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": os.environ.get("HF_XET_CACHE", f"{FACTORY_HF_HOME}/xet"),
    }
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    return env


def source_complete(dest: Path) -> bool:
    if not (dest / "config.json").is_file():
        return False
    index = dest / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        files = {dest / name for name in payload.get("weight_map", {}).values()}
        return bool(files) and all(p.is_file() and p.stat().st_size > 0 for p in files)
    shards = sorted(dest.glob("*.safetensors"))
    return len(shards) >= 1 and all(p.stat().st_size > 0 for p in shards)


def cmd_setup_optiq() -> None:
    py = Path("/Users/devop/.local/bin/python3.12")
    if not py.is_file():
        py = Path(sys.executable)
    if not (OPTIQ_VENV / "bin" / "python").is_file():
        log(f"create optiq venv {OPTIQ_VENV}")
        OPTIQ_VENV.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(py), "-m", "venv", str(OPTIQ_VENV)], check=True)
    pip = OPTIQ_VENV / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", "-U", "pip", "huggingface_hub"],
        check=True,
        env=hf_env(),
    )
    subprocess.run(
        [str(pip), "install", "-U", "mlx-optiq>=0.4.12", "mlx-lm"],
        check=True,
        env=hf_env(),
    )
    log("optiq venv ready")


def cmd_download_optiq() -> None:
    dest = Path(PACKS["optiq2"]["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_complete(dest):
        log(f"reuse OptiQ source {dest}")
        return
    hf = ROOT / ".venv" / "bin" / "hf"
    cmd = [
        str(hf) if hf.is_file() else "hf",
        "download",
        OPTIQ_ID,
        "--local-dir",
        str(dest),
    ]
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=hf_env(), cwd=str(ROOT))
    if not source_complete(dest):
        raise SystemExit(f"incomplete OptiQ download {dest}")


def _rss_mb() -> float | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return None


class _AxqBackend:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._mx = None
        self._mlx_lm = None

    def load(self, path: Path) -> None:
        import mlx.core as mx
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        self._mx = mx
        self._mlx_lm = mlx_lm
        self._make_sampler = make_sampler
        self._model, self._tokenizer = mlx_lm.load(str(path), lazy=False)
        mx.eval(self._model.parameters())

    def generate(self, prompt: str, max_tokens: int, seed: int) -> str:
        rendered = _render(self._tokenizer, prompt)
        self._mx.random.seed(seed)
        return str(
            self._mlx_lm.generate(
                self._model,
                self._tokenizer,
                rendered,
                verbose=False,
                max_tokens=max_tokens,
                sampler=self._make_sampler(temp=0.0),
            )
        )

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


class _OptiqBackend:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._mx = None
        self._generate = None
        self._make_sampler = None

    def load(self, path: Path) -> None:
        import mlx.core as mx
        import optiq  # noqa: F401
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        from optiq.runtime import moe_stream

        self._mx = mx
        self._generate = generate
        self._make_sampler = make_sampler
        self._model, self._tokenizer = moe_stream.load_streaming(str(path))

    def generate(self, prompt: str, max_tokens: int, seed: int) -> str:
        rendered = _render(self._tokenizer, prompt)
        self._mx.random.seed(seed)
        return str(
            self._generate(
                self._model,
                self._tokenizer,
                rendered,
                verbose=False,
                max_tokens=max_tokens,
                sampler=self._make_sampler(temp=0.0),
            )
        )

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def _render(tokenizer: Any, prompt: str) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            return str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
    return prompt


def _run_quality(backend: Any, key: str, *, score: bool) -> dict[str, Any]:
    datasets = Path(os.environ.get("DSV4_DATASETS", FACTORY_DATASETS))
    suites = {
        "agent-coding": datasets / "development-agent-coding" / "dataset.jsonl",
        "general": datasets / "development-general" / "dataset.jsonl",
    }
    load_quality_tasks = score_quality_task_output = None
    if score:
        load_quality_tasks, score_quality_task_output = _quality_lib()
    out: dict[str, Any] = {}
    for suite, path in suites.items():
        if not path.is_file():
            raise SystemExit(f"missing dataset {path}")
        if score:
            task_iter = [
                {
                    "task_id": task.task_id,
                    "category": task.category,
                    "prompt": task.prompt,
                    "obj": task,
                }
                for task in load_quality_tasks(path)
            ]
        else:
            task_iter = [
                {
                    "task_id": row["task_id"],
                    "category": row.get("category", suite),
                    "prompt": row["prompt"],
                    "obj": None,
                }
                for row in _load_task_rows(path)
            ]
        results = []
        scores: list[float] = []
        t0 = time.perf_counter()
        for index, task in enumerate(task_iter):
            try:
                text = backend.generate(task["prompt"], MAX_TOKENS_QA, SEED + index)
                err = None
            except Exception as exc:
                text, err = "", str(exc)
            if score:
                sc, checks = score_quality_task_output(task["obj"], text)
                scores.append(sc)
            else:
                sc, checks = None, {}
            results.append(
                {
                    "task_id": task["task_id"],
                    "category": task["category"],
                    "score": sc,
                    "check_scores": checks,
                    "output": text[:2000],
                    "error": err,
                }
            )
            log(f"{key} {suite} {index + 1}/{len(task_iter)} {task['task_id']} score={sc}")
        passed = sum(1 for item in results if item["score"] == 1.0)
        out[suite] = {
            "n": len(results),
            "mean_score": (sum(scores) / len(scores) if scores else None),
            "pass_rate": (passed / len(results) if results and score else None),
            "passed": passed if score else None,
            "seconds": time.perf_counter() - t0,
            "tasks": results,
        }
    return out


def _run_speed(backend: Any) -> dict[str, Any]:
    decode_prompt = "Continue listing incrementing integers, space-separated, starting at 1.\n"
    prefill_block = ("alpha bravo charlie delta echo foxtrot golf hotel " * 80).strip()
    cases = [
        ("decode-128", decode_prompt, MAX_TOKENS_DECODE),
        ("prefill-512-decode-8", prefill_block[:2000], 8),
        ("prefill-2k-decode-8", (prefill_block + " ") * 4, 8),
    ]
    # Warmup
    backend.generate("Say OK.", 8, SEED)
    rows = []
    for name, prompt, max_tokens in cases:
        t0 = time.perf_counter()
        text = backend.generate(prompt, max_tokens, SEED)
        elapsed = time.perf_counter() - t0
        gen_tokens = max(backend.count(text), 1)
        prompt_tokens = backend.count(_render(backend._tokenizer, prompt))
        rows.append(
            {
                "case": name,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "elapsed_seconds": elapsed,
                "tok_per_s": gen_tokens / elapsed if elapsed > 0 else None,
                "preview": text[:240],
            }
        )
        log(f"speed {name}: {rows[-1]['tok_per_s']:.3f} tok/s in {elapsed:.2f}s")
    return {"cases": rows, "rss_gb": (_rss_mb() or 0) / 1024}


def cmd_eval(key: str) -> None:
    _require_factory_host()
    item = PACKS[key]
    pack = Path(item["path"])
    if not (pack / "config.json").is_file():
        raise SystemExit(f"missing pack {pack}")
    work = work_dir()
    out = work / f"{key}.json"
    if out.is_file() and os.environ.get("DSV4_FORCE_EVAL") != "1":
        log(f"reuse {out}")
        return
    log(f"load {key} {item['runtime']} {pack}")
    t_load = time.perf_counter()
    backend: Any = _AxqBackend() if key == "axq2" else _OptiqBackend()
    backend.load(pack)
    load_s = time.perf_counter() - t_load
    log(f"loaded {key} in {load_s:.1f}s rss={_rss_mb()}")
    quality = _run_quality(backend, key, score=(key != "optiq2"))
    payload = {
        "key": key,
        "label": item["label"],
        "hub": item["hub"],
        "commit": item["commit"],
        "runtime": item["runtime"],
        "path": str(pack),
        "host_id": FACTORY_HOST_ID,
        "source": f"{SOURCE_ID}@{SOURCE_REV}",
        "seed": SEED,
        "max_tokens_qa": MAX_TOKENS_QA,
        "load_seconds": load_s,
        "quality": quality,
        "speed": None,
        "measured_at": datetime.now(UTC).isoformat(),
    }
    write_json(out, payload)
    log(f"wrote quality {out}")
    try:
        payload["speed"] = _run_speed(backend)
    except Exception as exc:
        payload["speed"] = {"error": f"{type(exc).__name__}: {exc}", "cases": []}
        log(f"speed failed {key}: {exc}")
    payload["measured_at"] = datetime.now(UTC).isoformat()
    write_json(out, payload)
    log(f"wrote {out}")


def _load_result(key: str) -> dict[str, Any]:
    path = work_dir() / f"{key}.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run eval first")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_score_optiq() -> None:
    load_quality_tasks, score_quality_task_output = _quality_lib()
    payload = _load_result("optiq2")
    datasets = Path(os.environ.get("DSV4_DATASETS", FACTORY_DATASETS))
    suites = {
        "agent-coding": datasets / "development-agent-coding" / "dataset.jsonl",
        "general": datasets / "development-general" / "dataset.jsonl",
    }
    for suite, path in suites.items():
        tasks = {task.task_id: task for task in load_quality_tasks(path)}
        block = payload["quality"][suite]
        scores: list[float] = []
        for item in block["tasks"]:
            task = tasks[item["task_id"]]
            sc, checks = score_quality_task_output(task, item.get("output") or "")
            item["score"] = sc
            item["check_scores"] = checks
            scores.append(sc)
        passed = sum(1 for item in block["tasks"] if item["score"] == 1.0)
        block["mean_score"] = sum(scores) / len(scores) if scores else 0.0
        block["passed"] = passed
        block["pass_rate"] = passed / len(block["tasks"]) if block["tasks"] else 0.0
        log(f"scored optiq2 {suite}: mean={block['mean_score']:.3f} pass={block['pass_rate']:.3f}")
    write_json(work_dir() / "optiq2.json", payload)


def cmd_report() -> None:
    axq = _load_result("axq2")
    optiq = _load_result("optiq2")
    work = work_dir()
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "host_id": FACTORY_HOST_ID,
        "status": "measured",
        "not_a_certification": True,
        "axq2": axq,
        "optiq2": optiq,
    }
    write_json(work / "summary.json", summary)
    raw_dir = Path(
        os.environ.get(
            "DSV4_EVAL_DIR",
            str(ROOT / "docs" / "eval" / "deepseek-v4-flash-0731-optiq2-vs-axq2-v190-macstudio-m2"),
        )
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / "summary.json", summary)
    shutil.copy2(work / "axq2.json", raw_dir / "axq2.json")
    shutil.copy2(work / "optiq2.json", raw_dir / "optiq2.json")

    def qa_row(suite: str) -> str:
        a = axq["quality"][suite]
        o = optiq["quality"][suite]
        return (
            f"| {suite} | {a['n']} | {a['passed']} / {a['n']} ({a['pass_rate'] * 100:.1f}%) "
            f"mean {a['mean_score']:.3f} | {o['passed']} / {o['n']} ({o['pass_rate'] * 100:.1f}%) "
            f"mean {o['mean_score']:.3f} |"
        )

    def speed_row(case: str) -> str:
        def find(blob: dict[str, Any]) -> dict[str, Any]:
            for row in blob["speed"]["cases"]:
                if row["case"] == case:
                    return row
            return {}

        a, o = find(axq), find(optiq)
        ar = a.get("tok_per_s")
        or_ = o.get("tok_per_s")
        ratio = (or_ / ar) if ar and or_ else None
        return (
            f"| {case} | {a.get('prompt_tokens')} / {a.get('generated_tokens')} / "
            f"{a.get('elapsed_seconds', 0):.2f}s / {ar:.2f} | "
            f"{o.get('prompt_tokens')} / {o.get('generated_tokens')} / "
            f"{o.get('elapsed_seconds', 0):.2f}s / {or_:.2f} | "
            f"{ratio:.2f}x |"
            if ar and or_
            else f"| {case} | n/a | n/a | n/a |"
        )

    a_mean = sum(axq["quality"][s]["mean_score"] for s in ("agent-coding", "general")) / 2
    o_mean = sum(optiq["quality"][s]["mean_score"] for s in ("agent-coding", "general")) / 2
    md = "\n".join(
        [
            "# DeepSeek V4 Flash-0731 — OptiQ 2-bit vs AXQ 2-bit (v1.9.0)",
            "",
            "| Field | Value |",
            "| --- | --- |",
            "| Status | Measured practical comparison; **not** a certification |",
            f"| Host | `{FACTORY_HOST_ID}` (Apple M2 Ultra, 192 GB) |",
            f"| Date | {datetime.now(UTC).date().isoformat()} |",
            "| Protocol | Greedy, temperature `0`, thinking off, factory development suites |",
            "",
            "Product question: on the same Mac Studio, how does the mlx-community "
            "OptiQ 2-bit streaming pack compare to the AXQuant 1.9.0 AXQ 2-bit resident "
            "pack for short QA and decode speed?",
            "",
            f"**Short answer:** factory-suite mean score AXQ `{a_mean:.3f}` vs OptiQ "
            f"`{o_mean:.3f}`. Speed is native-runtime: AXQ resident mlx-lm vs OptiQ "
            "SSD expert streaming.",
            "",
            "## Bound artifacts",
            "",
            "| Pack | Hub | Runtime | Local path |",
            "| --- | --- | --- | --- |",
            (
                f"| {axq['label']} | [`{axq['hub']}`](https://huggingface.co/{axq['hub']}) "
                f"@ `{axq['commit']}` | resident mlx-lm | `{Path(axq['path']).name}` |"
            ),
            (
                f"| {optiq['label']} | [`{optiq['hub']}`](https://huggingface.co/{optiq['hub']}) "
                f"| mlx-optiq stream | `{Path(optiq['path']).name}` |"
            ),
            "",
            f"Common source: `{SOURCE_ID}@{SOURCE_REV}`.",
            "",
            "## Quality (factory development suites)",
            "",
            f"Seed `{SEED}`, max new tokens `{MAX_TOKENS_QA}`. "
            "Pass = every check on the task scores 1.0.",
            "",
            "| Suite | N | AXQ 2-bit | OptiQ 2-bit |",
            "| --- | ---: | --- | --- |",
            qa_row("agent-coding"),
            qa_row("general"),
            "",
            "## Speed (native runtime, greedy)",
            "",
            "Columns: prompt tokens / generated tokens / wall / tok/s. Ratio is OptiQ / AXQ.",
            "",
            "| Case | AXQ 2-bit | OptiQ 2-bit | OptiQ / AXQ |",
            "| --- | --- | --- | ---: |",
            speed_row("decode-128"),
            speed_row("prefill-512-decode-8"),
            speed_row("prefill-2k-decode-8"),
            "",
            f"Load time: AXQ `{axq['load_seconds']:.1f}` s, OptiQ `{optiq['load_seconds']:.1f}` s. "
            f"Peak RSS (process): AXQ `{axq['speed'].get('rss_gb', 0):.1f}` GB, "
            f"OptiQ `{optiq['speed'].get('rss_gb', 0):.1f}` GB.",
            "",
            "## Notes",
            "",
            "- This is **not** checkpoint Tier 1 and **not** a retention-vs-BF16 claim.",
            "- AXQ 0731 2-bit remains **not certified** (dual-suite viability was "
            "previously skipped; AX Engine manifest fails on fused gate+up).",
            "- OptiQ streams routed experts from SSD; AXQ keeps the expert table "
            "resident. Speed is not a same-kernel A/B.",
            "- Suites: `development-agent-coding` and `development-general` on Ext12T.",
            "",
            "Runner: [`scripts/run_deepseek_v4_0731_optiq_vs_axq2.py`]"
            "(../scripts/run_deepseek_v4_0731_optiq_vs_axq2.py).",
            "Raw JSON: [`docs/eval/deepseek-v4-flash-0731-optiq2-vs-axq2-v190-macstudio-m2/`]"
            "(eval/deepseek-v4-flash-0731-optiq2-vs-axq2-v190-macstudio-m2/).",
            "",
        ]
    )
    report = Path(
        os.environ.get(
            "DSV4_REPORT",
            str(ROOT / "docs" / "deepseek-v4-flash-0731-optiq2-vs-axq2-v190.md"),
        )
    )
    report.write_text(md + "\n", encoding="utf-8")
    shutil.copy2(report, work / "report.md")
    log(f"wrote {report}")


def cmd_all() -> None:
    cmd_setup_optiq()
    cmd_download_optiq()
    if Path(sys.executable).resolve() != (OPTIQ_VENV / "bin" / "python").resolve():
        # AXQ eval in the axquant venv; OptiQ eval in the isolated optiq venv.
        cmd_eval("axq2")
        env = hf_env()
        env.pop("PYTHONPATH", None)
        subprocess.run(
            [
                str(OPTIQ_VENV / "bin" / "python"),
                str(ROOT / "scripts" / "run_deepseek_v4_0731_optiq_vs_axq2.py"),
                "eval-optiq",
            ],
            check=True,
            cwd=str(ROOT),
            env=env,
        )
        cmd_score_optiq()
    else:
        cmd_eval("optiq2")
        return
    cmd_report()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        choices=[
            "setup-optiq",
            "download-optiq",
            "eval-axq",
            "eval-optiq",
            "score-optiq",
            "report",
            "all",
        ],
    )
    args = parser.parse_args()
    {
        "setup-optiq": cmd_setup_optiq,
        "download-optiq": cmd_download_optiq,
        "eval-axq": lambda: cmd_eval("axq2"),
        "eval-optiq": lambda: cmd_eval("optiq2"),
        "score-optiq": cmd_score_optiq,
        "report": cmd_report,
        "all": cmd_all,
    }[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
