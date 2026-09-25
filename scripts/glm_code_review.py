#!/usr/bin/env python3
"""GLM-driven, provenance-bound code review harness for axquant.

The harness enumerates Python blobs under a pinned git revision, batches them by
cumulative line budget with per-file sha256 digests, dispatches every batch to a
GLM reviewer through the ``ax-code`` CLI in a read-only sandbox, and writes
fail-closed artifacts:

    inventory.json   every reviewed file exactly once with lines/sha256/batch_id
    manifest.json    batch provenance: prompt/source digests, exit code, sentinel
    findings.json    candidate findings with local triage verdicts
    report.md        English human-readable report
    prompts/         the exact prompt bytes sent for each batch
    logs/            captured stdout/stderr per batch

A batch is accepted only when the reviewer exits 0, ends with the terminal
completion sentinel, and returns parseable findings JSON. ``check`` recomputes
coverage, blob digests, batch status and every finding citation from git and
validates the ``.internal/bugs/`` entries for confirmed findings.

A triage verdict is decided locally (never by the reviewer) and applied with the
``triage`` subcommand from a ``triage.json`` file. This module is tooling only;
it never edits ``src/``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SENTINEL = "GLM_CODE_REVIEW_COMPLETE"
JSON_BEGIN = "BEGIN_GLM_CODE_REVIEW_JSON"
JSON_END = "END_GLM_CODE_REVIEW_JSON"

DEFAULT_MODEL = "defai-01-ax-trust-com/glm-5.3"
DEFAULT_SCOPE = "src/axquant"
DEFAULT_REPORT = ".internal/reports/glm-code-review"
DEFAULT_BUGS_DIR = ".internal/bugs"
DEFAULT_LINE_BUDGET = 2500
DEFAULT_TIMEOUT_S = 900.0

INVENTORY_SCHEMA = "axquant.glm-code-review-inventory.v1"
MANIFEST_SCHEMA = "axquant.glm-code-review-manifest.v1"
FINDINGS_SCHEMA = "axquant.glm-code-review-findings.v1"

VERDICTS = ("confirmed", "refuted", "needs-runtime-evidence")
SEVERITIES = ("high", "medium", "low")

# Default reviewer invocation. ``{model}``, ``{title}`` and ``{prompt_file}`` are
# substituted with whitespace-free values, then shlex-split (no shell is used).
DEFAULT_REVIEWER_CMD = (
    "ax-code run --model {model} --sandbox read-only --quiet "
    "--title {title} --prompt-file {prompt_file}"
)

_CITATION_RE = re.compile(r"(src/axquant/[\w./-]+\.py):(\d+)(?:\s*-\s*(\d+))?")


class ReviewError(RuntimeError):
    """Raised when the review harness cannot produce or validate valid evidence."""


# --------------------------------------------------------------------------- #
# git / hashing helpers
# --------------------------------------------------------------------------- #


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)
    if proc.returncode != 0:
        raise ReviewError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout


def repo_root(start: Path | None = None) -> Path:
    out = run_git(start or Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(out.decode().strip())


def resolve_revision(repo: Path, revision: str) -> str:
    out = run_git(repo, "rev-parse", f"{revision}^{{commit}}")
    return out.decode().strip()


def list_scope_files(repo: Path, revision: str, scope: str) -> list[str]:
    out = run_git(repo, "ls-tree", "-r", "--name-only", revision, scope)
    paths = [p for p in out.decode().splitlines() if p.endswith(".py")]
    return sorted(paths)


def read_blob(repo: Path, revision: str, path: str) -> bytes:
    return run_git(repo, "cat-file", "blob", f"{revision}:{path}")


def line_count(content: bytes) -> int:
    return len(content.splitlines())


def source_digest(entries: list[dict[str, object]]) -> str:
    payload = "\n".join(f"{e['path']}:{e['sha256']}" for e in entries)
    return sha256_hex(payload.encode())


# --------------------------------------------------------------------------- #
# inventory + batching
# --------------------------------------------------------------------------- #


@dataclass
class Batch:
    batch_id: str
    files: list[str]
    lines: int
    source_sha256: str


def build_inventory(
    repo: Path, revision: str, scope: str, line_budget: int
) -> tuple[dict[str, object], list[Batch], dict[str, str], dict[str, bytes]]:
    """Return (inventory, batches, path->source_text, path->blob_bytes)."""
    paths = list_scope_files(repo, revision, scope)
    if not paths:
        raise ReviewError(f"no Python files found under {scope} at {revision}")
    files: list[dict[str, object]] = []
    sources: dict[str, str] = {}
    blobs: dict[str, bytes] = {}
    for path in paths:
        raw = read_blob(repo, revision, path)
        blobs[path] = raw
        sources[path] = raw.decode("utf-8", "replace")
        files.append({"path": path, "lines": line_count(raw), "sha256": sha256_hex(raw)})

    batches: list[Batch] = []
    current: list[dict[str, object]] = []
    for entry in files:
        entry_lines = int(entry["lines"])
        if current and sum(int(e["lines"]) for e in current) + entry_lines > line_budget:
            batches.append(_make_batch(len(batches), current))
            current = []
        current.append(entry)
    if current:
        batches.append(_make_batch(len(batches), current))

    by_path = {str(e["path"]): b.batch_id for b in batches for e in _entries_for(b, files)}
    inventory: dict[str, object] = {
        "schema": INVENTORY_SCHEMA,
        "revision": revision,
        "scope": scope,
        "line_budget": line_budget,
        "file_count": len(files),
        "loc_total": sum(int(e["lines"]) for e in files),
        "files": [{**e, "batch_id": by_path[str(e["path"])]} for e in files],
    }
    return inventory, batches, sources, blobs


def _entries_for(batch: Batch, files: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = set(batch.files)
    return [e for e in files if e["path"] in wanted]


def _make_batch(index: int, entries: list[dict[str, object]]) -> Batch:
    return Batch(
        batch_id=f"b{index:03d}",
        files=[str(e["path"]) for e in entries],
        lines=sum(int(e["lines"]) for e in entries),
        source_sha256=source_digest(entries),
    )


# --------------------------------------------------------------------------- #
# prompt + reviewer contract
# --------------------------------------------------------------------------- #


def build_prompt(
    revision: str, batch: Batch, files: list[dict[str, object]], sources: dict[str, str]
) -> str:
    by_path = {str(e["path"]): e for e in files}
    parts = [
        "You are a senior Python reviewer performing a static, read-only review of",
        f"the axquant toolkit at git revision {revision}.",
        "",
        "Report only real defects: correctness bugs, resource/lifecycle leaks,",
        "fail-open error handling, unsafe assumptions, and clear contract",
        "violations. Do not report style, naming, or documentation nits. Do not",
        "speculate about code you were not shown.",
        "",
        "For every finding provide: severity (high|medium|low), file (the exact",
        "repo-relative path shown below), line (1-based number from the listing),",
        "snippet (the exact source text of that line), mechanism (why it is a",
        "defect), and suggested_fix.",
        "",
        "Output rules (follow exactly, English only):",
        f"1. The first line is exactly: {JSON_BEGIN}",
        '2. The next line is a single JSON object: {"findings": [...], "verdict":',
        '   "findings"|"no-findings"}. Each finding has keys severity, file, line,',
        "   snippet, mechanism, suggested_fix.",
        f"3. The next line is exactly: {JSON_END}",
        f"4. The final non-empty line is exactly: {SENTINEL}",
        "5. Emit no prose outside those markers.",
        "",
    ]
    for path in batch.files:
        entry = by_path[path]
        numbered = "\n".join(
            f"{i:>5}: {text}" for i, text in enumerate(sources[path].splitlines(), start=1)
        )
        parts += [
            f"=== FILE {path} (lines={entry['lines']} sha256={entry['sha256']}) ===",
            numbered,
            f"=== END FILE {path} ===",
            "",
        ]
    return "\n".join(parts)


def parse_contract(stdout: str) -> tuple[list[dict[str, object]], str]:
    """Enforce sentinel + JSON markers and return (findings, verdict)."""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines or lines[-1].strip() != SENTINEL:
        raise ReviewError(
            f"missing terminal completion sentinel {SENTINEL!r} "
            f"(last non-empty line: {lines[-1].strip()[:80]!r} if lines else '<empty>')"
        )
    text = stdout
    start = text.find(JSON_BEGIN)
    if start < 0:
        raise ReviewError(f"missing JSON start marker {JSON_BEGIN!r}")
    end = text.find(JSON_END, start + len(JSON_BEGIN))
    if end < 0:
        raise ReviewError(f"missing JSON end marker {JSON_END!r}")
    payload = text[start + len(JSON_BEGIN) : end].strip()
    payload = payload.strip("`").strip()
    if payload.startswith("json"):
        payload = payload[4:].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ReviewError(f"findings JSON is not parseable: {exc}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("findings"), list):
        raise ReviewError('findings JSON must be an object with a "findings" list')
    return list(parsed["findings"]), str(parsed.get("verdict", ""))


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


@dataclass
class BatchResult:
    batch_id: str
    files: list[str]
    lines: int
    source_sha256: str
    prompt_sha256: str
    prompt_file: str
    stdout_log: str
    stderr_log: str
    exit_code: int | None
    sentinel: bool
    duration_s: float
    findings: list[dict[str, object]] = field(default_factory=list)
    error: str | None = None


def reviewer_argv(template: str, model: str, title: str, prompt_file: Path) -> list[str]:
    for name, value in (("model", model), ("title", title), ("prompt_file", str(prompt_file))):
        if any(ch.isspace() for ch in value):
            raise ReviewError(f"reviewer {name} must not contain whitespace: {value!r}")
    rendered = template.format(model=model, title=title, prompt_file=str(prompt_file))
    return shlex.split(rendered)


def dispatch_batch(
    *,
    batch: Batch,
    prompt: str,
    report_dir: Path,
    revision: str,
    model: str,
    reviewer_template: str,
    timeout_s: float,
) -> BatchResult:
    prompts_dir = report_dir / "prompts"
    logs_dir = report_dir / "logs"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompts_dir / f"{batch.batch_id}.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    stdout_log = logs_dir / f"{batch.batch_id}.stdout.txt"
    stderr_log = logs_dir / f"{batch.batch_id}.stderr.txt"

    result = BatchResult(
        batch_id=batch.batch_id,
        files=list(batch.files),
        lines=batch.lines,
        source_sha256=batch.source_sha256,
        prompt_sha256=sha256_hex(prompt.encode()),
        prompt_file=str(prompt_file.relative_to(report_dir)),
        stdout_log=str(stdout_log.relative_to(report_dir)),
        stderr_log=str(stderr_log.relative_to(report_dir)),
        exit_code=None,
        sentinel=False,
        duration_s=0.0,
    )
    argv = reviewer_argv(reviewer_template, model, f"glm-review-{batch.batch_id}", prompt_file)
    started = time.monotonic()
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        result.duration_s = round(time.monotonic() - started, 3)
        result.error = f"reviewer timed out after {timeout_s:.0f}s"
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(result.error, encoding="utf-8")
        return result
    except OSError as exc:
        result.duration_s = round(time.monotonic() - started, 3)
        result.error = f"reviewer failed to start: {exc}"
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(result.error, encoding="utf-8")
        return result

    result.duration_s = round(time.monotonic() - started, 3)
    result.exit_code = proc.returncode
    stdout = proc.stdout.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8", "replace")
    stdout_log.write_text(stdout, encoding="utf-8")
    stderr_log.write_text(stderr, encoding="utf-8")

    if proc.returncode != 0:
        result.error = f"reviewer exited {proc.returncode}: {stderr.strip()[:200]}"
        return result
    try:
        findings, _verdict = parse_contract(stdout)
    except ReviewError as exc:
        result.error = str(exc)
        return result
    result.sentinel = True
    result.findings = findings
    return result


# --------------------------------------------------------------------------- #
# artifact writers
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def normalize_finding(raw: dict[str, object], index: int, batch_id: str) -> dict[str, object]:
    severity = str(raw.get("severity", "")).strip().lower()
    line_raw = raw.get("line", 0)
    try:
        line = int(line_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        line = 0
    return {
        "id": f"F{index:03d}",
        "batch_id": batch_id,
        "severity": severity,
        "file": str(raw.get("file", "")).strip(),
        "line": line,
        "snippet": str(raw.get("snippet", "")).strip(),
        "mechanism": str(raw.get("mechanism", "")).strip(),
        "suggested_fix": str(raw.get("suggested_fix", "")).strip(),
        "triage": "needs-runtime-evidence",
        "bug_doc": None,
        "triage_note": "",
    }


def collect_findings(results: list[BatchResult]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for result in results:
        for raw in result.findings:
            if isinstance(raw, dict):
                findings.append(normalize_finding(raw, len(findings) + 1, result.batch_id))
    return findings


def render_report(
    *,
    revision: str,
    model: str,
    inventory: dict[str, object],
    manifest: dict[str, object],
    findings: list[dict[str, object]],
) -> str:
    batches = manifest.get("batches", [])
    failed = [b for b in batches if b.get("exit_code") != 0 or not b.get("sentinel")]  # type: ignore[union-attr]
    verdict = "FINDINGS" if findings else "NO_FINDINGS"
    lines = [
        f"# GLM code review: {inventory['scope']} @ {revision}",
        "",
        f"Status: {verdict}",
        f"Model: {model}",
        f"Reviewed: {inventory['file_count']} files, {inventory['loc_total']} lines, "
        f"{manifest.get('batch_count')} batches (line budget {inventory.get('line_budget')})",
        f"Failed batches: {len(failed)}",
        "",
    ]
    if not findings:
        lines += [
            "NO_FINDINGS: GLM reported no candidate defects across every reviewed file.",
            "",
        ]
    for finding in findings:
        lines += [
            f"### {finding['id']} - {finding['severity'] or 'unknown'} - "
            f"{finding['file']}:{finding['line']}",
            "",
            f"- Triage: {finding['triage']}",
            f"- Batch: {finding['batch_id']}",
            f"- Snippet: `{finding['snippet']}`",
            f"- Mechanism: {finding['mechanism']}",
            f"- Suggested fix: {finding['suggested_fix']}",
        ]
        if finding.get("bug_doc"):
            lines.append(f"- Bug entry: {finding['bug_doc']}")
        if finding.get("triage_note"):
            lines.append(f"- Triage note: {finding['triage_note']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(
    *,
    report_dir: Path,
    revision: str,
    scope: str,
    model: str,
    reviewer_template: str,
    inventory: dict[str, object],
    results: list[BatchResult],
    findings: list[dict[str, object]],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "revision": revision,
        "scope": scope,
        "model": model,
        "reviewer_cmd": reviewer_template,
        "generated_at": _now_iso(),
        "file_count": inventory["file_count"],
        "loc_total": inventory["loc_total"],
        "batch_count": len(results),
        "batches": [asdict(r) for r in results],
    }
    write_json(report_dir / "inventory.json", inventory)
    write_json(report_dir / "manifest.json", manifest)
    payload = {
        "schema": FINDINGS_SCHEMA,
        "revision": revision,
        "model": model,
        "verdict": "findings" if findings else "no-findings",
        "findings": findings,
    }
    write_json(report_dir / "findings.json", payload)
    (report_dir / "report.md").write_text(
        render_report(
            revision=revision,
            model=model,
            inventory=inventory,
            manifest=manifest,
            findings=findings,
        ),
        encoding="utf-8",
    )
    return manifest


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
    repo = repo_root()
    revision = resolve_revision(repo, args.revision)
    report_dir = Path(args.report)
    inventory, batches, sources, _blobs = build_inventory(
        repo, revision, args.scope, args.line_budget
    )
    files = inventory["files"]
    assert isinstance(files, list)

    print(
        f"revision={revision} scope={args.scope} files={inventory['file_count']} "
        f"loc={inventory['loc_total']} batches={len(batches)} model={args.model}",
        flush=True,
    )

    def work(batch: Batch) -> BatchResult:
        prompt = build_prompt(revision, batch, files, sources)
        result: BatchResult | None = None
        for attempt in range(args.retries + 1):
            result = dispatch_batch(
                batch=batch,
                prompt=prompt,
                report_dir=report_dir,
                revision=revision,
                model=args.model,
                reviewer_template=args.reviewer_cmd,
                timeout_s=args.timeout,
            )
            if result.error is None:
                return result
            if attempt < args.retries:
                time.sleep(min(2**attempt, 10))
        assert result is not None
        return result

    if args.concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(work, batches))
    else:
        results = [work(b) for b in batches]

    failures = [r for r in results if r.error is not None]
    findings = collect_findings(results)
    write_artifacts(
        report_dir=report_dir,
        revision=revision,
        scope=args.scope,
        model=args.model,
        reviewer_template=args.reviewer_cmd,
        inventory=inventory,
        results=results,
        findings=findings,
    )
    verdict = "findings" if findings else "no-findings"
    print(
        f"batches_ok={len(results) - len(failures)}/{len(results)} "
        f"findings={len(findings)} verdict={verdict}",
        flush=True,
    )
    if failures:
        for r in failures:
            print(f"FAIL {r.batch_id}: {r.error}", file=sys.stderr)
        print(f"review run failed closed: {len(failures)} batch(es) failed", file=sys.stderr)
        return 2
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    repo = repo_root()
    revision = resolve_revision(repo, args.revision)
    snippet = "def ratio(a, b):\n    if b == 0:\n        return 0\n    return a / b\n"
    entry = {
        "path": "src/axquant/_smoke_sample.py",
        "lines": line_count(snippet.encode()),
        "sha256": sha256_hex(snippet.encode()),
    }
    batch = Batch(
        batch_id="smoke",
        files=[str(entry["path"])],
        lines=int(entry["lines"]),
        source_sha256=source_digest([entry]),
    )
    prompt = build_prompt(revision, batch, [entry], {str(entry["path"]): snippet})
    report_dir = Path(args.report)
    result = dispatch_batch(
        batch=batch,
        prompt=prompt,
        report_dir=report_dir / "smoke",
        revision=revision,
        model=args.model,
        reviewer_template=args.reviewer_cmd,
        timeout_s=args.timeout,
    )
    ok = result.error is None and result.exit_code == 0 and result.sentinel
    print(
        json.dumps(
            {
                "model": args.model,
                "revision": revision,
                "exit_code": result.exit_code,
                "sentinel": result.sentinel,
                "findings": len(result.findings),
                "duration_s": result.duration_s,
                "error": result.error,
            },
            indent=2,
        )
    )
    if not ok:
        print(f"smoke failed: {result.error}", file=sys.stderr)
        return 2
    return 0


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise ReviewError(f"missing required artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewError(f"invalid JSON in {path}: {exc}") from exc


def check_artifacts(
    *,
    repo: Path,
    report_dir: Path,
    revision: str,
    scope: str,
    bugs_dir: Path,
    require_passing: bool = True,
) -> dict[str, object]:
    """Validate every artifact against the pinned revision. Raises on failure."""
    revision = resolve_revision(repo, revision)
    inventory = _load_json(report_dir / "inventory.json")
    manifest = _load_json(report_dir / "manifest.json")
    findings_doc = _load_json(report_dir / "findings.json")
    for name, doc in (
        ("inventory.json", inventory),
        ("manifest.json", manifest),
        ("findings.json", findings_doc),
    ):
        if not isinstance(doc, dict):
            raise ReviewError(f"{name} must be a JSON object")

    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ReviewError("inventory schema mismatch")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ReviewError("manifest schema mismatch")
    if findings_doc.get("schema") != FINDINGS_SCHEMA:
        raise ReviewError("findings schema mismatch")
    if inventory.get("revision") != revision or manifest.get("revision") != revision:
        raise ReviewError(
            f"artifact revision mismatch: expected {revision}, "
            f"inventory={inventory.get('revision')} manifest={manifest.get('revision')}"
        )
    if findings_doc.get("revision") != revision:
        raise ReviewError("findings revision mismatch")
    if inventory.get("scope") != scope or manifest.get("scope") != scope:
        raise ReviewError("artifact scope mismatch")

    # --- coverage + digest recomputation from the pinned revision -------------
    expected_paths = list_scope_files(repo, revision, scope)
    inv_files = inventory.get("files")
    if not isinstance(inv_files, list):
        raise ReviewError("inventory.files must be a list")
    inv_by_path: dict[str, dict[str, object]] = {}
    for entry in inv_files:
        if not isinstance(entry, dict):
            raise ReviewError("inventory file entry must be an object")
        path = str(entry.get("path", ""))
        if path in inv_by_path:
            raise ReviewError(f"inventory lists {path} more than once")
        inv_by_path[path] = entry
    if sorted(inv_by_path) != expected_paths:
        missing = sorted(set(expected_paths) - set(inv_by_path))
        extra = sorted(set(inv_by_path) - set(expected_paths))
        raise ReviewError(
            f"inventory coverage mismatch at {revision}: missing={missing[:5]} extra={extra[:5]}"
        )

    loc_total = 0
    for path, entry in inv_by_path.items():
        raw = read_blob(repo, revision, path)
        real_lines = line_count(raw)
        real_sha = sha256_hex(raw)
        if int(entry.get("lines", -1)) != real_lines:
            raise ReviewError(f"{path}: line count {entry.get('lines')} != {real_lines}")
        if entry.get("sha256") != real_sha:
            raise ReviewError(f"{path}: blob sha256 mismatch (revision drift)")
        loc_total += real_lines
    if int(inventory.get("loc_total", -1)) != loc_total:
        raise ReviewError(f"inventory loc_total {inventory.get('loc_total')} != {loc_total}")
    if int(inventory.get("file_count", -1)) != len(expected_paths):
        raise ReviewError("inventory file_count mismatch")
    if int(manifest.get("loc_total", -1)) != loc_total:
        raise ReviewError("manifest loc_total mismatch")
    if int(manifest.get("file_count", -1)) != len(expected_paths):
        raise ReviewError("manifest file_count mismatch")

    # --- batch status + prompt/source digests ---------------------------------
    batches = manifest.get("batches")
    if not isinstance(batches, list):
        raise ReviewError("manifest.batches must be a list")
    seen_batch_files: list[str] = []
    for batch in batches:
        if not isinstance(batch, dict):
            raise ReviewError("manifest batch entry must be an object")
        batch_id = str(batch.get("batch_id", "?"))
        if batch.get("exit_code") != 0:
            raise ReviewError(f"batch {batch_id}: exit_code {batch.get('exit_code')} != 0")
        if batch.get("sentinel") is not True:
            raise ReviewError(f"batch {batch_id}: missing completion sentinel")
        prompt_rel = str(batch.get("prompt_file", ""))
        prompt_path = report_dir / prompt_rel
        if not prompt_rel or not prompt_path.is_file():
            raise ReviewError(f"batch {batch_id}: missing prompt artifact")
        if sha256_hex(prompt_path.read_bytes()) != batch.get("prompt_sha256"):
            raise ReviewError(f"batch {batch_id}: prompt sha256 mismatch")
        batch_files = batch.get("files")
        if not isinstance(batch_files, list) or not batch_files:
            raise ReviewError(f"batch {batch_id}: empty file list")
        seen_batch_files.extend(str(f) for f in batch_files)
        entries = [inv_by_path.get(str(f)) for f in batch_files]
        if any(e is None for e in entries):
            raise ReviewError(f"batch {batch_id}: cites a file absent from inventory")
        digest = source_digest([e for e in entries if e is not None])
        if digest != batch.get("source_sha256"):
            raise ReviewError(f"batch {batch_id}: source sha256 mismatch")
        if int(batch.get("lines", -1)) != sum(int(e["lines"]) for e in entries if e):
            raise ReviewError(f"batch {batch_id}: line total mismatch")
        for f in batch_files:
            if inv_by_path[str(f)].get("batch_id") != batch_id:
                raise ReviewError(f"{f}: inventory batch_id != {batch_id}")
    if sorted(seen_batch_files) != expected_paths:
        raise ReviewError("batches do not partition the reviewed files exactly once")

    # --- findings + citations -------------------------------------------------
    findings = findings_doc.get("findings")
    if not isinstance(findings, list):
        raise ReviewError("findings.findings must be a list")
    verdict = findings_doc.get("verdict")
    if verdict not in ("findings", "no-findings"):
        raise ReviewError(f"invalid overall verdict {verdict!r}")
    if bool(findings) != (verdict == "findings"):
        raise ReviewError("verdict disagrees with findings list")
    seen_ids: set[str] = set()
    confirmed = 0
    for finding in findings:
        if not isinstance(finding, dict):
            raise ReviewError("finding entry must be an object")
        fid = str(finding.get("id", "?"))
        if fid in seen_ids:
            raise ReviewError(f"duplicate finding id {fid}")
        seen_ids.add(fid)
        file = str(finding.get("file", ""))
        if file not in inv_by_path:
            raise ReviewError(f"{fid}: file {file!r} is not a reviewed file")
        entry = inv_by_path[file]
        total_lines = int(entry["lines"])
        line = int(finding.get("line", 0))
        if not 1 <= line <= total_lines:
            raise ReviewError(f"{fid}: line {line} out of range 1..{total_lines} for {file}")
        src_line = read_blob(repo, revision, file).decode("utf-8", "replace").splitlines()[line - 1]
        if _norm(str(finding.get("snippet", ""))) not in _norm(src_line):
            raise ReviewError(f"{fid}: snippet does not match {file}:{line} at {revision}")
        if str(finding.get("severity", "")) not in SEVERITIES:
            raise ReviewError(f"{fid}: invalid severity {finding.get('severity')!r}")
        if not str(finding.get("mechanism", "")).strip():
            raise ReviewError(f"{fid}: empty mechanism")
        triage = str(finding.get("triage", ""))
        if triage not in VERDICTS:
            raise ReviewError(f"{fid}: invalid triage verdict {triage!r}")
        if triage == "confirmed":
            confirmed += 1
            _validate_bug_doc(repo, revision, inv_by_path, bugs_dir, fid, finding)

    # --- report.md must list every finding with its verdict -------------------
    report = (report_dir / "report.md").read_text(encoding="utf-8")
    if revision not in report:
        raise ReviewError("report.md does not cite the pinned revision")
    if not findings and "NO_FINDINGS" not in report:
        raise ReviewError("report.md lacks the explicit NO_FINDINGS verdict")
    for finding in findings:
        fid = str(finding["id"])
        if f"### {fid} " not in report:
            raise ReviewError(f"report.md does not list finding {fid}")
        if f"- Triage: {finding['triage']}" not in report:
            raise ReviewError(f"report.md missing triage verdict for {fid}")

    return {
        "revision": revision,
        "scope": scope,
        "files": len(expected_paths),
        "loc_total": loc_total,
        "batches": len(batches),
        "findings": len(findings),
        "confirmed": confirmed,
        "snippet_mismatches": 0,
    }


def _validate_bug_doc(
    repo: Path,
    revision: str,
    inv_by_path: dict[str, dict[str, object]],
    bugs_dir: Path,
    fid: str,
    finding: dict[str, object],
) -> None:
    doc_rel = str(finding.get("bug_doc") or "")
    if not doc_rel:
        raise ReviewError(f"{fid}: confirmed finding has no bug_doc")
    doc_path = Path(doc_rel)
    if not doc_path.is_absolute():
        doc_path = repo / doc_path
    bugs_root = bugs_dir if bugs_dir.is_absolute() else repo / bugs_dir
    if bugs_root not in doc_path.parents:
        raise ReviewError(f"{fid}: bug_doc {doc_rel} is outside {bugs_dir}")
    if not doc_path.is_file():
        raise ReviewError(f"{fid}: bug_doc {doc_rel} does not exist")
    text = doc_path.read_text(encoding="utf-8")
    for required in (
        "Classification: confirmed",
        "## Evidence",
        "## Suggested Fix",
        "## Reproduction",
    ):
        if required not in text:
            raise ReviewError(f"{fid}: {doc_rel} missing required section {required!r}")
    citations = _CITATION_RE.findall(text)
    if not citations:
        raise ReviewError(f"{fid}: {doc_rel} quotes no src/axquant file:line citation")
    for path, start, end in citations:
        if path not in inv_by_path:
            raise ReviewError(f"{fid}: {doc_rel} cites unknown file {path}")
        total = int(inv_by_path[path]["lines"])
        start_i = int(start)
        end_i = int(end) if end else start_i
        if not 1 <= start_i <= end_i <= total:
            raise ReviewError(f"{fid}: {doc_rel} cites {path}:{start}-{end} outside 1..{total}")
    # The finding's own citation must also match the pinned revision.
    src = read_blob(repo, revision, str(finding["file"])).decode("utf-8", "replace")
    line_text = src.splitlines()[int(finding["line"]) - 1]
    if _norm(str(finding["snippet"])) not in _norm(line_text):
        raise ReviewError(f"{fid}: bug_doc-backed snippet does not match {revision}")


def cmd_check(args: argparse.Namespace) -> int:
    repo = repo_root()
    try:
        summary = check_artifacts(
            repo=repo,
            report_dir=Path(args.report),
            revision=args.revision,
            scope=args.scope,
            bugs_dir=Path(args.bugs_dir),
        )
    except ReviewError as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    print(
        f"check ok: files={summary['files']} loc={summary['loc_total']} "
        f"batches={summary['batches']} findings={summary['findings']} "
        f"confirmed={summary['confirmed']} snippet_mismatches=0"
    )
    return 0


# --------------------------------------------------------------------------- #
# triage
# --------------------------------------------------------------------------- #


def cmd_triage(args: argparse.Namespace) -> int:
    repo = repo_root()
    revision = resolve_revision(repo, args.revision)
    report_dir = Path(args.report)
    findings_doc = _load_json(report_dir / "findings.json")
    inventory = _load_json(report_dir / "inventory.json")
    manifest = _load_json(report_dir / "manifest.json")
    if not isinstance(findings_doc, dict) or not isinstance(inventory, dict):
        raise ReviewError("artifact shape error")
    if not isinstance(manifest, dict):
        raise ReviewError("manifest shape error")
    triage = _load_json(Path(args.triage))
    if not isinstance(triage, dict) or not isinstance(triage.get("entries"), dict):
        raise ReviewError("triage.json must be an object with an 'entries' mapping")
    entries: dict[str, object] = triage["entries"]
    findings = findings_doc.get("findings")
    if not isinstance(findings, list):
        raise ReviewError("findings.findings must be a list")
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        fid = str(finding.get("id", ""))
        entry = entries.get(fid)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ReviewError(f"{fid}: triage entry must be an object")
        verdict = str(entry.get("verdict", ""))
        if verdict not in VERDICTS:
            raise ReviewError(f"{fid}: invalid triage verdict {verdict!r}")
        finding["triage"] = verdict
        finding["triage_note"] = str(entry.get("note", ""))
        finding["bug_doc"] = entry.get("bug_doc")
    findings_doc["findings"] = findings
    write_json(report_dir / "findings.json", findings_doc)
    (report_dir / "report.md").write_text(
        render_report(
            revision=revision,
            model=str(manifest.get("model", DEFAULT_MODEL)),
            inventory=inventory,
            manifest=manifest,
            findings=findings,
        ),
        encoding="utf-8",
    )
    print(f"triage applied to {len(findings)} finding(s)")
    return 0


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    # Options are accepted after the subcommand (e.g. `check --revision <sha>`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--revision", default="HEAD", help="pinned git revision")
    common.add_argument("--scope", default=DEFAULT_SCOPE, help="path scope to review")
    common.add_argument("--report", default=DEFAULT_REPORT, help="artifact directory")
    common.add_argument("--model", default=DEFAULT_MODEL, help="reviewer model id")
    common.add_argument(
        "--reviewer-cmd",
        default=DEFAULT_REVIEWER_CMD,
        help="reviewer command template (placeholders: {model} {title} {prompt_file})",
    )
    common.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-batch timeout in seconds"
    )

    parser = argparse.ArgumentParser(description="GLM-driven, provenance-bound code review")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", parents=[common], help="run the full review and write artifacts")
    p_run.add_argument("--line-budget", type=int, default=DEFAULT_LINE_BUDGET)
    p_run.add_argument("--concurrency", type=int, default=1)
    p_run.add_argument("--retries", type=int, default=1, help="retries per failed batch")
    p_run.set_defaults(func=cmd_run)

    p_smoke = sub.add_parser(
        "smoke", parents=[common], help="single live round-trip through the reviewer"
    )
    p_smoke.set_defaults(func=cmd_smoke)

    p_check = sub.add_parser(
        "check", parents=[common], help="validate artifacts against the pinned revision"
    )
    p_check.add_argument("--bugs-dir", default=DEFAULT_BUGS_DIR)
    p_check.set_defaults(func=cmd_check)

    p_triage = sub.add_parser(
        "triage", parents=[common], help="apply local triage verdicts and re-render report"
    )
    p_triage.add_argument("--triage", default=DEFAULT_REPORT + "/triage.json")
    p_triage.set_defaults(func=cmd_triage)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ReviewError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
