"""Fail-closed tests for the GLM code-review harness.

These tests never call the network or the real ``ax-code`` CLI. They inject a
fake reviewer command that emits the contract-shaped output so the harness's
fail-closed behavior can be exercised deterministically:

* nonzero reviewer exit fails the run
* a missing terminal completion sentinel fails the run
* unparseable findings JSON fails the run
* a blob-digest mismatch fails ``check``
* a confirmed finding without a matching ``.internal/bugs`` document fails ``check``

The happy path runs the harness end to end against a small real scope and a
temporary bugs directory, then triages one finding to ``confirmed``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import glm_code_review as gcr

REPO = Path(__file__).resolve().parents[1]
SCOPE = "src/axquant/architectures"
LINE_BUDGET = 250

# Emits a finding for the first numbered source line of the first file in the
# prompt, so every citation the harness receives is a real line at the pinned
# revision. Behavior is selected by the GLM_TEST_MODE environment variable.
FAKE_REVIEWER = """\
import json
import os
import re
import sys

mode = os.environ.get("GLM_TEST_MODE", "ok")
sentinel = "GLM_CODE_REVIEW_COMPLETE"
begin = "BEGIN_GLM_CODE_REVIEW_JSON"
end = "END_GLM_CODE_REVIEW_JSON"

prompt_path = sys.argv[1]
prompt = open(prompt_path, encoding="utf-8").read()

findings = []
header = re.search(r"=== FILE (.+?) \\(lines=\\d+ sha256=[0-9a-f]+\\) ===", prompt)
if header:
    path = header.group(1)
    tail = prompt[header.end():]
    line = re.search(r"^\\s*(\\d+): (\\S.*)$", tail, re.M)
    if line:
        findings.append(
            {
                "severity": "low",
                "file": path,
                "line": int(line.group(1)),
                "snippet": line.group(2).strip(),
                "mechanism": "injected test finding",
                "suggested_fix": "none - test harness only",
            }
        )

if mode == "nonzero":
    sys.stderr.write("reviewer boom\\n")
    sys.exit(3)
if mode == "no_sentinel":
    print(begin)
    print(json.dumps({"findings": findings, "verdict": "findings"}))
    print(end)
    sys.exit(0)
if mode == "bad_json":
    print(begin)
    print("{not valid json")
    print(end)
    print(sentinel)
    sys.exit(0)

print(begin)
print(json.dumps({"findings": findings, "verdict": "findings" if findings else "no-findings"}))
print(end)
print(sentinel)
sys.exit(0)
"""


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.chdir(REPO)
    script = tmp_path / "fake_reviewer.py"
    script.write_text(FAKE_REVIEWER, encoding="utf-8")
    template = f"{sys.executable} {script} {{prompt_file}}"
    report = tmp_path / "report"
    bugs = tmp_path / "bugs"
    bugs.mkdir()

    def set_mode(mode: str) -> None:
        monkeypatch.setenv("GLM_TEST_MODE", mode)

    def run_argv() -> list[str]:
        return [
            "run",
            "--scope",
            SCOPE,
            "--report",
            str(report),
            "--revision",
            "HEAD",
            "--line-budget",
            str(LINE_BUDGET),
            "--reviewer-cmd",
            template,
        ]

    def check_argv() -> list[str]:
        return [
            "check",
            "--scope",
            SCOPE,
            "--report",
            str(report),
            "--revision",
            "HEAD",
            "--bugs-dir",
            str(bugs),
        ]

    return {
        "tmp": tmp_path,
        "report": report,
        "bugs": bugs,
        "template": template,
        "set_mode": set_mode,
        "run_argv": run_argv,
        "check_argv": check_argv,
    }


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_fails_closed_on_nonzero_exit(workspace: dict[str, object]) -> None:
    workspace["set_mode"]("nonzero")  # type: ignore[operator]
    rc = gcr.main(workspace["run_argv"]())  # type: ignore[operator]
    assert rc == 2
    manifest = _load(Path(workspace["report"]) / "manifest.json")  # type: ignore[arg-type]
    batches = manifest["batches"]
    assert isinstance(batches, list) and batches
    assert all(b["exit_code"] == 3 and b["sentinel"] is False for b in batches)  # type: ignore[index]


def test_run_fails_closed_on_missing_sentinel(workspace: dict[str, object]) -> None:
    workspace["set_mode"]("no_sentinel")  # type: ignore[operator]
    rc = gcr.main(workspace["run_argv"]())  # type: ignore[operator]
    assert rc == 2
    manifest = _load(Path(workspace["report"]) / "manifest.json")  # type: ignore[arg-type]
    assert all(b["sentinel"] is False for b in manifest["batches"])  # type: ignore[union-attr,index]


def test_run_fails_closed_on_unparseable_json(workspace: dict[str, object]) -> None:
    workspace["set_mode"]("bad_json")  # type: ignore[operator]
    rc = gcr.main(workspace["run_argv"]())  # type: ignore[operator]
    assert rc == 2


def test_parse_contract_requires_terminal_sentinel() -> None:
    body = f'{gcr.JSON_BEGIN}\n{{"findings": [], "verdict": "no-findings"}}\n{gcr.JSON_END}\n'
    with pytest.raises(gcr.ReviewError, match="sentinel"):
        gcr.parse_contract(body)
    findings, verdict = gcr.parse_contract(body + gcr.SENTINEL + "\n")
    assert findings == []
    assert verdict == "no-findings"


def test_check_fails_closed_on_blob_digest_mismatch(workspace: dict[str, object]) -> None:
    workspace["set_mode"]("ok")  # type: ignore[operator]
    assert gcr.main(workspace["run_argv"]()) == 0  # type: ignore[operator]
    inventory_path = Path(workspace["report"]) / "inventory.json"  # type: ignore[arg-type]
    inventory = _load(inventory_path)
    inventory["files"][0]["sha256"] = "0" * 64
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(gcr.ReviewError, match="sha256"):
        gcr.check_artifacts(
            repo=REPO,
            report_dir=Path(workspace["report"]),  # type: ignore[arg-type]
            revision="HEAD",
            scope=SCOPE,
            bugs_dir=Path(workspace["bugs"]),  # type: ignore[arg-type]
        )


def test_check_fails_closed_on_confirmed_without_bug_doc(workspace: dict[str, object]) -> None:
    workspace["set_mode"]("ok")  # type: ignore[operator]
    assert gcr.main(workspace["run_argv"]()) == 0  # type: ignore[operator]
    findings_path = Path(workspace["report"]) / "findings.json"  # type: ignore[arg-type]
    doc = _load(findings_path)
    assert doc["findings"], "fake reviewer should produce at least one finding"
    doc["findings"][0]["triage"] = "confirmed"
    findings_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(gcr.ReviewError, match="bug_doc"):
        gcr.check_artifacts(
            repo=REPO,
            report_dir=Path(workspace["report"]),  # type: ignore[arg-type]
            revision="HEAD",
            scope=SCOPE,
            bugs_dir=Path(workspace["bugs"]),  # type: ignore[arg-type]
        )


def test_happy_path_run_triage_and_check(workspace: dict[str, object]) -> None:
    workspace["set_mode"]("ok")  # type: ignore[operator]
    assert gcr.main(workspace["run_argv"]()) == 0  # type: ignore[operator]
    report = Path(workspace["report"])  # type: ignore[arg-type]
    doc = _load(report / "findings.json")
    finding = doc["findings"][0]
    assert finding["file"].startswith("src/axquant/")

    bugs = Path(workspace["bugs"])  # type: ignore[arg-type]
    bug_doc = bugs / "injected-fixture.md"
    bug_doc.write_text(
        "# Injected fixture finding\n\n"
        "Classification: confirmed\n\n"
        "## Evidence\n\n"
        f"- {finding['file']}:{finding['line']} quotes `{finding['snippet']}` and is "
        "recomputed from the pinned revision by the harness checker.\n\n"
        "## Suggested Fix\n\n"
        "No product fix; this document only exercises the checker.\n\n"
        "## Reproduction\n\n"
        "Reproduced by the fake reviewer in tests/test_glm_code_review.py; the "
        "checker accepts the citation because it matches the pinned blob.\n",
        encoding="utf-8",
    )
    triage_path = report / "triage.json"
    triage_path.write_text(
        json.dumps({"entries": {finding["id"]: {"verdict": "confirmed", "bug_doc": str(bug_doc)}}}),
        encoding="utf-8",
    )
    triage_rc = gcr.main(
        [
            "triage",
            "--report",
            str(report),
            "--triage",
            str(triage_path),
            "--revision",
            "HEAD",
        ]
    )
    assert triage_rc == 0
    updated = _load(report / "findings.json")
    assert updated["findings"][0]["triage"] == "confirmed"

    assert gcr.main(workspace["check_argv"]()) == 0  # type: ignore[operator]
    summary = gcr.check_artifacts(
        repo=REPO,
        report_dir=report,
        revision="HEAD",
        scope=SCOPE,
        bugs_dir=bugs,
    )
    assert summary["findings"] >= 1
    assert summary["confirmed"] == 1
    assert summary["snippet_mismatches"] == 0
