#!/usr/bin/env python3
"""Audit every published AXQ MTP Hub pack under its architecture-specific contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from axquant.hub_mtp_audit import (
    MtpHubAuditResult,
    MtpHubPackKind,
    audit_mtp_hub_snapshot,
    discover_axq_mtp_artifact_repositories,
    discover_axq_mtp_repositories,
    load_mtp_hub_snapshot,
)
from axquant.serde import write_data

_TRANSIENT_ISSUE_PREFIX = "[transient] "


def _classify_error(error: Exception) -> str:
    """Return ``"transient"`` for retryable Hub errors, else ``"real"``."""

    if isinstance(error, HfHubHTTPError):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        if status == 429 or (isinstance(status, int) and status >= 500):
            return "transient"
        return "real"
    if isinstance(error, (ConnectionError, TimeoutError)):
        return "transient"
    return "real"


def _is_transient_result(result: MtpHubAuditResult) -> bool:
    """A repository result is transient when its snapshot could not be loaded
    at all (no revision pinned) AND the only issue carries the transient
    marker. Real audit failures keep their original messages.
    """

    return bool(result.issues) and all(
        issue.startswith(_TRANSIENT_ISSUE_PREFIX) for issue in result.issues
    )


def _failed_result(repo_id: str, error: Exception, *, transient: bool) -> MtpHubAuditResult:
    prefix = _TRANSIENT_ISSUE_PREFIX if transient else ""
    return MtpHubAuditResult(
        repo_id=repo_id,
        revision="unknown",
        kind=MtpHubPackKind.UNKNOWN,
        issues=(f"{prefix}audit could not load repository: {type(error).__name__}: {error}",),
    )


def _print_line(result: MtpHubAuditResult) -> None:
    if _is_transient_result(result):
        print(
            f"SKIP\t{result.kind.value}\t{result.repo_id}  (audit unavailable; transient)",
            flush=True,
        )
        return
    status = "PASS" if result.passed else "FAIL"
    print(
        f"{status}\t{result.kind.value}\t{result.repo_id}@{result.revision}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", default="AutomatosX")
    parser.add_argument("--repo", action="append", default=[])
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)

    api = HfApi()
    discovery: dict[str, int] = {"name": 0, "manifest": 0, "union": 0}
    if args.repo:
        repo_ids = tuple(args.repo)
    else:
        # Union name-based and manifest-based discovery so a sidecar-bearing
        # pack whose name lacks the -mtp suffix is still audited.
        name_set = tuple(discover_axq_mtp_repositories(api, args.author))
        manifest_set = tuple(discover_axq_mtp_artifact_repositories(api, args.author))
        ordered = dict.fromkeys((*name_set, *manifest_set))
        repo_ids = tuple(ordered)
        discovery = {
            "name": len(name_set),
            "manifest": len(manifest_set),
            "union": len(repo_ids),
        }
        print(
            f"discovery: name={discovery['name']} "
            f"manifest={discovery['manifest']} union={discovery['union']}",
            flush=True,
        )

    results: list[MtpHubAuditResult] = []
    for repo_id in repo_ids:
        try:
            result = audit_mtp_hub_snapshot(load_mtp_hub_snapshot(api, repo_id))
        except Exception as exc:
            result = _failed_result(repo_id, exc, transient=_classify_error(exc) == "transient")
        results.append(result)
        _print_line(result)
        for issue in result.issues:
            print(f"  - {issue}", flush=True)

    if len(results) != len(repo_ids):
        print(
            f"ERROR\tinternal: audited {len(results)} of {len(repo_ids)} "
            "discovered repositories; refusing to continue",
            flush=True,
        )
        return 3

    transient_count = sum(1 for r in results if _is_transient_result(r))
    real_failed_count = sum(1 for r in results if not r.passed and not _is_transient_result(r))

    payload = {
        "schema_version": "axquant.mtp-hub-fleet-audit.v1",
        "author": args.author,
        "discovery": discovery,
        "repository_count": len(results),
        "passed_count": sum(item.passed for item in results),
        "failed_count": real_failed_count,
        "audit_unavailable_count": transient_count,
        "results": [item.model_dump() for item in results],
    }
    if args.json_output is not None:
        write_data(args.json_output, payload)
    print(json.dumps({key: payload[key] for key in payload if key != "results"}, sort_keys=True))
    if real_failed_count:
        return 1
    if transient_count:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
