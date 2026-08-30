#!/usr/bin/env python3
"""Audit every published AXQ MTP Hub pack under its architecture-specific contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

from axquant.hub_mtp_audit import (
    MtpHubAuditResult,
    MtpHubPackKind,
    audit_mtp_hub_snapshot,
    discover_axq_mtp_repositories,
    load_mtp_hub_snapshot,
)
from axquant.serde import write_data


def _failed_result(repo_id: str, error: Exception) -> MtpHubAuditResult:
    return MtpHubAuditResult(
        repo_id=repo_id,
        revision="unknown",
        kind=MtpHubPackKind.UNKNOWN,
        issues=(f"audit could not load repository: {type(error).__name__}: {error}",),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", default="AutomatosX")
    parser.add_argument("--repo", action="append", default=[])
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)

    api = HfApi()
    repo_ids = tuple(args.repo) or discover_axq_mtp_repositories(api, args.author)
    results: list[MtpHubAuditResult] = []
    for repo_id in repo_ids:
        try:
            result = audit_mtp_hub_snapshot(load_mtp_hub_snapshot(api, repo_id))
        except Exception as exc:
            result = _failed_result(repo_id, exc)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status}\t{result.kind.value}\t{result.repo_id}@{result.revision}",
            flush=True,
        )
        for issue in result.issues:
            print(f"  - {issue}", flush=True)

    payload = {
        "schema_version": "axquant.mtp-hub-fleet-audit.v1",
        "author": args.author,
        "repository_count": len(results),
        "passed_count": sum(item.passed for item in results),
        "failed_count": sum(not item.passed for item in results),
        "results": [item.model_dump() for item in results],
    }
    if args.json_output is not None:
        write_data(args.json_output, payload)
    print(json.dumps({key: payload[key] for key in payload if key != "results"}, sort_keys=True))
    return 1 if payload["failed_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
