#!/usr/bin/env python3
"""Merge multi-host Ext4T package indexes; report dups and a final layout plan.

Content identity for delete / alias renames is proven only by
``fingerprint_mode == "content"`` records from the indexer (full-file
sha256 per path). cheap, incomplete, path-size/manifest, and legacy
records are never used to recommend delete_duplicate.

Usage:
  python3 scripts/plan-ext4t-from-index.py \\
    --index /Volumes/Ext4T/logs/index/m3.jsonl \\
    --index /Volumes/Ext4T/logs/index/m5.jsonl \\
    --index /Volumes/Ext4T/logs/index/m2u.jsonl \\
    -o /Volumes/Ext4T/logs/index/plan.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Fingerprint modes that prove byte-identical package content. Only these
# may drive delete_duplicate / fingerprint-based alias renames.
_CONTENT_SAFE_MODES = frozenset({"content"})
# Modes that must never participate in content-identity grouping at all.
_NON_IDENTITY_MODES = frozenset({"cheap", "incomplete"})


def fingerprint_is_content_safe(rec: dict) -> bool:
    """True when the record's fingerprint proves full-file content identity."""
    mode = rec.get("fingerprint_mode")
    if mode not in _CONTENT_SAFE_MODES:
        return False
    fp = rec.get("manifest_sha256", "")
    # Defensive: indexer prefixes incomplete digests even if mode drifts.
    return not (isinstance(fp, str) and fp.startswith(("cheap:", "incomplete:")))


def content_identity_key(rec: dict) -> str:
    """Key used to group packages that share proven-identical content.

    Content-safe fingerprints group by digest. Weak fingerprints (cheap,
    incomplete, path-size/manifest, missing mode) each get a unique key so
    they never collide into a false duplicate group.
    """
    if fingerprint_is_content_safe(rec):
        return str(rec["manifest_sha256"])
    # Isolate weak/legacy records: same path+size layout must not imply
    # identical weights, and cheap/incomplete must never group for deletes.
    return f"unverified:{rec['host']}:{rec['path']}"


# Absolute root for Ext4T fleet layouts. Paths in indexes are absolute under this.
_EXT4T_ROOT = "/Volumes/Ext4T"


def final_absolute_path(final: str) -> str:
    """Absolute path for a relative final layout location (e.g. models/Foo)."""
    return f"{_EXT4T_ROOT}/{final}"


def is_at_final_path(path: str, final: str) -> bool:
    """True when *path* is exactly the fleet final location for *final*.

    Must not use endswith('/' + final): both ``models/Foo`` and
    ``axquant/models/Foo`` end with ``/models/Foo``, and treating the
    nested copy as already-final would keep the stray path and schedule
    delete_duplicate on the real top-level models tree.
    """
    return path == final_absolute_path(final)


def _category_matches(categories: set[str], prefix: str) -> bool:
    return any(c == prefix or c.startswith(prefix + "/") for c in categories)


# Preferred final locations by category heuristics
def preferred_location(name: str, categories: set[str], size: int) -> str:
    n = name.lower()

    def has(prefix: str) -> bool:
        return _category_matches(categories, prefix)

    # Host-local / reserved trees win over name heuristics so e.g.
    # axquant/logs/smoke-nightly stays under logs (not smokes/work).
    if has("axquant/logs"):
        return f"axquant/logs/{name}"
    if has("axquant/scripts"):
        return f"axquant/scripts/{name}"
    if has("axquant/axq-publish") or name.startswith("AX-"):
        return f"axquant/axq-publish/{name}"
    if has("axquant/smokes") or "smoke" in n or n.endswith("-test"):
        return f"axquant/smokes/{name}"
    if has("axquant/certification"):
        return f"axquant/certification/{name}"
    if has("models") or has("axquant/models"):
        # BF16 / factory sources stay top-level models, even when a stray
        # copy also sits under work/
        return f"models/{name}"
    if has("axquant/work") or "candidate" in n or "prep" in n or "tmp" in n:
        # keep grouping if present
        for c in sorted(categories):
            if c.startswith("axquant/work/"):
                return f"{c}/{name}"
        return f"axquant/work/{name}"
    if has("huggingface"):
        return f"huggingface/{name}"
    # default
    if size > 1_000_000_000 and ("bf16" in n or "mlx" in n or "model" in n):
        return f"models/{name}"
    return f"axquant/work/{name}"


def human(n: int) -> str:
    for unit, div in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n}B"


_REQUIRED_RECORD_KEYS = (
    "host",
    "category",
    "name",
    "path",
    "size_bytes",
    "file_count",
    "manifest_sha256",
)


def load_indexes(paths: list[Path]) -> list[dict]:
    recs: list[dict] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"ERROR: {p}:{line_number}: invalid JSON ({exc.msg}); the index "
                        "looks truncated or corrupt — re-run the indexer for this host"
                    ) from exc
                if not isinstance(obj, dict):
                    raise SystemExit(
                        f"ERROR: {p}:{line_number}: expected a JSON object, "
                        f"got {type(obj).__name__}"
                    )
                if obj.get("type") == "index_meta":
                    continue
                mode = obj.get("fingerprint_mode")
                # cheap HF rollups and incomplete walks never prove content
                # identity — drop them from the package plan entirely so they
                # cannot contribute to delete_duplicate or alias merges.
                if mode in _NON_IDENTITY_MODES:
                    continue
                missing = [key for key in _REQUIRED_RECORD_KEYS if key not in obj]
                if missing:
                    raise SystemExit(
                        f"ERROR: {p}:{line_number}: record is missing {missing}; "
                        "re-run the indexer for this host"
                    )
                recs.append(obj)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="append", required=True, help="JSONL index (repeatable)")
    ap.add_argument("-o", "--output", required=True, help="plan markdown path")
    ap.add_argument("--json-out", help="optional machine-readable plan JSON")
    args = ap.parse_args()

    recs = load_indexes([Path(p) for p in args.index])
    if not recs:
        print("no package records", file=sys.stderr)
        return 2

    # Group by proven content identity (weak fingerprints never merge)
    by_fp: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_fp[content_identity_key(r)].append(r)

    # Group by name (any content)
    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_name[r["name"]].append(r)

    hosts = sorted({r["host"] for r in recs})
    unverified_count = sum(1 for r in recs if not fingerprint_is_content_safe(r))

    # Unique content packages
    unique_content: list[dict] = []
    duplicate_groups: list[dict] = []
    for fp, group in sorted(by_fp.items(), key=lambda x: -x[1][0]["size_bytes"]):
        size = group[0]["size_bytes"]
        names = sorted({g["name"] for g in group})
        paths = [(g["host"], g["path"], g["category"]) for g in group]
        host_set = sorted({g["host"] for g in group})
        cats = {g["category"] for g in group}
        content_safe = all(fingerprint_is_content_safe(g) for g in group)
        # pick canonical name: prefer non-source slug / HF-style / AX- publish names
        canonical_name = _pick_canonical_name(names)
        final = preferred_location(canonical_name, cats, size)
        # Logs are host-specific operational records: they mutate between
        # index runs, so even a content-safe snapshot is not a durable
        # fleet identity. Prefer category over final path so a mis-bucketed
        # name heuristic cannot reclassify a log tree as fleet-syncable.
        host_local = final.startswith("axquant/logs/") or _category_matches(cats, "axquant/logs")
        entry = {
            "manifest_sha256": group[0]["manifest_sha256"],
            "identity_key": fp,
            "size_bytes": size,
            "file_count": group[0]["file_count"],
            "names": names,
            "canonical_name": canonical_name,
            "final_path": final,
            "host_local": host_local,
            # Only full-file content fingerprints may drive destructive
            # same-host reclaim or alias renames.
            "content_safe": content_safe,
            "fingerprint_mode": group[0].get("fingerprint_mode", "legacy"),
            "hosts": host_set,
            "instances": [{"host": h, "path": p, "category": c} for h, p, c in paths],
            "instance_count": len(group),
            "multi_host": len(host_set) > 1,
            "multi_name": len(names) > 1 and content_safe,
            "same_host_dup": len(group) > len(host_set) and content_safe,
        }
        if len(group) > 1 and content_safe:
            duplicate_groups.append(entry)
        unique_content.append(entry)

    # Name collisions with different content
    name_conflicts = []
    for name, group in sorted(by_name.items()):
        fps = {g["manifest_sha256"] for g in group}
        if len(fps) > 1:
            name_conflicts.append(
                {
                    "name": name,
                    "variants": [
                        {
                            "host": g["host"],
                            "path": g["path"],
                            "size": g["size_bytes"],
                            "fp": g["manifest_sha256"][:16],
                            "category": g["category"],
                        }
                        for g in group
                    ],
                }
            )

    # Transfer plan: for each unique content, need one copy at final_path on
    # each host. Prefer a host already at final_path; else host_priority order.
    # Peer/index host labels (SSH/index short names). Keep historical mbp-m5
    # for existing indexes; formal certification id df-macbookpro-m5 ranks
    # the same when operators reindex under the new name.
    host_priority = [
        "macstudio-m2u",
        "m3",
        "local",
        "df-macbookpro-m5",
        "mbp-m5",
        "AKDF-M3-MAX",
        "m5",
    ]

    def host_rank(h: str) -> int:
        return host_priority.index(h) if h in host_priority else len(host_priority)

    conflicted_names = {c["name"] for c in name_conflicts}
    transfers = []
    planned_targets: set[tuple[str, str]] = set()
    deleted_paths: set[tuple[str, str]] = set()
    local_moves = []
    reclaimable = 0
    for e in unique_content:
        final = e["final_path"]
        # instances already at exact final path (not nested lookalikes)
        already = [inst for inst in e["instances"] if is_at_final_path(inst["path"], final)]
        desired = final_absolute_path(final)
        # Reserve destinations already occupied so later renames/transfers
        # cannot schedule a second writer into the same absolute path.
        for inst in already:
            planned_targets.add((inst["host"], desired.lower()))

        # same-host dups reclaim — only when every instance has a content-safe
        # fingerprint. cheap / incomplete / path-size / legacy fingerprints
        # never justify delete_duplicate (false collisions on size layout).
        # Host-local logs are also excluded: even a content-safe snapshot is
        # not a durable identity for mutable operational trees.
        by_h: dict[str, list] = defaultdict(list)
        for inst in e["instances"]:
            by_h[inst["host"]].append(inst)
        for h, insts in by_h.items():
            if len(insts) > 1 and not e["host_local"] and e["content_safe"]:
                # keep the copy already at the final path, else one whose
                # basename matches the canonical name, else the first
                keep = None
                for inst in insts:
                    if is_at_final_path(inst["path"], final):
                        keep = inst
                        break
                if keep is None:
                    for inst in insts:
                        if Path(inst["path"]).name == e["canonical_name"]:
                            keep = inst
                            break
                if keep is None:
                    keep = insts[0]
                for inst in insts:
                    if inst is keep:
                        continue
                    local_moves.append(
                        {
                            "action": "delete_duplicate",
                            "host": h,
                            "path": inst["path"],
                            "keep": keep["path"],
                            "size_bytes": e["size_bytes"],
                            "reason": (
                                "same content-safe fingerprint on same host "
                                f"(mode={e['fingerprint_mode']})"
                            ),
                        }
                    )
                    deleted_paths.add((h, inst["path"]))
                    reclaimable += e["size_bytes"]

        # cross-host: hosts missing this content. Logs stay host-local.
        # Transfers also require content-safe fingerprints — otherwise a
        # path-size/legacy index would invent one "unique" package per host
        # and spuriously schedule fleet copies of unproven content.
        if not e["host_local"] and e["content_safe"]:
            present_hosts = set(e["hosts"])
            for h in hosts:
                if h not in present_hosts:
                    if e["canonical_name"] in conflicted_names:
                        # same name exists with different content somewhere in
                        # the fleet — transferring would clobber; leave for the
                        # manual-review section
                        continue
                    # prefer a source already at the final path, else one whose
                    # basename matches the canonical name, else any instance;
                    # break ties by host_priority order.
                    candidates = (
                        already
                        or [
                            inst
                            for inst in e["instances"]
                            if Path(inst["path"]).name == e["canonical_name"]
                        ]
                        or e["instances"]
                    )
                    src = min(candidates, key=lambda inst: host_rank(inst["host"]))
                    to_path = desired
                    if (h, to_path.lower()) in planned_targets:
                        # another content already planned onto this target path
                        continue
                    planned_targets.add((h, to_path.lower()))
                    transfers.append(
                        {
                            "content": e["canonical_name"],
                            "size_bytes": e["size_bytes"],
                            "from_host": src["host"],
                            "from_path": src["path"],
                            "to_host": h,
                            "to_path": to_path,
                        }
                    )

        # local rename if present but wrong path. Skip when the basename is a
        # name-conflict (different content shares the name) or when another
        # action already claims the destination — both would clobber.
        name_conflicted = e["canonical_name"] in conflicted_names
        for inst in e["instances"]:
            if (inst["host"], inst["path"]) in deleted_paths:
                continue  # already scheduled for deletion as a same-host dup
            if inst["path"] == desired:
                continue
            dest_key = (inst["host"], desired.lower())
            if name_conflicted:
                continue  # leave for the name-conflict manual-review section
            if dest_key in planned_targets:
                continue  # destination already occupied or claimed
            if Path(inst["path"]).name == e["canonical_name"]:
                # wrong parent only
                local_moves.append(
                    {
                        "action": "rename_or_move",
                        "host": inst["host"],
                        "from": inst["path"],
                        "to": desired,
                        "size_bytes": e["size_bytes"],
                        "reason": "align to final layout (same basename)",
                    }
                )
                planned_targets.add(dest_key)
            elif (
                e["multi_name"]
                # alias grouping requires a content-safe fingerprint; never
                # rename on cheap/incomplete/path-size collisions, and never
                # rename mutable host-local log trees
                and e["content_safe"]
                and not e["host_local"]
            ):
                # alias name — rename to canonical if this host is source of truth
                local_moves.append(
                    {
                        "action": "rename_to_canonical",
                        "host": inst["host"],
                        "from": inst["path"],
                        "to": desired,
                        "size_bytes": e["size_bytes"],
                        "reason": f"alias names {e['names']} -> {e['canonical_name']}",
                    }
                )
                planned_targets.add(dest_key)

    # Summaries
    total_unique = sum(e["size_bytes"] for e in unique_content)
    multi_host = [e for e in unique_content if e["multi_host"]]
    multi_name = [e for e in unique_content if e["multi_name"]]
    same_host_dups = [e for e in unique_content if e["same_host_dup"]]

    # Naive transfer volume if we only ship missing packages (one copy each)
    transfer_bytes = sum(t["size_bytes"] for t in transfers)
    host_local_entries = [e for e in unique_content if e["host_local"]]
    host_local_bytes = sum(e["size_bytes"] for e in host_local_entries)

    lines: list[str] = []
    lines.append("# Ext4T fleet index plan")
    lines.append("")
    lines.append(f"Hosts indexed: {', '.join(hosts)}")
    lines.append(f"Unique content packages: **{len(unique_content)}** ({human(total_unique)})")
    lines.append(f"Content present on multiple hosts: **{len(multi_host)}**")
    lines.append(f"Same content under multiple names: **{len(multi_name)}**")
    lines.append(f"Same-host duplicate instances: **{len(same_host_dups)}**")
    lines.append(f"Estimated reclaimable (same-host dups): **{human(reclaimable)}**")
    lines.append(
        f"Estimated residual transfer (missing packages only): **{human(transfer_bytes)}**"
    )
    lines.append(f"Name conflicts (same name, different content): **{len(name_conflicts)}**")
    lines.append(
        f"Host-local log packages (kept in place): **{len(host_local_entries)}** "
        f"({human(host_local_bytes)})"
    )
    lines.append(
        f"Unverified fingerprints (path-size/legacy; not used for deletes): **{unverified_count}**"
    )
    lines.append("")
    lines.append(
        "Delete/alias actions require `fingerprint_mode=content` (full-file "
        "sha256). cheap, incomplete, and path-size fingerprints never produce "
        "`delete_duplicate`."
    )
    lines.append("")
    lines.append("## Final layout standard")
    lines.append("")
    lines.append("```")
    lines.append("/Volumes/Ext4T/")
    lines.append("  huggingface/           # HF cache only (do not mix models here)")
    lines.append("  models/                # BF16 / factory sources (canonical names)")
    lines.append("  axquant/")
    lines.append("    axq-publish/         # published AX-* outputs")
    lines.append("    work/                # prep, candidates, temps")
    lines.append("    smokes/              # smoke candidates")
    lines.append("    certification/       # cert evidence")
    lines.append("    logs/                # host-local run logs (never fleet-synced)")
    lines.append("  logs/                  # host-local migration / sync / index logs")
    lines.append("```")
    lines.append("")
    lines.append("## Same content, multiple names (rename instead of re-copy)")
    lines.append("")
    if multi_name:
        lines.append("| Size | Canonical | Aliases | Hosts | Final path |")
        lines.append("|------|-----------|---------|-------|------------|")
        for e in sorted(multi_name, key=lambda x: -x["size_bytes"]):
            lines.append(
                f"| {human(e['size_bytes'])} | `{e['canonical_name']}` | "
                f"{', '.join(f'`{n}`' for n in e['names'] if n != e['canonical_name']) or '—'} | "
                f"{', '.join(e['hosts'])} | `{e['final_path']}` |"
            )
    else:
        lines.append("_None found._")
    lines.append("")
    lines.append("## Same-host duplicates (delete extras after verify)")
    lines.append("")
    lines.append(
        "Only emitted when every instance shares a **content-safe** fingerprint "
        "(`fingerprint_mode=content`). Re-run the indexer without "
        "`--fingerprint path-size` if this section is empty but you expect dups."
    )
    lines.append("")
    dups = [m for m in local_moves if m["action"] == "delete_duplicate"]
    if dups:
        lines.append("| Host | Size | Delete | Keep |")
        lines.append("|------|------|--------|------|")
        for m in sorted(dups, key=lambda x: -x["size_bytes"]):
            lines.append(
                f"| {m['host']} | {human(m['size_bytes'])} | `{m['path']}` | `{m['keep']}` |"
            )
    else:
        lines.append("_None found (or already unique per host)._")
    lines.append("")
    lines.append("## Residual transfers (content missing on a host)")
    lines.append("")
    if transfers:
        lines.append("| Size | Content | From | To host | To path |")
        lines.append("|------|---------|------|---------|---------|")
        for t in sorted(transfers, key=lambda x: -x["size_bytes"])[:80]:
            lines.append(
                f"| {human(t['size_bytes'])} | `{t['content']}` | "
                f"{t['from_host']}:`{t['from_path']}` | {t['to_host']} | `{t['to_path']}` |"
            )
        if len(transfers) > 80:
            lines.append(f"| … | _{len(transfers) - 80} more_ | | | |")
    else:
        lines.append("_All unique packages already present on every indexed host._")
    lines.append("")
    lines.append("## Host-local logs (kept in place)")
    lines.append("")
    lines.append(
        "Logs are host-specific operational records: they mutate between index runs, "
        "so even a content-safe snapshot is not a durable fleet identity. This plan "
        "therefore only moves them into `axquant/logs/` on their own host — no "
        "cross-host transfers, no duplicate deletion, no fingerprint-driven renames. "
        "To consolidate history later, copy into per-host subdirectories "
        "(`axquant/logs/<host>/…`) instead of mirroring."
    )
    lines.append("")
    if host_local_entries:
        lines.append("| Size | Package | Host(s) | Final path |")
        lines.append("|------|---------|---------|------------|")
        for e in sorted(host_local_entries, key=lambda x: -x["size_bytes"]):
            lines.append(
                f"| {human(e['size_bytes'])} | `{e['canonical_name']}` | "
                f"{', '.join(e['hosts'])} | `{e['final_path']}` |"
            )
    else:
        lines.append("_None indexed._")
    lines.append("")
    lines.append("## Name conflicts (manual review — do not auto-merge)")
    lines.append("")
    if name_conflicts:
        for c in name_conflicts:
            lines.append(f"### `{c['name']}`")
            for v in c["variants"]:
                lines.append(
                    f"- {v['host']}: `{v['path']}` "
                    f"({human(v['size'])}, fp={v['fp']}…, {v['category']})"
                )
            lines.append("")
    else:
        lines.append("_None._")
    lines.append("")
    lines.append("## Recommended execution order")
    lines.append("")
    lines.append("1. **Stop** bulk blind rsync of models/axquant (HF cache may continue).")
    lines.append("2. **Rename** alias packages to canonical names (same filesystem = instant).")
    lines.append("3. **Move** packages into final layout buckets (instant on same volume).")
    lines.append(
        "4. **Delete** same-host duplicates only where the plan lists "
        "`delete_duplicate` (content-safe fingerprints); spot-check first."
    )
    lines.append(
        "5. **Sync only missing** packages (rsync per package path, "
        "prefer content verify over `--size-only` when possible); "
        "logs stay host-local — never add them to sync jobs."
    )
    lines.append(
        "6. Re-run index with default `--fingerprint content` to verify every "
        "host has the same content-safe fingerprint set (logs excepted)."
    )
    lines.append("")

    text = "\n".join(lines) + "\n"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}", file=sys.stderr)

    if args.json_out:
        plan = {
            "hosts": hosts,
            "unique_content": unique_content,
            "local_moves": local_moves,
            "transfers": transfers,
            "name_conflicts": name_conflicts,
            "totals": {
                "unique_packages": len(unique_content),
                "unique_bytes": total_unique,
                "reclaimable_bytes": reclaimable,
                "transfer_bytes": transfer_bytes,
                "host_local_packages": len(host_local_entries),
                "host_local_bytes": host_local_bytes,
                "unverified_fingerprints": unverified_count,
            },
        }
        Path(args.json_out).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}", file=sys.stderr)
    return 0


def _pick_canonical_name(names: list[str]) -> str:
    def score(n: str) -> tuple:
        nl = n.lower()
        # prefer publish AX- names, then HF-style (dots/caps);
        # penalize -source and axquant- prefixes
        return (
            0 if n.startswith("AX-") else 1,
            0 if any(c.isupper() for c in n) and "." in n else 1,
            0 if "bf16" in nl and "source" not in nl else 1,
            1 if "source" in nl else 0,
            1 if "axquant" in nl else 0,
            len(n),
        )

    return sorted(names, key=score)[0]


if __name__ == "__main__":
    raise SystemExit(main())
