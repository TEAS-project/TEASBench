#!/usr/bin/env python3
"""Compact MoE workload_profile sidecars to the stable fields used by TEAS.

The historical workload_profile JSONs carried generator provenance, copied-source
notes, baseline deltas, null tool fields, and estimates. Downstream result tables
only need the run identity and exact workload token totals. This script rewrites
workload_profile*.json files to a lean schema:

{
  "schema_version": 3,
  "run_type": "moe",
  "run_path": "...",
  "workload_profile": {
    "prefill_tokens_total": 123,
    "decode_tokens_total": 456
  }
}

The rewrite is in place and lossy: every field outside that schema is dropped and
is recoverable only from version control. Writing therefore requires --apply;
without it the script reports, file by file, what it would rewrite and touches
nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _token_total(value: Any) -> int | float | None:
    if isinstance(value, dict):
        value = value.get("total")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, float):
        return value
    return None


def compact_profile(data: dict[str, Any], fallback_run_path: str | None = None) -> dict[str, Any]:
    profile = data.get("workload_profile") or {}
    # Accept both the historical nested schema and the compact v3 schema so the
    # cleaner is idempotent and can resume after an interrupted run.
    prefill_total = _token_total(profile.get("prefill_tokens"))
    if prefill_total is None:
        prefill_total = _token_total(profile.get("prefill_tokens_total"))
    decode_total = _token_total(profile.get("decode_tokens"))
    if decode_total is None:
        decode_total = _token_total(profile.get("decode_tokens_total"))
    if prefill_total is None:
        raise ValueError("missing prefill token total")
    if decode_total is None:
        raise ValueError("missing decode token total")

    return {
        "schema_version": 3,
        "run_type": data.get("run_type") or "moe",
        "run_path": data.get("run_path") or fallback_run_path,
        "workload_profile": {
            "prefill_tokens_total": prefill_total,
            "decode_tokens_total": decode_total,
        },
    }


def _sibling_metrics_totals(path: Path) -> tuple[int | float | None, int | float | None]:
    candidates = [path.with_name(path.name.replace("workload_profile", "metrics", 1))]
    candidates.extend(sorted(path.parent.glob("metrics*.json")))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate.read_text())
        except Exception:
            continue
        profile = data.get("batch_token_profile") or {}
        prefill = _token_total(profile.get("prefill_tokens"))
        decode = _token_total(profile.get("decode_generated_tokens"))
        if prefill is not None or decode is not None:
            return prefill, decode
    return None, None


def clean_file(path: Path, root: Path, dry_run: bool = False) -> bool:
    data = json.loads(path.read_text())
    try:
        fallback_run_path = str(path.parent.relative_to(root))
    except ValueError:
        fallback_run_path = str(path.parent)
    try:
        compact = compact_profile(data, fallback_run_path=fallback_run_path)
    except ValueError:
        prefill, decode = _sibling_metrics_totals(path)
        if prefill is None or decode is None:
            raise
        data = dict(data)
        data["workload_profile"] = {
            "prefill_tokens": {"total": prefill},
            "decode_tokens": {"total": decode},
        }
        compact = compact_profile(data, fallback_run_path=fallback_run_path)
    old = json.dumps(data, sort_keys=True, separators=(",", ":"))
    new = json.dumps(compact, sort_keys=True, separators=(",", ":"))
    if old == new:
        return False
    if not dry_run:
        path.write_text(json.dumps(compact, indent=2) + "\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Tree containing workload_profile*.json files")
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite the profiles. Without it the script only "
                             "reports, file by file, what it would write.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only. Now the default; kept so existing invocations still work.")
    args = parser.parse_args()

    # The rewrite is in place and lossy -- every field outside the compact schema is
    # dropped -- so writing is opt-in, and an explicit --dry-run still wins over --apply.
    dry_run = args.dry_run or not args.apply
    if dry_run:
        print("DRY RUN -- no file will be written. Re-run with --apply to write.")

    root = args.root.resolve()
    files = sorted(root.rglob("workload_profile*.json"))
    changed = 0
    action = "Would update" if dry_run else "Updated"
    for path in files:
        if clean_file(path, root, dry_run=dry_run):
            changed += 1
            print(f"{action} {path}")
    print(f"{action} {changed} workload profile files (scanned {len(files)})")
    if dry_run:
        print("DRY RUN -- nothing was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
