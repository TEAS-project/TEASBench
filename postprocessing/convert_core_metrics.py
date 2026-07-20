#!/usr/bin/env python3

"""Add unnormalized / throughput companions to every "e2e_s" entry in results.

Walks the results repo tree the same way aggregate_results.py does -- descending
into each timestamped run directory (see aggregate_results.parse_run_path for
what constitutes a valid run directory) -- and rewrites the .json files found in
each such directory in place. Wherever a key "e2e_s" appears (at any depth, and
possibly more than once), two sibling keys are added alongside it:

    "unnormalized_e2e" = e2e_s * num_tasks
    "request/s"        = 1.0 / e2e_s

Each occurrence of "e2e_s" gets its own copy of the two companion keys at the
same level in the structure, so duplicated e2e_s values yield duplicated
companions in their respective locations.

`num_tasks` is determined per run directory from that directory's metrics.json,
as the value of ["quality"]["total"], and is applied to every .json file in the
directory (including metrics.json itself).
"""

import argparse
import json
import pathlib

from aggregate_results import parse_run_path


def annotate_e2e(obj, num_tasks):
    """Recursively add "unnormalized_e2e" and "request/s" beside every "e2e_s".

    Descends through dicts and lists. Whenever a dict contains an "e2e_s" key
    with a numeric value, the two companion keys are inserted into that same
    dict, using `num_tasks` as the unnormalization multiplier. Returns True if
    any change was made anywhere in `obj`.
    """
    changed = False

    if isinstance(obj, dict):
        # Add companions at this level first (before recursing) so they land
        # alongside the e2e_s they describe.
        if "e2e_s" in obj:
            value = obj["e2e_s"]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                obj["unnormalized_e2e"] = value * num_tasks
                obj["request/s"] = (1.0 / value) if value != 0 else None
                changed = True
        for child in obj.values():
            changed |= annotate_e2e(child, num_tasks)

    elif isinstance(obj, list):
        for item in obj:
            changed |= annotate_e2e(item, num_tasks)

    return changed


def read_num_tasks(run_dir):
    """Return ["quality"]["total"] from run_dir/metrics.json, or None if it can't
    be determined (missing/unparseable file, missing keys, or non-numeric value).
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        print(f"No metrics.json in {run_dir}; skipping directory.")
        return None

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Unparseable metrics.json in {run_dir} ({exc}); skipping directory.")
        return None

    try:
        num_tasks = metrics["quality"]["total"]
    except (TypeError, KeyError):
        print(f"No ['quality']['total'] in {metrics_path}; skipping directory.")
        return None

    if not isinstance(num_tasks, (int, float)) or isinstance(num_tasks, bool):
        print(f"Non-numeric ['quality']['total'] ({num_tasks!r}) in {metrics_path}; "
              f"skipping directory.")
        return None

    return num_tasks


def process_file(path, num_tasks, indent):
    """Load, annotate, and (if changed) rewrite one json file in place.

    Returns True if the file was modified. Files that don't parse as json are
    reported and skipped.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Skipping unparseable json: {path} ({exc})")
        return False

    if not annotate_e2e(data, num_tasks):
        return False

    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
        f.write("\n")
    return True


def find_run_dirs(root):
    """Return the set of valid timestamped run directories under `root`.

    A directory qualifies if it contains at least one .json file whose path
    matches the layout aggregate_results relies on (parse_run_path). The set is
    returned sorted for deterministic, readable output.
    """
    run_dirs = set()
    for path in root.rglob("*.json"):
        if not path.is_file():
            continue
        if parse_run_path(path.relative_to(root).parts) is None:
            continue
        run_dirs.add(path.parent)
    return sorted(run_dirs)


def main(root_dir, indent):
    root = pathlib.Path(root_dir)
    modified = 0
    scanned = 0

    for run_dir in find_run_dirs(root):
        num_tasks = read_num_tasks(run_dir)
        if num_tasks is None:
            continue

        for path in sorted(run_dir.glob("*.json")):
            if not path.is_file():
                continue
            scanned += 1
            if process_file(path, num_tasks, indent):
                modified += 1
                print(f"Updated {path} (num_tasks={num_tasks})")

    print(f"Done. Scanned {scanned} run json file(s); updated {modified}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add unnormalized_e2e and request/s beside every e2e_s in run json files")
    parser.add_argument("--root-dir", type=str, required=True,
                        help="Top level directory containing the benchmark run results to convert")
    parser.add_argument("--indent", type=int, default=2,
                        help="Indentation (spaces) used when rewriting json files (default: 2)")

    args = parser.parse_args()

    main(args.root_dir, args.indent)
