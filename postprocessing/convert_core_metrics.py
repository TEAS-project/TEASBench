#!/usr/bin/env python3

"""Add unnormalized / throughput companions to every "e2e_s" entry in results.

Walks the results repo tree the same way aggregate_results.py does -- descending
into each timestamped run directory (see aggregate_results.parse_run_path for
what constitutes a valid run directory) -- and rewrites the .json files found in
each such directory in place.

Writing requires --apply. Without it the script dry-runs: it does all the same
work and reports, file by file, exactly what it would write, but touches
nothing. The rewrite is in place and one branch below replaces a measured
"e2e_s", so the default must not be to write.

Wherever a key "e2e_s" appears (at any depth, and possibly more than once), two
sibling keys are added alongside it:

    "unnormalized_e2e" = e2e_s * num_tasks
    "request/s"        = 1.0 / e2e_s

Each occurrence of "e2e_s" gets its own copy of the two companion keys at the
same level in the structure, so duplicated e2e_s values yield duplicated
companions in their respective locations.

`num_tasks` is determined per run directory from that directory's metrics.json,
as the value of ["quality"]["total"], and is applied to every .json file in the
directory (including metrics.json itself).

Schema safeguard
-----------------
The formula above assumes "e2e_s" is always the *per-request mean* latency
(equivalently, exactly 1 / request rate -- see compute_cost.py's handling of
the newer collection schema, which omits e2e_s and derives it that way). That
assumption silently breaks for any run directory where the raw metrics.json
instead recorded "e2e_s" as the *aggregate* wall-clock total for the whole run
(i.e. what "unnormalized_e2e" is meant to be). This happened for a batch of
AMD mi355x runs collected 2026-07-13 through 2026-07-16: applying the formula
above to them inflated unnormalized_e2e by an extra ~num_tasks (or
num_tasks/achieved_batch_size) and deflated request/s by the same factor --
small enough that several cells rounded to a stored request/s of 0.00 and
tripped audit.py's B7-bounds hard-fail gate in the sync-web-data.yml workflow.

`detect_e2e_schema()` below guards against a repeat: it cross-checks e2e_s
against an independent per-request estimate built from ttft, tpot, and the
batch_token_profile, and only trusts the usual "per-request mean" formula when
that check passes. When a run directory's e2e_s looks aggregate instead, its
"e2e_s" is normalized in place (divided by num_tasks) *before* the companion
keys are computed, so unnormalized_e2e and request/s end up correct and
"e2e_s" keeps its documented per-request meaning for every downstream consumer
(compute_cost.py included). Directories where the check can't be run (e.g.
agentic datasets without a decode token profile) fall back to the original,
assume-per-request behaviour, unchanged.
"""

import argparse
import json
import pathlib

from aggregate_results import parse_run_path

# How far e2e_s/single_request_estimate is allowed to drift from the expected
# concurrency-driven inflation (num_tasks / decode_avg_batch_size) before we
# still call it "aggregate". Kept loose (3x) because the estimate itself is a
# rough one (additive ttft + tpot*tokens, ignoring prefill/decode overlap).
_AGGREGATE_MATCH_TOLERANCE = 3.0
# Below this, "per-request" and "aggregate" are indistinguishable (concurrency
# barely inflates the total), so don't even attempt the call.
_MIN_DETECTABLE_INFLATION = 3.0


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def detect_e2e_schema(metrics):
    """Classify a run's "e2e_s" as 'per_request', 'aggregate', or None.

    'per_request'  -- e2e_s is the per-request mean latency (1 / request rate).
                      This is the assumption the rest of this script has always
                      made, and is by far the common case.
    'aggregate'    -- e2e_s is actually the whole-run wall-clock total (what
                      unnormalized_e2e is supposed to be), mislabeled.
    None           -- not enough information in metrics.json to tell (e.g. no
                      batch_token_profile, as with agentic datasets). Callers
                      should treat this the same as 'per_request' to preserve
                      prior behaviour.

    Method: compare e2e_s to an independent estimate of the per-request time,
    single_request_est = ttft + tpot * decode_generated_tokens_per_request.
    If e2e_s tracks that estimate directly, it's per-request. If instead it
    tracks single_request_est * (num_tasks / decode_avg_batch_size) -- i.e.
    the time the whole run would take at the reported concurrency -- it's an
    aggregate total mislabeled as e2e_s.
    """
    perf = metrics.get("performance", {})
    prof = metrics.get("batch_token_profile", {})
    quality = metrics.get("quality", {})

    e2e = perf.get("e2e_s")
    ttft = perf.get("ttft")
    tpot = perf.get("tpot")
    tpr = prof.get("decode_generated_tokens_per_request")
    batch_size = prof.get("decode_avg_batch_size")
    num_tasks = quality.get("total")

    if not (_is_number(e2e) and _is_number(tpot) and _is_number(tpr)
            and _is_number(batch_size) and _is_number(num_tasks) and batch_size > 0):
        return None
    if ttft is not None and not _is_number(ttft):
        return None

    single_request_est = (ttft or 0) + tpot * tpr
    if single_request_est <= 0:
        return None

    ratio = e2e / single_request_est
    expected_aggregate_ratio = num_tasks / batch_size

    if (expected_aggregate_ratio >= _MIN_DETECTABLE_INFLATION
            and (expected_aggregate_ratio / _AGGREGATE_MATCH_TOLERANCE)
                <= ratio
                <= (expected_aggregate_ratio * _AGGREGATE_MATCH_TOLERANCE)):
        return "aggregate"
    return "per_request"


def annotate_e2e(obj, num_tasks, is_aggregate):
    """Recursively add "unnormalized_e2e" and "request/s" beside every "e2e_s".

    Descends through dicts and lists. Whenever a dict contains an "e2e_s" key
    with a numeric value, that value is first normalized (divided by
    num_tasks) if `is_aggregate` says the raw figure is actually a whole-run
    total rather than a per-request mean -- then the two companion keys are
    inserted using the (now per-request) value. Returns True if any change was
    made anywhere in `obj`.
    """
    changed = False

    if isinstance(obj, dict):
        # Add companions at this level first (before recursing) so they land
        # alongside the e2e_s they describe.
        if "e2e_s" in obj:
            value = obj["e2e_s"]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if is_aggregate and num_tasks:
                    value = value / num_tasks
                    obj["e2e_s"] = value
                obj["unnormalized_e2e"] = value * num_tasks
                obj["request/s"] = (1.0 / value) if value != 0 else None
                changed = True
        for child in obj.values():
            changed |= annotate_e2e(child, num_tasks, is_aggregate)

    elif isinstance(obj, list):
        for item in obj:
            changed |= annotate_e2e(item, num_tasks, is_aggregate)

    return changed


def read_run_metadata(run_dir):
    """Return (num_tasks, e2e_schema) for a run directory, from its metrics.json.

    num_tasks is ["quality"]["total"]; e2e_schema is the result of
    detect_e2e_schema() run over the same metrics.json. Returns (None, None)
    if num_tasks can't be determined (missing/unparseable file, missing keys,
    or non-numeric value) -- the directory is skipped entirely in that case,
    matching prior behaviour.
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        print(f"No metrics.json in {run_dir}; skipping directory.")
        return None, None

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Unparseable metrics.json in {run_dir} ({exc}); skipping directory.")
        return None, None

    try:
        num_tasks = metrics["quality"]["total"]
    except (TypeError, KeyError):
        print(f"No ['quality']['total'] in {metrics_path}; skipping directory.")
        return None, None

    if not isinstance(num_tasks, (int, float)) or isinstance(num_tasks, bool):
        print(f"Non-numeric ['quality']['total'] ({num_tasks!r}) in {metrics_path}; "
              f"skipping directory.")
        return None, None

    schema = detect_e2e_schema(metrics)
    return num_tasks, schema


def process_file(path, num_tasks, is_aggregate, indent, apply_changes):
    """Load, annotate, and (if changed) rewrite one json file in place.

    Returns True if the file was modified -- or, when `apply_changes` is False,
    would have been; the annotation is computed either way and only the write
    is skipped, so a dry run reports exactly the set of files an apply run
    writes. Files that don't parse as json are reported and skipped.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Skipping unparseable json: {path} ({exc})")
        return False

    if not annotate_e2e(data, num_tasks, is_aggregate):
        return False

    if apply_changes:
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


def main(root_dir, indent, apply_changes):
    root = pathlib.Path(root_dir)
    modified = 0
    scanned = 0
    aggregate_dirs = []
    verb = "Updated" if apply_changes else "Would update"

    if not apply_changes:
        print("DRY RUN -- no file will be written. Re-run with --apply to write.\n")

    for run_dir in find_run_dirs(root):
        num_tasks, schema = read_run_metadata(run_dir)
        if num_tasks is None:
            continue

        is_aggregate = schema == "aggregate"
        if is_aggregate:
            aggregate_dirs.append(run_dir)
            print(f"WARNING: {run_dir} looks like e2e_s is an aggregate whole-run "
                  f"total, not a per-request mean; {'normalizing' if apply_changes else 'would normalize'} "
                  f"by num_tasks={num_tasks} before computing companions.")

        for path in sorted(run_dir.glob("*.json")):
            if not path.is_file():
                continue
            scanned += 1
            if process_file(path, num_tasks, is_aggregate, indent, apply_changes):
                modified += 1
                print(f"{verb} {path} (num_tasks={num_tasks})")

    print(f"Done. Scanned {scanned} run json file(s); "
          f"{'updated' if apply_changes else 'would update'} {modified}.")
    if aggregate_dirs:
        normalized = "normalized in place" if apply_changes else "would be normalized in place"
        print(f"\n{len(aggregate_dirs)} run director{'y' if len(aggregate_dirs) == 1 else 'ies'} "
              f"had an aggregate-schema e2e_s ({normalized}, see WARNINGs above):")
        for d in aggregate_dirs:
            print(f"  {d}")
    if not apply_changes:
        print("\nDRY RUN -- nothing was written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add unnormalized_e2e and request/s beside every e2e_s in run json files")
    parser.add_argument("--root-dir", type=str, required=True,
                        help="Top level directory containing the benchmark run results to convert")
    parser.add_argument("--indent", type=int, default=2,
                        help="Indentation (spaces) used when rewriting json files (default: 2)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite the json files. Without it the script only "
                             "reports, file by file, what it would write.")

    args = parser.parse_args()

    main(args.root_dir, args.indent, args.apply)
