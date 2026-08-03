#!/usr/bin/env python3

import argparse
import pathlib
import os
import re
import json
import math
import pandas as pd
from decimal import Decimal, ROUND_DOWN

# Path layout is the inverse of pipeline/utils.results_repo_dir(), with a run timestamp and
# the metrics file appended beneath the directory that function returns:
#
#   moe/eidf/<inference_engine>/<model>/<dataset>_<num_samples>samples/
#       <hw>x<num_hw>/batch-size-<batch_size>[_input<N>][_output<N>]/
#       <run_timestamp>/metrics.json
#
# Note (vs. a naive reading of the path):
#   * <model> and <hw> are lower-cased in the path (utils uses .lower()).
#   * the "dataset" level encodes the sample count: "<dataset>_<num_samples>samples".
#   * the batch-size level may carry "_input<N>" and/or "_output<N>" suffixes,
#     and <batch_size> may be the literal "default".
#   * results_repo_dir() stops at the batch-size level; the run timestamp and
#     metrics file live below it, so timestamp handling is kept flexible.
#
# To Do: It would be neater if nothing needs to be inferred from parsing the
#        path but instead all relevant metadata is extracted from the metadata.json
#        within each run directory, decoupling the need to track potential changes
#        to results repo structure
#

BATCH_DIR_RE = re.compile(
    r"^batch-size-(?P<batch_size>.+?)(?:_input(?P<input_length>\d+))?(?:_output(?P<output_length>\d+))?$"
)
DATASET_DIR_RE = re.compile(r"^(?P<dataset>.+)_(?P<num_samples>\d+)samples$")

# Leading column order for the combined all_metrics.csv. Rows are also sorted by
# these columns, in this order, as successive sort keys.
COMBINED_LEADING_COLS = [
    "model",
    "dataset",
    "platform",
    "hw_type x num_hw",
    "inference_engine",
    "batch_size",
]
# Remaining metadata columns, placed after the leading ones (before the metrics).
OTHER_META_COLS = ["num_samples", "input_length", "output_length", "run_timestamp"]


def parse_dataset_dir(dataset_dir):
    """'gsm8k_500samples' -> ('gsm8k', '500'). Falls back to (dir, None)."""
    m = DATASET_DIR_RE.match(dataset_dir)
    if not m:
        return dataset_dir, None
    return m.group("dataset"), m.group("num_samples")


def parse_batch_dir(batch_dir):
    """'batch-size-32_input1024_output256' -> ('32', '1024', '256').

    Returns (batch_size, input_length, output_length); the lengths are None when
    the suffixes are absent. Returns (None, None, None) on no match.
    """
    m = BATCH_DIR_RE.match(batch_dir)
    if not m:
        return None, None, None
    return m.group("batch_size"), m.group("input_length"), m.group("output_length")


def parse_run_path(rel_parts):
    """Turn a metrics file's path parts (relative to results_dir) into a metadata
    dict, or return None if the path doesn't match the expected layout.

    Fixed prefix (6 levels): platform / inference_engine / model /
    <dataset>_<n>samples / <hw>x<num_hw> / batch-size-...
    Everything between that prefix and the trailing filename is the run id
    (normally a single timestamp directory).
    """
    # 6 fixed dirs + at least the metrics filename.
    if len(rel_parts) < 7:
        return None

    platform, inference_engine, model, dataset_dir, hw_dir, batch_dir = rel_parts[:6]
    dataset, num_samples = parse_dataset_dir(dataset_dir)
    batch_size, input_length, output_length = parse_batch_dir(batch_dir)

    return {
        "platform": platform,
        "inference_engine": inference_engine,
        "model": model,
        "dataset": dataset,
        "num_samples": num_samples,
        "hw_type x num_hw": hw_dir,
        "batch_size": batch_size,
        "input_length": input_length,
        "output_length": output_length,
        "run_timestamp": "/".join(rel_parts[6:-1]),
    }


def find_metric_column(columns, metric):
    """Locate the column for a given metric after json_normalize.

    Handles a top-level key (e.g. "acc") and nested keys (e.g. "results.acc").
    """
    if metric in columns:
        return metric
    for col in columns:
        if col.split(".")[-1] == metric:
            return col
    return None


def trunc_fixed(decimals):
    """Return a transform that truncates a value toward zero to `decimals` decimal
    places and renders it as a string with exactly that many decimals.

    Fixed decimal places (rather than significant figures) are what make a column
    line up by the decimal point for the eye. Uses Decimal on the shortest
    round-trip repr so truncation matches the value as written, not its binary
    approximation. Non-finite / non-numeric inputs (NaN, None) pass through
    unchanged (written as blanks by to_csv).
    """
    quantum = Decimal(1).scaleb(-decimals)

    def _format(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return x
        if not math.isfinite(x):
            return x
        d = Decimal(repr(x)).quantize(quantum, rounding=ROUND_DOWN)
        return f"{d:.{decimals}f}"

    return _format


# Fixed decimal places used when writing the per model-dataset CSVs, so values
# line up by the decimal point. Tune per metric group as the scales warrant.
ACC_DECIMALS = 5
# Cost per request is sub-cent, so a couple of extra places keep some significant
# figures while still aligning by the decimal point.
COST_DECIMALS = 5

# Per-column decimal places for the performance CSVs, keyed by output column
# label. Each performance column (latency metrics plus the merged sparsity
# metrics) is truncated/formatted to its own fixed number of decimals, so columns
# on very different scales -- e.g. tokens/s in the thousands vs S_MFU well below 1
# -- can each be shown at an appropriate precision while still lining up by the
# decimal point within a column. PERF_DECIMALS_DEFAULT applies to any performance
# column not listed here.
PERF_DECIMALS_DEFAULT = 5
PERF_DECIMALS = {
    "unnormalized_e2e": 2,
    "request/s": 5,
    "ttft": 5,
    "tpot": 5,
    "prefill_avg_expert_activation": 2,
    "decode_avg_expert_activation": 2,
    "prefill_tokens_per_s": 2,
    "decode_output_tokens_per_s": 2,
    "prefill_S_MBU": 6,
    "decode_S_MBU": 6,
    "prefill_S_MFU": 6,
    "decode_S_MFU": 6,
    # Agentic runs (see AGENTIC_PERFORMANCE_VALUE_SPECS) have no sparsity/MoE
    # routing metrics, but do report per-task latency and agent workload stats.
    "avg_e2e_latency_s": 3,
    "p50_e2e_latency_s": 3,
    "p99_e2e_latency_s": 3,
    "avg_num_requests": 2,
    "avg_tool_call_count": 2,
    "avg_total_input_tokens": 2,
    "avg_total_output_tokens": 2,
}


def get_results(results_dir, pattern="metrics*", required=True):
    """Collect every file matching `pattern` under results_dir into one DataFrame.

    Each row is the json-normalized contents of one matched file, augmented with
    metadata derived from that file's location in the directory tree. `pattern`
    is an rglob glob: "metrics*" for the per-run metrics files, "*cost*" for the
    cost files (which appear as both "cost.json" and "cost_<dataset>_<ts>.json"
    depending on collection location).

    When `required` is False and nothing matches, returns None instead of raising,
    so optional metric families (e.g. cost, which is absent for the agentic runs)
    don't abort the whole aggregation.
    """
    results_root = pathlib.Path(results_dir)
    data_frames = []

    for path in sorted(results_root.rglob(pattern)):
        if not path.is_file():
            continue

        meta = parse_run_path(path.relative_to(results_root).parts)
        if meta is None:
            print(f"Skipping file with unexpected path: {path}")
            continue

        print(path)

        with open(path, "r") as file:
            data = json.loads(file.read())

        df = pd.json_normalize(data)
        for key, value in meta.items():
            df[key] = value

        data_frames.append(df)

    if not data_frames:
        if required:
            raise SystemExit(f"No files matching '{pattern}' found under {results_dir}")
        print(f"No files matching '{pattern}' found under {results_dir}; skipping.")
        return None

    return pd.concat(data_frames, ignore_index=True)


# Sort order for the hw_type portion of the "hw_type x num_hw" column. That
# column holds "<hw_type>x<num_hw>" (e.g. "h200x8", "mi355xx4"); wherever it is a
# sort key, rows are ordered by hw_type per this list, then by num_hw
# numerically. hw_types not listed here sort after all listed ones; values that
# don't parse sort last.
HW_TYPE_ORDER = ["a100", "h100", "h200", "b200", "b300", "gb10", "mi355x", "blackhole-p150b"]
HW_DIR_RE = re.compile(r"^(?P<hw_type>.+)x(?P<num_hw>\d+)$")


def hw_dir_sort_value(value):
    """Map a "<hw_type>x<num_hw>" value to a zero-padded string that sorts by
    hw_type (per HW_TYPE_ORDER) then num_hw. A string key (rather than a tuple)
    keeps it robust under pandas' object-dtype sorting."""
    m = HW_DIR_RE.match(str(value))
    if not m:
        return f"{len(HW_TYPE_ORDER) + 1:03d}|{value}"
    hw_type = m.group("hw_type")
    num_hw = int(m.group("num_hw"))
    rank = (HW_TYPE_ORDER.index(hw_type)
            if hw_type in HW_TYPE_ORDER else len(HW_TYPE_ORDER))
    return f"{rank:03d}|{num_hw:04d}|{hw_type}"


def descriptor_sort_key(col):
    """Key for DataFrame.sort_values: apply the custom hw ordering to the
    "hw_type x num_hw" column and leave every other sort column unchanged.
    sort_values applies this to each `by` column independently, and the Series
    carries its column name so the hw column can be singled out."""
    if col.name == "hw_type x num_hw":
        return col.map(hw_dir_sort_value)
    return col


def arrange_combined(df):
    """Order columns and sort rows for the combined all_metrics.csv.

    Leading columns are COMBINED_LEADING_COLS (in that order), followed by the
    remaining metadata columns, then everything else (metrics) in existing order.
    Rows are sorted by the leading columns as successive sort keys.
    """
    leading = [c for c in COMBINED_LEADING_COLS if c in df.columns]
    other_meta = [c for c in OTHER_META_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in leading + other_meta]

    df = df[leading + other_meta + rest]
    if leading:
        df = df.sort_values(by=leading, kind="stable", key=descriptor_sort_key,
                            ignore_index=True)
    return df


# Metadata columns (all produced by parse_run_path) that together uniquely
# identify a single run, used to join metrics from sibling files (e.g. sparsity)
# onto the metrics rows.
RUN_KEY_COLS = ["platform", "inference_engine", "model", "dataset", "num_samples",
                "hw_type x num_hw", "batch_size", "input_length", "output_length",
                "run_timestamp"]


def merge_run_metrics(left, right, value_cols):
    """Left-join the given `value_cols` from `right` onto `left`, matching rows by
    run identity (RUN_KEY_COLS).

    Both frames carry the same parse_run_path metadata, so a single "|"-joined
    string key is built from RUN_KEY_COLS on each side. Each column is cast to the
    nullable string dtype and its missing entries filled with a sentinel before
    concatenation: absent values (e.g. input/output lengths) then compare equal
    across the two frames instead of poisoning the whole key to NA (under pandas'
    StringDtype, NA + str == NA), which would otherwise collapse every key and
    cause a cross-join. Only `value_cols` actually present in `right` are pulled
    across; any missing ones stay absent and are reported by find_metric_column.
    """
    def run_key(df):
        parts = [df[col].astype("string").fillna("\x00") for col in RUN_KEY_COLS]
        key = parts[0]
        for part in parts[1:]:
            key = key + "|" + part
        return key

    present = [c for c in value_cols if c in right.columns]
    left = left.copy()
    left["_run_key"] = run_key(left)
    right_subset = right[present].copy()
    right_subset["_run_key"] = run_key(right)
    merged = left.merge(right_subset, on="_run_key", how="left")
    return merged.drop(columns="_run_key")


def apply_explicit_performance_fallbacks(df):
    """Fill derived display metrics from explicit per-run performance values.

    Some dense-model runs have no ``sparsity.json`` because expert activation is
    not applicable.  Their source ``metrics.json`` may still contain a measured
    prefill throughput.  Prefer the normal sparsity sidecar when present, and use
    the explicit performance value only when that sidecar value is absent.
    """
    fallbacks = {
        "sparsity.prefill.prefill_tokens_per_s": "performance.prefill_tokens_per_s",
    }
    result = df.copy()
    for target, source in fallbacks.items():
        if source not in result.columns:
            continue
        if target in result.columns:
            result[target] = result[target].combine_first(result[source])
        else:
            result[target] = result[source]
    return result


CORE_DESCRIPTOR_COLS = ["platform", "hw_type x num_hw", "batch_size", "inference_engine"]
OPTIONAL_DESCRIPTOR_COLS = ["num_samples", "input_length", "output_length"]

# Descriptor column order used inside the per-permutation CSVs, which also serves
# as the successive row-sort key order. Kept deliberately separate for accuracy and
# performance so they can diverge later. Any listed column that is fixed by the
# grouping (e.g. batch_size in the performance files) is dropped automatically.
ACCURACY_DESCRIPTOR_ORDER = ["inference_engine", "hw_type x num_hw", "batch_size", "platform"]
PERFORMANCE_DESCRIPTOR_ORDER = ["inference_engine", "hw_type x num_hw", "batch_size", "platform"]
COST_DESCRIPTOR_ORDER = ["inference_engine", "hw_type x num_hw", "batch_size", "platform"]
# Fixed-length CSVs omit batch_size (always "default") and lead with input_length
# / output_length instead (handled separately by write_fixed_length_per_permutation_csvs).
FIXED_LENGTH_DESCRIPTOR_ORDER = ["inference_engine", "hw_type x num_hw", "platform"]


def sanitize_filename_part(value):
    """Make a single path-safe filename component from a value."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))


def write_metric_per_permutation_csvs(df, output_dir, value_specs,
                                      group_cols=("model", "dataset"),
                                      descriptor_order=None,
                                      filename_fn=None, subdir_fn=None):
    """Write one CSV per unique combination of `group_cols`.

    Descriptor columns are CORE_DESCRIPTOR_COLS plus whichever OPTIONAL_DESCRIPTOR_COLS
    are present, EXCEPT any column already fixed by `group_cols` -- a column that is
    constant within every file (because it defines the file) is dropped, since it
    adds nothing. `value_specs` is a list of (output_column, source_metric,
    transform) tuples appended after the descriptors; `source_metric` is resolved
    via find_metric_column and `transform` is an optional per-value function (or
    None to write as-is).

    `filename_fn` maps the dict of group-key values to a filename stem (default:
    sanitized key values joined with "_"). `subdir_fn` optionally maps the same
    dict to a subdirectory (relative to output_dir) the file is placed in.

    `descriptor_order` lists the descriptor columns in the order they should appear
    (defaults to CORE_DESCRIPTOR_COLS). It also serves as the successive row-sort
    key order. Entries that are fixed by `group_cols` (and so dropped from output)
    are ignored; only listed columns appear, so omitting one removes it.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_cols = list(group_cols)

    # Resolve each metric to its actual (possibly nested) column name once.
    resolved_specs = []
    for out_name, source_metric, transform in value_specs:
        src = find_metric_column(df.columns, source_metric)
        if src is None:
            print(f"Warning: metric '{source_metric}' not found; column "
                  f"'{out_name}' will be NaN.")
        resolved_specs.append((out_name, src, transform))

    if descriptor_order is None:
        descriptor_order = CORE_DESCRIPTOR_COLS
    descriptor_cols = [c for c in descriptor_order
                       if c in CORE_DESCRIPTOR_COLS and c not in group_cols]
    optional_cols = [c for c in OPTIONAL_DESCRIPTOR_COLS if c not in group_cols]

    written = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_cols, keys))

        present_optional = [c for c in optional_cols if group[c].notna().any()]
        table = group[descriptor_cols + present_optional].copy()

        for out_name, src, transform in resolved_specs:
            if src is None:
                table[out_name] = pd.NA
            elif transform is not None:
                table[out_name] = group[src].apply(transform)
            else:
                table[out_name] = group[src]

        if descriptor_cols:
            table = table.sort_values(by=descriptor_cols, kind="stable",
                                      key=descriptor_sort_key, ignore_index=True)

        if filename_fn is not None:
            stem = filename_fn(key_values)
        else:
            stem = "_".join(sanitize_filename_part(k) for k in keys)

        target_dir = output_dir / subdir_fn(key_values) if subdir_fn else output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"{stem}.csv"

        table.to_csv(out_path, index=False)
        written.append(out_path)
        print(f"  wrote {out_path} ({len(table)} run(s))")

    return written


def is_fixed_length(df):
    """True for rows from fixed-length runs (batch-size-default_input<N>_output<N>),
    i.e. rows where input_length or output_length is set. Standard runs (batch-size-N
    or bare batch-size-default) have both as NA."""
    return df["input_length"].notna() | df["output_length"].notna()


def write_fixed_length_per_permutation_csvs(df_standard, df_fixed, output_dir,
                                             value_specs, descriptor_order=None,
                                             filename_fn=None):
    """Write per-(model, dataset) CSVs for fixed-length runs.

    Each CSV combines:
      - every fixed-length run for that (model, dataset) from df_fixed
        (batch-size-default_input<N>_output<N>, one row each)
      - the standard batch-size-default rows from df_standard for the same
        (model, dataset) (input_length and output_length are blank)

    Column layout: input_length, output_length, then descriptor_order columns
    (defaulting to FIXED_LENGTH_DESCRIPTOR_ORDER; batch_size is always "default"
    for every row so it is omitted), then num_samples if present, then metrics.

    Rows sort by (input_length, output_length) numerically with the standard rows
    (blank input/output) first, then by the remaining descriptor columns.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if descriptor_order is None:
        descriptor_order = FIXED_LENGTH_DESCRIPTOR_ORDER

    # Resolve value specs against the fixed-length frame's columns (both frames
    # come from the same file types, so columns are the same).
    resolved_specs = []
    for out_name, source_metric, transform in value_specs:
        src = find_metric_column(df_fixed.columns, source_metric)
        if src is None:
            print(f"Warning: metric '{source_metric}' not found; column "
                  f"'{out_name}' will be NaN.")
        resolved_specs.append((out_name, src, transform))

    def fl_sort_key(col):
        """Numeric sort for input/output length (None → -1, sorts first); HW
        ordering for hw_type x num_hw; natural sort for everything else."""
        if col.name in ("input_length", "output_length"):
            def to_int(v):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return -1
            return col.map(to_int)
        return descriptor_sort_key(col)

    written = []
    for keys, fixed_group in df_fixed.groupby(["model", "dataset"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        model, dataset = keys

        # Pull the standard batch-size-default rows for this (model, dataset),
        # restricted to configurations (inference_engine, hw_type x num_hw,
        # platform) that have at least one fixed-length run -- so only rows with
        # a direct fixed-length counterpart appear in the CSV.
        config_cols = [c for c in descriptor_order if c in fixed_group.columns]
        fixed_configs = set(
            map(tuple, fixed_group[config_cols].drop_duplicates().values)
        )
        std_mask = (
            (df_standard["model"] == model) &
            (df_standard["dataset"] == dataset) &
            (df_standard["batch_size"] == "default") &
            ~is_fixed_length(df_standard)
        )
        std_rows = df_standard[std_mask]
        std_rows = std_rows[
            std_rows[config_cols].apply(lambda r: tuple(r) in fixed_configs, axis=1)
        ]

        combined = pd.concat([std_rows, fixed_group], ignore_index=True)

        # Column order: descriptor cols, num_samples, input_length, output_length,
        # then metrics. input_length/output_length sit just before the metrics so
        # that standard and fixed-length rows for the same configuration are
        # adjacent and easy to compare.
        desc_cols = [c for c in descriptor_order if c in combined.columns]
        has_num_samples = (
            "num_samples" in combined.columns and combined["num_samples"].notna().any()
        )
        optional = ["num_samples"] if has_num_samples else []
        fl_cols = ["input_length", "output_length"]

        table = combined[desc_cols + optional + fl_cols].copy()

        for out_name, src, transform in resolved_specs:
            if src is None:
                table[out_name] = pd.NA
            elif transform is not None:
                table[out_name] = combined[src].apply(transform)
            else:
                table[out_name] = combined[src]

        # Sort by descriptor cols first so standard and fixed-length rows for the
        # same configuration are grouped together, then by input/output length
        # (None sorts first, placing the standard row above the fixed-length ones).
        sort_by = desc_cols + fl_cols
        table = table.sort_values(by=sort_by, kind="stable",
                                  key=fl_sort_key, ignore_index=True)

        if filename_fn is not None:
            stem = filename_fn({"model": model, "dataset": dataset})
        else:
            stem = f"{sanitize_filename_part(model)}_{sanitize_filename_part(dataset)}"

        out_path = output_dir / f"{stem}.csv"
        table.to_csv(out_path, index=False)
        written.append(out_path)
        print(f"  wrote {out_path} "
              f"({len(fixed_group)} fixed-length + {len(std_rows)} standard run(s))")

    return written


def write_accuracy_per_permutation_csvs(df, output_dir):
    """Per (model, dataset) CSVs whose final column is accuracy ("acc"),
    truncated to a fixed ACC_DECIMALS decimal places for aligned display.
    Batch size remains a column, since each file spans all batch sizes."""
    return write_metric_per_permutation_csvs(
        df, output_dir,
        value_specs=[("acc", "acc", trunc_fixed(ACC_DECIMALS))],
        group_cols=("model", "dataset"),
        descriptor_order=ACCURACY_DESCRIPTOR_ORDER,
    )


# Sparsity metrics appended to the performance CSVs, as (output_label,
# source_column) in display order. Source columns are the full json_normalize
# paths into each run's sparsity file -- the full path is required because S_MBU
# and S_MFU each occur under both "prefill" and "decode", so the bare-name
# fallback in find_metric_column would be ambiguous. These are merged onto the
# metrics rows by run identity before the performance CSVs are written (see
# merge_run_metrics).
SPARSITY_VALUE_SPECS = [
    ("prefill_avg_expert_activation", "sparsity.activation.avg_expert_activation_prefill"),
    ("decode_avg_expert_activation",  "sparsity.activation.avg_expert_activation_decode"),
    ("prefill_tokens_per_s",          "sparsity.prefill.prefill_tokens_per_s"),
    ("decode_output_tokens_per_s",    "sparsity.decode.output_tokens_per_s"),
    ("prefill_S_MBU",                 "sparsity.prefill.S_MBU"),
    ("decode_S_MBU",                  "sparsity.decode.S_MBU"),
    ("prefill_S_MFU",                 "sparsity.prefill.S_MFU"),
    ("decode_S_MFU",                  "sparsity.decode.S_MFU"),
]

# Latency/throughput metrics that lead the performance CSVs for MoE (batch
# inference) runs, as (output_label, source_column) in display order, followed by
# the merged sparsity metrics. Both new-style keys are top-level under
# "performance" in every metrics file, so find_metric_column resolves them to
# "performance.<key>".
PERFORMANCE_VALUE_SPECS = [
    ("unnormalized_e2e", "unnormalized_e2e"),
    ("request/s", "request/s"),
    ("ttft", "ttft"),
    ("tpot", "tpot"),
] + SPARSITY_VALUE_SPECS

# Latency and agent-workload metrics that lead the performance CSVs for agentic
# runs. Agentic metrics files have no MoE routing/sparsity data (there is no
# batch inference server to profile), but each task's LLM interaction is still
# characterized by e2e/ttft/tpot latency plus request/tool-call/token counts
# under the "agentic.*" key, which take the place of the sparsity metrics.
AGENTIC_PERFORMANCE_VALUE_SPECS = [
    ("avg_e2e_latency_s", "avg_e2e_latency_s"),
    ("p50_e2e_latency_s", "p50_e2e_latency_s"),
    ("p99_e2e_latency_s", "p99_e2e_latency_s"),
    ("ttft", "ttft"),
    ("tpot", "tpot"),
    ("avg_num_requests", "avg_num_requests"),
    ("avg_tool_call_count", "avg_tool_call_count"),
    ("avg_total_input_tokens", "avg_total_input_tokens"),
    ("avg_total_output_tokens", "avg_total_output_tokens"),
]


def write_performance_per_permutation_csvs(df, output_dir, value_specs=PERFORMANCE_VALUE_SPECS):
    """Per (model, dataset, batch_size) CSVs whose final columns are the given
    performance `value_specs` in order (PERFORMANCE_VALUE_SPECS for MoE runs,
    AGENTIC_PERFORMANCE_VALUE_SPECS for agentic runs -- see main()). Each column
    is truncated/formatted to its own fixed number of decimals from PERF_DECIMALS
    (falling back to PERF_DECIMALS_DEFAULT). Files are bucketed into a
    "batch-size-<n>" subdirectory per batch size, so batch_size appears in
    neither the columns nor the filename."""
    return write_metric_per_permutation_csvs(
        df, output_dir,
        value_specs=[
            (label, src, trunc_fixed(PERF_DECIMALS.get(label, PERF_DECIMALS_DEFAULT)))
            for label, src in value_specs
        ],
        group_cols=("model", "dataset", "batch_size"),
        descriptor_order=PERFORMANCE_DESCRIPTOR_ORDER,
        subdir_fn=lambda k: "batch-size-" + sanitize_filename_part(k["batch_size"]),
        filename_fn=lambda k: "_".join([
            sanitize_filename_part(k["model"]),
            sanitize_filename_part(k["dataset"]),
        ]),
    )


def write_fixed_length_performance_per_permutation_csvs(df_standard, df_fixed,
                                                         output_dir,
                                                         value_specs=PERFORMANCE_VALUE_SPECS):
    """Fixed-length performance CSVs under batch-size-default_fixed-length/.

    Same metrics as the standard performance CSVs (`value_specs` with per-column
    PERF_DECIMALS), preceded by input_length and output_length columns."""
    return write_fixed_length_per_permutation_csvs(
        df_standard, df_fixed, output_dir,
        value_specs=[
            (label, src, trunc_fixed(PERF_DECIMALS.get(label, PERF_DECIMALS_DEFAULT)))
            for label, src in value_specs
        ],
        descriptor_order=FIXED_LENGTH_DESCRIPTOR_ORDER,
        filename_fn=lambda k: "_".join([
            sanitize_filename_part(k["model"]),
            sanitize_filename_part(k["dataset"]),
        ]),
    )


# Every run's cost file carries two independent cost models side by side, under
# top-level "rent" and "buy" keys: "rent.cost.*" is the cost at the platform's
# quoted market rental price, "buy.cost.*" is the cost under a capital
# amortization + electricity model as if the hardware were purchased outright.
# Both blocks report the same metric names (e.g. "avg_cost_per_request_usd"), so
# the bare metric name is ambiguous under find_metric_column's suffix matching --
# each spec below therefore uses the full "rent.cost.<metric>"/"buy.cost.<metric>"
# path to pick out the right one unambiguously, with output labels prefixed
# "rent_"/"buy_" so the two are clearly distinguished in the CSV columns too.

# Cost metrics for MoE runs, whose natural unit of work is a single inference
# request.
MOE_COST_VALUE_SPECS = [
    ("rent_avg_cost_per_request_usd", "rent.cost.avg_cost_per_request_usd"),
    ("buy_avg_cost_per_request_usd", "buy.cost.avg_cost_per_request_usd"),
    ("rent_avg_cost_per_1M_output_tokens_usd", "rent.cost.avg_cost_per_1M_output_tokens_usd"),
    ("buy_avg_cost_per_1M_output_tokens_usd", "buy.cost.avg_cost_per_1M_output_tokens_usd"),
]

# Cost metrics for agentic runs, whose natural unit of work is a whole agent
# task (e.g. one SWE-bench-lite instance), which is why the per-request cost
# metric is named "avg_cost_per_task_usd" instead of "avg_cost_per_request_usd".
# "buy.cost.avg_cost_per_task_usd" reflects the run's default_cost_mode (normally
# "active_resource"); the alternative "reserved_worker" accounting mode nested
# under "buy.cost.reserved_worker.*" is not surfaced as a separate column here.
AGENTIC_COST_VALUE_SPECS = [
    ("rent_avg_cost_per_task_usd", "rent.cost.avg_cost_per_task_usd"),
    ("buy_avg_cost_per_task_usd", "buy.cost.avg_cost_per_task_usd"),
    ("rent_avg_cost_per_1M_output_tokens_usd", "rent.cost.avg_cost_per_1M_output_tokens_usd"),
    ("buy_avg_cost_per_1M_output_tokens_usd", "buy.cost.avg_cost_per_1M_output_tokens_usd"),
]


def write_cost_per_permutation_csvs(df, output_dir, value_specs=MOE_COST_VALUE_SPECS):
    """Per (model, dataset, batch_size) CSVs whose final columns are the given
    cost `value_specs` (MOE_COST_VALUE_SPECS for MoE runs, AGENTIC_COST_VALUE_SPECS
    for agentic runs -- see main()), each truncated to a fixed COST_DECIMALS
    decimal places. Files are bucketed into a "batch-size-<n>" subdirectory per
    batch size, so batch_size appears in neither the columns nor the filename --
    mirroring the performance CSVs."""
    fmt = trunc_fixed(COST_DECIMALS)
    return write_metric_per_permutation_csvs(
        df, output_dir,
        value_specs=[(label, src, fmt) for label, src in value_specs],
        group_cols=("model", "dataset", "batch_size"),
        descriptor_order=COST_DESCRIPTOR_ORDER,
        subdir_fn=lambda k: "batch-size-" + sanitize_filename_part(k["batch_size"]),
        filename_fn=lambda k: "_".join([
            sanitize_filename_part(k["model"]),
            sanitize_filename_part(k["dataset"]),
        ]),
    )


def write_fixed_length_cost_per_permutation_csvs(df_standard, df_fixed, output_dir,
                                                  value_specs=MOE_COST_VALUE_SPECS):
    """Fixed-length cost CSVs under batch-size-default_fixed-length/.

    Same metrics as the standard cost CSVs, preceded by input_length and
    output_length columns."""
    fmt = trunc_fixed(COST_DECIMALS)
    return write_fixed_length_per_permutation_csvs(
        df_standard, df_fixed, output_dir,
        value_specs=[(label, src, fmt) for label, src in value_specs],
        descriptor_order=FIXED_LENGTH_DESCRIPTOR_ORDER,
        filename_fn=lambda k: "_".join([
            sanitize_filename_part(k["model"]),
            sanitize_filename_part(k["dataset"]),
        ]),
    )


def hw_type_of(hw_dir):
    """Extract hw_type from a "hw_type x num_hw" value, e.g. "h200x8" -> "h200"."""
    m = HW_DIR_RE.match(str(hw_dir))
    return m.group("hw_type") if m else None


# Batch sizes shown as columns in README.md, in display order, mapped to their
# column headers.
README_BATCH_SIZES = ["default", "1"]
README_BATCH_COLUMN_LABELS = {"default": "batch-size-default", "1": "batch-size-1"}

# HW types physically available on each platform, used to restrict each
# platform's table in README.md to only the hw_types it can actually run.
# Platforms not listed here fall back to the full HW_TYPE_ORDER.
PLATFORM_HW_TYPES = {
    "eidf": ["a100", "h100", "h200"],
    "vastai": ["h200", "b200", "b300"],
    "dgx-spark": ["gb10"],
    "amd": ["mi355x"],
    "tenstorrent": ["blackhole-p150b"],
}

# Display order for platform tables in README.md. Platforms not listed here sort
# after all listed ones, alphabetically.
PLATFORM_ORDER = ["eidf", "vastai", "amd", "dgx-spark", "tenstorrent"]

# Inference engines available on a given platform, used to restrict that
# platform's table rows accordingly. Platforms not listed here fall back to
# DEFAULT_INFERENCE_ENGINES.
PLATFORM_INFERENCE_ENGINES = {
    "tenstorrent": ["kai"],
}
DEFAULT_INFERENCE_ENGINES = ["sglang", "vllm"]


def platform_sort_key(platform):
    return (PLATFORM_ORDER.index(platform) if platform in PLATFORM_ORDER
            else len(PLATFORM_ORDER), platform)


def write_readme(df, root_dir):
    """Write README.md under root_dir: one section per (model, dataset) combination
    present in df, and within each section one table per platform that has at
    least one result for that (model, dataset).

    Each table only lists the hw_types available on that platform (per
    PLATFORM_HW_TYPES) that actually have a result there -- in particular,
    vastai's H200 row is only shown if vastai itself has an H200 result, since
    H200 is normally run on eidf instead. Rows are sorted by hw_type first (per
    HW_TYPE_ORDER), then by inference_engine (per PLATFORM_INFERENCE_ENGINES,
    falling back to DEFAULT_INFERENCE_ENGINES), so successive rows share a
    hw_type across inference engines before moving to the next hw_type. One
    column per README_BATCH_SIZES entry holds a bold tick mark where at least one
    run exists for that platform/hw_type/inference_engine/batch_size (any
    num_hw), left blank otherwise.

    If the same hw_type has results on more than one platform for a given
    (model, dataset) -- e.g. H200 on both eidf and vastai -- a warning naming the
    hw_type, platforms, model and dataset is printed, since each hw_type is
    normally expected to be run on only one platform.
    """
    df = df.copy()
    df["_hw_type"] = df["hw_type x num_hw"].map(hw_type_of)

    group_cols = ["model", "dataset"]
    combos = df[group_cols].drop_duplicates().sort_values(by=group_cols, kind="stable")

    lines = ["# Results\n"]
    header = "| hw_type | inference_engine | " + " | ".join(
        README_BATCH_COLUMN_LABELS[b] for b in README_BATCH_SIZES) + " |"
    separator = "|---|---|" + "|".join(":---:" for _ in README_BATCH_SIZES) + "|"

    for _, key in combos.iterrows():
        model, dataset = key["model"], key["dataset"]
        subset = df[(df["model"] == model) & (df["dataset"] == dataset)]

        for hw_type in HW_TYPE_ORDER:
            platforms_with_hw = sorted(
                subset.loc[subset["_hw_type"] == hw_type, "platform"].dropna().unique())
            if len(platforms_with_hw) > 1:
                print(f"Warning: results for the hw_type {hw_type} on multiple "
                      f"platforms ({', '.join(platforms_with_hw)}) "
                      f"for {model}/{dataset}")

        lines.append(f"## {model} / {dataset}\n")
        platforms = sorted(subset["platform"].dropna().unique(), key=platform_sort_key)
        for platform in platforms:
            platform_subset = subset[subset["platform"] == platform]
            platform_hw_types = PLATFORM_HW_TYPES.get(platform, HW_TYPE_ORDER)

            hw_types_to_show = [g for g in HW_TYPE_ORDER if g in platform_hw_types]
            if platform == "vastai":
                hw_types_to_show = [
                    g for g in hw_types_to_show
                    if g != "h200" or (platform_subset["_hw_type"] == "h200").any()
                ]
            platform_engines = PLATFORM_INFERENCE_ENGINES.get(platform, DEFAULT_INFERENCE_ENGINES)

            lines.append(f"### {platform}\n")
            lines.append(header)
            lines.append(separator)
            for hw_type in hw_types_to_show:
                for inference_engine in platform_engines:
                    engine_subset = platform_subset[platform_subset["inference_engine"] == inference_engine]
                    cells = [
                        "**✓**" if ((engine_subset["_hw_type"] == hw_type) &
                                  (engine_subset["batch_size"] == batch_size)).any() else ""
                        for batch_size in README_BATCH_SIZES
                    ]
                    lines.append(f"| {hw_type.upper()} | {inference_engine} | " +
                                 " | ".join(cells) + " |")
            lines.append("")

    readme_path = pathlib.Path(root_dir) / "README.md"
    readme_path.write_text("\n".join(lines))
    print(f"Wrote {readme_path}")
    return readme_path


def main(results_dir, output_dir):

    df = get_results(results_dir, pattern="metrics*")
    df = arrange_combined(df)

    # Agentic runs report an "agentic.*" block of per-task request/tool-call/token
    # stats that MoE (batch inference) runs never have; its presence is what
    # distinguishes the two result sets and selects which performance/cost
    # metric names to look for below (moe and agentic name some metrics
    # differently -- e.g. avg_cost_per_request_usd vs avg_cost_per_task_usd --
    # since a MoE run's unit of work is a request and an agentic run's is a
    # whole task).
    is_agentic = any(col == "agentic" or col.startswith("agentic.") for col in df.columns)
    performance_value_specs = AGENTIC_PERFORMANCE_VALUE_SPECS if is_agentic else PERFORMANCE_VALUE_SPECS
    cost_value_specs = AGENTIC_COST_VALUE_SPECS if is_agentic else MOE_COST_VALUE_SPECS

    # e2e_s has been superseded by unnormalized_e2e / request/s and is no longer
    # reported, so drop it from the combined all_metrics.csv dump too.
    e2e_cols = [c for c in df.columns if c.split(".")[-1] == "e2e_s"]
    df = df.drop(columns=e2e_cols)

    write_readme(df, results_dir)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "all_metrics.csv"), index=False)
    print(f"Combined results collected in {output_dir}.")

    # Split into standard runs (no fixed context length) and fixed-length runs
    # (batch-size-default_input<N>_output<N>). Fixed-length rows are written to
    # separate CSVs and must not pollute the standard per-batch-size CSVs.
    df_std = df[~is_fixed_length(df)].copy()
    df_fl  = df[ is_fixed_length(df)].copy()

    accuracy_per_permutation_dir = pathlib.Path(output_dir) / "accuracy/by_model_dataset"
    print(f"Writing accuracy per model-dataset CSVs to {accuracy_per_permutation_dir} ...")
    write_accuracy_per_permutation_csvs(df_std, accuracy_per_permutation_dir)

    performance_per_permutation_dir = pathlib.Path(output_dir) / "performance/by_model_dataset"

    # Sparsity metrics live in separate sibling files ("sparsity.json" or
    # "sparsity_<dataset>_<ts>.json"), present only for MoE runs, so they are
    # collected in their own pass and joined onto the metrics rows by run
    # identity. Agentic runs have no MoE routing to profile, so this pass is
    # skipped entirely for them. The merge uses RUN_KEY_COLS (which includes
    # input_length / output_length) so sparsity is matched correctly for both
    # standard and fixed-length rows.
    performance_df = df
    if not is_agentic:
        sparsity_df = get_results(results_dir, pattern="sparsity*.json", required=False)
        if sparsity_df is not None:
            performance_df = merge_run_metrics(
                df, sparsity_df, [src for _, src in SPARSITY_VALUE_SPECS])
        performance_df = apply_explicit_performance_fallbacks(performance_df)

    perf_std = performance_df[~is_fixed_length(performance_df)].copy()
    perf_fl  = performance_df[ is_fixed_length(performance_df)].copy()

    print(f"Writing performance per model-dataset CSVs to {performance_per_permutation_dir} ...")
    write_performance_per_permutation_csvs(perf_std, performance_per_permutation_dir,
                                            value_specs=performance_value_specs)

    if not perf_fl.empty:
        fl_perf_dir = performance_per_permutation_dir / "batch-size-default_fixed-length"
        print(f"Writing fixed-length performance CSVs to {fl_perf_dir} ...")
        write_fixed_length_performance_per_permutation_csvs(
            perf_std, perf_fl, fl_perf_dir, value_specs=performance_value_specs)

    # Cost lives in separate cost files ("cost.json" or "cost_<dataset>_<ts>.json"),
    # present only for a subset of runs (both MoE and agentic runs carry cost
    # files; some ad hoc runs of either kind may not), so it is collected in its
    # own pass and is non-fatal when absent.
    cost_df = get_results(results_dir, pattern="*cost*.json", required=False)
    if cost_df is not None:
        cost_std = cost_df[~is_fixed_length(cost_df)].copy()
        cost_fl  = cost_df[ is_fixed_length(cost_df)].copy()

        cost_per_permutation_dir = pathlib.Path(output_dir) / "cost/by_model_dataset"
        print(f"Writing cost per model-dataset CSVs to {cost_per_permutation_dir} ...")
        write_cost_per_permutation_csvs(cost_std, cost_per_permutation_dir,
                                         value_specs=cost_value_specs)

        if not cost_fl.empty:
            fl_cost_dir = cost_per_permutation_dir / "batch-size-default_fixed-length"
            print(f"Writing fixed-length cost CSVs to {fl_cost_dir} ...")
            write_fixed_length_cost_per_permutation_csvs(cost_std, cost_fl, fl_cost_dir,
                                                          value_specs=cost_value_specs)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect results from jobs")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Top level directory containing the results to aggregate")
    parser.add_argument("--output_dir", type=str, required=False, default=None,
                        help="Output dir for CSV files with aggregated results (defaults to results_dir/aggregate_results)")

    args = parser.parse_args()

    if args.output_dir is None:
         args.output_dir = os.path.join(args.results_dir, "aggregate_results")
    

    main(args.results_dir, args.output_dir)
