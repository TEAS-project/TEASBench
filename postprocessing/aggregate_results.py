#!/usr/bin/env python3

import argparse
import pathlib
import os
import re
import json
import pandas as pd

# Path layout is the inverse of pipeline/utils.results_repo_dir(), with a run timestamp and
# the metrics file appended beneath the directory that function returns:
#
#   moe/eidf/<inference_engine>/<model>/<dataset>_<num_samples>samples/
#       <gpu>x<num_gpu>/batch-size-<batch_size>[_input<N>][_output<N>]/
#       <run_timestamp>/metrics.json
#
# Note (vs. a naive reading of the path):
#   * <model> and <gpu> are lower-cased in the path (utils uses .lower()).
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
    "gpu_type x num_gpu",
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
    <dataset>_<n>samples / <gpu>x<num_gpu> / batch-size-...
    Everything between that prefix and the trailing filename is the run id
    (normally a single timestamp directory).
    """
    # 6 fixed dirs + at least the metrics filename.
    if len(rel_parts) < 7:
        return None

    platform, inference_engine, model, dataset_dir, gpu_dir, batch_dir = rel_parts[:6]
    dataset, num_samples = parse_dataset_dir(dataset_dir)
    batch_size, input_length, output_length = parse_batch_dir(batch_dir)

    return {
        "platform": platform,
        "inference_engine": inference_engine,
        "model": model,
        "dataset": dataset,
        "num_samples": num_samples,
        "gpu_type x num_gpu": gpu_dir,
        "batch_size": batch_size,
        "input_length": input_length,
        "output_length": output_length,
        "run_timestamp": "/".join(rel_parts[6:-1]),
    }


def find_acc_column(columns):
    """Locate the accuracy ("acc") column after json_normalize.

    Handles a top-level "acc" key and nested keys such as "results.acc".
    """
    if "acc" in columns:
        return "acc"
    for col in columns:
        if col.split(".")[-1] == "acc":
            return col
    return None


def get_results(results_dir):
    """Collect every metrics file under results_dir into a single DataFrame.

    Each row is the json-normalized contents of one metrics file, augmented with
    metadata derived from that file's location in the directory tree.
    """
    results_root = pathlib.Path(results_dir)
    data_frames = []

    for path in sorted(results_root.rglob("metrics*")):
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
        raise SystemExit(f"No metrics files found under {results_dir}")

    return pd.concat(data_frames, ignore_index=True)


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
        df = df.sort_values(by=leading, kind="stable", ignore_index=True)
    return df


def write_accuracy_per_permutation_csvs(df, output_dir):
    """Write one CSV per unique (model, dataset) pair.

    Core columns are platform, gpu config, batch size, inference engine and
    accuracy. num_samples / input_length / output_length are included as well
    when present, so rows stay unambiguous across the full sweep.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc_col = find_acc_column(df.columns)
    if acc_col is None:
        print("Warning: no 'acc' column found in metrics; per-permutation files "
              "will report accuracy as NaN.")

    core_cols = ["platform", "gpu_type x num_gpu", "batch_size", "inference_engine"]
    optional_cols = ["num_samples", "input_length", "output_length"]

    written = []
    for (model, dataset), group in df.groupby(["model", "dataset"]):
        present_optional = [c for c in optional_cols if group[c].notna().any()]
        table = group[core_cols + present_optional].copy()
        table["acc"] = group[acc_col] if acc_col is not None else pd.NA

        safe_model = re.sub(r"[^0-9A-Za-z._-]+", "_", str(model))
        safe_dataset = re.sub(r"[^0-9A-Za-z._-]+", "_", str(dataset))
        out_path = output_dir / f"{safe_model}_{safe_dataset}.csv"

        table.to_csv(out_path, index=False)
        written.append(out_path)
        print(f"  wrote {out_path} ({len(table)} run(s))")

    return written


def main(results_dir, output_dir):

    df = get_results(results_dir)
    df = arrange_combined(df)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "all_metrics.csv"), index=False)
    print(f"Combined results collected in {output_dir}.")

    accuracy_per_permutation_dir = pathlib.Path(output_dir) / "accuracy/by_model_dataset"

    print(f"Writing accuracy per model-dataset CSVs to {accuracy_per_permutation_dir} ...")
    write_accuracy_per_permutation_csvs(df, accuracy_per_permutation_dir)

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
