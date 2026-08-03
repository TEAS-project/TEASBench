#!/bin/python3

import argparse
import os
import pathlib
import pandas as pd
from utils import (EIDF_GPU_MAP, HF_MODEL_MAP, benchmark_family, get_run_name,
                   k8s_friendlify, local_model_path)
from template import Template as yaml_template

def write_yaml_files(target_dir, file_name, file_content):

    with open(f"{target_dir}/{file_name}", "w") as f:
        f.write(file_content)


def build_params(row):
    """Turn a CSV row into the experiment-parameter dict consumed by the
    template engine. All columns flow through generically (so adding a new
    sweep dimension to a CSV needs no change here); a handful of parameters
    are derived from the maps in utils, and a couple of columns get a default
    value when a CSV omits them entirely (older MoE CSVs) or leaves a row
    blank (agentic CSVs)."""
    p = {}
    for key, value in row.to_dict().items():
        p[key] = None if (isinstance(value, float) and pd.isna(value)) else value

    # MoE optional length columns: keep them present and integer-typed so run
    # names read '_i128' rather than '_i128.0'.
    for col in ("input_length", "output_length"):
        p[col] = int(p[col]) if p.get(col) is not None else None

    # 'family' is the leading column of every experiments CSV and selects the
    # pipeline family; benchmark_family() validates it and raises on a missing
    # or unrecognised value. 'benchmark' names which agentic benchmark the row
    # is (MoE rows use 'dataset' instead, and leave this empty).
    p.setdefault("benchmark", None)
    benchmark_family(p)  # fail fast, with the offending row's own message
    # 'platform' defaults to 'eidf' for CSVs that predate the column.
    p.setdefault("platform", "eidf")

    # batch_size/concurrency: default when the CSV omits the column, or a row
    # leaves it blank. MoE CSVs always supply batch_size, so this is a no-op
    # for them; agentic CSVs may omit concurrency and rely on this default.
    if p.get("batch_size") is None:
        p["batch_size"] = "default"
    p["concurrency"] = int(p["concurrency"]) if p.get("concurrency") is not None else 4

    # Derived parameters
    p["hf_model_path"] = HF_MODEL_MAP[p["model"]]
    p["gpu_product"] = EIDF_GPU_MAP[p["gpu"]]
    p["tensor_parallel_size"] = p["num_gpu"]
    p["model_path"] = local_model_path(p["model"])
    return p


def main(experiments_csv, yaml_target_dir, results_repo):

    pathlib.Path(yaml_target_dir).mkdir(parents=True, exist_ok=True)

    print("Reading from", experiments_csv)
    df = pd.read_csv(experiments_csv)

    # Generate and write one K8s config per experiment row.
    for _, row in df.iterrows():
        params = build_params(row)
        for file_name, content, mode in yaml_template().get_artifacts(
                params, results_repo=results_repo):
            write_yaml_files(yaml_target_dir, file_name, content)
            # Driver scripts are meant to be executed directly.
            os.chmod(os.path.join(yaml_target_dir, file_name), mode)
            print(f"  wrote {file_name}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Path to experiments CSV file")
    parser.add_argument("--target_dir", type=str, default="./", required=False, help="Target directory to save generated YAML files - defaults to current directory")
    parser.add_argument("--results_repo", type=str, default="TEAS_Development_Results_Private", required=False, help="Name of results repository (not the URL) - defaults to TEAS_Development_Results_Private")
    parser.add_argument("--vast", action="store_true", help="Generate Vast.ai launch scripts instead of EIDF YAML configs")
    parser.add_argument("--private-image", action="store_true",
                         help="Add a --login to the generated script (reading GHCR_USERNAME/GHCR_PAT from the "
                              "environment at run time), since the container image is currently private on ghcr.io.")
    args = parser.parse_args()

    if args.vast:
        from vast_generate import generate_vast_scripts
        generate_vast_scripts(args.csv_file, args.target_dir, private_image=args.private_image)
    else:
        main(args.csv_file, args.target_dir, args.results_repo)
