#!/bin/python3

import argparse
import pathlib
import pandas as pd
from utils import (EIDF_GPU_MAP, HF_MODEL_MAP, get_run_name, k8s_friendlify,
                   local_model_path)
from template import Template as yaml_template

def write_yaml_files(target_dir, file_name, file_content):

    with open(f"{target_dir}/{file_name}", "w") as f:
        f.write(file_content)


def build_params(row):
    """Turn a CSV row into the experiment-parameter dict consumed by the
    template engine. All columns flow through generically (so adding a sweep
    dimension to the CSV needs no change here); a few derived parameters are
    computed from the maps in utils."""
    p = {}
    for key, value in row.to_dict().items():
        p[key] = None if (isinstance(value, float) and pd.isna(value)) else value

    # MoE optional length columns: keep them present and integer-typed so run
    # names read '_i128' rather than '_i128.0'.
    for col in ("input_length", "output_length"):
        p[col] = int(p[col]) if p.get(col) is not None else None

    # Family selector defaults to MoE when the column is absent (legacy CSVs).
    p.setdefault("benchmark", None)

    # Derived parameters
    p["hf_model_path"] = HF_MODEL_MAP[p["model"]]
    p["gpu_product"] = EIDF_GPU_MAP[p["gpu"]]
    p["tensor_parallel_size"] = p["num_gpu"]
    p["model_path"] = local_model_path(p["model"])
    return p


def main(experiments_csv, yaml_target_dir, results_repo):

    pathlib.Path(yaml_target_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(experiments_csv)

    # Generate and write one K8s config per experiment row.
    for _, row in df.iterrows():
        params = build_params(row)
        yaml_content = yaml_template().get(params, results_repo=results_repo)
        file_name = k8s_friendlify(get_run_name(params) + ".yaml")
        write_yaml_files(yaml_target_dir, file_name, yaml_content)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Path to experiments CSV file")
    parser.add_argument("--target_dir", type=str, default="./", required=False, help="Target directory to save generated YAML files - defaults to current directory")
    parser.add_argument("--results_repo", type=str, default="TEAS_Development_Results_Private", required=False, help="Name of results repository (not the URL) - defaults to TEAS_Development_Results_Private")
    args = parser.parse_args()

    main(args.csv_file, args.target_dir, args.results_repo)
