#!/bin/python3

import argparse
import pathlib
import pandas as pd
from utils import EIDF_GPU_MAP, HF_MODEL_MAP, get_run_name, k8s_friendlify
from template import Template as yaml_template

def write_yaml_files(target_dir, file_name, file_content):
    
    with open(f"{target_dir}/{file_name}", "w") as f:
        f.write(file_content)


def main(experiments_csv, yaml_target_dir, results_repo):

    pathlib.Path(yaml_target_dir).mkdir(parents=True, exist_ok=True)

    print("Reading from", experiments_csv)
    df = pd.read_csv(experiments_csv)

    # Generate K8s config(s) using templates based on experiment parameters from CSV file
    df["yaml"] = df.apply(lambda row: yaml_template().get({"inference_engine": row.inference_engine,
                                                           "model": row.model,
                                                           "hf_model_path": HF_MODEL_MAP[row.model],
                                                           "dataset": row.dataset,
                                                           "num_samples": row.num_samples,
                                                           "gpu": row.gpu,
                                                           "gpu_product": EIDF_GPU_MAP[row.gpu],
                                                           "num_gpu": row.num_gpu,
                                                           "tensor_parallel_size": row.num_gpu,
                                                           "batch_size": row.batch_size,
                                                           "input_length": int(row.input_length) if pd.notna(row.input_length) else None,
                                                           "output_length": int(row.output_length) if pd.notna(row.output_length) else None},
                                                          results_repo=results_repo),
                          axis=1)
    
    # # Write K8s config(s) to yaml file
    df.apply(lambda row: write_yaml_files(target_dir=yaml_target_dir,
                                          file_name=k8s_friendlify(get_run_name({"inference_engine": row.inference_engine,
                                                                                 "model": row.model,
                                                                                 "dataset": row.dataset,
                                                                                 "num_samples": row.num_samples,
                                                                                 "gpu": row.gpu,
                                                                                 "num_gpu": row.num_gpu,
                                                                                 "batch_size": row.batch_size,
                                                                                 "input_length": int(row.input_length) if pd.notna(row.input_length) else None,
                                                                                 "output_length": int(row.output_length) if pd.notna(row.output_length) else None})
                                                                   + ".yaml"),
                                          file_content=row.yaml),
             axis=1)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Path to experiments CSV file")
    parser.add_argument("--target_dir", type=str, default="./", required=False, help="Target directory to save generated files - defaults to current directory")
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
