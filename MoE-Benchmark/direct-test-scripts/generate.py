#!/bin/python3

import argparse
import pathlib
import pandas as pd
from template import Template as yaml_template
from utils import get_run_name, GPU_MAP, TOKEN_LENGTH_MAP

def write_yaml_files(target_dir, \
    file_content, \
    model_name, \
    gpu, \
    num_gpu, \
    target_input_tokens, \
    target_output_tokens, \
    batch_size, \
    dataset):

    run_name = get_run_name(model_name, gpu, num_gpu, target_input_tokens, target_output_tokens, batch_size, dataset)
    file_name = f"{model_name.split("/")[1]}_{gpu}x{num_gpu}_{target_input_tokens}_{target_output_tokens}_bs{batch_size}_{dataset}.yaml"

    with open(f"{target_dir}/{file_name}", "w") as f:
        f.write(file_content)

def main(experiments_csv, yaml_target_dir):
    pathlib.Path(yaml_target_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(experiments_csv)
    df["yaml"] = df.apply(lambda row: yaml_template().get(model_name=row.model_name, \
                                tensor_parallel_size=row.num_gpu, \
                                dataset=row.dataset, \
                                target_input_tokens=TOKEN_LENGTH_MAP[row.target_input_tokens], \
                                target_output_tokens=TOKEN_LENGTH_MAP[row.target_output_tokens], \
                                num_samples=row.num_samples, \
                                batch_size=row.batch_size, \
                                num_gpu=row.num_gpu, \
                                gpu_product=GPU_MAP[row.gpu]), axis=1)
    
    df.apply(lambda row: write_yaml_files(target_dir=yaml_target_dir, \
                                            file_content=row.yaml, \
                                            model_name=row.model_name, \
                                            gpu=row.gpu, \
                                            num_gpu=row.num_gpu, \
                                            target_input_tokens=row.target_input_tokens, \
                                            target_output_tokens=row.target_output_tokens, \
                                            batch_size=row.batch_size, \
                                            dataset=row.dataset), axis=1)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Path to experiments CSV file")
    parser.add_argument("--target_dir", type=str, required=True, help="Target directory to save generated YAML files")
    args = parser.parse_args()

    main(args.csv_file, args.target_dir)