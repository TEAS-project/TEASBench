#!/bin/python3

import os
import argparse
import pathlib
import pandas as pd
from utils import get_run_name, GPU_MAP, TOKEN_LENGTH_MAP

def write_yaml_files(target_dir, \
    file_content, \
    inference_engine, \
    gpu, \
    num_gpu, \
    csv_file_name):

    file_name = f"{inference_engine}_{gpu}x{num_gpu}_{csv_file_name}.yaml"

    with open(f"{target_dir}/{file_name}", "w") as f:
        f.write(file_content)

def main(experiments_csv, yaml_target_dir, inference_engine):

    if inference_engine=="sglang":
        from template_sglang_loop import Template as yaml_template
    elif inference_engine=="vllm":
        #from template_vllm_loop import Template as yaml_template
        print("Inference engine: '", inference_engine, "' under construction")
        raise SystemExit(1)
    else:
        print("Inference engine: '", inference_engine, "' not supported")
        raise SystemExit(1)


    # strip suffix off of input file name
    experiments_csv_clean = os.path.basename(experiments_csv).split(".")[0]

    pathlib.Path(yaml_target_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(experiments_csv)

    # Analyse the input file

    ## Get number of entries for each combination of type and numebr of GPU

    gpu_series = pd.Series(df["gpu"]).unique()

    num_gpu_series = pd.Series(df["num_gpu"]).unique()

    ## Create new data frame

    output_key = []
    value_gpu = []
    value_num_gpu = []

    for i in gpu_series:
        for j in num_gpu_series:
            output_key.append(f"{i}x{j}")
            value_gpu.append(i)
            value_num_gpu.append(j)

    output_df = pd.DataFrame(output_key)
    output_df["gpu"] = value_gpu
    output_df["num_gpu"] = value_num_gpu

    ## Identify which lines each GPU combiantion is used and add to a list

    df_gpu_list = []
    for g in gpu_series:
        df_gpu_list.append(df[df['gpu'] == g])

    df_gpu_num_list = []
    for d in df_gpu_list:
      for n in num_gpu_series:
        df_gpu_num_list.append(d[d['num_gpu'] == n])

    ### Calculate combinations and index list

    df_gpu_num_list_counts = []
    for i in df_gpu_num_list:
      df_gpu_num_list_counts.append(pd.Series(i.index).count())

    df_gpu_num_list_indices = []
    for i in df_gpu_num_list:
      df_gpu_num_list_indices.append(' '.join(str(j) for j in i.index.to_list()))

    ## Add a row to data frame for each combination of type and numebr of GPU

    output_df["counts"] = df_gpu_num_list_counts
    output_df["indices"] = df_gpu_num_list_indices
    output_df["input_file"] = experiments_csv_clean


    print("Full data frame: ", output_df)

    output_df_nz = output_df[output_df["counts"] != 0]

    print("Non-zero data frame: ", output_df_nz)

    output_df_nz["yaml"] = output_df_nz.apply(lambda row: yaml_template().get(tensor_parallel_size=row.num_gpu, \
                                num_gpu=row.num_gpu, \
                                gpu_product=GPU_MAP[row.gpu], \
                                completions=row.counts, \
                                line_array=row.indices, \
                                filename=row.input_file), axis=1)

    output_df_nz.apply(lambda row: write_yaml_files(target_dir=yaml_target_dir, \
                                            file_content=row.yaml, \
                                            inference_engine=inference_engine, \
                                            gpu=row.gpu, \
                                            num_gpu=row.num_gpu, \
                                            csv_file_name=experiments_csv_clean), axis=1)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Path to experiments CSV file")
    parser.add_argument("--target_dir", type=str, required=True, help="Target directory to save generated YAML files")
    parser.add_argument("--inference_engine", type=str, required=True, help="inference engine")
    args = parser.parse_args()

    main(args.csv_file, args.target_dir, args.inference_engine)
