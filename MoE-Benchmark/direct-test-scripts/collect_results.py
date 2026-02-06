#!/bin/python3

import argparse
import pathlib
import pandas as pd
from generate import get_run_name
from datetime import datetime
import json

def get_result(results_dir, output_jsonl, model_name, gpu, num_gpu, target_input_tokens, target_output_tokens, batch_size, dataset):
    model_split = model_name.split("/")
    model_name_clean = model_split[1] #.replace(".", "-")
    model_vendor = model_split[0]

    run_name = get_run_name(model_name, gpu, num_gpu, target_input_tokens, target_output_tokens, batch_size, dataset, token_abbrev=False)
    search_dir = f"{results_dir}/{run_name}/{model_vendor}/{model_name_clean}"

    cap_metrics_files = pathlib.Path(search_dir).glob("cap_metrics_*")

    """ example
    {
        "prefill_smbu": 0.05201683731988759,
        "prefill_smfu": 0.5374358145882385,
        "decoding_smbu": 0.6763836654156693,
        "decoding_smfu": 0.18298034271389818,
        "kv_size": 50181.427791487324,
        "decoding_throughput": 1944.6293711297642,
        "prefill_tp": 110278.12140045782,
        "ttft": 0.08999655082688379,
        "tpot": 0.03402787310123347,
        "total_requests": 1319,
        "successful_requests": 1319,
        "failed_requests": 0,
        "exact_match": 0.1106899166034875,
        "correct": 146,
        "total": 1319,
        "no_answer": 904,
        "cost": null,
        "model_name": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "method": "sglang",
        "precision": "bfloat16",
        "e2e_s": 13104.24,
        "batch_size": null,
        "gpu_type": "1xH200",
        "dataset": "gsm8k",
        "ignore_eos": true,
        "server_batch_size": 64,
        "model_type": "instruct"
    }
    """

    for f in cap_metrics_files:
        print(f)
        mydict = {}
        with open(f, "r") as f_json:
            data = f_json.read()
            data = json.loads(data)

        try:
            mydict["model_name"] = data["model_name"]
            mydict["gpu"] = gpu
            mydict["num_gpu"] = num_gpu
            mydict["batch_size"] = data["server_batch_size"]
            mydict["dataset"] = data["dataset"]
            mydict["ttft"] = data["ttft"]
            mydict["tpot"] = data["tpot"]
        except:
            print(f"Error parsing {f}, skipping...")
            continue

        with open(output_jsonl, "a") as f_out:
            f_out.write(str(mydict) + "\n")
        break # first run results for now
    return

def main(experiments_csv, results_dir):
    df = pd.read_csv(experiments_csv)
    # model_name,gpu,num_gpu,target_input_tokens,target_output_tokens,batch_size,dataset,num_samples
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_jsonl = f"sweep_{timestamp}.jsonl"
    
    df.apply(lambda row: get_result(results_dir, \
                                    output_jsonl, \
                                    row.model_name, \
                                    row.gpu, \
                                    row.num_gpu, \
                                    row.target_input_tokens, \
                                    row.target_output_tokens, \
                                    row.batch_size, \
                                    row.dataset), axis=1)
    
    print(f"Results collected in {output_jsonl}. Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect results from sweep jobs")
    parser.add_argument("--experiments_csv", type=str, required=True, help="Path to the CSV file containing the experiment configurations")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing the results from sweep jobs")

    args = parser.parse_args()

    main(args.experiments_csv, args.results_dir)