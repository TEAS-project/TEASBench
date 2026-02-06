# Generate Sweep

This repository contains a benchmark sweep to evaluate MoE-Benchmark performance on [EIDF](https://edinburgh-international-data-facility.ed.ac.uk/). The contained [k8s configurations](configs) are created with the provided Python generator based on the parameters specified in [data](data).

- check out the table of experiments as csv
- generate yaml configs from csv
- launch jobs on eidf
- inspect generated performance data

## Experiments

The list of experiments with parameters to be found in [data/experiments.csv](data/experiments.csv).

## Generate k8s configs

Run [generate.sh](generate.sh) to generate k8s config files. Yaml configurations will be written to `--target_dir`:

```
#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install pandas

python3 generate.py --csv_file=data/experiments.csv --target_dir=configs
```

## Create k8s jobs on EIDF

Run [submit_jobs.sh](submit_jobs.sh). K8s job IDs are written to the file `k8s_jobs.csv`. Note parameters to selectively create jobs based on str matching (model):

```
#!/bin/bash

K8S_configs=($(ls configs/*.yaml))
K8S_JOBS="k8s_jobs.csv"

RUN_QWEN1_5=1
RUN_MIXTRAL_8x7B=0
RUN_MIXTRAL_8x22B=0
RUN_DEEPSEEK_V2_LITE=1
RUN_DEEPSEEK_R1=0

RUN_A100=1
RUN_H100=1
RUN_H200=1

RUN_1GPU=1
RUN_8GPU=0

RUN_BS1=1
RUN_BS32=0
RUN_BS64=0
RUN_BS128=1

echo "yaml,job_id" > ${K8S_JOBS} 
for yaml in "${K8S_configs[@]}"; do
	if [[ $yaml == *"Qwen1.5"* && $RUN_QWEN1_5 -eq 0 ]]; then
		continue
	elif [[ $yaml == *"Mixtral-8x7B"* && $RUN_MIXTRAL_8x7B -eq 0 ]]; then
		continue
	elif [[ $yaml == *"Mixtral-8x22B"* && $RUN_MIXTRAL_8x22B -eq 0 ]]; then
		continue
	elif [[ $yaml == *"DeepSeek-V2-Lite"* && $RUN_DEEPSEEK_V2_LITE -eq 0 ]]; then
		continue
	elif [[ $yaml == *"DeepSeek-R1"* && $RUN_DEEPSEEK_R1 -eq 0 ]]; then
		continue
	fi

	if [[ $yaml == *"A100"* && $RUN_A100 -eq 0 ]]; then
		continue
	elif [[ $yaml == *"H100"* && $RUN_H100 -eq 0 ]]; then
		continue
	elif [[ $yaml == *"H200"* && $RUN_H200 -eq 0 ]]; then
		continue
	fi

	if [[ $yaml == *"x1"* && $RUN_1GPU -eq 0 ]]; then
		continue
	elif [[ $yaml == *"x8"* && $RUN_8GPU -eq 0 ]]; then
		continue
	fi

	if [[ $yaml == *"_bs1_"* && $RUN_BS1 -eq 0 ]]; then
		continue
	elif [[ $yaml == *"_bs32_"* && $RUN_BS32 -eq 0 ]]; then
		continue
	elif [[ $yaml == *"_bs64_"* && $RUN_BS64 -eq 0 ]]; then
		continue
	elif [[ $yaml == *"_bs128_"* && $RUN_BS128 -eq 0 ]]; then
		continue
	fi

	job=$(kubectl -n eidf230ns create -f ${yaml} | sed -n 's/.*job\.batch\/\([a-z0-9-]\+\).*/\1/p')
	echo "${yaml},${job}" >> ${K8S_JOBS}
done
```

## Profiling and performance data

The output directory is specified as command line parameter to the profiler (`moe_cap.runner.openai_api_profile`) in the respective yaml configuration. The base path for the output directory is set to `/mnt/ceph/tmp/MoE-CAP-outputs/` which can be mounted into a pod.

For instance:
```
# list the files in the following dir
ls /mnt/ceph/tmp/MoE-CAP-outputs/Qwen1.5-MoE-A2.7B-Chat_A100x1_4000_1000_bs1_gsm8k

# which lists the following two files
cap_metrics_gsm8k_20260203_151054.json  
detailed_results_gsm8k.jsonl
```

<details>

<summary>cap_metrics_*</summary>

```
$ cat cap_metrics_gsm8k_20260203_151054.json 
{
    "total_requests": 100,
    "successful_requests": 100,
    "failed_requests": 0,
    "exact_match": 0.0,
    "correct": 0,
    "total": 100,
    "no_answer": 4,
    "cost": null,
    "model_name": "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "method": "sglang",
    "precision": "bfloat16",
    "e2e_s": 54.04,
    "batch_size": null,
    "gpu_type": "1xA100",
    "dataset": "gsm8k",
    "ignore_eos": true,
    "server_batch_size": 1,
    "model_type": "thinking"
}
```
</details>

<details>

<summary>detailed_results_*</summary>

```
$ cat detailed_results_gsm8k.jsonl
{"index": 0, "prompt_length": 4031, "success": true, "output_len": 1, "ttft": 35.56820008531213, "latency": 35.56820281967521, "itl": [], "error": ""}
{"index": 1, "prompt_length": 4031, "success": true, "output_len": 1, "ttft": 35.565046813338995, "latency": 35.565052550286055, "itl": [], "error": ""}
{"index": 2, "prompt_length": 4031, "success": true, "output_len": 1, "ttft": 35.56619094312191, "latency": 35.56619391031563, "itl": [], "error": ""}
{"index": 3, "prompt_length": 4031, "success": true, "output_len": 1, "ttft": 35.56578540802002, "latency": 35.56579086370766, "itl": [], "error": ""}
...
{"index": 99, "prompt_length": 4031, "success": true, "output_len": 1, "ttft": 54.00317149795592, "latency": 54.00317497737706, "itl": [], "error": ""}
```
</details>
