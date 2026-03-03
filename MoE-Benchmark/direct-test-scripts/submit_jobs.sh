#!/bin/bash

K8S_configs=($(ls configs/*.yaml))
K8S_JOBS_MAPPING="k8s_jobs_mapping.csv"
K8S_JOBS="k8s_jobs.csv"

RUN_DEEPSEEK_R1=0
RUN_QWEN3=0
RUN_KIMI=0
RUN_GPT=0

RUN_A100=1
RUN_H100=1
RUN_H200=1

RUN_1GPU=1
RUN_2GPU=0
RUN_4GPU=0
RUN_8GPU=0

RUN_BS1=1
RUN_BS128=1

echo "yaml,job_id" > ${K8S_JOBS} 
for yaml in "${K8S_configs[@]}"; do
	if [[ $yaml == *"Qwen3"* && $RUN_QWEN3 -eq 0 ]]; then
		continue
	elif [[ $yaml == *"Kimi"* && $RUN_KIMI -eq 0 ]]; then
		continue
	elif [[ $yaml == *"gpt"* && $RUN_GPT -eq 0 ]]; then
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
	elif [[ $yaml == *"x2"* && $RUN_2GPU -eq 0 ]]; then
		continue	
	elif [[ $yaml == *"x4"* && $RUN_4GPU -eq 0 ]]; then
		continue	
	elif [[ $yaml == *"x8"* && $RUN_8GPU -eq 0 ]]; then
		continue
	fi

	if [[ $yaml == *"_bs1_"* && $RUN_BS1 -eq 0 ]]; then
		continue
	elif [[ $yaml == *"_bs128_"* && $RUN_BS128 -eq 0 ]]; then
		continue
	fi

	job=$(kubectl -n eidf230ns create -f ${yaml} | sed -n 's/.*job\.batch\/\([a-z0-9-]\+\).*/\1/p')
	echo "${yaml},${job}" >> ${K8S_JOBS_MAPPING}
	echo "${job}" >> ${K8S_JOBS}
done
