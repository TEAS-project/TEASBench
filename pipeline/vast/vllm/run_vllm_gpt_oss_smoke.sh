#!/bin/bash

# Derived from the raw GPT-OSS 20B smoke test with vLLM.

set -e

# Benchmark parameters. Expect these to be set externally as environment
# variables passed through Vast.ai or as k8s secrets.
# They will likely be determined by the top level pipeline script,
# so it should be a matter of copying and pasting the Vast CLI commands.
INFERENCE_ENGINE="vllm"
GPU="h100"
NUM_GPUS=1
MODEL_NAME="gpt-oss-20b"
DATASET_NAMES="gsm8k longbench_v1 arena-hard"
BATCH_SIZE="default"
TEASBENCH_COMMIT="4c63323"
NUM_SAMPLES=1
GIT_TOKEN=""
OPENROUTER_API_KEY=""

job_start_time=$(date +%s)

timestamp=$( date +%Y%m%d-%H%M )
RUN_DIR=/dev/shm/$timestamp
mkdir -p $RUN_DIR
cd $RUN_DIR

exec &> $RUN_DIR/stdout_stderr.log

echo "Starting run."

# CONTAINER SHOULD INCLUDE THIS SETUP
# -----------------------------------
# env_start_time=$(date +%s)

# apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout="30" update
# apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout="30" -y install git
# git clone  https://github.com/Auto-CAP/MoE-CAP.git /dev/shm/MoE-CAP
# cd /dev/shm/MoE-CAP
# MOE_CAP_COMMIT=$(git rev-parse --short HEAD)
# pip install -e .
# # pip install gputil
# apt-get -y install jq

# env_end_time=$(date +%s)
# env_duration=$((env_end_time - env_start_time))
# echo "Environment setup took $env_duration seconds"
# jq -n --arg env_duration $env_duration '{environment_setup: $env_duration}'  >> $RUN_DIR/timings.json

# git config user.email "autopush@vast"
# git config user.name "Pipeline Autopush"
# -----------------------------------

cd /dev/shm/MoE-CAP
MOE_CAP_COMMIT=$(git rev-parse --short HEAD)


# Function to run single benchmark on given dataset.
# Arena-hard requires extra arguments to judge results, but for others
# this argument might just be empty.
run_benchmark() {
    local dataset=$1
    local extra_args=$2

    echo "Starting to run $dataset benchmark (sending http requests)..."
    client_start_time=$(date +%s)
    python3 -m moe_cap.runner.openai_api_profile \
    --model_name unsloth/$MODEL_NAME \
    --datasets $dataset \
    --num-samples 1 \
    --api-url http://localhost:30000/v1/completions \
    --use-chat-api \
    --output_dir $RUN_DIR \
    --backend vllm \
    $extra_args \
    &> $RUN_DIR/client_$dataset.log
    CLIENT_SUCCESS=$?  # not 100% robust, could be failed redirection

    if [ "$CLIENT_SUCCESS" == 0 ]; then
        echo "$dataset benchmark run finished"
        client_end_time=$(date +%s)
        client_duration=$((client_end_time - client_start_time))
        echo "Client took $client_duration seconds"
        jq -n --arg client_duration $client_duration '{client: $client_duration}'  >> $RUN_DIR/timings.json
    else
        echo "Client crashed"
    fi
    return $CLIENT_SUCCESS
}

# Function to move output into the right directory and upload to results repo.
# I think it makes sense to do the pull/push after every successful benchmark run.
push_results() {
    ##################
    # Saving results #
    ##################

    local dataset=$1

    # Tidy up - gather MoE-CAP generated results from subdir
    cd $RUN_DIR
    cp unsloth/${MODEL_NAME}/* .
    rm -r `dirname unsloth/${MODEL_NAME}`
    mv metadata*.json metadata.json
    mv metrics*.json metrics.json
    mv detailed_results* detailed_results.jsonl
    mv output_data*.jsonl output_data.jsonl

    # Copy data into the correct directory within the repo.
    RUN_OUTPUT_DIR=/root/run_output/TEASBench-container-dev-results/moe/vast/vllm/${MODEL_NAME}/${dataset}_${NUM_SAMPLES}samples/${GPU}x${NUM_GPUS}/batch-size-${BATCH_SIZE}/$timestamp
    mkdir -p $RUN_OUTPUT_DIR
    cp -R $RUN_DIR/* $RUN_OUTPUT_DIR/

    # We've also moved our log so change where we are writing
    exec &>> $RUN_OUTPUT_DIR/stdout_stderr.log
    cd $RUN_OUTPUT_DIR

    echo "Files copied to pvc at $RUN_OUTPUT_DIR"

    # Add commit IDs of MoE-CAP and TEASBench to metadata.json
    echo "Adding commit IDs to metadata.json"
    jq --arg moe_cap_commit $MOE_CAP_COMMIT --arg teasbench_commit $TEASBENCH_COMMIT '.system_environment += {"teasbench_commit": $teasbench_commit, "moe_cap_commit": $moe_cap_commit}' $RUN_OUTPUT_DIR/metadata.json > tmp.json && mv tmp.json $RUN_OUTPUT_DIR/metadata.json
    # rm $RUN_OUTPUT_DIR/tmp.json

    job_end_time=$(date +%s)
    job_duration=$((job_end_time - job_start_time))
    echo "Job took $job_duration seconds"
    jq -n --arg job_duration $job_duration '{job: $job_duration}'  >> $RUN_OUTPUT_DIR/timings.json

    # Pull to refresh before committing and pushing
    GIT_TOKEN="${GIT_TOKEN%$'\n'}"   # remove trailing newline that causes issues
    git pull "https://oauth2:${GIT_TOKEN}@github.com/welucas2/TEASBench-container-dev-results.git"

    # Commit data to results repository
    git add metrics*.json metadata*.json *.yaml timings.json
    git commit -m "auto: ${INFERENCE_ENGINE}-${MODEL_NAME}-${dataset}-${NUM_SAMPLES}-${GPU}x${NUM_GPUS}-bs${BATCH_SIZE}"
    git push "https://oauth2:${GIT_TOKEN}@github.com/welucas2/TEASBench-container-dev-results.git"
}

server_init_start_time=$(date +%s)

# Start server
echo "Initialising server"
python3 -m moe_cap.systems.vllm \
  --model unsloth/$MODEL_NAME \
  --port 30000 \
  --host 0.0.0.0 \
  --enable-expert-distribution-metrics \
  --tensor-parallel-size 1 \
  --reasoning-parser openai_gptoss &> $RUN_DIR/server.log &
SERVER_PID=$!

# Wait until the /health endpoint returns HTTP 200
echo "Waiting for server to be ready..."

until curl -s -f http://localhost:30000/health > /dev/null; do
    # Check if the server has crashed (if so the health check will never succeed)
    if ! kill -0 "$SERVER_PID" 2> /dev/null; then
        echo -e "\n[ERROR] Server process died before becoming ready to use."
        SERVER_STATUS="dead"
        break
    fi

    echo -n "."
    sleep 2
done

if [ "$SERVER_STATUS" != "dead" ]; then
    server_init_end_time=$(date +%s)
    server_init_duration=$((server_init_end_time - server_init_start_time))
    echo "Server is ready!"
    echo "Server initialisation took $server_init_duration seconds"
    jq -n --arg server_init_duration $server_init_duration '{server_initialisation: $server_init_duration}'  >> $RUN_DIR/timings.json

    # If we'll be benchmarking arena-hard, download the sample answers before we
    # get going.
    if [[ $DATASET_NAMES == *"arena-hard"* ]]; then
        echo "Downloading sample answers for arena-hard..."
        curl -L --output-dir $RUN_DIR -O https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v0.1/model_answer/gpt-4-0613.jsonl
    fi

    # Loop over datasets.
    for dataset in $DATASET_NAMES; do
        if [ "$dataset" == "arena-hard" ]; then
            extra_args="--judge-api-url https://openrouter.ai/api/v1/chat/completions \
                  --judge-model openai/gpt-4.1 \
                  --judge-api-key $OPENROUTER_API_KEY \
                  --baseline-answers-path $RUN_DIR/gpt-4-0613.jsonl"
            echo "Benchmarking arena-hard with extra arguments: $extra_args"
        else
            extra_args=""
            echo "Benchmarking $dataset with no extra arguments"
        fi
        run_benchmark $dataset "$extra_args"
        RUN_SUCCESS=$?

        # If run is successful, stage results and push to repo. Otherwise, skip.
        if [ $RUN_SUCCESS -eq 0 ]; then
            push_results $dataset
        else
            echo "Benchmark run for $dataset failed, skipping pushing results"
        fi
        sleep 5  # short break between benchmarks
    done

    echo "Shutting down server..."  # regardless of client success
    kill $SERVER_PID
    wait $SERVER_PID
else
    echo "Not starting client because server is not available"
fi

# Premature exit for now -- don't save results, just see if the benchmarks run.
exit 0

echo "Finalising run."