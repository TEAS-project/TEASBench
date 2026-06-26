#!/bin/bash

# This runs outside the container.

# Encode CSV contents with base64.
csv_file="smoke_tests_vllm_simple.csv"
CONTAINER_CSV=$(base64 -w 0 $csv_file)
echo "Contents of $csv_file:"
echo "$CONTAINER_CSV"

# Now heading into container.

set -e

start_timestamp=$( date +%Y%m%d-%H%M )
BASE_DIR=/dev/shm/$start_timestamp
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

exec &> "$BASE_DIR/stdout_stderr.log"

RESULTS_REPO="TEASBench-container-dev-results"
RESULTS_REPO_USER="welucas2"
RESULTS_REPO_URL="https://oauth2:$GIT_TOKEN@github.com/$RESULTS_REPO_USER/$RESULTS_REPO.git"
echo "Cloning results repo $RESULTS_REPO"
git clone "$RESULTS_REPO_URL"

# Function to run single benchmark with given options. Extra arguments might be
# judge arguments for arena-hard or set the batch size, but for others this
# argument might just be empty.
run_benchmark() {
    local run_dir=$1
    local model_name=$2
    local dataset=$3
    local num_samples=$4
    # Move the first four args out of the way and store the remainder as the
    # extra arguments to the MoE-CAP client.
    shift 4
    local extra_args=("$@")

    echo "Starting to run benchmark (sending http requests)..."
    client_start_time=$(date +%s)
    python3 -m moe_cap.runner.openai_api_profile \
      --model_name unsloth/"$model_name" \
      --datasets "$dataset" \
      --num-samples "$num_samples" \
      --api-url http://localhost:30000/v1/completions \
      --use-chat-api \
      --output_dir "$run_dir" \
      --backend vllm \
      "${extra_args[@]}" \
      &> "$run_dir/client_$dataset.log"
    CLIENT_SUCCESS=$?  # not 100% robust, could be failed redirection

    if [[ $CLIENT_SUCCESS -eq 0 ]]; then
        echo "$dataset benchmark run finished"
        client_end_time=$(date +%s)
        client_duration=$((client_end_time - client_start_time))
        echo "Client took $client_duration seconds"
        jq -n --arg client_duration $client_duration '{client: $client_duration}'  >> "$run_dir/timings.json"
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

    local run_dir=$1
    local run_subdir=$2
    local job_description=$3
    local model_name=$4
    local job_duration=$5

    # Tidy up - gather MoE-CAP generated results from subdir
    cd "$run_dir"
    cp unsloth/"$model_name"/* .
    rm -r "$(dirname unsloth/"$model_name")"
    mv metadata*.json metadata.json
    mv metrics*.json metrics.json
    mv detailed_results* detailed_results.jsonl
    mv output_data*.jsonl output_data.jsonl

    cd "$BASE_DIR/TEASBench-container-dev-results"
    output_dir="$BASE_DIR/TEASBench-container-dev-results/$run_subdir"
    mkdir -p "$output_dir"
    cp -r "$run_dir"/* "$output_dir"/

    echo "Files copied into repo at $output_dir."

    # Add commit IDs of MoE-CAP and TEASBench to metadata.json
    echo "Adding commit IDs to metadata.json"
    jq --arg moe_cap_commit "$MOE_CAP_COMMIT" --arg teasbench_commit "$TEASBENCH_COMMIT" '.system_environment += {"teasbench_commit": $teasbench_commit, "moe_cap_commit": $moe_cap_commit}' "$output_dir/metadata.json" > tmp.json && mv tmp.json "$output_dir/metadata.json"

    echo "Job took $job_duration seconds"
    jq -n --arg job_duration "$job_duration" '{job: $job_duration}'  >> "$output_dir/timings.json"

    # Pull to refresh before committing and pushing
    git pull "$RESULTS_REPO_URL" main

    # Commit and push data to results repository
    git add "$output_dir/metrics*.json" "$output_dir/metadata*.json" "$output_dir/timings.json"
    git commit -m "auto: ${INFERENCE_ENGINE}-${MODEL_NAME}-${dataset}-${NUM_SAMPLES}-${GPU}x${NUM_GPUS}-bs${BATCH_SIZE}"
    git push "https://oauth2:${GIT_TOKEN}@github.com/$RESULTS_REPO_USER/$RESULTS_REPO.git"
    # echo "Would be pushing to results repo here, but skipping for now."
}

# Get MoE-CAP commit hash for reproducibility.
cd /dev/shm/MoE-CAP
MOE_CAP_COMMIT=$(git rev-parse --short HEAD)

# Configuration
REQUIRED_HEADERS=("inference_engine" "model" "dataset" "num_samples" "gpu" "num_gpu" "batch_size")
ALLOWED_ENGINE="vllm" # Change to SGLang for that container, or whatever else we may have in the future.

# Decode the CSV and grab the header line
IFS=',' read -r -a HEADERS < <(echo "$CONTAINER_CSV" | base64 -d | head -n 1)

# Verify that we have all the column headers we're expecting.
for req in "${REQUIRED_HEADERS[@]}"; do
    if [[ ! " ${HEADERS[@]} " =~ " ${req} " ]]; then
        echo "ERROR: Missing required column header: '$req'" >&2
        exit 1
    fi
done

declare -A row
line_number=1

# Stream the raws from the encoded CSV, skipping the first line (headers).
while IFS=',' read -r -a VALUES; do
    ((line_number++))

    # Verify that we have the same number of entries as headers.
    if [[ "${#VALUES[@]}" -ne "${#HEADERS[@]}" ]]; then
        echo "ERROR on line $line_number: Column count mismatch" >&2
        exit 1
    fi

    # Clear array from previous line
    row=()

    # Map raw values to headers.
    for i in "${!VALUES[@]}"; do
        row["${HEADERS[$i]}"]="${VALUES[$i]}"
    done

    # Clean up any potential trailing carriage returns or spaces in the value
    current_engine=$(echo "${row[inference_engine]}" | xargs | tr -d '\r')
    # then check it's the engine we've said is allowed.
    if [[ "$current_engine" != "$ALLOWED_ENGINE" ]]; then
        echo "ERROR on line $line_number: Invalid inference_engine '$current_engine'." >&2
        echo "   This container only accepts benchmarks for '$ALLOWED_ENGINE'." >&2
        exit 1
    fi

    # Also raise error if any value in the row is a blank.
    for req in "${REQUIRED_HEADERS[@]}"; do
        if [[ -z "${row[$req]}" ]]; then
            echo "ERROR on line $line_number: Required field '$req' is empty." >&2
            exit 1
        fi
    done

    # The parameters for this benchmark look good, so go ahead and run it.
    echo "Running ${row[inference_engine]} ${row[model]} ${row[dataset]} ${row[num_samples]} ${row[gpu]} ${row[num_gpu]} ${row[batch_size]}"

    job_start_time=$(date +%s)
    SERVER_STATUS=""

    cd "$BASE_DIR"

    job_timestamp=$( date +%Y%m%d-%H%M )
    RUN_SUBDIR="moe/vast/${row[inference_engine]}/${row[model]}/${row[dataset]}_${row[num_samples]}samples/${row[gpu]}x${row[num_gpu]}/batch-size-${row[batch_size]}/$job_timestamp"
    RUN_DIR="${BASE_DIR}/${RUN_SUBDIR}"
    # RUN_DIR=/dev/shm/$timestamp
    mkdir -p "$RUN_DIR"
    cd "$RUN_DIR"

    # Extra arguments for the server.
    if [[ "${row[batch_size]}" == "1" ]]; then
        server_extra_args=("--max_num_seqs" "1")
    else
        server_extra_args=()
    fi

    echo "Starting server..."
    echo "with extra arguments: ${server_extra_args[@]}"
    echo "${row[model]}"
    server_init_start_time=$(date +%s)
    python3 -m moe_cap.systems.vllm \
      --model "unsloth/${row[model]}" \
      --port 30000 \
      --host 0.0.0.0 \
      --enable-expert-distribution-metrics \
      --tensor-parallel-size 1 \
      "${server_extra_args[@]}" \
      --reasoning-parser openai_gptoss &> "$RUN_DIR/server.log" &
    SERVER_PID=$!

    # Wait until the /health endpoint returns HTTP 200
    echo "Waiting for server $SERVER_PID to be ready..."

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

    # Server has started, so run the benchmark.
    if [[ "$SERVER_STATUS" != "dead" ]]; then
        server_init_end_time=$(date +%s)
        server_init_duration=$((server_init_end_time - server_init_start_time))
        echo "Server is ready!"
        echo "Server initialisation took $server_init_duration seconds"
        jq -n --arg server_init_duration $server_init_duration '{server_initialisation: $server_init_duration}'  >> "$RUN_DIR/timings.json"

        # If we'll be benchmarking arena-hard, download the sample answers before we
        # get going.
        if [[ "${row[dataset]}" == "arena-hard" ]]; then
            echo "Downloading sample answers for arena-hard..."
            curl -L --output-dir "$BASE_DIR" -O https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v0.1/model_answer/gpt-4-0613.jsonl
        fi

        # Determine the extra arguments to the client.
        client_extra_args=()
        if [[ "${row[batch_size]}" != "default" ]]; then
            client_extra_args+=(--server-batch-size "${row[batch_size]}")
        fi
        if [[ "${row[dataset]}" == "arena-hard" ]]; then
            client_extra_args+=(--judge-api-url https://api.openai.com/v1/chat/completions)
            client_extra_args+=(--judge-model openai/gpt-4.1)
            client_extra_args+=(--judge-api-key "$OPENAI_API_KEY")
            client_extra_args+=(--baseline-answers-path "$BASE_DIR/gpt-4-0613.jsonl")
        fi
        echo "Benchmarking with extra arguments: ${client_extra_args[@]}"
        run_benchmark "$RUN_DIR" "${row[model]}" "${row[dataset]}" "${row[num_samples]}" "${client_extra_args[@]}"
        RUN_SUCCESS=$?

        # If run is successful, stage results and push to repo. Otherwise, skip.
        if [[ $RUN_SUCCESS -eq 0 ]]; then
            job_end_time=$(date +%s)
            job_duration=$((job_end_time - job_start_time))
            job_description="{row[inference_engine]}-${row[model]}-${row[dataset]}-${row[num_samples]}-${row[gpu]}x${row[num_gpu]}-bs${row[batch_size]}"
            push_results "$RUN_DIR" "$RUN_SUBDIR" "$job_description" "${row[model]}" "$job_duration"
            echo "Benchmark succeeded; push results to repo here."
        else
            echo "Benchmark run for $dataset failed, skipping pushing results."
        fi

        echo "Shutting down server..."  # regardless of client success
        kill $SERVER_PID
        wait $SERVER_PID
    else
        echo "Not starting client because server is not available"
    fi

    sleep 5  # short break between benchmarks

done < <(echo "$CONTAINER_CSV" | base64 -d | tail -n +2)

benchmark_end_timestamp=$( date +%Y%m%d-%H%M )
echo "-----------------------------------------"
echo "Benchmarking completed at $benchmark_end_timestamp."
echo "-----------------------------------------"
