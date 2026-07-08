#!/usr/bin/env bash

# This general purpose script runs benchmarks for all vLLM and SGLang configurations from the
# experiment CSV passed in $BENCHMARK_CSV. The server and client commands for each row are built
# at runtime by /opt/teasbench/pipeline/vast/resolve_commands.py, which applies the rules in the
# config.yaml in /opt/teasbench/pipeline/configs.

set -euo pipefail

# Check required environment variables.
if [ -z "${BENCHMARK_CSV:-}" ]; then
  echo "ERROR: BENCHMARK_CSV environment variable is not set" >&2
  exit 1
fi

if [ -z "${GIT_TOKEN:-}" ]; then
  echo "ERROR: GIT_TOKEN environment variable is not set" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN environment variable is not set" >&2
  exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY environment variable is not set" >&2
  exit 1
fi

# In normal use the container ID (identifying the current instance) and the container API key
# (granting access to the Vast.ai API) should have been injected into the container by Vast.ai
# itself. If something's gone wrong, `exit 1` now, as this is our only way of automating the
# destruction of the instance once the benchmarks are complete.
if [ -z "${CONTAINER_ID:-}" ]; then
  echo "ERROR: CONTAINER_ID environment variable is not set (should be injected by Vast.ai)" >&2
  exit 1
fi
if [ -z "${CONTAINER_API_KEY:-}" ]; then
  echo "ERROR: CONTAINER_API_KEY environment variable is not set (should be injected by Vast.ai)" >&2
  exit 1
fi

# Start benchmarking.
start_timestamp=$( date +%Y%m%d-%H%M )
BASE_DIR=/dev/shm/$start_timestamp
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# For now, leave output unredirected so we can see it in the logs on the Vast.ai. console.
#exec &> "$BASE_DIR/stdout_stderr.log"

RESULTS_REPO="TEASBench-container-dev-results"
RESULTS_REPO_USER="welucas2"
RESULTS_REPO_URL="https://oauth2:$GIT_TOKEN@github.com/$RESULTS_REPO_USER/$RESULTS_REPO.git"
echo "Cloning results repo $RESULTS_REPO"
git clone "$RESULTS_REPO_URL"

# Function to run a single benchmark. Takes the run directory, the dataset name,
# and the row's client command (base64-encoded, as printed by resolve_commands.py).
run_benchmark() {
    local run_dir=$1
    local dataset=$2
    local client_command
    client_command=$(echo "$3" | base64 -d)

    echo "Starting to run benchmark (sending http requests)..."
    echo "Client command: $client_command"
    client_start_time=$(date +%s)
    # Briefly turn off set -e so that a non-zero exit from the client doesn't kill
    # the script before we can capture it. Once we've done so, just below, set -e again
    set +e
    # and benchmark.
    eval "$client_command" &> "$run_dir/client_$dataset.log"
    CLIENT_SUCCESS=$?
    set -e

    if [[ $CLIENT_SUCCESS -eq 0 ]]; then
        echo "$dataset benchmark run finished"
        client_end_time=$(date +%s)
        client_duration=$((client_end_time - client_start_time))
        echo "Client took $client_duration seconds"
        jq -n --arg client_duration $client_duration '{client: $client_duration}'  >> "$run_dir/timings.json"
    else
        echo "Client crashed"
        echo "----- client_$dataset.log -----"
        cat "$run_dir/client_$dataset.log"
        echo "----- end client_$dataset.log -----"
    fi
    return $CLIENT_SUCCESS
}

# Function to move output into the right directory and upload to results repo.
push_results() {
    ##################
    # Saving results #
    ##################

    local run_dir=$1
    local run_subdir=$2
    local job_description=$3
    local job_duration=$4
    local row_name=$5
    declare -n r="$row_name"
    local model_name="${r[model]}"

    # Tidy up - gather MoE-CAP generated results from the subdir named after the
    # HF model path (e.g. Qwen/Qwen3-..., resolved by resolve_commands.py — the
    # org prefix varies by model).
    cd "$run_dir"
    cp "$HF_MODEL_PATH"/* .
    rm -r "$(dirname "$HF_MODEL_PATH")"
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
    git commit -m "auto: ${r[inference_engine]}-${model_name}-${r[dataset]}-${r[num_samples]}-${r[gpu]}x${r[num_gpu]}-bs${r[batch_size]}"
    git push "https://oauth2:${GIT_TOKEN}@github.com/$RESULTS_REPO_USER/$RESULTS_REPO.git"
    # echo "Would be pushing to results repo here, but skipping for now."
}

# Get MoE-CAP commit hash for reproducibility.
cd /dev/shm/MoE-CAP
MOE_CAP_COMMIT=$(git rev-parse --short HEAD)

# Configuration
REQUIRED_HEADERS=("inference_engine" "model" "dataset" "num_samples" "gpu" "num_gpu" "batch_size")

# Decode the CSV and grab the header line
IFS=',' read -r -a HEADERS < <(echo "$BENCHMARK_CSV" | base64 -d | head -n 1)

# Verify that we have all the column headers we're expecting.
for req in "${REQUIRED_HEADERS[@]}"; do
    found=0
    for header in "${HEADERS[@]}"; do
        if [[ "$header" == "$req" ]]; then
            found=1
            break
        fi
    done
    if [[ $found -eq 0 ]]; then
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
        echo "WARNING on line $line_number: inference_engine '$current_engine' does not match '$ALLOWED_ENGINE'; skipping." >&2
        continue
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

    # Resolve the server and client commands for this row from the config.yaml
    # baked into the image, via template.py's rule engine. resolve_commands.py
    # prints three lines: the HF model path, then the server and client commands
    # (base64-encoded, since they contain line continuations).
    resolve_args=(
        --inference-engine "${row[inference_engine]}"
        --model "${row[model]}"
        --dataset "${row[dataset]}"
        --num-samples "${row[num_samples]}"
        --gpu "${row[gpu]}"
        --num-gpu "${row[num_gpu]}"
        --batch-size "${row[batch_size]}"
    )
    # input_length/output_length are optional CSV columns; only pass them if present.
    if [[ -n "${row[input_length]:-}" ]]; then
        resolve_args+=(--input-length "${row[input_length]}")
    fi
    if [[ -n "${row[output_length]:-}" ]]; then
        resolve_args+=(--output-length "${row[output_length]}")
    fi
    resolved=$(python3 /opt/teasbench/pipeline/vast/resolve_commands.py "${resolve_args[@]}")
    { read -r HF_MODEL_PATH; read -r server_b64; read -r client_b64; } <<< "$resolved"

    server_command=$(echo "$server_b64" | base64 -d)
    echo "Starting server: $server_command"
    server_init_start_time=$(date +%s)
    # eval is needed for the command's embedded line continuations; backgrounding
    # inside the eval string means $! still captures the server's PID.
    eval "$server_command &> \"$RUN_DIR/server.log\" &"
    SERVER_PID=$!

    # Wait until the /health endpoint returns HTTP 200
    echo "Waiting for server $SERVER_PID to be ready..."

    until curl -s -f http://localhost:30000/health > /dev/null; do
        # Check if the server has crashed (if so the health check will never succeed)
        if ! kill -0 "$SERVER_PID" 2> /dev/null; then
            echo -e "\n[ERROR] Server process died before becoming ready to use."
            echo "----- server.log -----"
            cat "$RUN_DIR/server.log"
            echo "----- end server.log -----"
            if grep -q "unsupported display driver / cuda driver combination" "$RUN_DIR/server.log"; then
                echo "[HINT] This looks like a host-level GPU driver/kernel-module mismatch on this specific Vast.ai machine."
                echo "Destroy this instance and try a different offer."
            fi
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
        # get going. The resolved client command's --baseline-answers-path points at
        # $RUN_DIR (from config.yaml), so download into the run directory.
        if [[ "${row[dataset]}" == "arena-hard" ]]; then
            echo "Downloading sample answers for arena-hard..."
            if ! curl -L --output-dir "$RUN_DIR" -O https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v0.1/model_answer/gpt-4-0613.jsonl; then
                echo "WARNING: failed to download arena-hard sample answers. This benchmark will likely fail." >&2
            fi
        fi

        # Try to get the error code without triggering set -e so we can handle it properly below
        # without killing the whole process.
        if run_benchmark "$RUN_DIR" "${row[dataset]}" "$client_b64"; then
            RUN_SUCCESS=0
        else
            RUN_SUCCESS=$?
        fi

        # If run is successful, stage results and push to repo. Otherwise, skip.
        if [[ $RUN_SUCCESS -eq 0 ]]; then
            job_end_time=$(date +%s)
            job_duration=$((job_end_time - job_start_time))
            job_description="{row[inference_engine]}-${row[model]}-${row[dataset]}-${row[num_samples]}-${row[gpu]}x${row[num_gpu]}-bs${row[batch_size]}"
            echo "Benchmark run completed, pushing results to repo..."
            if push_results "$RUN_DIR" "$RUN_SUBDIR" "$job_description" "$job_duration" row; then
                PUSH_SUCCESS=0
            else
                PUSH_SUCCESS=$?
            fi
            if [[ $PUSH_SUCCESS -eq 0 ]]; then
                echo "Push complete."
            else
                echo "Push failed, results may not have been saved to the repo."
            fi
        else
            echo "Benchmark run failed, skipping pushing results."
        fi

        echo "Shutting down server..."  # regardless of client success
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    else
        echo "Not starting client because server is not available"
    fi

    sleep 5  # short break between benchmarks

done < <(echo "$BENCHMARK_CSV" | base64 -d | tail -n +2)

benchmark_end_timestamp=$( date +%Y%m%d-%H%M )
echo "-----------------------------------------"
echo "Benchmarking completed at $benchmark_end_timestamp."
echo "-----------------------------------------"

# Self-destruct the instance or else Vast.ai will restart at the entrypoint until the instance is
# externally destroyed. Use an API call through curl to avoid having to bake the Vast.ai CLI into
# the image. We also need to retry on failure, and as a last resort, sleep forever instead of exiting.
# If the script were to exit instead, without the instance being destroyed, Vast.ai would just
# start the container up again and loop it indefinitely.
echo "Destroying this instance..."
destroyed=0
for attempt in 1 2 3 4 5; do
    if curl -sf -X DELETE "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" -H "Authorization: Bearer $CONTAINER_API_KEY"; then
        destroyed=1
        break
    fi
    echo "WARNING: failed to destroy instance (attempt $attempt/5), retrying in 30s..." >&2
    sleep 30
done
if [[ $destroyed -eq 0 ]]; then
    echo "ERROR: could not destroy this instance via the Vast.ai API. Idling so the benchmarks don't re-run on container restart. Please destroy this instance manually." >&2
    sleep infinity
fi
