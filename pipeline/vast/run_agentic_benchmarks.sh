#!/usr/bin/env bash

# This script runs agentic benchmarks (imo-answerbench, mcp-atlas, swe-bench-lite) for all
# vLLM and SGLang configurations from the experiment CSV passed in $BENCHMARK_CSV. It is the
# agentic sibling of run_benchmarks.sh (MoE): same CSV-loop shape, same defensive habits, same
# reliance on /opt/teasbench/pipeline/vast/resolve_commands.py to build commands from
# /opt/teasbench/pipeline/configs/config.yaml's rule engine -- but it drives agent_cap.agents
# instead of the MoE-CAP server+client split, and has two Vast.ai-specific differences from the
# MoE script that fall out of "no sidecar containers, no k8s" on this platform (see
# docs/agentic-pipeline-design.md):
#   - a benchmark that needs a tool server (currently only mcp-atlas) starts it as a background
#     process in this same container, instead of a k8s pod sidecar;
#   - swe-bench-lite provisions its per-task sandboxes via Modal (native to swe-rex) instead of
#     a TEASBench k8s sandbox provider -- there is no sandbox/exec provider flag at all here.
#
# The script should be platform agnostic, except for the presence of CONTAINER_ID and CONTAINER_API_KEY environment
# variables which are injected by Vast.ai into the container at runtime, and the use of the Vast.ai API to destroy
# the instance at the very end of the script. Running outside Vast.ai would require changing this behaviour.

set -euo pipefail

# Check that required environment variables have been set.
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

# Needed by the imo-answerbench judge (see pipeline/configs/config.yaml's imo-answerbench
# rules). Not every row needs it (mcp-atlas judges via OpenRouter, swe-bench-lite has no LLM
# judge), but it's validated up front regardless, the same way run_benchmarks.sh validates
# OPENAI_API_KEY up front even though only arena-hard rows use it.
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "ERROR: GEMINI_API_KEY environment variable is not set" >&2
  exit 1
fi

# Needed by the mcp-atlas GTFA judge (OpenRouter, google/gemini-3.1-flash-lite -- see the
# mcp-atlas extra_setup rule in pipeline/configs/config.yaml). Only checked when the CSV
# actually contains mcp-atlas rows, so runs of the other benchmarks don't need an OpenRouter
# key provisioned.
if echo "$BENCHMARK_CSV" | base64 -d | grep -q "mcp-atlas" && [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY environment variable is not set (required for mcp-atlas rows)" >&2
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

# MODAL_TOKEN_ID/MODAL_TOKEN_SECRET are only needed for swe-bench-lite rows (Modal is the
# sandbox substrate there -- see docs/agentic-pipeline-design.md); validated per-row below,
# not here, since a CSV that never runs swe-bench-lite shouldn't need them at all.

# Start benchmarking.
start_timestamp=$( date +%Y%m%d-%H%M )
BASE_DIR=/dev/shm/$start_timestamp
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

# For now, leave output unredirected so we can see it in the logs on the Vast.ai console.
#exec &> "$BASE_DIR/stdout_stderr.log"

RESULTS_REPO="TEAS_Development_Results_Private"
RESULTS_REPO_USER="TEAS-project"
RESULTS_REPO_URL="https://oauth2:$GIT_TOKEN@github.com/$RESULTS_REPO_USER/$RESULTS_REPO.git"
echo "Cloning results repo $RESULTS_REPO"
# Shallow, single-branch, no-checkout up front, then a cone-mode sparse-checkout that
# push_results (below) grows by one directory per row it publishes. --filter=blob:none
# makes it a partial clone so blobs for the repo's whole accumulated history of prior
# runs are never downloaded -- sparse-checkout alone only stops them being written to
# the working tree, not fetched.
git clone --quiet --branch main --single-branch --depth 1 --filter=blob:none --no-checkout "$RESULTS_REPO_URL" "$RESULTS_REPO"
git -C "$RESULTS_REPO" sparse-checkout init --cone
git -C "$RESULTS_REPO" checkout --quiet main

# Get AgentCAP commit hash for reproducibility (AgentCAP is git-cloned into /opt/AgentCAP at
# image build time -- see pipeline/vast/{sglang,vllm}/Dockerfile.agentic -- so its .git dir is
# still there at runtime, unlike TEASBench's pipeline/ files, which are copied in individually
# and so carry their commit via the TEASBENCH_COMMIT image env var instead; see that Dockerfile).
AGENTCAP_COMMIT=$(git -C /opt/AgentCAP rev-parse --short HEAD)

# Function to start a tool server (if this row needs one) and run the client command. Takes the
# run directory, the tool-server setup+wait block (base64, empty if none needed), and the
# client command (base64) -- both as printed by resolve_commands.py.
run_benchmark() {
    local run_dir=$1
    local tool_server_b64=$2
    local client_b64=$3
    local client_command
    client_command=$(echo "$client_b64" | base64 -d)

    # tool_server_setup is "bash .../mcp-server/start.sh &" followed by a readiness poll (see
    # resolve_commands.py's docstring) -- i.e. it backgrounds the server itself and then blocks
    # until it's ready (or exits 1 on timeout). Run it in a subshell so a timeout only fails
    # this row (falls through to RUN_SUCCESS!=0 below) rather than the whole script -- the
    # backgrounded server process itself is unaffected by the subshell exiting, since `&`
    # detaches it from the subshell's job table rather than being tied to its lifetime.
    local tool_server_setup
    tool_server_setup=$(echo "$tool_server_b64" | base64 -d)
    if [[ -n "$tool_server_setup" ]]; then
        echo "Starting tool server..."
        if ! ( eval "$tool_server_setup" ); then
            echo "Tool server did not become ready; not starting the client." >&2
            return 1
        fi
    fi

    echo "Starting to run benchmark (sending http requests)..."
    echo "Client command: $client_command"
    client_start_time=$(date +%s)
    # Briefly turn off set -e so that a non-zero exit from the client doesn't kill
    # the script before we can capture it. Once we've done so, just below, set -e again
    set +e
    # and benchmark.
    eval "$client_command" &> "$run_dir/client.log"
    CLIENT_SUCCESS=$?
    set -e

    if [[ $CLIENT_SUCCESS -eq 0 ]]; then
        echo "Benchmark run finished"
        client_end_time=$(date +%s)
        client_duration=$((client_end_time - client_start_time))
        echo "Client took $client_duration seconds"
        jq -n --arg client_duration $client_duration '{client: $client_duration}'  >> "$run_dir/timings.json"
    else
        echo "Client crashed"
        echo "----- client.log -----"
        cat "$run_dir/client.log"
        echo "----- end client.log -----"
    fi
    return $CLIENT_SUCCESS
}

# Function to move output into the right directory and upload to results repo.
push_results() {
    local run_dir=$1
    local run_subdir=$2
    local job_duration=$3
    local row_name=$4
    local sandbox_provider=$5
    declare -n r="$row_name"

    output_dir="$BASE_DIR/$RESULTS_REPO/$run_subdir"
    mkdir -p "$output_dir"

    # AgentCAP's TEAS writer lands its outputs in $OUTPUT_ROOT ($run_dir/agentic); the
    # results-repo layout is flat (see TEAS_Results_Private/agentic/**), so copy their contents
    # directly into output_dir rather than preserving the agentic/ subdirectory -- same file
    # set pipeline/templates/agentic.yaml (the K8s template) collects. nullglob means a pattern
    # with no match (e.g. metadata_*.json for imo-answerbench, which never calls
    # write_teas_outputs) simply disappears instead of leaving a literal unmatched glob.
    shopt -s nullglob
    RESULT_FILES=(
        "$run_dir"/agentic/metrics*.json "$run_dir"/agentic/metadata*.json
        "$run_dir"/agentic/detailed-results_*.jsonl "$run_dir"/agentic/output-data*.jsonl
        "$run_dir"/agentic/results.jsonl
        "$run_dir"/timings.json "$run_dir"/*.log
    )
    shopt -u nullglob
    if [ ${#RESULT_FILES[@]} -gt 0 ]; then
        cp "${RESULT_FILES[@]}" "$output_dir"/
    fi

    echo "Files copied into repo at $output_dir."

    echo "Job took $job_duration seconds"
    jq -n --arg job_duration "$job_duration" '{job: $job_duration}'  >> "$output_dir/timings.json"

    # Provenance: commits, timings, and the run metadata needed to interpret the numbers --
    # sandbox placement is part of the measured scenario (see
    # docs/agentic-pipeline-design.md's measurement caveat), same as the K8s template records.
    jq -n \
      --arg teasbench_commit "${TEASBENCH_COMMIT:-unknown}" \
      --arg agentcap_commit "$AGENTCAP_COMMIT" \
      --arg benchmark "${r[benchmark]}" \
      --arg model "${r[model]}" \
      --argjson num_tasks "${r[num_tasks]}" \
      --argjson concurrency "${r[concurrency]}" \
      --arg sandbox_provider "$sandbox_provider" \
      --slurpfile timings "$output_dir/timings.json" \
      '{teasbench_commit: $teasbench_commit, agentcap_commit: $agentcap_commit, benchmark: $benchmark, model: $model, num_tasks: $num_tasks, concurrency: $concurrency, sandbox_provider: (if $sandbox_provider == "" then null else $sandbox_provider end), timings: ($timings | add)}' \
      > "$output_dir/provenance.json"

    # All git work for this row happens in the clone.
    cd "$BASE_DIR/$RESULTS_REPO"

    # Bring this row's subdirectory into the sparse-checkout scope -- a brand-new
    # directory not yet in scope silently fails to stage with a plain `git add`.
    git sparse-checkout add "$run_subdir"

    shopt -s nullglob
    ADD_FILES=(
        "$output_dir"/metrics*.json "$output_dir"/metadata*.json
        "$output_dir"/detailed-results_*.jsonl "$output_dir"/output-data*.jsonl
        "$output_dir"/results.jsonl "$output_dir"/timings.json "$output_dir"/provenance.json
        "$output_dir"/*.log
    )
    shopt -u nullglob
    if [ ${#ADD_FILES[@]} -gt 0 ]; then
        git add -f "${ADD_FILES[@]}"
    fi
    git commit -m "auto: ${r[inference_engine]}-${r[model]}-${r[benchmark]}-nt${r[num_tasks]}-${r[gpu]}x${r[num_gpu]}"
    # Push first and only reconcile if another instance got there first: fetching up
    # front costs a round trip per CSV row even when nothing has changed. Every row
    # writes its own timestamped directory, so a rejected push is never a real
    # conflict and the rebase is a trivial replay of an add-only commit.
    local pushed=0
    for attempt in 1 2 3 4 5 6; do
        if git push -q "$RESULTS_REPO_URL" HEAD:main; then pushed=1; break; fi
        echo "push attempt $attempt rejected; rebasing onto origin/main"
        git pull --rebase -q "$RESULTS_REPO_URL" main || break
        sleep $(( (RANDOM % 20) + 5 ))
    done
    [ $pushed -eq 1 ]
}

# These are the headers we always need to be present in the CSV, otherwise we can't run the
# benchmark. concurrency/batch_size may still be blank on individual rows (defaulted below,
# matching generate.py's build_params), but the columns themselves must exist.
REQUIRED_HEADERS=("family" "benchmark" "inference_engine" "model" "gpu" "num_gpu" "num_tasks" "concurrency" "batch_size")
# These must be non-blank on every row -- no sensible default exists for them.
REQUIRED_NONBLANK=("benchmark" "inference_engine" "model" "gpu" "num_gpu" "num_tasks")

# Decode the CSV and grab the header line
IFS=',' read -r -a HEADERS < <(echo "$BENCHMARK_CSV" | base64 -d | head -n 1)

# Verify that we have all the required column headers.
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

# Main benchmarking loop.
# Stream the rows from the encoded CSV, skipping the first line (headers).
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
    row[inference_engine]="$current_engine"

    # Also raise error if any required-non-blank value in the row is blank.
    for req in "${REQUIRED_NONBLANK[@]}"; do
        if [[ -z "${row[$req]}" ]]; then
            echo "ERROR on line $line_number: Required field '$req' is empty." >&2
            exit 1
        fi
    done

    # Defaults for optional-per-row fields, matching generate.py's build_params (agentic runs
    # don't batch in the MoE sense; batch_size exists only so the results-repo path keeps its
    # usual batch-size-<...> level).
    : "${row[batch_size]:=default}"
    : "${row[concurrency]:=4}"

    benchmark=$(echo "${row[benchmark]}" | xargs | tr -d '\r')
    case "$benchmark" in
        imo-answerbench|mcp-atlas|swe-bench-lite) ;;
        *)
            echo "ERROR on line $line_number: unknown benchmark '$benchmark' (expected imo-answerbench, mcp-atlas or swe-bench-lite)" >&2
            exit 1
            ;;
    esac

    # Modal is the sandbox substrate for swe-bench-lite on Vast.ai (no TEASBench sandbox
    # provider at all here -- see docs/agentic-pipeline-design.md). Fail fast if this row needs
    # Modal auth and doesn't have it, same treatment as any other missing-required-field row.
    if [[ "$benchmark" == "swe-bench-lite" ]]; then
        if [[ -z "${MODAL_TOKEN_ID:-}" || -z "${MODAL_TOKEN_SECRET:-}" ]]; then
            echo "ERROR on line $line_number: benchmark 'swe-bench-lite' requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables" >&2
            exit 1
        fi
    fi

    # The parameters for this benchmark look good, so go ahead and run it.
    echo "Running ${row[inference_engine]} ${row[model]} ${benchmark} nt${row[num_tasks]} ${row[gpu]} ${row[num_gpu]}"

    job_start_time=$(date +%s)
    SERVER_STATUS=""

    cd "$BASE_DIR"

    job_timestamp=$( date +%Y%m%d-%H%M )
    # model/gpu are lowercased to match utils.results_repo_dir's convention (and hence
    # postprocessing/aggregate_results.py's parser) -- see §2 of the implementation spec.
    model_lower=$(echo "${row[model]}" | tr '[:upper:]' '[:lower:]')
    gpu_lower=$(echo "${row[gpu]}" | tr '[:upper:]' '[:lower:]')
    RUN_SUBDIR="agentic/vastai/${row[inference_engine]}/${model_lower}/${benchmark}/${gpu_lower}x${row[num_gpu]}/batch-size-${row[batch_size]}/$job_timestamp"
    RUN_DIR="${BASE_DIR}/${RUN_SUBDIR}"
    mkdir -p "$RUN_DIR"
    cd "$RUN_DIR"

    # Resolve the agentic server/client commands (and supporting env/setup blocks) for this row
    # from the config.yaml baked into the image, via template.py's rule engine -- exactly the
    # same mechanism run_benchmarks.sh uses for MoE rows, extended to the agentic family. See
    # resolve_commands.py's module docstring for the full eleven-line contract (the trailing
    # END_AGENTIC_CONTRACT terminator matters: several fields -- extra_setup, the tool-server
    # block, teas_env_exports -- are routinely empty all at once, e.g. for imo-answerbench, and
    # command substitution strips *all* trailing newlines, so without a fixed non-empty
    # terminator those trailing empty fields would silently disappear and the reads below would
    # run out of input).
    resolve_args=(
        --family "${row[family]}"
        --inference-engine "${row[inference_engine]}"
        --model "${row[model]}"
        --gpu "${row[gpu]}"
        --num-gpu "${row[num_gpu]}"
        --batch-size "${row[batch_size]}"
        --benchmark "$benchmark"
        --num-tasks "${row[num_tasks]}"
        --concurrency "${row[concurrency]}"
        --platform vastai
    )
    resolved=$(python3 /opt/teasbench/pipeline/vast/resolve_commands.py "${resolve_args[@]}")
    {
        read -r FAMILY
        read -r HF_MODEL_PATH
        read -r AGENTIC_ENGINE_VERSION
        read -r TEAS_GPU_NAME
        read -r server_b64
        read -r client_b64
        read -r env_b64
        read -r extra_setup_b64
        read -r tool_server_b64
        read -r teas_env_b64
        read -r CONTRACT_TERMINATOR
    } <<< "$resolved"

    if [[ "$FAMILY" != "agentic" ]]; then
        echo "ERROR on line $line_number: resolve_commands.py did not return the expected 'agentic' family marker (got '$FAMILY')" >&2
        exit 1
    fi
    if [[ "$CONTRACT_TERMINATOR" != "END_AGENTIC_CONTRACT" ]]; then
        echo "ERROR on line $line_number: resolve_commands.py output did not end with the expected terminator (got '$CONTRACT_TERMINATOR') -- fields may have been misaligned" >&2
        exit 1
    fi

    # Apply any environment variables from config.yaml's extra_container_env rules for this row
    # before starting the server which may need them (e.g. SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR).
    env_exports=$(echo "$env_b64" | base64 -d)
    if [[ -n "$env_exports" ]]; then
        eval "$env_exports"
    fi

    # SWE-bench-lite: Modal is the sandbox substrate; SWEBENCH_HARNESS_MODAL=1 makes
    # SWEBenchEvaluator.finalize (workstream A) pass --modal true to the grading harness. There
    # is no --sandbox-provider/--exec-provider at all on this platform -- Modal is native to
    # swe-rex (see docs/agentic-pipeline-design.md §5, "why no broker").
    if [[ "$benchmark" == "swe-bench-lite" ]]; then
        export SWEBENCH_HARNESS_MODAL=1
        export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
        sandbox_provider_for_provenance="modal"
    else
        sandbox_provider_for_provenance=""
    fi

    # Benchmark-specific setup, run before the server starts (e.g. verifying SWE-agent + the
    # streaming patch for swe-bench-lite). Empty for imo-answerbench and mcp-atlas (the
    # mcp-atlas judge is configured by the judge: block in its agents yaml, not env vars).
    extra_setup=$(echo "$extra_setup_b64" | base64 -d)
    if [[ -n "$extra_setup" ]]; then
        echo "Running extra setup for $benchmark..."
        eval "$extra_setup"
    fi

    server_command=$(echo "$server_b64" | base64 -d)
    echo "Starting server: $server_command"
    server_init_start_time=$(date +%s)
    # eval is needed for the command's embedded line continuations; backgrounding
    # inside the eval string means $! still captures the server's PID.
    eval "$server_command &> \"$RUN_DIR/server.log\" &"
    SERVER_PID=$!

    # Wait until the /v1/models endpoint returns HTTP 200 (matches the K8s template's own
    # server-readiness check for the agentic server; MoE's run_benchmarks.sh polls /health
    # instead, which the MoE-CAP server implementations expose but vllm/sglang's own OpenAI
    # front-ends don't).
    echo "Waiting for server $SERVER_PID to be ready..."

    until curl -s -f http://127.0.0.1:8000/v1/models > /dev/null; do
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

        export OUTPUT_ROOT="$RUN_DIR/agentic"
        mkdir -p "$OUTPUT_ROOT"

        # TEAS_* env vars, read by agent_cap.agents.teas_output at run end (metadata_*.json /
        # metrics_*.json hardware+model fields) -- same set the K8s template exports.
        export TEAS_ENGINE="${row[inference_engine]}"
        export TEAS_ENGINE_VERSION="$AGENTIC_ENGINE_VERSION"
        export TEAS_GPU_TYPE="$TEAS_GPU_NAME"
        export TEAS_NUM_GPUS="${row[num_gpu]}"
        export TEAS_TP="${row[num_gpu]}"
        export TEAS_MODEL_NAME="$HF_MODEL_PATH"
        export TEAS_PRECISION="mxfp4"
        export TEAS_CPU_TYPE="$(lscpu 2>/dev/null | grep 'Model name' | head -1 | cut -d: -f2 | xargs || true)"
        export TEAS_NUM_CPUS="$(nproc 2>/dev/null || echo 0)"
        teas_env_exports=$(echo "$teas_env_b64" | base64 -d)
        if [[ -n "$teas_env_exports" ]]; then
            eval "$teas_env_exports"
        fi

        # Try to get the error code without triggering set -e so we can handle it properly below
        # without killing the whole process.
        if run_benchmark "$RUN_DIR" "$tool_server_b64" "$client_b64"; then
            RUN_SUCCESS=0
        else
            RUN_SUCCESS=$?
        fi

        # Stop the tool server (if one was started for this row) so its port is free for the
        # next row. It execs into the uvicorn process (see AgentCAP-teasbench/mcp-server/start.sh),
        # so match on that rather than the start.sh command line, which no longer exists once exec runs.
        if [[ -n "$(echo "$tool_server_b64" | base64 -d)" ]]; then
            pkill -f "agent_environment.main:app" 2>/dev/null || true
        fi

        # If run is successful, stage results and push to repo. Otherwise, skip.
        if [[ $RUN_SUCCESS -eq 0 ]]; then
            job_end_time=$(date +%s)
            job_duration=$((job_end_time - job_start_time))
            echo "Benchmark run completed, pushing results to repo..."
            if push_results "$RUN_DIR" "$RUN_SUBDIR" "$job_duration" row "$sandbox_provider_for_provenance"; then
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

 # End of main benchmarking loop, reading next CSV row.
done < <(echo "$BENCHMARK_CSV" | base64 -d | tail -n +2)

benchmark_end_timestamp=$( date +%Y%m%d-%H%M )
echo "-----------------------------------------"
echo "Benchmarking completed at $benchmark_end_timestamp."
echo "-----------------------------------------"

# Self-destruct the instance or else Vast.ai will restart at the entrypoint until the instance is
# externally destroyed. Use an API call through curl to avoid having to bake the Vast.ai CLI into
# the image. We also need to retry on failure, and as a last resort, sleep forever instead of exiting.
# If the script were to exit instead, without the instance being destroyed, Vast.ai would just
# restart the container and run the set of benchmarks again.
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
