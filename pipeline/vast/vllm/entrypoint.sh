#!/usr/bin/env bash
set -euo pipefail

# Copy MoE-CAP into /dev/shm from its build-time location in /opt.
echo "Copying MoE-CAP from /opt/MoE-CAP into /dev/shm and installing it..."
cp -a /opt/MoE-CAP /dev/shm/
python3 -m pip install -e /dev/shm/MoE-CAP
echo "Finished copying and installing MoE-CAP."

# We're running in a vLLM-based container, so specify for run_benchmarks.sh
# that we should only run parameter sets from the CSV for vLLM.
export ALLOWED_ENGINE="vllm"

# Potentially set anything else specific for vLLM benchmarks here.

# Run the benchmarks.
exec /root/run_benchmarks.sh
