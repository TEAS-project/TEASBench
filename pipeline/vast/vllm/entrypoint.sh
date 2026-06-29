#!/usr/bin/env bash
set -euo pipefail

# Copy MoE-CAP into /dev/shm from its build-time location in /opt.
echo "MoE-CAP from /opt/MoE-CAP into /dev/shm..."
cp -a /opt/MoE-CAP /dev/shm/
echo "Copied /opt/MoE-CAP to /dev/shm/MoE-CAP"

# If we need to set anything specific for vLLM before running the benchmark, do it here.

# Run the benchmarks.
exec /root/run_benchmarks.sh
