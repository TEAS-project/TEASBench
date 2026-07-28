#!/usr/bin/env bash
set -euo pipefail

# Agentic-family entrypoint (see Dockerfile.agentic). The MoE entrypoint.sh
# alongside this one is untouched by the agentic work; the two families ship as
# separate images so agentic dependencies can never destabilise a MoE sweep.

# Install AgentCAP from its build-time location in /opt. Unlike MoE-CAP in the
# MoE image it is not copied into /dev/shm first: nothing writes into the
# checkout at runtime, and /dev/shm is where the run directories live.
echo "Installing AgentCAP from /opt/AgentCAP..."
python3 -m pip install -e /opt/AgentCAP
echo "Finished installing AgentCAP."

# We're running in an SGLang-based container, so specify for
# run_agentic_benchmarks.sh that we should only run parameter sets from the CSV
# for SGLang.
export ALLOWED_ENGINE="sglang"

# Potentially set anything else specific for SGLang benchmarks here.

# Run the benchmarks.
exec /root/run_agentic_benchmarks.sh
