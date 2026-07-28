#!/usr/bin/env bash
#
# Sanity-check the AgentCAP vLLM image before launching the agentic runner.
# Unlike the sglang image, no torch/torchvision repair is needed here, so this
# just verifies the key libraries import and CUDA is visible. Versioned inside
# TEASBench so the check lives in one auditable place; template.py inlines the
# contents of this file into the generated agentic job at generation time.
#
# NOTE: keep this script heredoc-free. Its lines are re-indented when inlined
# into the YAML block scalar, which would break an unquoted heredoc terminator.
#
set -euo pipefail

echo "Checking vLLM environment..."
python -c "import torch, vllm; print('torch', torch.__version__, '| cuda', torch.version.cuda); print('vllm', vllm.__version__); print('cuda available', torch.cuda.is_available())"
