#!/usr/bin/env bash
#
# Repair the torch / torchvision mismatch baked into the AgentCAP sglang image
# before launching SGLang. Versioned here (inside TEASBench) so the fix lives in
# exactly one auditable place; template.py inlines the contents of this file into
# the generated agentic job at generation time.
#
# This replaces the ~55 lines of inline pip-repair Python that were hand-pasted
# into the original imo_answerbench_sglang_gptoss_A100x2.yaml.
#
# NOTE: keep this script heredoc-free. Its lines are re-indented when inlined
# into the YAML block scalar, which would break an unquoted heredoc terminator.
#
set -euo pipefail

echo "Checking and repairing torch / torchvision before SGLang launch..."

echo "Before repair:"
python -m pip show torch torchvision sglang transformers || true

echo "Installing torchvision==0.24.1 for torch==2.9.1 CUDA 13.0..."
python -m pip install --force-reinstall --no-deps torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

echo "After repair / import checks:"
python -c "import torch, torchvision, torchvision.transforms, transformers, sglang; from transformers import AutoProcessor; print('torch', torch.__version__, '| cuda', torch.version.cuda); print('torchvision', torchvision.__version__); print('transformers', transformers.__version__); print('sglang', getattr(sglang, '__version__', 'unknown')); print('cuda available', torch.cuda.is_available()); print('AutoProcessor / torchvision.transforms import OK')"
