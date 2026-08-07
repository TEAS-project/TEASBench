# tt-llm-run.sh

Brings up a [tt-inference-server](https://github.com/tenstorrent/tt-inference-server) vLLM server on Tenstorrent hardware and smoke-tests it with a completion request. Handles weight download/staging, offline setup, card reset, launch, and auth.

**Validated:** Llama-3.1-8B-Instruct on a single Blackhole p150.
**Pinned:** repo `v0.10.0` + image `0.10.0-55fd115-aa4ae1e` (kept as a matched pair).

Based on Tenstorrent's [Deploying LLMs with vLLM](https://docs.tenstorrent.com/getting-started/vLLM-servers.html) guide.

## Prerequisites

- Tenstorrent software stack installed (tt-metal built, `tt-smi` working) — see [Deploying LLMs with vLLM](https://docs.tenstorrent.com/getting-started/vLLM-servers.html) and its base-install guide.
- `huggingface_hub` on PATH (`pip install --user huggingface_hub`) and a HuggingFace token with model access (Llama is gated).

## Quick start

```bash
# environment (adjust paths to your install)
export TT_METAL_RUNTIME_ROOT=$HOME/tt-metal/
export PATH="/opt/tenstorrent/.tenstorrent-venv/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/tt-metal/build_Release/lib:$LD_LIBRARY_PATH"

export HF_TOKEN="<your-hugging-face-access-token>"
export JWT_SECRET="testing"

./tt-llm-run.sh --clone      # first run: clones the repo, downloads weights, launches
```

Subsequent runs: `./tt-llm-run.sh` (reuses a live server) or `./tt-llm-run.sh --force` (clean relaunch).

## Options

| Flag | Effect |
|---|---|
| `--clone` | Clone the repo into `REPO_DIR` if missing |
| `--force` | Tear down a running server and relaunch |
| `--test-only` | Smoke-test an already-running server |

Any CONFIG value can be overridden via env, e.g. `MODEL=... DEVICE=... ./tt-llm-run.sh`.

## Notes

- If a request hangs indefinitely, reset the card (`tt-smi -r`) and relaunch with `--force`.
- Run from a plain shell (not an activated venv) so the correct interpreter is picked for `run.py`.
- On success the script prints a ready-to-use token + `curl` for sending your own requests.
