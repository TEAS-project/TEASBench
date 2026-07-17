#!/bin/python3

import argparse
import base64
import pathlib
import shlex
import sys

# template.py, utils.py and configs/ live one directory up, both in the repo
# (pipeline/) and in the container image (/opt/teasbench/pipeline/).
PIPELINE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

import yaml

from template import Template
from utils import HF_MODEL_MAP

def resolve_env_exports(template, config, matching_rules, parameters):
    """Translate any extra_container_env rules from the config into
    'export NAME=VALUE' shell lines.
    Entries that use valueFrom (e.g. secretKeyRef) are skipped: those only
    resolve inside a k8s pod, and on Vast.ai the variables they'd populate
    (e.g. OPENAI_API_KEY) are already required directly as container env vars.
    """
    raw = template.resolve_generic_variable(
        "extra_container_env", config, matching_rules, parameters
    )
    if not raw:
        return ""
    entries = yaml.safe_load(raw) or []
    exports = [
        f"export {entry['name']}={shlex.quote(str(entry['value']))}"
        for entry in entries
        if "value" in entry
    ]
    return "\n".join(exports)

def main():
    """
    run_benchmarks.sh calls this script in order to use the same template.py code
    used in the EIDF pipeline to build the server and client commands. This means
    we can use the same pipeline/configs/config.yaml included in the image to
    determine those commands, avoiding the two pipelines falling out of sync
    with one another.

    It also translates from the short-form model names used in the CSV files
    to the full Huggingface paths needed by MoE-CAP.

    Prints exactly four lines to stdout which run_benchmarks.sh picks up:
      1. the Huggingface model path (e.g. unsloth/gpt-oss-20b)
      2. the server command, base64-encoded
      3. the client command, base64-encoded
      4. env var exports needed before starting the server, base64-encoded
    The commands are base64-encoded because they contain backslash-newline
    continuations, which otherwise break line-based parsing in bash.
    """

    # run_benchmarks.sh provides these args directly from the current row of the CSV.
    parser = argparse.ArgumentParser(
        description="Resolve server/client commands for one benchmark run using config.yaml."
    )
    parser.add_argument("--inference-engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-samples", required=True)
    parser.add_argument("--gpu", required=True)
    # num_gpu must be an int in order to match with the YAML rules in config.yaml.
    parser.add_argument("--num-gpu", required=True, type=int)
    # batch_size stays a string: config.yaml rules match "1"/"default" as strings.
    parser.add_argument("--batch-size", required=True)
    parser.add_argument("--input-length", default=None)
    parser.add_argument("--output-length", default=None)
    args = parser.parse_args()

    if args.model not in HF_MODEL_MAP:
        sys.exit(f"ERROR: unknown model {args.model!r}; add it to HF_MODEL_MAP in utils.py")

    parameters = {
        "inference_engine": args.inference_engine,
        "model": args.model,
        "hf_model_path": HF_MODEL_MAP[args.model],
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "gpu": args.gpu,
        "num_gpu": args.num_gpu,
        "tensor_parallel_size": args.num_gpu,
        "batch_size": args.batch_size,
        "input_length": args.input_length,
        "output_length": args.output_length,
    }

    with open(PIPELINE_DIR / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    template = Template()
    matching_rules = template.get_matching_rules(config.get("rules", []), parameters)
    server_cmd = template.build_command("server", config, parameters, matching_rules)
    client_cmd = template.build_command("client", config, parameters, matching_rules)
    env_exports = resolve_env_exports(template, config, matching_rules, parameters)

    print(parameters["hf_model_path"])
    print(base64.b64encode(server_cmd.encode()).decode())
    print(base64.b64encode(client_cmd.encode()).decode())
    print(base64.b64encode(env_exports.encode()).decode())

if __name__ == "__main__":
    main()
