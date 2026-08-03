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
from utils import AGENTIC_BENCHMARKS, HF_MODEL_MAP, TEAS_GPU_NAME_MAP, benchmark_family, local_model_path

def resolve_env_exports(template, config, matching_rules, parameters):
    """Translate any extra_container_env rules from the config into
    'export NAME=VALUE' shell lines.
    Entries that use valueFrom (e.g. secretKeyRef) are skipped: those only
    resolve inside a k8s pod, and on Vast.ai the variables they'd populate
    (e.g. OPENAI_API_KEY) are already required directly as container env vars.

    Shared verbatim between the MoE and agentic branches of main() -- the
    extra_container_env variable and its valueFrom-skipping caveat apply
    identically to both families (e.g. the sglang expert-distribution-dir
    rule for MoE, or a future agentic rule that needs the same treatment).
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

def b64(text):
    return base64.b64encode(text.encode()).decode()

def resolve_moe(template, config, args):
    """Resolve the server/client commands for one MoE-family CSV row.

    Prints five lines to stdout:
      1. the Huggingface model path (e.g. unsloth/gpt-oss-20b)
      2. the server command, base64-encoded
      3. the client command, base64-encoded
      4. env var exports needed before starting the server, base64-encoded
      5. a fixed terminator, "END_MOE_CONTRACT"
    The commands are base64-encoded because they contain backslash-newline
    continuations, which otherwise break line-based parsing in bash.
    We also need the terminator (line 5) to help bash parse this output
    when field 4 is empty (i.e. no environment variables). Otherwise,
    it's possible for run_benchmarks.sh to run out of input and crash.
    Very similar to the terminator returned by resolve_agentic().
    """
    if args.dataset is None or args.num_samples is None:
        sys.exit("ERROR: --dataset and --num-samples are required for MoE rows "
                  "(rows with no --benchmark; see benchmark_family in utils.py)")

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

    matching_rules = template.get_matching_rules(config.get("rules", []), parameters)
    server_cmd = template.build_command("server", config, parameters, matching_rules)
    client_cmd = template.build_command("client", config, parameters, matching_rules)
    env_exports = resolve_env_exports(template, config, matching_rules, parameters)

    print(parameters["hf_model_path"])
    print(b64(server_cmd))
    print(b64(client_cmd))
    print(b64(env_exports))
    print("END_MOE_CONTRACT")

def resolve_agentic(template, config, args):
    """Resolve the agentic_server/agentic_client commands (and the
    supporting env/setup blocks) for one agentic-family CSV row, using
    exactly the same config.yaml rule engine as pipeline/template.py's
    _agentic() (Template.build_command / Template.resolve_generic_variable)
    -- no rule logic is reimplemented here, only assembled and printed for
    run_agentic_benchmarks.sh to consume.

    Prints a leading line naming the family ("agentic"), followed by nine
    more lines:
      1. "agentic"
      2. the Huggingface model path (e.g. unsloth/gpt-oss-120b)
      3. the agentic inference-engine version (config.yaml's
         agentic_inference_engine_version, e.g. "0.5.12.post1") -- for
         TEAS_ENGINE_VERSION
      4. the human-readable GPU name (utils.TEAS_GPU_NAME_MAP, e.g.
         "NVIDIA A100") -- for TEAS_GPU_TYPE
      5. the agentic server start command, base64-encoded
      6. the agentic client (agent_cap.agents) command, base64-encoded
      7. env var exports needed before starting the server
         (config.yaml's extra_container_env), base64-encoded
      8. @extra_setup@ -- benchmark-specific setup run after the AgentCAP
         checkout (e.g. SWE-agent + the streaming patch), base64-encoded
      9. the tool-server startup block, base64-encoded: Vast.ai's
         capability-equivalent of the EIDF template's sidecar_containers is
         config.yaml's tool_server_setup (e.g. "bash
         /opt/AgentCAP/mcp-server/start.sh &"); its readiness-poll half is
         pure bash with no k8s-specific content, so it is NOT duplicated --
         this line is tool_server_setup concatenated with the SAME
         sidecar_wait value the EIDF template also renders. Empty (blank
         line, base64 of "") for benchmarks that need no tool server
         (imo-answerbench, swe-bench-lite).
      10. @teas_env_exports@ -- the benchmark-specific TEAS_BACKEND export,
          base64-encoded
      11. a fixed terminator, "END_AGENTIC_CONTRACT". Unlike the MoE
          contract (where the last field, env exports, is sometimes the
          empty string), several of these later fields are routinely empty
          together (e.g. imo-answerbench's extra_setup, tool-server block,
          AND teas_env_exports are all "" at once). `resolved=$(...)`
          command substitution in bash strips *all* trailing newlines from
          captured output, which would silently swallow those trailing
          empty lines and make run_agentic_benchmarks.sh's `read` group run
          out of input (confirmed by reproducing it: an all-empty run of
          fields 8-10 collapses to nothing after the last non-empty field,
          and the extra `read`s past it return non-zero, which -- under
          `set -e` -- aborts the whole script). Printing one final,
          always-non-empty line stops it from ever being a *trailing*
          newline, so every real (possibly empty) line before it survives.
    All commands/blocks are base64-encoded because they may contain
    backslash-newline continuations or multi-line bash, which otherwise
    break line-based parsing in bash. The leading "agentic" line lets
    run_agentic_benchmarks.sh assert it got the family of output it
    expected, rather than silently misreading fields if it were ever called
    for the wrong row shape.
    """
    if args.benchmark not in AGENTIC_BENCHMARKS:
        sys.exit(f"ERROR: --benchmark {args.benchmark!r} is not one of {sorted(AGENTIC_BENCHMARKS)}")
    if args.num_tasks is None:
        sys.exit("ERROR: --num-tasks is required when --benchmark is set")

    parameters = {
        "inference_engine": args.inference_engine,
        "model": args.model,
        "hf_model_path": HF_MODEL_MAP[args.model],
        "gpu": args.gpu,
        "num_gpu": args.num_gpu,
        "tensor_parallel_size": args.num_gpu,
        "batch_size": args.batch_size,
        "benchmark": args.benchmark,
        "num_tasks": args.num_tasks,
        "concurrency": args.concurrency if args.concurrency is not None else 4,
        "platform": args.platform,
        "model_path": local_model_path(args.model),
    }

    matching_rules = template.get_matching_rules(config.get("rules", []), parameters)
    server_cmd = template.build_command("agentic_server", config, parameters, matching_rules)
    client_cmd = template.build_command("agentic_client", config, parameters, matching_rules)
    env_exports = resolve_env_exports(template, config, matching_rules, parameters)
    extra_setup = template.resolve_generic_variable("extra_setup", config, matching_rules, parameters)
    teas_env_exports = template.resolve_generic_variable("teas_env_exports", config, matching_rules, parameters)

    # tool_server_setup (Vast.ai: start as a background process) + sidecar_wait
    # (platform-agnostic readiness poll, reused verbatim -- see docstring).
    tool_server_start = template.resolve_generic_variable("tool_server_setup", config, matching_rules, parameters)
    sidecar_wait = template.resolve_generic_variable("sidecar_wait", config, matching_rules, parameters)
    tool_server_setup = "\n".join(s.strip() for s in (tool_server_start, sidecar_wait) if s.strip())

    agentic_engine_version = template.resolve_generic_variable(
        "agentic_inference_engine_version", config, matching_rules, parameters)
    teas_gpu_name = TEAS_GPU_NAME_MAP.get(args.gpu, args.gpu)

    print("agentic")
    print(parameters["hf_model_path"])
    print(agentic_engine_version)
    print(teas_gpu_name)
    print(b64(server_cmd))
    print(b64(client_cmd))
    print(b64(env_exports))
    print(b64(extra_setup))
    print(b64(tool_server_setup))
    print(b64(teas_env_exports))
    # Always-non-empty terminator -- see the "11." docstring entry above for why.
    print("END_AGENTIC_CONTRACT")

def main():
    """
    run_benchmarks.sh (MoE) and run_agentic_benchmarks.sh (agentic) both call
    this script in order to use the same template.py code used in the EIDF
    pipeline to build their server/client commands. This means both Vast.ai
    scripts stay driven by the one pipeline/configs/config.yaml included in
    the image, instead of duplicating its rules and falling out of sync with
    the EIDF pipeline (see the "Further work" note this replaced in
    pipeline/vast/README.md, and docs/agentic-pipeline-design.md §4).

    It also translates from the short-form model names used in the CSV files
    to the full Huggingface paths needed by MoE-CAP / agent_cap.agents.

    Family is selected the same way generate.py selects it: by whether
    --benchmark is set to one of utils.AGENTIC_BENCHMARKS (see
    utils.benchmark_family). MoE rows (no --benchmark) print a five-line
    contract ending in a fixed terminator -- see resolve_moe's docstring.
    Agentic rows print a ten-line contract with a leading "agentic" line
    -- see resolve_agentic's docstring.
    """

    # run_benchmarks.sh / run_agentic_benchmarks.sh provide these args
    # directly from the current row of the CSV.
    parser = argparse.ArgumentParser(
        description="Resolve server/client commands for one benchmark run using config.yaml."
    )
    parser.add_argument("--inference-engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", required=True)
    # num_gpu must be an int in order to match with the YAML rules in config.yaml.
    parser.add_argument("--num-gpu", required=True, type=int)
    # batch_size stays a string: config.yaml rules match "1"/"default" as strings.
    parser.add_argument("--batch-size", required=True)
    # Only meaningful for the agentic family (selects platform-scoped
    # config.yaml rules, e.g. swe-bench-lite's sandbox/exec provider choice).
    # Vast.ai is the only platform that runs this script, hence the default.
    parser.add_argument("--platform", default="vastai")
    # The row's pipeline family, straight from the CSV's leading 'family'
    # column. Passed explicitly rather than inferred from which other flags are
    # present, so the container resolves exactly the family the CSV declares.
    parser.add_argument("--family", required=True)
    # MoE-only: required (checked in resolve_moe) when family is 'moe'.
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--num-samples", default=None)
    parser.add_argument("--input-length", default=None)
    parser.add_argument("--output-length", default=None)
    # Agentic-only: required (checked in resolve_agentic) when family is 'agentic'.
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    if args.model not in HF_MODEL_MAP:
        sys.exit(f"ERROR: unknown model {args.model!r}; add it to HF_MODEL_MAP in utils.py")

    with open(PIPELINE_DIR / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    template = Template()

    if benchmark_family({"family": args.family, "benchmark": args.benchmark}) == "agentic":
        resolve_agentic(template, config, args)
    else:
        resolve_moe(template, config, args)

if __name__ == "__main__":
    main()
