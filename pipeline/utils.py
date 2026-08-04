#!/bin/python3

"""
Supported GPU products on EIDF:
    nvidia.com/gpu.product: 'NVIDIA-A100-SXM4-80GB'
    nvidia.com/gpu.product: 'NVIDIA-A100-SXM4-40GB'
    nvidia.com/gpu.product: 'NVIDIA-A100-SXM4-40GB-MIG-3g.20gb'
    nvidia.com/gpu.product: 'NVIDIA-A100-SXM4-40GB-MIG-1g.5gb'
    nvidia.com/gpu.product: 'NVIDIA-H100-80GB-HBM3'
    nvidia.com/gpu.product: 'NVIDIA-H200'
"""

MODEL_SHORT_NAME_MAP={
    "gpt-oss-20b": "gptoss20b",
    "gpt-oss-120b": "gptoss120b",
    "Qwen3-235B-A22B-Instruct-2507": "qwen3-235b",
    "Qwen3-235B-A22B-Instruct-2507-FP8": "qwen3-235b-fp8",
    "DeepSeek-R1": "deepseek-r1",
    "Kimi-K2.5": "kimi-k2.5",
    "Qwen3-4B": "qwen3-4b"
}

DATASET_SHORT_NAME_MAP={
    "gsm8k": "gsm8k",
    "arena-hard": "arena-hard",
    "longbench_v1": "longbench",
    # Agentic benchmarks are not actually shortened in run names/paths (see
    # get_run_name / results_repo_dir below, which use the raw 'benchmark'
    # value directly) -- entries are kept here too purely so any code that
    # looks a benchmark up in this map, by analogy with the MoE 'dataset'
    # column, doesn't KeyError.
    "imo-answerbench": "imo-answerbench",
    "mcp-atlas": "mcp-atlas",
    "swe-bench-lite": "swe-bench-lite",
    }

HF_MODEL_MAP={
    "gpt-oss-20b": "unsloth/gpt-oss-20b",
    "gpt-oss-120b": "unsloth/gpt-oss-120b",
    "Qwen3-235B-A22B-Instruct-2507": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-235B-A22B-Instruct-2507-FP8": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "Kimi-K2.5": "moonshotai/Kimi-K2.5",
    "Qwen3-4B": "Qwen/Qwen3-4B"
}

# Container disc space to request per model on Vast.ai (--disk, in GB).
# Require space for full Python, CUDA, inference engine installation but also
# the model which varies quite a bit across those we use.
# Verify model requirement with `hf download --dry-run <model name>`
# and add some extra as buffer for the rest of the software.
MODEL_DISK_GB_MAP={
    "gpt-oss-20b": 60,                          # model = 27.5 GB
    "gpt-oss-120b": 160,                        # model = 130.5 GB
    "Qwen3-235B-A22B-Instruct-2507": 550,       # model = 470.2 GB
    "Qwen3-235B-A22B-Instruct-2507-FP8": 280,   # model = 236.4 GB
    "DeepSeek-R1": 800,                         # model = 688.6 GB
    "Kimi-K2.5": 700                            # model = 595.2 GB
}


EIDF_GPU_MAP={
    "A100":"NVIDIA-A100-SXM4-80GB",
    "H100":"NVIDIA-H100-80GB-HBM3",
    "H200":"NVIDIA-H200"
}

# Directory name under $TEAS_OUTPUT_DIR on the PVC where jobs archive their full run
# output. Deliberately distinct from the results_repo (git) name: the PVC copy is a
# plain archive, never a git working tree, so it must not collide with a directory a
# job's throwaway /dev/shm clone of the results repo might also be named after.
PVC_ARCHIVE_DIR = "TEAS_Development_Results_Private-archive-nogit"

VAST_GPU_MAP={
    "A100": "A100_SXM4",
    "H100": "H100_SXM",
    "H200": "H200_SXM",
}

# Human-readable GPU display names, used for the TEAS_GPU_TYPE env var read by
# agent_cap.agents.teas_output (see pipeline/templates/agentic.yaml).
TEAS_GPU_NAME_MAP={
    "A100": "NVIDIA A100",
    "H100": "NVIDIA H100",
    "H200": "NVIDIA H200",
}

# Platforms the pipeline can target.
PLATFORMS = {"eidf", "vastai"}

# Root under which models are staged on the cluster PVC. Combined with
# HF_MODEL_MAP this yields a local on-disk path some engines/benchmarks can
# load a model from instead of downloading it via HF_HOME, e.g.
# gpt-oss-120b -> /llm-cache-pvc/models/unsloth/gpt-oss-120b
MODELS_ROOT = "/llm-cache-pvc/models"

# Benchmarks that run as a single all-in-one agentic command (agent_cap.agents
# manages its own inference server / sandbox / evaluator) rather than the MoE
# server+client split. These are carried on the CSV 'benchmark' column, kept
# separate from the MoE 'dataset' column so the two families never collide.
AGENTIC_BENCHMARKS = {"imo-answerbench", "mcp-atlas", "swe-bench-lite"}

# The MCP Atlas server set. This *is* the mcp-atlas benchmark definition --
# AgentCAP's docs are explicit that dropping any server changes the results --
# so both platforms must enable exactly this list or their numbers stop being
# comparable. EIDF passes it to the sidecar container and Vast.ai writes it into
# the generated .env; tests/test_mcp_env.py asserts the two stay in step.
MCP_ENABLED_SERVERS = (
    "arxiv,brave-search,calculator,cli-mcp-server,"
    "clinicaltrialsgov-mcp-server,context7,ddg-search,desktop-commander,"
    "fetch,filesystem,git,github,mcp-code-executor,mcp-server-code-runner,"
    "memory,met-museum,open-library,osm-mcp-server,pubmed,weather,whois,"
    "wikipedia"
)

# Pipeline families, declared per row in the leading CSV 'family' column.
# "moe" is the basic server+client benchmark family (gsm8k, arena-hard,
# longbench_v1); "agentic" is AGENTIC_BENCHMARKS above. The values match the
# top-level directories in the results repo (moe/... and agentic/...).
FAMILIES = {"moe", "agentic"}


def benchmark_family(p: dict):
    """Return the pipeline family for an experiment row, from its `family`
    column.

    Every experiment CSV declares its family explicitly in a leading `family`
    column, whose value is one of FAMILIES. It is deliberately not inferred
    from the presence of other columns: the family selects which job template,
    which in-container runner, and which results-repo tree a row uses, and
    that is too consequential to leave implicit. A row whose family is missing
    or unrecognised is an error, not a default.
    """
    family = p.get("family")
    if isinstance(family, str):
        family = family.strip()
    if family not in FAMILIES:
        raise ValueError(
            f"experiment row has family {family!r}; expected one of "
            f"{sorted(FAMILIES)}. Every experiments CSV needs a leading "
            f"'family' column - see pipeline/README.md."
        )
    if family == "agentic" and p.get("benchmark") not in AGENTIC_BENCHMARKS:
        raise ValueError(
            f"family 'agentic' row has benchmark {p.get('benchmark')!r}; "
            f"expected one of {sorted(AGENTIC_BENCHMARKS)}."
        )
    return family


def needs_login_node_driver(p: dict):
    """True when this row cannot run as an unattended in-cluster Job.

    EIDF does not grant pods RBAC, so a benchmark whose driver must create
    Kubernetes objects mid-run (currently only swe-bench-lite, for its per-task
    sandbox and eval Jobs) has to be driven from a login node using the user's
    own credentials. The engine still runs on GPUs as an ordinary Job; only the
    driver process moves. See docs/DEVELOPER_GUIDE.md 5.

    imo-answerbench and mcp-atlas never touch the Kubernetes API, so they stay
    unattended Jobs on every platform.
    """
    return (p.get("platform", "eidf") == "eidf"
            and p.get("benchmark") == "swe-bench-lite")


def swe_bench_lite_k8s(p: dict):
    """True for swe-bench-lite on EIDF specifically, for now -- scoped to
    "== eidf" rather than "!= vastai" until a second k8s platform actually
    exists, so this can't accidentally claim to cover one that hasn't been
    validated. Revisit this check when that happens.

    Deliberately NOT the same condition as needs_login_node_driver, even
    though the two happen to agree everywhere today: this one is about which
    engine build/launch-flags to use (see agentic_engine_image /
    agentic_engine_server_flags in configs/config.yaml, and _agentic()) --
    a property of the benchmark, not of where the driver process runs.
    needs_login_node_driver is a topology question (RBAC availability) that
    will diverge from this one the day EIDF gets RBAC or a second k8s
    platform is added; conflating them would silently drop the validated
    swe-bench-lite engine recipe for such a row.

    False for Vast.ai: that platform never reaches this code path at all
    (pipeline/vast/resolve_commands.py calls Template.build_command directly
    with cmd_type "agentic_server", not through _agentic()), but the name
    still says so explicitly in case that ever changes.
    """
    return (p.get("benchmark") == "swe-bench-lite"
            and p.get("platform", "eidf") == "eidf")


def local_model_path(model: str):
    return f"{MODELS_ROOT}/{HF_MODEL_MAP[model]}"


def get_run_name(p: dict):
    if benchmark_family(p) == "agentic":
        return (f"{p['inference_engine']}_{MODEL_SHORT_NAME_MAP[p['model']]}"
                f"_{p['benchmark']}_nt{p['num_tasks']}_{p['gpu']}x{p['num_gpu']}")

    name = f"{p['inference_engine']}_{MODEL_SHORT_NAME_MAP[p['model']]}_{DATASET_SHORT_NAME_MAP[p['dataset']]}_ns{p['num_samples']}_{p['gpu']}x{p['num_gpu']}"

    if p['batch_size'] == "default":
        name += "_bsd"
    else:
        name += f"_bs{p['batch_size']}"
    if p['input_length'] != None:
        name += f"_i{p['input_length']}"
    if p['output_length'] != None:
        name += f"_o{p['output_length']}"
    return name

def k8s_friendlify(unfriendly_string):
    return unfriendly_string.replace("_", "-").lower()

def results_repo_dir(p: dict):
    platform = p.get("platform", "eidf")

    if benchmark_family(p) == "agentic":
        # agentic/<platform>/<engine>/<model>/<benchmark>/<hw>x<num_gpu>/batch-size-<batch_size>
        # (matches TEAS_Results_Private/agentic/** and the 6-level parser in
        # postprocessing/aggregate_results.py:parse_run_path -- no '_Ntasks'
        # suffix on the benchmark directory, and batch-size- is NOT omitted).
        return (f"agentic/{platform}/{p['inference_engine']}/{p['model'].lower()}"
                f"/{p['benchmark']}/{p['gpu'].lower()}x{p['num_gpu']}"
                f"/batch-size-{p['batch_size']}")

    dir = f"moe/{platform}/{p['inference_engine']}/{p['model'].lower()}/{p['dataset']}_{p['num_samples']}samples/{p['gpu'].lower()}x{p['num_gpu']}"
    if p['batch_size'] == "default":
        dir += f"/batch-size-default"
    else:
        dir += f"/batch-size-{p['batch_size']}"
    if p['input_length'] != None:
        dir += f"_input{p['input_length']}"
    if p['output_length'] != None:
        dir += f"_output{p['output_length']}"

    return dir


