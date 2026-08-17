#!/bin/python3

"""Maps and predicates shared by the pipeline generator.

Everything here is site-independent. Anything that varies per site -- namespace,
Kueue queue, PVC names, GPU node-label values, model staging root, whether pods
get RBAC -- lives in a site profile under configs/sites/ and is reached through
load_site() below.
"""

import functools
import os
import re

import yaml

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
    "Qwen3-4B": 60,                             # model = 8.1 GB; the software stack dominates here
    "gpt-oss-20b": 60,                          # model = 27.5 GB
    "gpt-oss-120b": 160,                        # model = 130.5 GB
    "Qwen3-235B-A22B-Instruct-2507": 550,       # model = 470.2 GB
    "Qwen3-235B-A22B-Instruct-2507-FP8": 280,   # model = 236.4 GB
    "DeepSeek-R1": 800,                         # model = 688.6 GB
    "Kimi-K2.5": 700                            # model = 595.2 GB
}


# Directory name under $TEAS_OUTPUT_DIR on the PVC where jobs archive their full run
# output. Deliberately distinct from the results_repo (git) name: the PVC copy is a
# plain archive, never a git working tree, so it must not collide with a directory a
# job's throwaway /dev/shm clone of the results repo might also be named after.
PVC_ARCHIVE_DIR = "TEAS_Development_Results_Private-archive-nogit"

# Human-readable GPU display names, used for the TEAS_GPU_TYPE env var read by
# agent_cap.agents.teas_output (see pipeline/templates/agentic.yaml).
TEAS_GPU_NAME_MAP={
    "A100": "NVIDIA A100",
    "H100": "NVIDIA H100",
    "H200": "NVIDIA H200",
}

# Site profiles live one per file in configs/sites/. A row's `platform` column
# names the profile it runs under, so `platform` keeps meaning the *site* -- the
# label results are published under -- while the *mechanism* comes from that
# profile's `orchestrator`. See configs/sites/eidf.yaml for the full rationale.
SITES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "configs", "sites")

# Orchestrators a site profile may declare: how work is launched, as opposed to
# where it runs. Deliberately separate from the set of sites, since any number
# of sites can share one orchestrator -- that separation is the whole point.
ORCHESTRATORS = {"k8s", "vastai"}


def available_sites():
    """Site names with a profile in configs/sites/ (the valid `platform` values)."""
    if not os.path.isdir(SITES_DIR):
        return set()
    return {f[:-5] for f in os.listdir(SITES_DIR) if f.endswith(".yaml")}


@functools.lru_cache(maxsize=None)
def load_site(name):
    """Read and validate configs/sites/<name>.yaml.

    Cached: generate.py resolves the site once per CSV row, and a sweep is
    hundreds of rows against the same one or two profiles.
    """
    path = os.path.join(SITES_DIR, f"{name}.yaml")
    if not os.path.isfile(path):
        raise ValueError(
            f"unknown platform {name!r}: no site profile at {path}. "
            f"Known sites: {sorted(available_sites())}. Add one by copying an "
            f"existing profile in pipeline/configs/sites/."
        )
    with open(path) as f:
        site = yaml.safe_load(f) or {}

    orchestrator = site.get("orchestrator")
    if orchestrator not in ORCHESTRATORS:
        raise ValueError(
            f"site profile {path} declares orchestrator {orchestrator!r}; "
            f"expected one of {sorted(ORCHESTRATORS)}."
        )
    # A k8s site without these renders a Job with an empty namespace or an
    # unmountable volume -- both fail late and confusingly, so fail here instead.
    if orchestrator == "k8s":
        for key in ("namespace", "models_root", "pvcs", "gpu_products"):
            if not site.get(key):
                raise ValueError(
                    f"site profile {path} has orchestrator 'k8s' but no {key!r}."
                )
    return site

# Benchmarks that run as a single all-in-one agentic command (agent_cap.agents
# manages its own inference server / sandbox / evaluator) rather than the MoE
# server+client split. These are carried on the CSV 'benchmark' column, kept
# separate from the MoE 'dataset' column so the two families never collide.
AGENTIC_BENCHMARKS = {"imo-answerbench", "mcp-atlas", "swe-bench-lite"}

# The MCP Atlas server set. This *is* the mcp-atlas benchmark definition --
# AgentCAP's docs are explicit that dropping any server changes the results --
# so both platforms must enable exactly this list or their numbers stop being
# comparable. The K8s path passes it to the sidecar container and Vast.ai writes it into
# the generated .env; tests/test_mcp_env.py asserts the two stay in step.
MCP_ENABLED_SERVERS = (
    "arxiv,brave-search,calculator,cli-mcp-server,"
    "clinicaltrialsgov-mcp-server,context7,ddg-search,desktop-commander,"
    "fetch,filesystem,git,github,mcp-code-executor,mcp-server-code-runner,"
    "memory,met-museum,open-library,osm-mcp-server,pubmed,weather,whois,"
    "wikipedia"
)

# Identity stamped into every leaf of the controlled repeatability /
# engine-build study. Rows opt in via a 'study_block' CSV column (E1..E6);
# see experiments/replication-study-eidf.csv and
# pipeline/k8s/helpers/run_study_block.sh. The x2 suffix separates this fresh
# six-block design from earlier development pilots that reused E1 on x1
# hardware; study_id + block_id must never pool those populations.
STUDY_ID = "controlled-variation-2026-x2"

# Exact within-block order for the frozen six-block x2 study. Each pair lists
# (engine, dataset, first build, second build); expansion below assigns orders
# 1..12. Keeping this independent of the CSV makes malformed or ad-hoc study
# rows fail generation instead of silently acquiring study provenance.
STUDY_PAIR_PLANS = {
    "e1": (
        ("vllm", "gsm8k", "0.16.0", "0.21.0"),
        ("vllm", "arena-hard", "0.16.0", "0.21.0"),
        ("vllm", "longbench_v1", "0.16.0", "0.21.0"),
        ("sglang", "gsm8k", "0.5.9", "0.5.12.post1"),
        ("sglang", "arena-hard", "0.5.9", "0.5.12.post1"),
        ("sglang", "longbench_v1", "0.5.9", "0.5.12.post1"),
    ),
    "e2": (
        ("sglang", "arena-hard", "0.5.12.post1", "0.5.9"),
        ("sglang", "longbench_v1", "0.5.12.post1", "0.5.9"),
        ("sglang", "gsm8k", "0.5.12.post1", "0.5.9"),
        ("vllm", "arena-hard", "0.21.0", "0.16.0"),
        ("vllm", "longbench_v1", "0.21.0", "0.16.0"),
        ("vllm", "gsm8k", "0.21.0", "0.16.0"),
    ),
    "e3": (
        ("vllm", "longbench_v1", "0.21.0", "0.16.0"),
        ("vllm", "gsm8k", "0.21.0", "0.16.0"),
        ("vllm", "arena-hard", "0.21.0", "0.16.0"),
        ("sglang", "longbench_v1", "0.5.12.post1", "0.5.9"),
        ("sglang", "gsm8k", "0.5.12.post1", "0.5.9"),
        ("sglang", "arena-hard", "0.5.12.post1", "0.5.9"),
    ),
    "e4": (
        ("sglang", "gsm8k", "0.5.9", "0.5.12.post1"),
        ("sglang", "longbench_v1", "0.5.9", "0.5.12.post1"),
        ("sglang", "arena-hard", "0.5.9", "0.5.12.post1"),
        ("vllm", "gsm8k", "0.16.0", "0.21.0"),
        ("vllm", "longbench_v1", "0.16.0", "0.21.0"),
        ("vllm", "arena-hard", "0.16.0", "0.21.0"),
    ),
    "e5": (
        ("vllm", "longbench_v1", "0.21.0", "0.16.0"),
        ("vllm", "arena-hard", "0.21.0", "0.16.0"),
        ("vllm", "gsm8k", "0.21.0", "0.16.0"),
        ("sglang", "longbench_v1", "0.5.12.post1", "0.5.9"),
        ("sglang", "arena-hard", "0.5.12.post1", "0.5.9"),
        ("sglang", "gsm8k", "0.5.12.post1", "0.5.9"),
    ),
    "e6": (
        ("sglang", "arena-hard", "0.5.9", "0.5.12.post1"),
        ("sglang", "gsm8k", "0.5.9", "0.5.12.post1"),
        ("sglang", "longbench_v1", "0.5.9", "0.5.12.post1"),
        ("vllm", "arena-hard", "0.16.0", "0.21.0"),
        ("vllm", "gsm8k", "0.16.0", "0.21.0"),
        ("vllm", "longbench_v1", "0.16.0", "0.21.0"),
    ),
}


def _study_coordinates():
    coordinates = {}
    for block, pairs in STUDY_PAIR_PLANS.items():
        leaves = []
        for engine, dataset, first, second in pairs:
            leaves.extend(((engine, first, dataset), (engine, second, dataset)))
        coordinates[block] = dict(enumerate(leaves, 1))
    return coordinates


STUDY_COORDINATES = _study_coordinates()
STUDY_HARDWARE = {
    "e1": "A100", "e2": "A100", "e3": "A100",
    "e4": "H100", "e5": "H100", "e6": "H100",
}

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


def site_of(p: dict):
    """The site profile for an experiment row, named by its `platform` column.

    Defaults to eidf for CSVs that predate the column, matching generate.py.
    """
    return load_site(p.get("platform", "eidf"))


def orchestrator_of(p: dict):
    """How this row is launched: "k8s" or "vastai". See ORCHESTRATORS."""
    return site_of(p)["orchestrator"]


def needs_login_node_driver(p: dict):
    """True when this row cannot run as an unattended in-cluster Job.

    On a cluster whose site profile sets grants_pod_rbac: false, a pod cannot
    create Kubernetes objects through its ServiceAccount. A benchmark whose
    driver must create them mid-run (currently only swe-bench-lite, for its
    per-task sandbox and eval Jobs) therefore has to be driven from a login node
    using the user's own credentials. The engine still runs on GPUs as an
    ordinary Job; only the driver process moves. See docs/DEVELOPER_GUIDE.md 5.

    imo-answerbench and mcp-atlas never touch the Kubernetes API, so they stay
    unattended Jobs everywhere. Vast.ai profiles report grants_pod_rbac: false
    too, but never reach this predicate -- vast_generate.py is a separate path.
    """
    site = site_of(p)
    return (site["orchestrator"] == "k8s"
            and not site.get("grants_pod_rbac", False)
            and p.get("benchmark") == "swe-bench-lite")


def swe_bench_lite_on_k8s(p: dict):
    """True for swe-bench-lite launched as Kubernetes Jobs, on any cluster.

    Deliberately NOT the same condition as needs_login_node_driver, even though
    the two agree on every site defined today: this one is about which engine
    build/launch-flags to use (see swebench_k8s_engine_image /
    swebench_k8s_engine_server_command in configs/config.yaml, and _agentic()) --
    a property of the benchmark and the orchestrator, not of where the driver
    process runs. needs_login_node_driver is a topology question (does this
    cluster grant pods RBAC), and the two diverge on a cluster that does:
    conflating them would silently drop the validated swe-bench-lite engine
    recipe for such a row.

    False for Vast.ai: that orchestrator never reaches this code path at all
    (pipeline/vast/resolve_commands.py calls Template.build_command directly
    with cmd_type "agentic_server", not through _agentic()), but the name
    still says so explicitly in case that ever changes.
    """
    return (p.get("benchmark") == "swe-bench-lite"
            and orchestrator_of(p) == "k8s")


def local_model_path(model: str, site: dict):
    """On-disk path this site stages `model` at, under its models_root.

    Empty for a site that stages nothing: Vast.ai rents a bare instance and
    downloads from HF at run time, so it has no models_root and nothing on
    that path consumes model_path (only hf_model_path reaches a CLI flag).
    """
    root = site.get("models_root")
    return f"{root}/{HF_MODEL_MAP[model]}" if root else ""


def study_fields(p: dict):
    """Return a validated frozen-study (block, version), or None.

    Every study coordinate is allowlisted here, independently of the CSV.
    Study-only fields without a block are rejected too, so a direct generator
    call cannot smuggle an arbitrary image version into an ordinary run.
    """
    block = p.get("study_block")
    if block is None or block == "":
        stray = [key for key in ("engine_version", "study_order", "moe_cap_ref")
                 if p.get(key) not in (None, "")]
        if stray:
            raise ValueError(
                f"study-only fields {stray} require a valid study_block")
        return None
    if p.get("family") != "moe":
        raise ValueError("study_block is only supported for family 'moe' rows")
    if not re.fullmatch(r"E[1-6]", str(block)):
        raise ValueError(f"study_block {block!r}: expected E1..E6")
    block = str(block).lower()
    version = p.get("engine_version")
    if version is None or str(version).strip() == "":
        raise ValueError(
            f"study row (block {block}) has no engine_version; study rows "
            "must pin the build explicitly")
    def decimal_integer(key):
        value = p.get(key)
        if not re.fullmatch(r"[0-9]+", str(value)):
            raise ValueError(f"study row {block}: {key} must be a decimal integer")
        parsed = int(str(value), 10)
        if str(value) != str(parsed):
            raise ValueError(f"study row {block}: {key} must use canonical decimal form")
        return parsed

    order = decimal_integer("study_order")
    num_samples = decimal_integer("num_samples")
    num_gpu = decimal_integer("num_gpu")
    fixed = {
        "model": "gpt-oss-120b",
        "num_samples": 256,
        "gpu": STUDY_HARDWARE[block],
        "num_gpu": 2,
        "batch_size": "default",
    }
    actual = {
        "model": p.get("model"),
        "num_samples": num_samples,
        "gpu": p.get("gpu"),
        "num_gpu": num_gpu,
        "batch_size": p.get("batch_size"),
    }
    if actual != fixed:
        raise ValueError(f"study row {block}/{order}: frozen fields {fixed}, got {actual}")
    expected = STUDY_COORDINATES[block].get(order)
    coordinate = (p.get("inference_engine"), str(version), p.get("dataset"))
    if expected is None or coordinate != expected:
        raise ValueError(
            f"study row {block}/{order}: expected {expected}, got {coordinate}")
    return block, str(version)


def study_version_token(version: str):
    """Dot-free engine version for k8s names: '0.5.12.post1' -> '0512p1'."""
    return str(version).replace(".", "").replace("post", "p")


def compatibility_preflight_fields(p: dict):
    """Return (engine, version) for one of the four excluded E1 preflights."""
    marker = p.get("compatibility_preflight")
    if marker in (None, "", False):
        return None
    if marker is not True and str(marker).lower() != "true":
        raise ValueError("compatibility_preflight must be true when specified")
    study = study_fields(p)
    order = int(str(p["study_order"]), 10)
    if (study is None or study[0] != "e1" or p.get("dataset") != "longbench_v1"
            or order not in (5, 6, 11, 12)):
        raise ValueError(
            "compatibility_preflight is limited to the four E1 LongBench A100x2 recipes")
    return p["inference_engine"], study[1]


def get_run_name(p: dict):
    if benchmark_family(p) == "agentic":
        return (f"{p['inference_engine']}_{MODEL_SHORT_NAME_MAP[p['model']]}"
                f"_{p['benchmark']}_nt{p['num_tasks']}_{p['gpu']}x{p['num_gpu']}")

    preflight = compatibility_preflight_fields(p)
    if preflight:
        engine, version = preflight
        return (f"preflight_{engine}_{study_version_token(version)}"
                f"_{MODEL_SHORT_NAME_MAP[p['model']]}"
                f"_{DATASET_SHORT_NAME_MAP[p['dataset']]}"
                f"_{p['gpu']}x{p['num_gpu']}")

    study = study_fields(p)
    if study:
        block, version = study
        # No ns/bsd segments (frozen at 256 samples, batch-default) so the
        # job name + generateName suffix stays under the 63-char label limit.
        return (f"study_{block}_{p['inference_engine']}"
                f"_{study_version_token(version)}"
                f"_{MODEL_SHORT_NAME_MAP[p['model']]}"
                f"_{DATASET_SHORT_NAME_MAP[p['dataset']]}"
                f"_{p['gpu']}x{p['num_gpu']}")

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

    # Study ingestion marker: .../batch-size-default/study-e1/<timestamp>.
    # A directory level, not a timestamp suffix — downstream parsers treat
    # everything below batch-size as the run id but need a PURE timestamp dir.
    study = study_fields(p)
    if compatibility_preflight_fields(p):
        dir += "/compatibility-preflight"
    elif study:
        dir += f"/study-{study[0]}"

    return dir
