#!/bin/python3

import base64
import io
import pathlib
import textwrap

import pandas as pd

from utils import MODEL_DISK_GB_MAP, benchmark_family, load_site

# A rented Vast.ai host's driver must support a minimum CUDA version that is compatible with
# the version in the base images in pipeline/vast/*/Dockerfile (currently vllm/vllm-openai:v0.16.0, which uses 12.9).
# Note that I've observed that sometimes the version specified in an offer doesn't appear to be accurate; in this case,
# cancel and start with a new offer. run_benchmark.sh tries to spot if this happens in logs and comments in the log.
MIN_CUDA_VERSION = "12.9"


def _image_for_engine(engine, family="moe"):
    """
    Return the image name depending on the inference engine and pipeline family.

    The two families ship as separate images (Dockerfile vs Dockerfile.agentic).
    The agentic one adds SWE-agent, swebench, swe-rex and modal on top of a base
    image whose torch/transformers pins are brittle; keeping them apart means
    that dependency risk can never destabilise a MoE sweep.
    """
    if engine not in ("vllm", "sglang"):
        raise ValueError(f"Unknown inference engine: {engine!r}")
    suffix = "bench" if family == "moe" else "agentic"
    return f"ghcr.io/teas-project/{engine}-{suffix}:latest"


def _encode_csv_group(group_df, all_columns):
    """
    Encode the whole input CSV with base64 and return it.
    """
    buf = io.StringIO()
    group_df[all_columns].to_csv(buf, index=False)
    return base64.b64encode(buf.getvalue().encode()).decode()


def _generate_script_text(engine, gpu, num_gpu, disk_gb, encoded_csv, private_image: bool = False,
                          family: str = "moe", benchmarks=None):
    """
    Generate a bash script for launching a Vast.ai instance with the specified engine, GPU, and number of GPUs.
    We pass the encoded CSV containing the benchmarks to run through to it as a environment variable.
    We rely on:
    1) the user having installed the Vast.ai CLI locally
    2) the image being accessible (i.e. public, or else you must set your GHCR_USERNAME/GHCR_PAT at runtime)
    3) the GIT_TOKEN, HF_TOKEN, OPENAI_API_KEY secrets set in Vast.ai
    """
    vast_gpu = load_site("vastai")["gpu_products"][gpu]
    image = _image_for_engine(engine, family)

    # Add registry login info, if the image is private. All images live on ghcr.io (see
    # _image_for_engine), so we only need the registry username and password/PAT.
    # Check that we've got these at runtime from the environment variables GHCR_USERNAME
    # and GHCR_PAT.
    login_line = ""
    guard_block = ""
    registry_comment = ""
    if private_image:
        registry = image.split("/")[0]  # For us this is likely always going to be ghcr.io.
        login_line = f"\n          --login \"-u $GHCR_USERNAME -p $GHCR_PAT {registry}\" \\"
        registry_comment = "\n        #   - GHCR_USERNAME and GHCR_PAT must be set in *your* environment (to access image)"
        guard_block = textwrap.dedent(f"""\
            # {image} is private; GHCR_USERNAME and GHCR_PAT must be set environment variables.
            if [ -z "$GHCR_USERNAME" ] || [ -z "$GHCR_PAT" ]; then
              echo "Error: GHCR_USERNAME and GHCR_PAT must be set in your environment (required while {image} is private)." >&2
              exit 1
            fi

            """)

    # Secrets differ by family. The MoE client judges arena-hard with OpenAI; the
    # imo-answerbench judge uses Gemini, the mcp-atlas GTFA judge uses OpenRouter, and
    # swe-bench-lite additionally needs Modal credentials because Modal is the sandbox
    # substrate on Vast.ai (swe-rex supports it natively, so unlike K8s there is no
    # sandbox provider here).
    if family == "agentic":
        secrets = "GIT_TOKEN, HF_TOKEN, GEMINI_API_KEY"
        if benchmarks and "swe-bench-lite" in benchmarks:
            secrets += ", MODAL_TOKEN_ID, MODAL_TOKEN_SECRET"
        if benchmarks and "mcp-atlas" in benchmarks:
            secrets += ", OPENROUTER_API_KEY"
        # No family env var is needed: the agentic image's entrypoint only ever
        # runs the agentic runner.
        family_env = ""
        # e.g. "swe-bench-lite: sglang on H200x1"
        listed = ", ".join(sorted(benchmarks)) if benchmarks else "agentic"
        title = f"{listed}: {engine} on {gpu}x{num_gpu}"
    else:
        secrets = "GIT_TOKEN, HF_TOKEN, OPENAI_API_KEY"
        family_env = ""
        # Unchanged from before the agentic work, so MoE launch scripts stay
        # byte-for-byte identical to those the MoE-only generator produced.
        title = f"{engine} on {gpu}x{num_gpu}"

    # The --args (with nothing following) argument is required to put the instance into args/entrypoint launch mode;
    # without it, vastai defaults to injecting its own SSH server instead of going to the image's ENTRYPOINT.
    header = textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # Vast.ai launch script — {title}
        # Generated by generate.py --site vastai
        #
        # Prerequisites:
        #   - vastai CLI installed and authenticated (vastai login)
        #   - Container image pushed to an accessible registry
        #   - The following environment variables must be set as secrets in Vast.ai:
        #       {secrets}{registry_comment}

        """)
    body = textwrap.dedent(f"""\
        BENCHMARK_CSV="{encoded_csv}"

        # 1. Search for available offers matching the hardware requested from the CSV.
        #    Offers will be sorted by dollars per hour rental cost, increasing.
        vastai search offers 'gpu_name={vast_gpu} num_gpus={num_gpu} verified=true rentable=true cuda_vers>={MIN_CUDA_VERSION}' -o 'dph+'

        # 2. Prompt for an offer ID from the results above.
        read -rp "Enter offer ID: " OFFER_ID

        # 3. Launch the instance.
        vastai create instance "$OFFER_ID" \\
          --image "{image}" \\
          --disk {disk_gb} \\{login_line}
          --env "-e BENCHMARK_CSV=${{BENCHMARK_CSV}}" \\{family_env}
          --args
    """)
    return header + guard_block + body


def generate_vast_scripts(csv_file, target_dir, private_image: bool = False):
    pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    all_columns = list(df.columns)

    # Split by pipeline family first, from each row's explicit `family` column.
    # The two families use different images, run different scripts inside the
    # container (run_benchmarks.sh vs run_agentic_benchmarks.sh, selected by the
    # BENCHMARK_FAMILY env var) and need different secrets, so they can never
    # share one launch script even when they'd share hardware.
    df["_family"] = df.apply(
        lambda row: benchmark_family(
            {"family": row.get("family"), "benchmark": row.get("benchmark")}),
        axis=1,
    )

    # Loop through groups from the dataframe based on the family-engine-gpu-num_gpu tuple.
    # This way we're automatically going to build an encoded CSV for each tuple alone (by which I mean not including
    # the rows for other families / inference engines / GPU configs).
    for (family, engine, gpu, num_gpu), group in df.groupby(
            ["_family", "inference_engine", "gpu", "num_gpu"]):
        if gpu not in load_site("vastai")["gpu_products"]:
            raise ValueError(
                f"GPU {gpu!r} not in the vastai site profile. Add it to "
                f"pipeline/configs/sites/vastai.yaml under gpu_products."
            )

        # Size the instance's disk to the largest model in this group, since the model
        # weights are downloaded onto it (into the HF cache).
        disk_gb = max(MODEL_DISK_GB_MAP[model] for model in group["model"].unique())

        # The helper column is ours, not part of the experiment definition -- don't
        # ship it to the container in the encoded CSV.
        columns = [c for c in all_columns if c != "_family"]
        benchmarks = (set(group["benchmark"].dropna().unique())
                      if "benchmark" in group.columns else set())

        # Generate the encoded CSV for this group and then write the script to submit
        # this to a Vast.ai instance.
        encoded_csv = _encode_csv_group(group, columns)
        script = _generate_script_text(engine, gpu, num_gpu, disk_gb, encoded_csv,
                                  private_image=private_image,
                                  family=family, benchmarks=benchmarks)

        # Write the script and for convenience make it executable right away. MoE script
        # names are unchanged, so existing workflows keep working; agentic ones are
        # prefixed to keep the two families' scripts distinguishable in a target dir.
        #
        # A single-benchmark agentic group also names its benchmark. Without that, the
        # three per-benchmark agentic CSVs all produce the same filename for a given
        # engine/GPU and silently overwrite one another when generated into one target
        # dir. A group spanning several benchmarks (one instance running them back to
        # back, as the archived Vast.ai runs did) keeps the plain name.
        if family == "moe":
            stem = f"vast_{engine}_{gpu}x{num_gpu}"
        elif len(benchmarks) == 1:
            stem = f"vast_agentic_{next(iter(benchmarks))}_{engine}_{gpu}x{num_gpu}"
        else:
            stem = f"vast_agentic_{engine}_{gpu}x{num_gpu}"
        out_path = pathlib.Path(target_dir) / f"{stem}.sh"
        out_path.write_text(script)
        out_path.chmod(0o755)
        print(f"Generated {out_path}  ({len(group)} benchmark row(s))")
