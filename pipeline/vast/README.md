# Pipeline execution on Vast.ai

## Contents

This directory contains the resources needed to run the pipeline through Vast.ai.

We will provide containers derived from the same vLLM and SGLang images used on the EIDF, with a runscript and MoE-CAP baked in. Environment variables passed through the Vast.ai interface parameterise the benchmarks to be run and provide necessary secrets. The pipeline script will then run through the set of benchmarks, run them one-by-one and push the results to GitHub.

A script [run_vllm_gpt_oss_smoke.sh](vllm/run_vllm_gpt_oss_smoke.sh) can be found in the vllm directory. This runs through three datasets for the gpt-oss-20b benchmark. There is also [encode-csv-keyed-errors.sh](vllm/encode-csv-keyed-errors.sh); this is a test script which shows the contents of a CSV file being turned into a variable which can be passed through Vast.ai's interface into the container, then used to loop through the benchmark parameters on each line, error checking along the way. These two scripts now need to be combined.

Alternatively, allow an inclusive CSV within the container but only run those benchmarks which match current hardware and inference engine. It's a question of whether to do this selection outside or inside the container. I'm of the opinion at the moment that it would be easier to do outside in the Python layer.

## Planned workflow

1. Run the pipeline `generate.py` script locally, likely with a new `--vast` option to tell it to generate commands for Vast.ai and not the EIDF.
2. This generates a bash script or scripts containing the required commands to reserve the requested resource on Vast.ai. Likely multiple scripts, separated by hardware and inference engine (since we need to ask Vast.ai to reserve given instances running given images). They will include the tokenised CSV file and the ability to retrieve and pass through as environment variables any secrets.
3. Run the bash script to submit a 'job' to Vast.ai. Needs to be some decision-making about how to determine whether to reserve a given hardware option (given price in particular).
4. The container will go through the entrypoint script, running all the benchmarks for this GPU/inference engine combination, pushing to GitHub as it goes.

## Agentic benchmarks

The same instance model also runs the agentic family — `imo-answerbench`, `mcp-atlas`
and `swe-bench-lite`. Every experiments CSV declares its family in a leading `family` column, whose value is
`moe` or `agentic` (matching the results-repo top-level directories). It is required and
never inferred: the family selects the image, the in-container runner and the results
tree, which is too consequential to leave implicit. A row with a missing or unrecognised
family is an error.

```bash
python3 generate.py --csv_file=../experiments/swe-bench-lite-vastai.csv --vast
```

That writes `vast_agentic_<benchmark>_<engine>_<gpu>x<n>.sh`, pointing at the agentic
image (`ghcr.io/teas-project/<engine>-agentic`), whose entrypoint runs
`run_agentic_benchmarks.sh`. MoE rows keep their original script names and the original
`<engine>-bench` image.

The two families ship as **separate images** (`Dockerfile` vs `Dockerfile.agentic`) rather
than one image with a runtime switch. The agentic image adds SWE-agent, swebench, swe-rex
and modal on top of an engine base image whose torch/transformers pins are brittle - the
EIDF agentic path needs a whole torch/torchvision repair script for exactly this reason.
Sharing one image would put that dependency risk on MoE sweeps that need none of it.

Both families resolve their commands through `resolve_commands.py`, which drives
`template.py`'s rule engine against the *same* `configs/config.yaml` the EIDF pipeline
uses. Nothing about a benchmark is specified twice.

### How this differs from EIDF, and why

The pipeline abstracts by *capability*, not by Kubernetes mechanism, which is what lets
one config serve both platforms:

| Capability | EIDF | Vast.ai |
|---|---|---|
| Tool server (mcp-atlas) | sidecar container in the pod | background process in the one container |
| SWE-bench sandboxes | k8s provider (`teasbench.sandbox.k8s:InClusterK8sProvider`) | **Modal** — native to swe-rex, so no provider at all |
| SWE-bench grading | exec containers via the k8s provider | `SWEBENCH_HARNESS_MODAL=1` → harness `--modal true` |

Kubernetes is the only substrate needing a sandbox provider; `modal` and `docker` are
native swe-rex deployment types. This mirrors what the archived reference runs actually
did (`TEAS_Results_Private/agentic/vastai/.../run.sh`).

### Required Vast.ai secrets

Beyond `GIT_TOKEN` and `HF_TOKEN`, which both families need:

- `GEMINI_API_KEY` — the judge for `imo-answerbench` and `mcp-atlas`
- `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` — `swe-bench-lite` only

The generated script's header lists exactly the set its rows need.

### MCP Atlas tool credentials

`mcp-atlas` additionally needs the MCP tool credentials. `mcp-server/start.sh`
requires an `.env` file and exits without one, so
`pipeline/scripts/write_mcp_env.sh` builds it at run start from environment
variables: for every key in the submodule's `env.template`, a same-named
non-empty env var supplies the value. Set whichever you have as Vast.ai secrets
(`GITHUB_TOKEN`, `BRAVE_API_KEY`, `ALCHEMY_API_KEY`, …); the key names come from
the template, so a new upstream key is picked up automatically.

**Missing credentials do not fail the run.** Servers with blank keys still
start and error only at tool-call time, so an incomplete set produces a lower
accuracy that looks like model behaviour rather than a config problem. The
script therefore logs which keys were supplied and which were left empty — names
only, never values — and that log is pushed with the results. Check it before
comparing a run against the archived reference numbers.

`ENABLED_SERVERS` is pinned to `utils.MCP_ENABLED_SERVERS` and deliberately
ignores any value in the environment: the server set *is* the benchmark
definition. `tests/test_mcp_env.py` asserts EIDF's sidecar and Vast.ai's `.env`
enable the identical 22 servers.

### Known gap: Blackwell GPUs

`VAST_GPU_MAP` in `pipeline/utils.py` currently maps A100/H100/H200 only. The archived
reference agentic runs are on B200 and B300, so reproducing them needs entries for those
— along with `MODEL_DISK_GB_MAP` and `TEAS_GPU_NAME_MAP`. The Vast.ai `gpu_name` strings
must be taken from `vastai search offers` rather than guessed: an incorrect name yields
a search that silently matches no offers. The checked-in `*-vastai.csv` files therefore
use H100/H200.

## Further work

Currently, the entrypoint scripts living inside the container are decoupled from the EIDF pipeline, in particular the config.yaml options. This means any changes made to one pipeline (EIDF/VAST) need to be manually ported to the other. It's also rather ad hoc at the moment; doing something more programmatic would be much preferred.

This isn't ideal. Longer term, it may be a good idea to turn container entrypoint scripts into templates to be modified much in the same way as the EIDF YAMLs with the correct option sets for different benchmarks.

(The agentic path above already closes part of this gap: `resolve_commands.py` reuses
`template.py` and `config.yaml` rather than duplicating them. The remaining ad-hockery is
in the runner shell scripts themselves.)