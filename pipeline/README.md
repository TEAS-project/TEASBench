# Generate Sweep

This repository contains a benchmark sweep to evaluate MoE-Benchmark performance on [EIDF](https://edinburgh-international-data-facility.ed.ac.uk/). The contained [k8s configurations](configs) are created with the provided Python generator based on the parameters specified in [data](data).

- check out the table of experiments as csv
- generate yaml configs from csv
- launch jobs on eidf
- inspect generated performance data

## Experiments

The list of experiments with parameters to be found in [data/experiments.csv](data/experiments.csv).

**Note: data/smoke_test.csv allows for a rapid test of the setup**

## Generate k8s configs

Run [generate.sh](generate.sh) or generate.py directly to generate k8s config files.
Yaml configurations will be written to current working directory by default, can specify different using `--target_dir`:

```
#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install pandas pyyaml re

python3 generate.py --csv_file=data/moe-smoke-tests-eidf.csv
```


## Create k8s jobs on EIDF

To run a job:  `./submit.sh sglang-gpt-oss-20b-gsm8k-ns1-a100x1-bs1.yaml`


## Agentic benchmarks

Three agentic benchmarks are also generated from this pipeline: `imo-answerbench`,
`mcp-atlas` and `swe-bench-lite`. Unlike the MoE server+client split above, each
agentic benchmark runs `agent_cap.agents` (from
[AgentCAP](https://github.com/Auto-CAP/AgentCAP)) against a separately-started
inference-engine server, all inside one k8s Job driven by a single template,
[`templates/agentic.yaml`](templates/agentic.yaml). See
[`../docs/agentic-pipeline-design.md`](../docs/agentic-pipeline-design.md) for
the full architecture (why one template covers all three benchmarks, the two
platforms, and the sandbox-placement measurement caveat).

### CSV schema

```
family,benchmark,inference_engine,model,gpu,num_gpu,num_tasks,concurrency,batch_size
```

`family` leads every experiments CSV, MoE and agentic alike, and is `moe` or
`agentic` (matching the results-repo top-level directories). It is required and
never inferred from the other columns: the family selects the job template, the
runner and the results tree, so leaving it implicit is too consequential. A row
with a missing or unrecognised family is an error, not a default.

`benchmark` then says *which* agentic benchmark the row is -- one of
`imo-answerbench`, `mcp-atlas`, `swe-bench-lite` (see `utils.AGENTIC_BENCHMARKS`
/ `utils.benchmark_family`). MoE rows name their dataset in `dataset` instead
and leave `benchmark` empty. `batch_size` defaults to
`default` and `concurrency` defaults to `4` if the column is omitted or a row
leaves it blank (agentic runs don't batch in the MoE sense; `batch_size` exists
only so the results-repo path keeps its usual `batch-size-<...>` level). See:

- [`../experiments/imo-answerbench-eidf.csv`](../experiments/imo-answerbench-eidf.csv)
- [`../experiments/mcp-atlas-eidf.csv`](../experiments/mcp-atlas-eidf.csv)
- [`../experiments/swe-bench-lite-eidf.csv`](../experiments/swe-bench-lite-eidf.csv)
- [`../experiments/agentic-smoke-tests-eidf.csv`](../experiments/agentic-smoke-tests-eidf.csv) --
  two tiny (`num_tasks: 2`) rows (mcp-atlas, swe-bench-lite) for a quick pipeline shakeout

### Generate and submit

Generation and submission work exactly as for MoE above -- the same CSV-driven
`generate.py` dispatches to `templates/agentic.yaml` instead of
`templates/template.yaml` whenever a row's `benchmark` column is set to one of
the three agentic benchmarks:

```
python3 generate.py --csv_file=../experiments/mcp-atlas-eidf.csv --target_dir=./
./submit_job.sh sglang-gptoss120b-mcp-atlas-nt60-a100x2.yaml
```

### RBAC prerequisite for swe-bench-lite

`swe-bench-lite` provisions one k8s Job per SWE-bench task (a sandbox pod
running the official SWE-bench instance image) via
`teasbench.sandbox.k8s:InClusterK8sProvider`, using a `teasbench-runner`
ServiceAccount (`@service_account@` in the template). Before running any
`swe-bench-lite` experiment,
[`../eidf/rbac/teasbench-runner-rbac.yaml`](../eidf/rbac/teasbench-runner-rbac.yaml)
must be applied once by someone with rights in the namespace:

```
kubectl apply -f ../eidf/rbac/teasbench-runner-rbac.yaml
```

If the EIDF project does not permit creating ServiceAccounts/Roles/RoleBindings,
drive `swe-bench-lite` from a login node with the `PortForwardK8sProvider`
fallback instead (see `teasbench/sandbox/k8s.py`) rather than
`InClusterK8sProvider`.

### k8s secrets required

| secret name | key | used by |
|---|---|---|
| `gemini-api-key` | `key` | the imo-answerbench and mcp-atlas judges (Gemini, via `GEMINI_API_KEY`) |
| `teas-develop-results-private-ap` | `token` | pushing results to the results repo (`GITHUB_RESULTS_TOKEN`) |
| `teasbench-pat` | `token` | cloning TEASBench itself inside the job (`TEASBENCH_TOKEN`), needed on `PYTHONPATH` for the swe-bench-lite sandbox provider |
| `mcp-atlas-github-token` | `token` | the mcp-atlas tool-server sidecar's `github` MCP server |
| `mcp-atlas-brave-api-key` | `key` | the mcp-atlas tool-server sidecar's `brave-search` MCP server |

All five are referenced by name (`secretKeyRef`) in `templates/agentic.yaml` and
`configs/config.yaml`'s mcp-atlas sidecar -- never inlined. Note that mcp-atlas'
own evaluation judge (`evaluator: gtfa`) is configured differently from the
other two benchmarks: it reads `EVAL_LLM_MODEL` / `EVAL_LLM_BASE_URL` /
`EVAL_LLM_API_KEY` from the environment rather than from the agent config
file's `judge:` block, set from the `gemini-api-key` secret by the mcp-atlas
rule's `extra_setup` -- see
[`../docs/agentic-pipeline-design.md`](../docs/agentic-pipeline-design.md).









