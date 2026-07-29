# TEASBench pipeline — user guide

How to run benchmark experiments on **EIDF** (Kubernetes) or **Vast.ai** (rented
GPU instances), for both the **MoE** and **agentic** benchmark families.

For how it works internally, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

---

## 1. The idea in 60 seconds

You describe experiments as **rows in a CSV**. A generator turns each row into
something you can launch:

- **EIDF** → one Kubernetes Job YAML per row. Submit it; it runs unattended and
  pushes its own results.
- **Vast.ai** → one bash launch script per (family, engine, GPU) group. Run it;
  it rents an instance, runs every row in that group, pushes results, and
  destroys itself.

Two families:

| Family | Benchmarks | Shape |
|---|---|---|
| `moe` | `gsm8k`, `arena-hard`, `longbench_v1` | inference server + client |
| `agentic` | `imo-answerbench`, `mcp-atlas`, `swe-bench-lite` | server + agent loop (+ tools/sandboxes) |

---

## 2. Prerequisites

### Everywhere

```bash
~/pyvenvs/teasbench/bin/python --version
```

Generation needs `pandas` and `pyyaml`. **The system `python3` lacks pandas** —
use the venv above. Run the generator **from inside the git repo** (it embeds
the TEASBench commit for provenance).

### EIDF only

- `kubectl` configured for your project namespace (default `eidf230ns`)
- These cluster secrets must exist:

| Secret | Key | Needed for |
|---|---|---|
| `teas-develop-results-private-ap` | `token` | pushing results (all runs) |
| `teasbench-pat` | `token` | cloning TEASBench in-job (agentic) |
| `gemini-api-key` | `key` | judge for `imo-answerbench`, `mcp-atlas` |
| `mcp-atlas-github-token` | `token` | `mcp-atlas` tool servers |
| `mcp-atlas-brave-api-key` | `key` | `mcp-atlas` tool servers |

- **SWE-bench Lite** additionally needs a Python environment on the login node
  with `agent_cap`, `swe-rex` and `swebench` installed, plus a SWE-agent checkout
  patched by AgentCAP's `scripts/patch_sweagent_streaming.py`. See §4.8 — on EIDF
  this benchmark runs from the login node, not as an unattended Job.

> **EIDF does not grant pods RBAC**, so `eidf/rbac/teasbench-runner-rbac.yaml`
> and the in-cluster preflight (§4.6) do not apply here. They are kept for a
> cluster that does grant it. On EIDF, SWE-bench always uses
> `PortForwardK8sProvider` — see §4.8.

### Vast.ai only

- `vastai` CLI installed and authenticated (`vastai login`)
- The container images built and pushed (§6.1)
- Instance secrets set in the Vast.ai console (§6.2)

---

## 3. The experiments CSV

Every CSV starts with a **`family`** column — `moe` or `agentic`. It is required
and never guessed.

**MoE:**
```csv
family,inference_engine,model,dataset,num_samples,gpu,num_gpu,batch_size
moe,sglang,gpt-oss-120b,gsm8k,256,A100,1,default
```
Optional: `input_length`, `output_length` (fixed-length mode), `platform`.

**Agentic:**
```csv
family,benchmark,inference_engine,model,gpu,num_gpu,num_tasks,concurrency,batch_size
agentic,swe-bench-lite,sglang,gpt-oss-120b,H100,2,100,4,default
```

| Column | Notes |
|---|---|
| `benchmark` | `imo-answerbench`, `mcp-atlas`, `swe-bench-lite` |
| `num_tasks` | 100 / 60 / 100 respectively, to match reference runs |
| `concurrency` | parallel tasks; defaults to 4 |
| `batch_size` | `default` — agentic runs don't batch in the MoE sense; it only keeps the results path level |
| `platform` | `eidf` (default) or `vastai` |

Existing CSVs live in [`experiments/`](../experiments/).

> **Note:** a CSV without a leading `family` column now fails with an explicit
> error. Add `family,` to the header and the value to each row.

---

## 4. Running on EIDF

### 4.1 Generate

```bash
cd pipeline
~/pyvenvs/teasbench/bin/python generate.py --csv_file=../experiments/moe-experiments-eidf.csv --target_dir=./out
```

Options: `--target_dir` (default `./`), `--results_repo` (default
`TEAS_Development_Results_Private`).

One YAML per row, named after the run:

```
sglang-gptoss120b-gsm8k-ns256-a100x1-bsd.yaml
sglang-gptoss120b-swe-bench-lite-nt100-h100x2.yaml
```

### 4.2 Inspect before submitting

Worth doing at least once — the manifest is fully self-describing:

```bash
grep -E "image:|nvidia.com/gpu:|serviceAccountName" out/<run>.yaml
grep -A3 "output_repo_dir\|RUN_OUTPUT_DIR=" out/<run>.yaml
```

### 4.3 Submit

```bash
./submit_job.sh out/sglang-gptoss120b-gsm8k-ns256-a100x1-bsd.yaml
```

This creates the Job, copies the YAML to the shared job-configs dir (so it ends
up alongside the results as provenance), and appends the name to
`submitted_jobs.log`.

Submit several by looping:

```bash
for f in out/*.yaml; do ./submit_job.sh "$f"; done
```

### 4.4 Watch

```bash
kubectl -n eidf230ns get jobs
kubectl -n eidf230ns get pods -w
kubectl -n eidf230ns logs -f <pod>
```

Helpers in [`eidf/scripts/`](../eidf/scripts/): `k8_pod_log.sh`,
`k8_job_desc.sh`, `k8_pod_bash_login.sh`.

For SWE-bench Lite you will also see transient sandbox pods appear and vanish:

```bash
kubectl -n eidf230ns get pods -l app=teasbench-sandbox
```

### 4.5 Agentic specifics

**IMO AnswerBench** — nothing extra; one container.

**MCP Atlas** — the pod gets a second container (the tool server) on port 1984.
Check both:

```bash
kubectl -n eidf230ns logs <pod> -c mcp-atlas-sidecar
```

**SWE-bench Lite** — *not* an unattended Job on EIDF. See §4.8: the driver runs
on a login node and creates the Jobs itself. The GPUs are still used through a
Kubernetes Job, as always — only the driver process sits outside the cluster.

### 4.6 Preflight for in-cluster mode (not applicable on EIDF)

> **Skip this on EIDF.** It tests `InClusterK8sProvider`, which needs pod RBAC
> that EIDF does not grant. Kept for a cluster that does. On EIDF go to §4.7
> (mechanism check) and §4.8 (running it).

In-cluster SWE-bench depends on two facts about the cluster that are worth
confirming *before* a GPU job queues, because both fail late and confusingly:

1. **Pod IPs are routable within the namespace** — the driver talks to a sandbox
   at `http://<podIP>:9999` with no port-forward. A restrictive NetworkPolicy
   breaks this.
2. **`kubectl` works in-cluster from the ServiceAccount token**, with no
   kubeconfig, and the RBAC grants the verbs the provider uses.

```bash
kubectl -n eidf230ns create -f eidf/preflight/teasbench-preflight.yaml
kubectl -n eidf230ns logs -f job/teasbench-preflight
```

The preflight does exactly what `InClusterK8sProvider.acquire()` does — same
kubectl bootstrap, same Job-spec shape, same jsonpath to read the pod IP, same
port — but against a busybox target instead of a multi-GB SWE-bench image, and
with **no GPU**. It schedules immediately and costs no GPU time.

Expected output:

```
  PASS  kubectl v1.xx.x installed
  PASS  kubectl authenticates and can list pods
  PASS  can create jobs
  ...
  PASS  read pod IP via the provider's jsonpath: 10.42.3.17
  PASS  pod IP is routable on port 9999 -- no port-forward needed
ALL CHECKS PASSED -- in-cluster mode is good to go.
```

It exits non-zero on any failure and names which assumption broke. Two quick
manual cross-checks if you want them independently:

```bash
kubectl -n eidf230ns get networkpolicy          # empty = nothing blocking pod-to-pod
kubectl -n eidf230ns auth can-i create jobs \
        --as=system:serviceaccount:eidf230ns:teasbench-runner
```

The second impersonates the ServiceAccount from your own session, so it checks
the RBAC without running anything. Note it needs impersonation rights, which not
every project grants — the preflight Job needs none, since it *is* the
ServiceAccount.

If the routability check fails, in-cluster mode is unusable on this cluster; use
`PortForwardK8sProvider` (§7.3), which needs neither assumption.

### 4.7 Preflight: check the port-forward mechanism

Everything SWE-bench does on EIDF rests on `PortForwardK8sProvider`, so check it
before committing GPU time. This runs **on a login node**, not as a Job, because
that is where the provider itself runs. It uses your own kubectl credentials and
needs no ServiceAccount and no RBAC manifest — which is exactly why this is the
path EIDF can support.

**Fast probe** (~1 min, no GPU):

```bash
python3 eidf/preflight/preflight_portforward.py --namespace eidf230ns
```

It drives the **real** `PortForwardK8sProvider` — OS port allocation, the
`kubectl port-forward` spawn, the readiness poll, the tunnel-babysitter thread
and cleanup on release — plus the `kubectl cp` / `kubectl exec` path the
SWE-bench evaluator uses. Only the sandbox container's payload is substituted
(busybox serving an `is_alive` file instead of a multi-GB image pip-installing
swe-rex), because what is under test is the tunnel mechanism, not swe-rex.

The check most worth reading is this one:

```
  PASS  tunnel still alive after 10s (babysitter working)
```

A `kubectl port-forward` that dies quietly mid-task is the failure the
babysitter thread exists to prevent, and the one that would otherwise surface as
an inexplicable task failure deep into a long run.

The permission unique to this path is `create pods/portforward`; the in-cluster
provider never needs it, because it talks to pod IPs directly.

**Full-fidelity probe** (minutes, still no GPU):

```bash
python3 eidf/preflight/preflight_portforward.py --namespace eidf230ns --real-image
```

`--real-image` drops the busybox substitution for a genuine
`docker.io/swebench/sweb.eval.x86_64.*` image, additionally proving the multi-GB
pull works and that `swe-rex` installs and runs inside the instance image — an
old conda env where a dependency clash is plausible. Pass an instance id to
override the default (`--real-image django__django-11099`).

### 4.8 SWE-bench Lite on EIDF

**EIDF does not grant pods RBAC**, so SWE-bench here always uses
`PortForwardK8sProvider`, with the **driver** running on a login node.

#### Where everything actually runs

The driver moving off the cluster does **not** mean Kubernetes is bypassed. The
model runs on GPUs through a Job, exactly as every other TEASBench run does.
Four components, three of them Kubernetes Jobs, all created with *your*
credentials — and all of it handled for you:

| Component | Where | GPU | Created by |
|---|---|---|---|
| **Driver** (`agent_cap.agents`) | login-node VM | no | you, by running the generated script |
| **Engine** (sglang/vllm serving the model) | Kubernetes Job | **yes** | the driver, at start-up |
| **Sandbox** (swe-rex, one per task) | Kubernetes Job | no | the driver, at run time |
| **Eval** (official grading) | Kubernetes Job | no | the driver, at run time |

The driver reaches the engine and each sandbox over `kubectl port-forward`.

Why this is allowed when in-cluster mode is not: *you* may create Jobs and
port-forward — §4.7 confirms both on EIDF. What EIDF refuses is granting those
rights to a **pod's ServiceAccount**. Running the driver as yourself sidesteps
that entirely.

IMO AnswerBench and MCP Atlas are unaffected and still run as ordinary
unattended Jobs, because neither touches the Kubernetes API.

#### Running it

The pipeline handles all of it, including the engine. Generation emits **two**
files for a SWE-bench row on EIDF instead of one:

```bash
cd pipeline
~/pyvenvs/teasbench/bin/python generate.py \
    --csv_file=../experiments/swe-bench-lite-eidf.csv --target_dir=./out
```

```
sglang-gptoss120b-swe-bench-lite-nt100-h200x1.sh           <- run this
sglang-gptoss120b-swe-bench-lite-nt100-h200x1.engine.yaml  <- submitted for you
```

Then start it and walk away:

```bash
bash out/sglang-gptoss120b-swe-bench-lite-nt100-h200x1.sh
```

The script checks prerequisites, submits the engine Job, waits for the model to
load, opens and babysits the tunnel, runs the benchmark, pushes the results, and
**deletes the engine Job on exit** — success, failure or Ctrl-C — so an aborted
run cannot leave GPUs allocated. You never start an engine or submit the engine
manifest by hand.

Useful flags: `--resume`, `--no-push`, `--namespace`, `--output-root`. Override
`SWEAGENT_DIR` / `AGENTCAP_DIR` if your checkouts are not in the default places.

#### Smoke test first

The 2-task row in `experiments/smoke_tests_agentic.csv` is the same thing at
small scale — same driver, same engine Job, same sandboxes, same grading:

```bash
cd pipeline
~/pyvenvs/teasbench/bin/python generate.py \
    --csv_file=../experiments/smoke_tests_agentic.csv --target_dir=./out
bash out/vllm-gptoss120b-swe-bench-lite-nt2-a100x1.sh
```

A low or zero accuracy on 2 tasks is normal and not a failure signal; what
matters is that every stage ran and wrote its outputs.

Run §4.7 first — if the mechanism is broken, this fails for that reason after
queueing for a GPU.
---

## 5. Running MoE on Vast.ai

### 5.1 Generate

```bash
cd pipeline
~/pyvenvs/teasbench/bin/python generate.py \
    --csv_file=../experiments/moe-experiments-vastai.csv \
    --target_dir=./out --vast
```

Add `--private-image` if the image is still private on ghcr.io; the generated
script then expects `GHCR_USERNAME` / `GHCR_PAT` in *your* environment.

Produces one script per (engine, GPU, num_gpu) group, e.g.
`vast_sglang_H200x8.sh`, with the CSV rows base64-embedded.

### 5.2 Launch

```bash
bash out/vast_sglang_H200x8.sh
```

It lists matching offers cheapest-first, prompts for an offer ID, and creates
the instance. The instance then works through its rows, pushing results as it
goes, and **destroys itself** at the end.

### 5.3 Secrets

Set in the Vast.ai console: `GIT_TOKEN`, `HF_TOKEN`, `OPENAI_API_KEY`.

`CONTAINER_ID` and `CONTAINER_API_KEY` are injected by Vast.ai automatically —
they're how the instance destroys itself. Without them it would restart and
rerun the whole sweep.

---

## 6. Running agentic on Vast.ai

### 6.1 Build and push the agentic images (once)

The agentic family uses **separate images** from MoE. Build context is
`pipeline/`, from the repo root:

```bash
docker build --platform=linux/amd64 \
  --build-arg TEASBENCH_COMMIT=$(git rev-parse --short HEAD) \
  -t ghcr.io/teas-project/sglang-agentic:latest \
  -f pipeline/vast/sglang/Dockerfile.agentic pipeline/

docker push ghcr.io/teas-project/sglang-agentic:latest
```

Same for `vllm/Dockerfile.agentic` → `vllm-agentic`.

> These images have **not yet been built in anger**. They add uv, Node 20, an
> npm MCP package preinstall and a streaming-patched SWE-agent on top of a
> version-brittle engine base image. Build one before renting a GPU.

### 6.2 Secrets

| Secret | Needed for |
|---|---|
| `GIT_TOKEN`, `HF_TOKEN` | everything |
| `GEMINI_API_KEY` | judge for `imo-answerbench`, `mcp-atlas` |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | `swe-bench-lite` only (Modal sandboxes) |
| `GITHUB_TOKEN`, `BRAVE_API_KEY`, `ALCHEMY_API_KEY`, … | `mcp-atlas` tool servers |

The generated script's header lists exactly what its rows need.

For `mcp-atlas`, any key in the mcp-atlas `env.template` is picked up from a
same-named environment variable — set whichever you have.

### 6.3 Generate and launch

```bash
cd pipeline
~/pyvenvs/teasbench/bin/python generate.py \
    --csv_file=../experiments/swe-bench-lite-vastai.csv \
    --target_dir=./out --vast

bash out/vast_agentic_swe-bench-lite_sglang_H200x1.sh
```

Agentic scripts are named `vast_agentic_<benchmark>_<engine>_<gpu>x<n>.sh` when
the group is a single benchmark, so per-benchmark CSVs don't overwrite each
other in one output directory.

### 6.4 What differs from EIDF

| | EIDF | Vast.ai |
|---|---|---|
| MCP Atlas tool server | sidecar container | background process, same container |
| SWE-bench sandboxes | Kubernetes pods | **Modal** |
| SWE-bench grading | exec pods | Modal (`--modal true`) |

---

## 7. Troubleshooting

### 7.1 Generation

| Symptom | Cause |
|---|---|
| `experiment row has family None` | CSV is missing the leading `family` column |
| `family 'agentic' row has benchmark 'gsm8k'` | agentic rows need an agentic benchmark |
| `GPU 'B200' not in VAST_GPU_MAP` | add it to `pipeline/utils.py` — take the exact `gpu_name` from `vastai search offers`, don't guess |
| `ModuleNotFoundError: pandas` | using system `python3`; use `~/pyvenvs/teasbench/bin/python` |
| `fatal: not a git repository` | run the generator from inside the repo |

### 7.2 EIDF runs

**Job pending forever** — usually queue/quota. `kubectl -n eidf230ns describe job <name>`.

**Server never becomes ready** — read `server.log` in the run dir, or the pod
logs. Model download and load can take tens of minutes.

**Nothing pushed to the results repo** — results are copied to the PVC *before*
any git operation, so look there first:
`$TEAS_OUTPUT_DIR/<results_repo>/<output_repo_dir>/<timestamp>/`.

### 7.3 SWE-bench Lite permissions

If sandbox creation fails with a `kubectl` permissions error, either apply
`eidf/rbac/teasbench-runner-rbac.yaml`, or switch to the login-node fallback by
pointing the run at `PortForwardK8sProvider` instead of `InClusterK8sProvider` —
that provider uses *your* kubectl credentials via port-forwarding rather than
the pod's ServiceAccount. Confirm it works first with §4.7.

### 7.4 MCP Atlas scores lower than expected

**Check the credential log first.** Tool servers with blank API keys still start
and fail only at tool-call time, so missing credentials look like poor model
performance. The run logs which keys were supplied and which were empty (names
only, never values):

```
credentials supplied: BRAVE_API_KEY GITHUB_TOKEN
left empty: ALCHEMY_API_KEY EXA_API_KEY ...
```

Also confirm the server set matches — it is pinned to the same 22 servers on
both platforms, because the server set *is* the benchmark definition.

### 7.5 Agentic metrics are empty

SWE-agent wasn't streaming-patched. Both platforms assert the
`AGENTCAP_STREAMING_PATCH_APPLIED` marker and should fail loudly; if you see
empty TTFT/TPOT, that check was bypassed or the image is stale — rebuild.

---

## 8. Where results go

```
<family>/<platform>/<engine>/<model>/<dataset-or-benchmark>/<hw>x<n>/batch-size-<bs>/<timestamp>/
```

```
moe/eidf/sglang/gpt-oss-120b/gsm8k_256samples/a100x1/batch-size-default/20260727-1432/
agentic/vastai/sglang/gpt-oss-120b/swe-bench-lite/h200x1/batch-size-default/20260727-1432/
```

Each run directory holds `metrics_*.json`, `metadata_*.json`,
`detailed-results_*.jsonl`, `output-data_*.jsonl`, `timings.json`, the job YAML
or provenance, and logs.

Aggregate with:

```bash
~/pyvenvs/teasbench/bin/python postprocessing/aggregate_results.py --results_dir <repo>/moe
```

Point `--results_dir` at the `moe` or `agentic` subdirectory, **not** the repo
root.

---

## 9. Common tasks

**Smoke test before a real sweep**

```bash
cd pipeline
~/pyvenvs/teasbench/bin/python generate.py --csv_file=../experiments/smoke_tests.csv --target_dir=./out
~/pyvenvs/teasbench/bin/python generate.py --csv_file=../experiments/smoke_tests_agentic.csv --target_dir=./out
```

**Add a model** — add to `HF_MODEL_MAP`, `MODEL_SHORT_NAME_MAP` and (for
Vast.ai) `MODEL_DISK_GB_MAP` in `pipeline/utils.py`.

**Add a GPU** — add to `EIDF_GPU_MAP` and/or `VAST_GPU_MAP`, plus
`TEAS_GPU_NAME_MAP`. For Vast.ai, take the string from `vastai search offers`.

**Change engine flags for one case** — add a rule to
`pipeline/configs/config.yaml`. Rules match on any parameter (`benchmark`,
`platform`, `gpu`, `model`, `inference_engine`, …); more specific rules override
more general ones.

**Change agent sampling parameters** — edit the relevant
`pipeline/configs/agents/<benchmark>_<model>_<engine>.yaml`.

**Verify you haven't broken anything**

```bash
~/pyvenvs/teasbench/bin/python -m pytest tests/ -q
```

Two failures in `test_agentic_compute_cost_cli.py` and
`test_moe_compute_sparsity_cli.py` are pre-existing and unrelated.
