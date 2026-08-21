# TEASBench

Uniting Models, Algorithms, and System Innovators with Top-Down Evolutionary Benchmarks

🌐 **Website:** [www.teasbench.com](https://www.teasbench.com)

**TEASBench** is a benchmark suite and toolkit developed to measure the **cost**, **accuracy**, and **performance** of AI inference on **realistic state-of-the-art workloads** running on **diverse hardware architectures**. It is developed in the TEAS (**T**racking **E**volving **A**I and **S**ystems) project funded by **[ARIA](https://aria.org.uk/)** as part of the ["Scaling compute"](https://aria.org.uk/opportunity-spaces/nature-computes-better/scaling-compute/) programme. 

In contrast to benchmarks that use fixed (short) context lengths, TEASBench does not artifically constrain input and output sequences to fixed numbers of tokens.  This enables assessment of accuracy and means TEASBench is more capable of exposing **real-world hardware limitations**. 

TEASBench is continuously evolving to track new and emergent workloads and hardware following a 6-monthly release cycle. The August 2026 release of TEASBench capture recent shifts towards **sparse Mixture-of-Experts**, **reasoning models**, and **agentic tool use**. 

---

## TEASBench Workloads, Models, and Datasets

The benchmarks in this release cover two workload classes / families that jointly characterise contemporary production traffic while stressing inference systems in distinct ways. The first consists of **basic tasks** (single-turn chat) on relatively short input and output contexts. Secondly, to track the state of the art we include a workload based on **reasoning and agentic** workflows. The long outputs, multi-turn structure, and tool-calling traces in this workload yield latency and cost profiles unlike traditional inference workloads. 

The TEAS benchmarks use mostly Mixture of Experts (MoE) models as they represent the majority of state-of-the-art open-source LLMs, as well as a small dense model (`qwen3-4b`) as a low memory requirement control that enables comparison across a wide range of devices. These are served using vLLM or SGLang on a range of GPUs including NVIDIA A100, H100, H200, B200, B300, GB10, AMD MI355X (as well as custom engines for emerging hardware, see [§10](#10-support-for-emerging-hardware)).

| Family/workload | Models | Benchmark Datasets |
|---|---|---|
| moe (basic tasks/single turn) | `gpt-oss-120b`, `qwen3-235b-a22b-instruct-fp8`, `deepseek-r1`, `kimi-k2.5` | `gsm8k`, `arena-hard`, `longbench_v1` |
| agentic (multi-turn, tools) | `gpt-oss-120b`, `deepseek-v3.2` | `imo-answerbench`, `mcp-atlas`, `swe-bench-lite` |

For more information, see the [page on Methods at teasbench.com](https://www.teasbench.com/methods).

## 1. The TEASBench benchmarking pipeline

The TEASBench pipeline tool provided in this repository encodes many of the settings and parameters used to produce the results shown on the [TEASBench website](https://www.teasbench.com), though does it not cover all cases, for example where part of the software stack used is not yet publicly available or currently somewhat fragile / difficult to standardise on. <!-- link to results repository -->

This README describes how to use the TEASBench pipeline provided in this repository to run the TEAS benchmark experiments on a Kubernetes (K8s) GPU cluster such as EIDF (the [Edinburgh International Data Facility](https://eidf.ed.ac.uk/)) or a cloud provider such as Vast.ai (rented GPU instances). 

Each experiment is described by a **row in a CSV file** (see `./experiments/` for example CSV files). 

The pipeline generator (`./pipeline/generator.py`) turns each row in given CSV file into something you can launch:

- **K8s cluster (e.g. EIDF)** → one Kubernetes Job YAML per row. Submit it; it runs unattended and
  pushes its own results.
- **Vast.ai** → one bash launch script per (family, engine, GPU) group. Run it; it rents an instance, runs every row in that group, pushes results, and
  destroys itself.


---
## 2. Prerequisites

### Everywhere

Generation needs a Python environment with `pandas` and `pyyaml`. Run the generator **from inside the git repo** (it embeds the TEASBench commit for provenance).

### K8s only

- `kubectl` configured for your project namespace 
- These cluster secrets must exist:

| Secret | Key | Needed for |
|---|---|---|
| `gemini-api-key` | `key` | judge for `imo-answerbench` |
| `openrouter-api-key` | `key` | judge for `mcp-atlas` (`kubectl create secret generic openrouter-api-key --from-literal=key=...`) |
| `mcp-atlas-github-token` | `token` | `mcp-atlas` tool servers |
| `mcp-atlas-brave-api-key` | `key` | `mcp-atlas` tool servers |

**SWE-bench Lite**

SWE-bench Lite needs a Python environment on the login node and several environment variables set. One script builds it, once:

```bash
bash pipeline/k8s/setup/setup_swebench_env.sh
```

  This installs `agent_cap`, `swe-rex` and `swebench` into a Python venv, clones SWE-agent and applies
  and verifies AgentCAP's streaming patch, then writes `env.sh` (environment setup) and `versions.json` (recorded into each run's metadata). 
  
On EIDF and other clusters that do not grant pods role-based access control (RBAC), SWE-bench Lite benchmarks are launched not as an unattended K8s Job but using a bash driver script on the login node that can be run interactively or backgrounded - see [§4.6](#46-preflight-check-for-portforwardk8sprovider-mode-clusters-without-pod-rbac) and [§4.7](#47-swe-bench-lite-on-a-k8s-cluster). 

> On EIDF and other clusters that do not grant pods RBAC, SWE-bench experiments are run using TEASBench's `PortForwardK8sProvider` mechanism described in [§4.6](#46-preflight-check-for-portforwardk8sprovider-mode-clusters-without-pod-rbac) and [§4.7](#47-swe-bench-lite-on-a-k8s-cluster). On clusters that permit RBAC the alternative `InClusterK8sProvider` mechanism is available - see [§4.5](#45-preflight-check-for-inclusterk8sprovider-mode-clusters-that-grant-pod-rbac), however this has not been tested. 

### Vast.ai only

- `vastai` CLI installed and authenticated (`vastai login`)
- The container images built and pushed ([§6.1](#61-vastai-setup))
- Instance secrets set in the Vast.ai console ([§6.2](#62-generate-and-launch))

---

## 3. The experiments CSV

Predefined validated benchmark parameters are provided in CSV files in [`./experiments/`](./experiments/).

One file is marked differently. `moe-experiments-vastai-beta.csv` holds the B200 and B300
coordinates behind published results, reconstructed from those runs' own records. **Beta means it
has not been re-executed through this pipeline**, unlike the other CSVs here the rows describe
hardware we measured on, not a matrix we have re-run end to end. Marketplace supply and price move
constantly, and the largest node sizes are the thinnest, so each generated script opens by searching
for matching offers: read what that returns before committing to a run.


**MoE:**

Ordered column header fields headers that define an MoE experiment:

```
family,inference_engine,model,dataset,num_samples,gpu,num_gpu,batch_size,input_length,output_length
```

Example row for an MoE experiment:

```
moe,sglang,gpt-oss-120b,gsm8k,256,A100,1,default
```

> Note: `input_length` and `output_length` are optional and only used for fixed-length mode as referenced in [TEASBench Insights](https://www.teasbench.com/insights))

**Agentic:**

Ordered column header fields that define an agentic experiment:

```
family,benchmark,inference_engine,model,gpu,num_gpu,num_tasks,concurrency,batch_size
```

Example row for an agentic experiment:

```
agentic,swe-bench-lite,sglang,gpt-oss-120b,H100,2,100,4,default
```

---

## 4. Running on K8s cluster
### 4.1 Generate

```bash
cd pipeline
python3 generate.py --csv_file=../experiments/moe-experiments-eidf.csv
```

Options: 

* `--target_dir` specifies where to output job yaml files generated (default `./`)
* `--results_repo` specifies a repository to which to commit results (default
`TEAS_Development_Results_Private`).
* `--site` selects which cluster to generate for (default `eidf`) — see below.

**Targeting a different cluster.** Everything specific to one cluster — namespace,
Kueue queue, PVC names, GPU node labels, model staging root, whether pods are
granted RBAC — lives in a site profile at
[`pipeline/configs/sites/<site>.yaml`](./pipeline/configs/sites/). Nothing else in
the pipeline names a cluster. To run on another K8s cluster, copy
[`eidf.yaml`](./pipeline/configs/sites/eidf.yaml), edit the values, and pass
`--site <name>`; no code changes are needed. The site name is also the directory
results are published under, so keep it distinct per cluster.

Generate creates one YAML per row, named after the run:

```
sglang-gptoss120b-gsm8k-ns256-a100x1-bsd.yaml
sglang-gptoss120b-swe-bench-lite-nt100-h100x2.yaml
```

### 4.2 Submit

```bash
./submit_job.sh sglang-gptoss120b-gsm8k-ns256-a100x1-bsd.yaml
```

This creates the Job, copies the job yaml to a job-config-dir, and appends the name to `submitted_jobs.log`. You will need to run `submit_job.sh` and set JOB\_CONFIGS\_DIR to a location that is accessible from your job so that it can copy and store (commit) the job yaml alongside the results for the sake of provenance. 

You can submit several jobs by looping, i.e.:

```bash
for f in out/*.yaml; do ./submit_job.sh "$f"; done
```

### 4.3 Watch

```bash
kubectl -n <namespace> get jobs
kubectl -n <namespace> get pods -w
kubectl -n <namespace> logs -f <pod>
```

Helpers in [`pipeline/k8s/helpers/`](../pipeline/k8s/helpers/): `k8_pod_log.sh`,
`k8_job_desc.sh`, `k8_pod_bash_login.sh`.

For SWE-bench Lite you will also see transient sandbox pods appear and vanish:

```bash
kubectl -n <namespace> get pods -l app=teasbench-sandbox
```

### 4.4 Agentic specifics

**IMO AnswerBench**: nothing extra; one container.

**MCP Atlas**: the pod gets a second container (the tool server) on port 1984.
Check both:

```bash
kubectl -n <namespace> logs <pod> -c mcp-atlas-sidecar
```

**SWE-bench Lite**: *not* an unattended Job on a cluster without pod RBAC (EIDF
among them). See [§4.7](#47-swe-bench-lite-on-a-k8s-cluster): the driver runs
on a login node and creates the Jobs itself. The GPUs are still used through a
Kubernetes Job, as always, only the driver process sits outside the cluster.

### 4.5 Preflight check for InClusterK8sProvider mode (clusters that grant pod RBAC)

> **Only for clusters that grant pods RBAC.** It tests `InClusterK8sProvider`,
> which needs that. EIDF does not grant it, so skip this section there and go to
> [§4.6](#46-preflight-check-for-portforwardk8sprovider-mode-clusters-without-pod-rbac)
> (mechanism check) and [§4.7](#47-swe-bench-lite-on-a-k8s-cluster) (running it).

In-cluster SWE-bench depends on two facts about the cluster that are worth
confirming *before* a GPU job queues, because both fail late and confusingly:

1. **Pod IPs are routable within the namespace**, the driver talks to a sandbox
   at `http://<podIP>:9999` with no port-forward. A restrictive NetworkPolicy
   breaks this.
2. **`kubectl` works in-cluster from the ServiceAccount token**, with no
   kubeconfig, and the RBAC grants the verbs the provider uses.

```bash
kubectl -n <namespace> create -f pipeline/k8s/preflight/teasbench-preflight.yaml
kubectl -n <namespace> logs -f job/teasbench-preflight
```

The preflight does exactly what `InClusterK8sProvider.acquire()` does; same
kubectl bootstrap, same Job-spec shape, same jsonpath to read the pod IP, same
port, but against a busybox target instead of a multi-GB SWE-bench image, and
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
kubectl -n <namespace> get networkpolicy          # empty = nothing blocking pod-to-pod
kubectl -n <namespace> auth can-i create jobs \
        --as=system:serviceaccount:<namespace>:teasbench-runner
```

The second impersonates the ServiceAccount from your own session, so it checks
the RBAC without running anything. Note it needs impersonation rights, which not
every project grants, the preflight Job needs none, since it *is* the
ServiceAccount.

If the routability check fails, in-cluster mode is unusable on this cluster; use
`PortForwardK8sProvider` [§4.6](#46-preflight-check-for-portforwardk8sprovider-mode-clusters-without-pod-rbac), which needs neither assumption.

### 4.6 Preflight check for PortForwardK8sProvider mode (clusters without pod RBAC)

Everything SWE-bench does on a K8s cluster without pod RBAC (EIDF among them) rests on `PortForwardK8sProvider`, so check it
before committing GPU time. This runs **on a login node**, not as a Job, because
that is where the provider itself runs. It uses your own kubectl credentials and
needs no ServiceAccount and no RBAC manifest, which is exactly why this is the
path such a cluster can support.

**Fast probe** (~1 min, no GPU):

```bash
python3 pipeline/k8s/preflight/preflight_portforward.py --namespace <namespace>
```

It drives the **real** `PortForwardK8sProvider` OS port allocation, the
`kubectl port-forward` spawn, the readiness poll, the tunnel-babysitter thread
and cleanup on release, plus the `kubectl cp` / `kubectl exec` path the
SWE-bench evaluator uses. Only the sandbox container's payload is substituted
(busybox serving an `is_alive` file instead of a multi-GB image pip-installing
swe-rex), because what is under test is the tunnel mechanism, not swe-rex.

The check most worth reading is this one — it doesn't just wait and hope, it
kills the tunnel and checks that the babysitter actually puts it back:

```
  [3b] fault injection: killing kubectl port-forward mid-task
        killed kubectl port-forward pid 84213 (local port 41235)
        waiting up to 90s for the babysitter to notice and respawn the tunnel
        on the SAME local port (41235)...
  PASS  tunnel recovered on the same local port (41235) after 6s
  PASS  babysitter journalled a pf_drop/running row for this drop
```

It kills the local `kubectl port-forward` process out from under a live
sandbox and checks two things recover, reported separately so a failure says
which half broke: the tunnel itself, back up on the **same** local port
(SWE-agent is handed that port once at launch and has no way to learn a new
one), and a `pf_drop` / `phase: "running"` row in the drop journal for the
drop — that exact row is what makes a task eligible for retry (see the
developer guide, §5, for the journal schema and the retry classifier). A
killed local process isn't quite the production failure that motivated the
babysitter rewrite — a dropped `kubectl` is not the same as an apiserver-side
stream reset that leaves `kubectl` running but silently stops forwarding —
but it's the closest fault this preflight can inject from outside the
cluster, and it exercises a recovery path that used to be entirely
unverified outside of production. It runs by default and adds ~90s; skip it
with `--no-fault-injection` when you only want the faster checks.

A `kubectl port-forward` that dies quietly mid-task is the failure the
babysitter thread exists to prevent, and the one that would otherwise surface as
an inexplicable task failure deep into a long run.

The permission unique to this path is `create pods/portforward`; the in-cluster
provider never needs it, because it talks to pod IPs directly.

**Full-fidelity probe** (minutes, still no GPU):

```bash
python3 pipeline/k8s/preflight/preflight_portforward.py --namespace <namespace> --real-image
```

`--real-image` drops the busybox substitution for a genuine
`docker.io/swebench/sweb.eval.x86_64.*` image, additionally proving the multi-GB
pull works and that `swe-rex` installs and runs inside the instance image, an
old conda env where a dependency clash is plausible. Pass an instance id to
override the default (`--real-image django__django-11099`).

### 4.7 SWE-bench Lite on a K8s cluster

On a cluster that **does not grant pods RBAC** — EIDF among them — SWE-bench
always uses `PortForwardK8sProvider`, with the **driver** running on a login
node. This is the validated path; the in-cluster alternative is [§4.5](#45-preflight-check-for-inclusterk8sprovider-mode-clusters-that-grant-pod-rbac).

#### Where everything actually runs

The driver moving off the cluster does **not** mean Kubernetes is bypassed. The
model runs on GPUs through a Job, exactly as every other TEASBench run does.
Four components, three of them Kubernetes Jobs, all created with *your*
credentials, and all of it handled for you:

| Component | Where | GPU | Created by |
|---|---|---|---|
| **Driver** (`agent_cap.agents`) | login-node VM | no | you, by running the generated script |
| **Engine** (sglang/vllm serving the model) | Kubernetes Job | **yes** | the driver, at start-up |
| **Sandbox** (swe-rex, one per task) | Kubernetes Job | no | the driver, at run time |
| **Eval** (official grading) | Kubernetes Job | no | the driver, at run time |

The driver reaches the engine and each sandbox over `kubectl port-forward`.

Why this is allowed when in-cluster mode is not: *you* may create Jobs and
port-forward, and [§4.6](#46-preflight-check-for-portforwardk8sprovider-mode-clusters-without-pod-rbac) confirms both. What such a cluster refuses is granting those
rights to a **pod's ServiceAccount**. Running the driver as yourself sidesteps
that entirely.

IMO AnswerBench and MCP Atlas are unaffected and still run as ordinary
unattended Jobs, because neither touches the Kubernetes API.

#### Running it

The pipeline handles all of it, including the engine. Generation emits **two**
files for a SWE-bench row on such a cluster instead of one:

```bash
cd pipeline
python3 generate.py \
    --csv_file=../experiments/swe-bench-lite-eidf.csv --target_dir=./out
```

```
sglang-gptoss120b-swe-bench-lite-nt100-h200x1.sh           <- driver script 
sglang-gptoss120b-swe-bench-lite-nt100-h200x1.engine.yaml  <- driver script submits for you
```

Any shell that wants to launch the driver script to run an `swe-bench-lite` experiment must first source the resulting environment initialisation script:

```
source ~/teasbench-env/env.sh
```

Then start it and walk away:

```bash
bash out/sglang-gptoss120b-swe-bench-lite-nt100-h200x1.sh
```

The script checks prerequisites, submits the engine Job, waits for the model to
load, opens and babysits the tunnel, runs the benchmark (retrying tasks that
were only lost to a dropped tunnel, see below), checks the run is complete
enough to publish, pushes the results, and **deletes the engine Job on exit**;
success, failure or Ctrl-C. This means an aborted run cannot leave GPUs
allocated. You never start an engine or submit the engine manifest by hand.

Useful flags: `--no-push`, `--namespace`, `--output-root`.

There is no `--resume` flag: an earlier version parsed one but never acted on
it, and since `$TIMESTAMP` is recomputed on every invocation there was never a
previous run directory for it to resume into. Resuming a run that lost tasks
to a dropped tunnel now happens automatically, inside the script — see
"Retrying dropped sandbox tunnels" below.

The driver contains **no install paths of its own**: `TEASBENCH_ROOT`,
`AGENTCAP_DIR`, `SWEAGENT_DIR` and the interpreter all come from `env.sh`, and it
refuses to start if that has not been sourced. A generated script is therefore
portable between machines: relocate a checkout and re-run the setup script rather
than editing anything generated.

#### Retrying dropped sandbox tunnels

A dropped `kubectl port-forward` tunnel to a sandbox kills the task using it
— but used to leave the run looking like a clean success anyway: a row still
landed in `results.jsonl` with an empty patch, and the script exited 0. The
driver now runs the client in a bounded retry loop instead of once, controlled
by two env vars (set them before invoking the script; they are not CLI flags,
since they tune the internal loop rather than anything a one-off run needs to
override):

| Env var | Default | Meaning |
|---|---|---|
| `MAX_ATTEMPTS` | `6` | backstop on total client invocations, including the first. The loop normally exits earlier -- when the retry list is empty, or when an attempt fails to shrink it |
| `RETRY_TIMEOUTS` | `1` | also retry tasks killed by the outer per-task `timeout` (`sweagent_rc == 124`), once each |
| `SKIP_PREFLIGHT` | `0` | set to `1` to bypass the real-image sandbox preflight in step `[1b]`. That gate pulls a real SWE-bench instance image and proves a sandbox pod actually serves swe-rex before any GPU is claimed — it is the only check that exercises the pod command, which otherwise fails invisibly until every task has burned its 600s sandbox timeout |

After each attempt — while attempts remain and the engine is still serving
`/v1/models` — `swebench_run_audit retry-list` decides which tasks to re-run:
only ones with positive evidence of an infrastructure failure (a tunnel drop
seen by the babysitter after it was already up, a `k8s sidecar failed:`
error, a tunnel-drop signature in the SWE-agent logs, or, with
`RETRY_TIMEOUTS=1`, a first-time outer timeout). A task the agent itself gave
up on — ran out of cost, format, or context budget, or finished and submitted
nothing — is never retried; doing so would give it a second sample and bias
accuracy upward. `swebench_run_audit prune` then archives `results.jsonl` and
clears the retried tasks' stream stats and trajectory files so a retry
can't inherit an earlier attempt's numbers or patch, and the loop re-invokes
the client with AgentCAP's own `--resume`. Full retry/do-not-retry rules are
in the developer guide, §5.

Retrying a whole task is the fallback, not the first line of defence: a tunnel
drop should not cost the task in the first place. swe-rex already ships
everything needed for that — `RemoteRuntime._request` has a retry loop, and
every request carries an `X-Request-ID` the server treats as an idempotency
key — but `num_retries` defaults to `0` and no caller overrides it, so the
first `ServerDisconnectedError` is fatal even though the babysitter restores
the tunnel in a millisecond or two. `pipeline/k8s/setup/patch_swerex_retries.py` turns
the retries on, and makes the server register a request as in-flight *before*
running it so a retry that arrives mid-execution waits for the original
instead of putting a second command on the same shell. The client half is
applied on the login node by `setup_swebench_env.sh`; the server half is
applied inside each sandbox pod, which pip-installs swe-rex at startup.

| Env var | Default | Meaning |
|---|---|---|
| `SWEREX_NUM_RETRIES` | `3` | transport-level retries per swe-rex request. `0` restores stock behaviour |

#### Completeness gate

Before pushing anything, the driver runs `swebench_run_audit report`, which
writes `$RUN_DIR/completeness.json` and exits non-zero if any task is still
infrastructure-incomplete after the retry loop, or if `predictions.json` /
`eval_k8s_results.json` are short of the number of patched tasks in
`results.jsonl`. This is the check the evidence run that motivated all of
this would have failed: it exited 0 with `status: "completed"` and
`acc: 0.200` while 55 of 100 tasks had silently produced no patch.

**A failed gate means nothing is published.** The script exits 1 without
pushing — the engine Job is still torn down first, since the `EXIT` trap runs
regardless, so a gate failure costs no stranded GPU time. Everything stays in
`$RUN_DIR`; inspect `completeness.json` for exactly which tasks are still
incomplete and why, then decide by hand whether to re-run or push manually.

A run directory for a `PortForwardK8sProvider` SWE-bench run now additionally
contains:

| Path | What |
|---|---|
| `portforward-events.jsonl` | the drop journal — one JSON line per tunnel start/drop/restart/release event, engine and sandbox tunnels alike |
| `portforward/` | per-sandbox `kubectl port-forward` stderr, one log per task (previously discarded to `/dev/null`) |
| `completeness.json` | the completeness report written by the gate above |
| `results.attempt-N.jsonl` | `results.jsonl` as it stood before attempt `N+1`'s retries — one archive per retried attempt |

#### Provenance

After the run the driver stamps `versions.json` into every `metadata_*.json`, so
a directory in the results repo records the exact code that produced it without
reference to anything outside it.


#### Smoke test first

The 2-task row in `experiments/agentic-smoke-tests-eidf.csv` is the same thing at
small scale; same driver, same engine Job, same sandboxes, same grading:

```bash
cd pipeline
python3 generate.py \
    --csv_file=../experiments/agentic-smoke-tests-eidf.csv --target_dir=./out
bash out/vllm-gptoss120b-swe-bench-lite-nt2-a100x1.sh
```

A low or zero accuracy on 2 tasks is normal and not a failure signal; what
matters is that every stage ran and wrote its outputs.

**Run [§4.6](#46-preflight-check-for-portforwardk8sprovider-mode-clusters-without-pod-rbac) first:  if the mechanism is broken, this fails for that reason after queueing for a GPU.**


## 5. Running MoE benchmarks on Vast.ai

Most of the engine driving the benchmarks is built into the container images we use
in the Vast.ai pipeline.

### 5.1 Vast.ai setup

On Vast.ai itself, either through the command line interface or the web console,
you will need to set the following environment variables to be injected into any
instances you create:

- `GIT_TOKEN`, a GitHub personal access token with read access to the TEASBench repository. This is used to push new results to 
  the repository inside the instance.
- `HF_TOKEN`, a Hugging Face token used to download the models inside the instance.
- `OPENAI_API_KEY`, an OpenAI API key used for accessing OpenAI services when judging `arena-hard` benchmark results.

### 5.2 Local prerequisites

Aside from the prerequisites listed above, you will need to install the Vast.ai CLI and authenticate it: https://docs.vast.ai/cli/.
The Vast.ai pipeline uses the CLI to search for offers and launch instances; from that point on, a benchmark run will manage itself.

### 5.3 Generate launch scripts

The scripts to launch Vast.ai instances are created in a similar fashion to the K8s Job YAMLs. `--site vastai`
selects the Vast.ai site profile, whose orchestrator makes the generator produce launch scripts rather than Job
YAMLs. For example, the following command generates
Vast.ai scripts in the `out/` directory for the MoE experiments described in `moe-experiments-vastai.csv`:

```bash
cd pipeline
python3 generate.py \
    --csv_file=../experiments/moe-experiments-vastai.csv \
    --target_dir=./out \
    --site vastai
```

This produces one script per (engine, GPU, num_gpu) triplet, e.g.
`vast_sglang_H200x8.sh`, with the relevant rows from the original CSV file base64 encoded within.

### 5.4 Launch benchmarks

Run a generated script to launch a Vast.ai instance running one of the TEASBench images. This will
automatically run the benchmarks described in the original CSV file corresponding to this
script's (engine, GPU, num_gpu) triplet and push results to GitHub.

The scripts take no arguments. For example, to launch the example script generated just above,
for a `sglang` engine on a `H200` instance with 8 GPUs, run:

```bash
bash out/vast_sglang_H200x8.sh
```

The script will show current offers from Vast.ai matching the requested GPU type and number,
prompting you to select one of the offer IDs. The offers shown are sorted by price, cheapest first.

Selecting an offer will launch the instance, printing its ID to the terminal.

The running instance can be monitored through the Vast.ai web console or CLI. The benchmarks will run
sequentially, pushing results to GitHub as they complete. Once all are complete, the instance will
automatically destroy itself.

---

## 6. Running agentic benchmarks on Vast.ai

The process for running agentic benchmarks on Vast.ai is broadly similar to the MoE benchmarks, with some
additional environment variables required for certain benchmarks. The process is outlined here.

### 6.1 Vast.ai setup

As with the MoE benchmarks, some environment variables within the container must be provided
through the Vast.ai interface so they can be picked up and used within the instance. Exactly
which are required depend on the benchmark(s) being run. Use the following table:

| Environment variable                                  | Purpose                          | Needed for benchmark?          |
|-------------------------------------------------------|----------------------------------|--------------------------------|
| `GIT_TOKEN`                                           | Push results to repo             | everything                     |
| `HF_TOKEN`                                            | Download models from Hugging Face | everything                     |
| `GEMINI_API_KEY`                                      | Judge                            | `imo-answerbench`              |
| `OPENROUTER_API_KEY`                                  | Judge                            | `mcp-atlas`                    |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`                | Run Modal sandboxes              | `swe-bench-lite`               |
| `GITHUB_TOKEN`, `BRAVE_API_KEY`, `ALCHEMY_API_KEY`, … | Tool server API keys (see below) | `mcp-atlas`                    |

For `mcp-atlas`, any tool server API key in the [mcp-atlas `env.template`](https://github.com/scaleapi/mcp-atlas/blob/main/env.template)
is picked up from a same-named environment variable, set whichever you have, using the template as a guide.

### 6.2 Generate and launch

Running an agentic benchmark suite is otherwise identical to the MoE benchmark process.
Provide a CSV file identical to those used for K8s runs and use the `generate.py` script
with `--site vastai` to generate bash scripts:

```bash
cd pipeline
python3 generate.py \
    --csv_file=../experiments/swe-bench-lite-vastai.csv \
    --target_dir=./out \
    --site vastai
```

Then, run one of the generated bash scripts:

```bash
bash out/vast_agentic_swe-bench-lite_sglang_H200x1.sh
```

This will use the Vast.ai CLI to find and list appropriate offers and prompt you to select one.
The appropriate benchmark container will run on the instance you select, pushing benchmark results
to GitHub as they complete.

### 6.3 What differs from a K8s cluster

Due to the way agents run, there are some differences in how they are executed on Vast.ai instances vs a K8s cluster.
The following table summarises those differences:

|                       | K8s cluster       | Vast.ai                            |
|-----------------------|-------------------|------------------------------------|
| MCP Atlas tool server | sidecar container | background process, same container |
| SWE-bench sandboxes   | Kubernetes pods   | Modal                              |
| SWE-bench grading     | exec pods         | Modal                              |

---

## 7. Troubleshooting

### 7.1 SWE-bench Lite permissions (if using InClusterK8sProvider) 

If sandbox creation fails with a `kubectl` permissions error, either apply
`pipeline/k8s/rbac/teasbench-runner-rbac.yaml`, or switch to the login-node fallback by
pointing the run at `PortForwardK8sProvider` instead of `InClusterK8sProvider` 
that provider uses *your* kubectl credentials via port-forwarding rather than
the pod's ServiceAccount. Confirm it works first with [§4.6](#46-preflight-check-for-portforwardk8sprovider-mode-clusters-without-pod-rbac).

### 7.2 MCP Atlas scores lower than expected

**Check the credential log first.** Tool servers with blank API keys still start
and fail only at tool-call time, so missing credentials look like poor model
performance. The run logs which keys were supplied and which were empty (names
only, never values):

```
credentials supplied: BRAVE_API_KEY GITHUB_TOKEN
left empty: ALCHEMY_API_KEY EXA_API_KEY ...
```

Also confirm the server set matches, it is pinned to the same 22 servers on
both platforms, because the server set *is* the benchmark definition.

---

## 8. Where results go

```
<family>/<platform>/<engine>/<model>/<dataset-or-benchmark>/<gpu_type>x<num_gpu>/batch-size-<default-or-1>/<timestamp>/
```

Example paths:

```
moe/eidf/sglang/gpt-oss-120b/gsm8k_256samples/a100x1/batch-size-default/20260727-1432/
agentic/vastai/sglang/gpt-oss-120b/swe-bench-lite/h200x1/batch-size-default/20260727-1432/
```

Each timestamped run directory holds `metrics_*.json`, `metadata_*.json`,
`detailed-results_*.jsonl`, `output-data_*.jsonl`, `timings.json`, the job YAML and/or driver run script
or provenance, and logs.

Aggregate with:

```bash
python3 postprocessing/aggregate_results.py --results_dir <repo>/moe
```

Point `--results_dir` at the `moe` or `agentic` subdirectory, **not** the repo
root.

---

## 9. Common tasks

**Smoke test before a real sweep**

```bash
cd pipeline
python3 generate.py --csv_file=../experiments/moe-smoke-tests-eidf.csv --target_dir=./out
python3 generate.py --csv_file=../experiments/agentic-smoke-tests-eidf.csv --target_dir=./out
```

**Add a model**

Add to `HF_MODEL_MAP`, `MODEL_SHORT_NAME_MAP` and (for
Vast.ai) `MODEL_DISK_GB_MAP` in `pipeline/utils.py`.

**Add a GPU or other device** 

Add it to `gpu_products` in the relevant site profile
(`pipeline/configs/sites/<site>.yaml`), plus `TEAS_GPU_NAME_MAP` in
`pipeline/utils.py`. For a K8s site the value is the `nvidia.com/gpu.product`
node label; for Vast.ai, take the string from `vastai search offers`.

**Change inference engine, client, environment, or other parameters for one case** 

Add a rule to `pipeline/configs/config.yaml`. Rules match on any parameter or combinations of parameters (`benchmark`,
`platform`, `gpu`, `model`, `inference_engine`, …); more specific rules override
more general ones.

**Change agent sampling parameters** 

Edit the relevant
`pipeline/configs/agents/<benchmark>_<model>_<engine>.yaml`.

**Verify you haven't broken anything**

```bash
python3 -m pytest tests/ -q
```

## 10. Support for emerging hardware 

### Tenstorrent 

Currently the TEASBench pipeline does not support running the TEASBench benchmarks on Tenstorrent accelerators due to limited [Tenstorrent inference engine model support](https://github.com/tenstorrent/tt-inference-server/blob/main/docs/model_support/llm/README.md). As model support matures we expect to extend the pipeline to target Tenstorrent hardware building on [our approach](./pipeline/dev/tenstorrent/README.md) running a simpler dense model, Llama3.1-8b-Instruct, on a Tenstorrent Blackhole p150b served using [`pipeline/dev/tenstorrent/tt-llm-run.sh`](./pipeline/dev/tenstorrent/tt-llm-run.sh), which serves as a concrete example based on the documentation on [how to deploy LLMs using tt-inference](https://docs.tenstorrent.com/getting-started/vLLM-servers.html). 

Note: [https://www.teasbench.com](https://www.teasbench.com) includes benchmark results for `qwen3-4B` -  another dense model - on Tenstorrent Blackhole p150, however this was produced using custom kernels under development rather than the publicly available Tenstorrent software stack (`tt-inference`) available at time of writing (August 2026). 

