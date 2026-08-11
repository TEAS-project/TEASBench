# Agentic pipeline design

This document explains how TEASBench runs *agentic* benchmarks (as opposed to
the MoE server+client sweeps the rest of this repo was originally built for):
which parts live where, why, and what to watch out for when reading the
numbers. It assumes no prior context beyond general familiarity with k8s and
LLM inference serving.

## 1. The AgentCAP / TEASBench split

Two separate projects cooperate to run an agentic benchmark:

- **[AgentCAP](https://github.com/Auto-CAP/AgentCAP) owns benchmark
  semantics.** Agent strategy (how the model is prompted and looped), call
  limits, streaming metrics (TTFT/TPOT capture), official grading (e.g. the
  real SWE-bench test harness), dataset loading, and the TEAS output schema
  (`metrics_*.json`, `metadata_*.json`, etc.) all live in AgentCAP.
- **TEASBench owns deployment scenario.** Which platform (EIDF vs Vast.ai),
  which hardware, which inference engine, how containers get provisioned, and
  where results end up all live in TEASBench -- this repository.

Neither project needs to know the other's internals. The interface between
them is a small number of **endpoints**:

1. One OpenAI-compatible LLM endpoint (`--base-url`) -- the inference-engine
   server this pipeline starts.
2. Per SWE-bench task, one **sandbox endpoint** `{host, port, auth_token}`
   speaking the swe-rex protocol (SWE-agent's `remote` deployment type) --
   supplied by a TEASBench-owned *sandbox provider*.
3. Per SWE-bench evaluation, an **exec container** (upload a file / run a
   command) built from the official SWE-bench instance image -- supplied by a
   TEASBench-owned *exec provider*.

AgentCAP consumes these endpoints through a small, deliberately generic seam
(`get_sandbox_provider` / `get_exec_provider` in
`agent_cap/agents/sandbox_providers.py`, resolved from a CLI flag): a name that
is either a built-in registry key, an `http(s)://` URL, or a **dotted Python
path** `"package.module:ClassName"` that gets imported and instantiated. This
pipeline's k8s Job template selects TEASBench's own implementation with:

```
--sandbox-provider k8s_pod_providers:InClusterK8sProvider
--exec-provider    k8s_pod_providers:InClusterK8sProvider
```

Making `k8s_pod_providers` importable is nothing more than putting one
directory of TEASBench's own checkout on `PYTHONPATH`
(`PYTHONPATH=/dev/shm/TEASBench/pipeline/k8s/lib`, since the job clones
TEASBench there anyway to pick up `configs/agents/*.yaml`). That `lib/` level
exists so the path entry contains *only* the provider package: pointing it at
`pipeline/k8s/` instead would put `setup`, `preflight`, `rbac` and `helpers` on
`sys.path` as PEP 420 namespace packages, ahead of `site-packages`, where they
could shadow a real module of the same name in agent_cap, swe-agent or swe-rex.
TEASBench is not pip-installed and carries no
`pyproject.toml` -- this dotted-path trick is the entire packaging story, by
design.

## 2. Three benchmark tiers

The three agentic benchmarks this pipeline generates jobs for increase in
mechanical complexity, in this order:

| | IMO AnswerBench | MCP Atlas | SWE-bench Lite |
|---|---|---|---|
| AgentCAP strategy | `single` | `single --tool-backend mcp` | `sweagent` |
| AgentCAP evaluator | `imo` | `gtfa` | `swebench-k8s` (K8s) |
| tool server needed | no | yes, one fixed instance (port 1984) | no |
| dynamic containers | no | no | yes, ~100 distinct per-task images |

IMO AnswerBench is the simple case: one model endpoint, a math-tool backend
that runs in-process, an LLM judge for semantic answer equivalence. MCP Atlas
adds exactly one extra moving part -- a tool server the agent talks to over
HTTP -- but that server's identity and count never change at runtime. SWE-bench
Lite is qualitatively different: each of the ~100 tasks in the curated subset
needs its own sandbox, built from a different (large) instance image, spun up
and torn down per task while the run is in flight.

All three run through the *same* `pipeline/templates/agentic.yaml` k8s Job
template. Nothing about the template is forked per benchmark; instead,
`pipeline/configs/config.yaml` carries composable, benchmark-keyed rules
(sidecar containers, wait-for-readiness scripts, extra setup steps, RBAC
requirements, ...) that render empty for the benchmarks that don't need them.
See `pipeline/README.md` for the CSV schema and how to generate a job.

## 3. Two platforms

The same three benchmarks also run on two platforms, which differ in how they
provide the two capabilities that vary at all: a tool server, and dynamic
sandboxes.

| | K8s cluster | Vast.ai |
|---|---|---|
| unit of work | one k8s Job per experiment, generated YAML | one rented instance per (engine, gpu, num_gpu) group, running a CSV loop |
| tool server (MCP Atlas) | a **sidecar container in the same pod** | a **background process in the same container** |
| SWE-bench sandboxes | the TEASBench k8s provider (dotted path) | **Modal** -- native to swe-rex, no provider needed |
| SWE-bench eval | `swebench-k8s` evaluator + exec provider | `swebench` evaluator with `--modal true` |

## 4. Why capability, not mechanism

The design rule that keeps this from turning into a combinatorial mess:
**abstract by capability, never by implementation mechanism.**

The concept this codebase reasons about is "the tool-server endpoint" and "the
sandbox substrate" -- not "a pod sidecar" or "a kubectl subprocess". A pod
sidecar is merely Kubernetes's way of providing the tool-server capability; a
background process is merely Vast.ai's way of providing the *same* capability.
Code (and rules, and templates) should ask "does this benchmark need a tool
server?" and "does this platform provide one as a sidecar or as a background
process?" -- never "is this k8s?". This is why `pipeline/templates/agentic.yaml`
has exactly one template with composable blocks, not one template per
benchmark, and why the sandbox/exec provider seam in AgentCAP is a dotted
Python path rather than a k8s-specific parameter.

The previous (abandoned) attempt at this got the *idea* right -- a sidecar
container for the tool server -- but implemented it as a second, hand-forked
300-line template with invalid YAML (a duplicated top-level `containers:` key,
which silently discarded the sidecar rather than erroring) and inlined literal
secret values. Forking the template per variant does not survive a third
variant; this design deliberately avoids doing that again.

## 5. Why no HTTP broker

An earlier design considered introducing a small HTTP service to broker
sandbox allocation across platforms. That idea was deliberately dropped:

- Modal's own SDK already *is* the broker on Vast.ai -- there is nothing left
  for a bespoke broker to do there.
- `docker` and `modal` deployment types are native to swe-rex (SWE-agent's
  remote-execution library); no extra layer is needed for either.
- Kubernetes is the *only* substrate swe-rex cannot provision on its own,
  which is exactly why the sandbox/exec provider seam exists in the first
  place -- it is a k8s-shaped gap, not a general "we need a broker" problem.

AgentCAP's existing `HttpSandboxProvider` (which *would* talk to such a
broker) is kept as an unused escape hatch, not deleted, in case a future
platform genuinely needs it. It should not be extended on the strength of this
decision alone.

## 6. Measurement caveat: sandbox placement is part of the scenario

For SWE-bench Lite, every tool call the agent makes against its sandbox is a
network round trip: agent process → sandbox pod (or Modal container) → back.
That round-trip time (RTT) is not a side channel -- it lands directly inside
the per-task end-to-end latency AgentCAP records, exactly like any other
inference latency. Two runs with identical model, engine, hardware and task
set can therefore report different latency numbers purely because one placed
its sandboxes in-cluster (`InClusterK8sProvider`, pod-IP-routable, low
RTT) and the other reached them through a port-forward tunnel from a login
node, or through Modal's own network path.

Practically, this means **sandbox placement is part of the measured
scenario**, not an implementation detail that can be abstracted away after the
fact. This pipeline's job template writes the sandbox/exec provider choice
(and, on a K8s cluster, whether it went through `InClusterK8sProvider` or the
`PortForwardK8sProvider` fallback) into `provenance.json` alongside the
TEASBench/AgentCAP commit hashes and per-phase timings, precisely so that
later analysis can tell these runs apart instead of quietly averaging over a
confound.
 