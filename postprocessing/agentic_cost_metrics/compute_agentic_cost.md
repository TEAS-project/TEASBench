# `compute_agentic_cost.py` — Rent + Buy cost computation for TEAS_Development AGENTIC runs

Computes per-run rent + buy cost for agentic benchmarks (mcp-atlas, swe-bench-lite, imo-answerbench, …) where each task is a multi-turn LLM-plus-tool-call session.

Two metrics, reported at **avg / p50 / p99**:

- `cost_per_task` — total cost for one task
- `cost_per_1M_output_tokens` — same cost amortized onto output tokens

Buy cost is further **split into GPU vs CPU contributions** and reports two explicit accounting modes: active-resource attribution (default) and reserved-worker upper bound. (See §3 below for the formula.)

For the GPU/CPU specs, sources, and buy formula derivation see [`../moe/compute_cost.md`](../moe/compute_cost.md) — they are identical to the MoE script.

---

## 1. Inputs

### Directory layout

```
<root>/<location>/<framework>/<model>/<dataset>/<gpu>x<n>/<batch>/<timestamp>/metrics(...).json
```

Same GPU-dir regex as the MoE script: `A100x1`, `a100x1_batch-size1`, `mi355xx8` all parse.

### Required metrics fields

From `metrics_<…>.json` / `metrics.json`:

```jsonc
{
  "performance": {
    "avg_e2e_latency_s": <float>,   // wall time per task (LLM + tool calls)
    "p50_e2e_latency_s": <float>,   // optional, used for p50 cost
    "p99_e2e_latency_s": <float>,   // optional, used for p99 cost
    "ttft":              <float>,   // avg time-to-first-token per LLM call
    "p99_ttft":          <float>,   // optional
    "tpot":              <float>,   // avg time per output token
    "p99_tpot":          <float>    // optional
  },
  "agentic": {
    "avg_num_requests":         <float>,  // avg LLM calls per task
    "avg_total_output_tokens":  <float>,  // avg output tokens per task
    "avg_tool_call_count":      <float>   // for reporting only
  }
}
```

Optional `metadata.json` is read for `system_environment.inference_engine_version` (or `hardware.vllm_version` / `hardware.sglang_version`) and copied into `run.framework_version`.

---

## 2. Time decomposition

For each task the wall clock is split into LLM-active vs tool-wait:

```
llm_active_s = avg_num_requests × ttft + avg_total_output_tokens × tpot
tool_wait_s  = max(0, avg_e2e_latency_s − llm_active_s)
```

Intuition:
- `num_req × ttft` ≈ prefill time summed over all turns.
- `out_tok × tpot` ≈ decode time summed over all turns.
- Whatever is left of the wall clock is attributed to tool wait / CPU-side work (network, shell, sandbox, external service latency …). During this interval SGLang-style continuous batching can often use the GPU for other requests.

`max(0, …)` guards against rounding/jitter where the estimate slightly exceeds wall time.

### p99 estimate

The metrics file does not provide `p99_num_requests` / `p99_output_tokens`, so p99 LLM-active is approximated with **avg counts × p99 per-event rates**:

```
p99_llm_active_s = avg_num_requests × p99_ttft + avg_total_output_tokens × p99_tpot
p99_tool_wait_s  = max(0, p99_e2e_latency_s − p99_llm_active_s)
```

This tends to **under-estimate** p99 LLM-active for token-heavy long-tail tasks, which in turn over-counts `p99_tool_wait_s`. Treat p99 numbers as upper bounds for tool-bound cost.

---

## 3. Cost formulas

### Rent (lumped GPU+CPU — provider charges a single hourly rate)

```
$/s              = price_per_GPU_per_hour × num_gpus / 3600

cost_per_task        = avg_e2e_latency_s × $/s
cost_per_1M_out_tok  = avg_e2e_latency_s × $/s × (1e6 / avg_total_output_tokens)

p50_cost_per_task        = p50_e2e_latency_s × $/s         (if p50 present)
p99_cost_per_task        = p99_e2e_latency_s × $/s         (if p99 present)
p99_cost_per_1M_out_tok  = p99_e2e_latency_s × $/s × (1e6 / avg_total_output_tokens)
```

### Buy (split GPU + CPU, two accounting modes)

Per-hour rates (each = capital amortization + energy):

```
gpu_$/h  = (gpu_$ × N × scale_other_capital) / lifetime_hours
         + (gpu_W × N) / 1000 × electricity_$_per_kWh
cpu_$/h  = (cpu_$ × M × scale_other_capital) / lifetime_hours
         + (cpu_W × M) / 1000 × electricity_$_per_kWh
```

The script reports **both** buy-cost modes in every `buy.cost` block. The top-level `buy.cost.avg_cost_per_task_usd` mirrors `--buy-cost-mode` (default `active-resource`) for backward-friendly consumers, while the nested mode blocks are always present.

#### Mode A: `active_resource` (default)

Use this for per-resource active-time attribution, especially continuous batching / multiplexed serving where another request can use the GPU while this task waits on tools.

```
gpu_billable_s = llm_active_s
cpu_billable_s = tool_wait_s

gpu_cost_per_task   = gpu_$/h × gpu_billable_s / 3600
cpu_cost_per_task   = cpu_$/h × cpu_billable_s / 3600
total_cost_per_task = gpu_cost_per_task + cpu_cost_per_task
```

#### Mode B: `reserved_worker` (upper bound)

Use this for a single-task / batch-size-1 worker when the whole machine is treated as reserved for the full request.

```
gpu_billable_s = avg_e2e_latency_s
cpu_billable_s = avg_e2e_latency_s

gpu_cost_per_task   = gpu_$/h × gpu_billable_s / 3600
cpu_cost_per_task   = cpu_$/h × cpu_billable_s / 3600
total_cost_per_task = gpu_cost_per_task + cpu_cost_per_task
```

For per-1M-tokens, multiply each cost by `(1e6 / avg_total_output_tokens)`.

The old mixed convention (`GPU = full e2e`, `CPU = tool_wait`) is intentionally not used because it mixes reserved-capacity accounting for GPU with active-time accounting for CPU. If the GPU is charged as reserved for full e2e, the CPU should be too; if the CPU is charged only while active, the GPU should also be charged only while active.

---

## 4. CLI

```
python3 compute_agentic_cost.py --root <dir> [rent flags] [buy flags] [--dry-run]
```

All rent + buy flags are identical to `../moe/compute_cost.py`. See [`../moe/compute_cost.md`](../moe/compute_cost.md) §3 for the full table. Quick summary:

| Group | Flag | Meaning |
|---|---|---|
| Common | `--root` | Tree to scan |
| Common | `--dry-run` | Print would-be output paths |
| Rent | `--rent-price gpu=usd` | Per-GPU $/h, repeatable |
| Rent | `--rent-prices-json` | JSON dict gpu_key → $/h |
| Rent | `--rent-price-source gpu=url` | Override `rent.price_source` URL |
| Rent | `--rent-price-quote-time` | ISO time stamp |
| Buy | `--buy-gpu-price gpu=usd` | Override default GPU price |
| Buy | `--buy-gpu-tdp gpu=W` | Override default GPU TDP |
| Buy | `--buy-cpu-for gpu=cpu_key` | Change CPU paired with a GPU |
| Buy | `--buy-num-cpus gpu=N` | Change CPU count per platform |
| Buy | `--buy-cpu-price cpu_key=usd` | Override CPU price |
| Buy | `--buy-cpu-tdp cpu_key=W` | Override CPU TDP |
| Buy | `--buy-lifetime-hours` | Tier default: 43800 datacentre / 26280 workstation; explicit value overrides both |
| Buy | `--utilisation` / `--utilization` | Tier default: 0.9 datacentre / 0.4 workstation; explicit value overrides both |
| Buy | `--buy-electricity-usd-per-kwh` | Default 0.15 |
| Buy | `--buy-scale-other-capital` | Default 1.2 (MoE-CAP) |
| Buy | `--buy-cost-mode` | Which mode is mirrored at `buy.cost` top level when active-resource timing is evidenced: `active-resource` (default) or `reserved-worker` |

---

## 5. Built-in defaults

GPU and CPU spec tables (price + TDP + source URLs) are identical to the MoE script. See [`../moe/compute_cost.md`](../moe/compute_cost.md) §4.

Quick refresher:

| GPU key | $/unit | TDP (W) | Default host |
|---|---:|---:|---|
| `a100`   | 18,000 | 400  | 2× Intel Xeon Platinum 8468 |
| `h100`   | 25,000 | 700  | 2× Intel Xeon Platinum 8468 |
| `h200`   | 30,000 | 700  | 2× Intel Xeon Platinum 8468 |
| `b200`   | 35,000 | 1000 | 2× Intel Xeon Platinum 8468 |
| `b300`   | 42,500 | 1400 | 2× Intel Xeon Platinum 8558 |
| `mi355x` | 30,000 | 1400 | 2× AMD EPYC 7713P |

| CPU key | $ | TDP (W) |
|---|---:|---:|
| `xeon-8468`  | 7,214 | 350 |
| `xeon-8558`  | 5,208 | 330 |
| `epyc-7713p` | 5,010 | 225 |

Override any of these with the `--buy-*` flags above (per-GPU or per-CPU).

---

## 6. Output schema

Sidecar JSON written to the same directory as the metrics file:

| Input | Output |
|---|---|
| `metrics.json` | `cost.json` |
| `metrics_<suffix>.json` | `cost_<suffix>.json` |

```jsonc
{
  "recorded_at": "<UTC ISO time of this run>",
  "run": {
    "location":          "amd | eidf | vastai | ...",
    "framework":         "vllm | sglang",
    "framework_version": "0.16.0",
    "model":             "gpt-oss-120b",
    "dataset":           "mcp-atlas",
    "gpu_key":           "mi355x",
    "num_gpus":          1
  },
  "performance": {
    "avg_e2e_latency_s": <float>,
    "p50_e2e_latency_s": <float>,
    "p99_e2e_latency_s": <float>,
    "ttft_s":            <float>,
    "p99_ttft_s":        <float>,
    "tpot_s":            <float>,
    "p99_tpot_s":        <float>
  },
  "agentic": {
    "avg_num_requests":        <float>,
    "avg_tool_call_count":     <float>,
    "avg_total_output_tokens": <float>,
    "llm_active_s_est":        <float>,   // avg num_req * ttft + out_tok * tpot
    "tool_wait_s_est":         <float>,   // max(0, avg_e2e - llm_active)
    "p99_llm_active_s_est":    <float>,   // avg num_req * p99_ttft + out_tok * p99_tpot
    "p99_tool_wait_s_est":     <float>    // max(0, p99_e2e - p99_llm_active)
  },
  "rent": {
    "price_quote_time":       "<ISO>",
    "price_per_gpu_hour_usd":  <float>,
    "total_hourly_rate_usd":   <float>,
    "price_per_second_usd":    <float>,
    "price_source":            "<URL>",
    "cost": {
      "avg_cost_per_task_usd":              <float>,
      "avg_cost_per_1M_output_tokens_usd":  <float>,
      "p50_cost_per_task_usd":              <float>,
      "p50_cost_per_1M_output_tokens_usd":  <float>,
      "p99_cost_per_task_usd":              <float>,
      "p99_cost_per_1M_output_tokens_usd":  <float>
    }
  },
  "buy": {
    "lifetime_hours":          <float>,
    "electricity_usd_per_kwh": <float>,
    "scale_other_capital":     <float>,
    "gpu": {
      "key": "<gpu_key>", "num": <int>,
      "price_per_unit_usd": <float>, "price_source": "<URL>",
      "tdp_w":              <float>, "tdp_source":   "<URL>",
      "capital_usd":              <float>,
      "amortized_usd_per_hour":   <float>,
      "energy_usd_per_hour":      <float>,
      "effective_hourly_rate_usd": <float>
    },
    "cpu": { /* same shape as gpu, plus "model" */ },
    "default_cost_mode": "active_resource | reserved_worker",
    "accounting_modes": {
      "active_resource": { "description": "...", "avg_gpu_billable_s": <float>, "avg_cpu_billable_s": <float> },
      "reserved_worker": { "description": "...", "avg_gpu_billable_s": <float>, "avg_cpu_billable_s": <float> }
    },
    "cost": {
      // top-level fields mirror --buy-cost-mode for backward-friendly readers
      "gpu_billable_s":                    <float>,
      "cpu_billable_s":                    <float>,
      "gpu_cost_per_task_usd":             <float>,
      "cpu_cost_per_task_usd":             <float>,
      "avg_cost_per_task_usd":             <float>,   // gpu + cpu
      "gpu_cost_per_1M_output_tokens_usd": <float>,
      "cpu_cost_per_1M_output_tokens_usd": <float>,
      "avg_cost_per_1M_output_tokens_usd": <float>,

      "active_resource": { /* same cost fields; GPU=llm_active, CPU=tool_wait */ },
      "reserved_worker": { /* same cost fields; GPU=e2e, CPU=e2e */ }
    }
  }
}
```

A `rent` or `buy` block is **omitted** (not zeroed) when prices/specs are missing for that GPU.

Every pass replaces the prior sidecar using only current evidence. Missing TTFT or TPOT preserves E2E-based rent and reserved-worker buy costs but omits prefill, LLM-active, tool-wait, and active-resource fields; missing output-token count preserves per-task costs but omits per-token costs. The exact old sidecar is removed only when no E2E statistic remains costable. `--dry-run` reports the write or removal without changing files.

---

## 7. Example run

```bash
quote_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 compute_agentic_cost.py \
  --root TEAS_Development_Results_Private/agentic_results \
  --rent-price a100=0.5630  \
  --rent-price h100=2.1622  \
  --rent-price h200=3.6864  \
  --rent-price b200=4.3771  \
  --rent-price b300=5.0059  \
  --rent-price mi355x=2.650 \
  --rent-price-source mi355x=https://www.vultr.com/products/cloud-gpu/amd-mi355x/ \
  --rent-price-quote-time "$quote_time"
```

### Worked example — mcp-atlas / gpt-oss-120b / 1× MI355X

From the metrics file:
```
avg_e2e_latency_s = 44.31      p99_e2e_latency_s = 475.37
ttft = 0.1360                  p99_ttft = 2.450
tpot = 0.004701                p99_tpot = 0.00691
avg_num_requests = 13.02       avg_total_output_tokens = 3464
```

**Time decomposition**
```
llm_active     = 13.02 × 0.136  + 3464 × 0.00470 = 1.77 + 16.29 = 18.06 s
tool_wait      = 44.31 − 18.06  = 26.25 s        (59% of avg wall time)
p99_llm_active = 13.02 × 2.45   + 3464 × 0.00691 = 31.89 + 23.94 = 55.83 s
p99_tool_wait  = 475.37 − 55.83 = 419.54 s       (88% of p99 wall time)
```

**Rent** (Vultr MI355X, $2.65/GPU/h → $/s = 0.000736)
| | $/task | $/1M tok |
|---|---:|---:|
| avg | 0.03262 | 9.42 |
| p50 | 0.01605 | 4.63 |
| p99 | 0.34993 | 101.02 |

**Buy — historical example** (former global 3-year/100% utilisation basis, $0.15/kWh, scale=1.2; current runs use the tier defaults in the options table above, so the rates and results below are not current defaults)
- GPU effective rate = $1.580/h (= $1.370 amort + $0.210 energy)
- CPU effective rate = $0.525/h (= $0.458 amort + $0.068 energy)

Active-resource attribution (default):

| | gpu $/task | cpu $/task | total |
|---|---:|---:|---:|
| avg | 0.00793 | 0.00383 | **0.01176** |
| p99 | 0.02451 | 0.06119 | **0.08570** |

Reserved-worker upper bound:

| | gpu $/task | cpu $/task | total |
|---|---:|---:|---:|
| avg | 0.01945 | 0.00646 | **0.02591** |
| p99 | 0.20862 | 0.06934 | **0.27796** |

---

## 8. Caveats

1. **p99 uses avg counts.** `p99_num_requests` / `p99_output_tokens` are not in the metrics file, so `p99_llm_active_s_est` uses average counts with p99 per-event rates. For long-tail tasks (more turns, more tokens) the real p99 LLM-active is larger and `p99_tool_wait_s_est` is correspondingly over-estimated. Treat p99 tool-wait (and thus p99 CPU cost) as an **upper bound**.
2. **Two buy accounting modes are reported.** `active_resource` is the default attribution mode for multiplexed/continuous-batching serving; `reserved_worker` is a conservative upper bound for single-task workers. Do not mix GPU-full-e2e with CPU-tool-wait as a single metric.
3. **Per-task wall time and concurrency.** With concurrency > 1, active-resource attribution is usually the cleaner per-task view because idle GPU time during tool waits can be used by other requests. The reserved-worker mode intentionally ignores that multiplexing and treats the full worker as occupied.
4. **All the MoE caveats still apply** — tier-specific utilisation and lifetime, `scale_other_capital = 1.2`, electricity price, and GPU/CPU estimates versus realised deals. See [`../moe/compute_cost.md`](../moe/compute_cost.md) §7.
5. **`avg_total_output_tokens` is per-task, not per-LLM-call.** All per-1M-token figures are amortized over the full per-task output budget.
