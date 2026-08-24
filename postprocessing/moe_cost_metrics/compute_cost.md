# `compute_cost.py` — Rent + Buy cost computation for TEAS_Development MoE runs

Computes two cost metrics per benchmark run, for both **rent** and **buy**:

- `avg_cost_per_request_usd`
- `avg_cost_per_1M_output_tokens_usd`

Following [MoE-CAP, arXiv 2412.07067 v6](https://arxiv.org/html/2412.07067v6) (Eqs. 1-3) for the buy model; rent uses the live per-GPU hourly price you look up on vast.ai (or any other provider). Built-in purchase prices are curated current/recent-average market estimates: public pricing for datacenter accelerators is inconsistent across vendors and TEASBench releases should prefer an explicit curated estimate over treating any single blog/OEM listing as canonical.

> **User-facing change notice.** Buy-cost runs print their resolved assumptions per accelerator. Published defaults are tier-specific: datacentre hardware uses a 5-year calendar life at 90% average utilisation, while workstation hardware uses 3 years at 40%. An explicit `--buy-lifetime-hours` or `--utilisation` value overrides that dimension for every discovered accelerator, and the output records `base_lifetime_hours`, effective `lifetime_hours`, and `utilisation`.

Required Installation:
```bash
git clone https://github.com/Auto-CAP/MoE-CAP.git
cd MoE-CAP
pip install -e .
```

The script writes one sidecar JSON next to each `metrics_*.json` / `metrics.json` file in the input tree. If a metrics leaf lacks a required cost input and must be skipped, its exact old cost sidecar is removed; `--dry-run` reports the removal without changing files.

---

## 1. Inputs

### Directory layout

The script `rglob`s for `metrics_*.json` and `metrics.json`. Each match is mapped to a run via its path:

```
<root>/<location>/<framework>/<model>/<dataset>/<gpu>x<n>/<batch>/<timestamp>/metrics(...).json
```

- `<gpu>` is matched case-insensitively against `^([a-z][a-z0-9]*?)x(\d+)(?:[_-].*)?$` so `A100x1`, `a100x1_batch-size1`, `mi355xx8` all work.
- `<location>` (`vastai` / `eidf` / `amd` / …) is taken as the path part 4 levels above `<gpu>`.

### Metrics file content

Required fields (subset of MoE-CAP runner output):

```jsonc
{
  "performance": {
    "e2e_s": <float>,    // end-to-end seconds per request
    "ttft":  <float>,    // time-to-first-token, seconds
    "tpot":  <float>     // time-per-output-token, seconds
  }
}
```

If `metadata.json` / `metadata_<…>.json` exists alongside, the script picks up `system_environment.inference_engine_version` and writes it as `run.framework_version`.

---

## 2. Formulas

Let `num_gpus = N`, `num_cpus = M`.

### Rent (`rent` block)

```
total_hourly_rate_$/h = price_per_GPU_per_hour × N
price_per_second_$/s  = total_hourly_rate_$/h / 3600

if batch_token_profile is available:
  prefill_seconds_per_request = ttft / prefill_avg_batch_size
  decode_seconds_per_request  = tpot × decode_generated_tokens_per_request / decode_avg_batch_size
  avg_cost_per_request_usd          = (prefill_seconds_per_request + decode_seconds_per_request) × price_per_second_$/s
  avg_cost_per_1M_output_tokens_usd = (tpot / decode_avg_batch_size) × 1e6 × price_per_second_$/s
else fallback:
  avg_cost_per_request_usd          = e2e_s      × price_per_second_$/s
  avg_cost_per_1M_output_tokens_usd = tpot × 1e6 × price_per_second_$/s
```

### Buy (`buy` block, MoE-CAP Eqs. 1-3)

```
capital_$        = (gpu_$ × N + cpu_$ × M) × scale_other_capital
power_W          =  gpu_W × N + cpu_W × M

effective_lifetime_hours = lifetime_hours × utilisation
amortized_$/h           = capital_$ / effective_lifetime_hours
energy_$/h              = (power_W / 1000) × electricity_$_per_kWh
effective_$/h    = amortized_$/h + energy_$/h
effective_$/s    = effective_$/h / 3600

if batch_token_profile is available:
  prefill_seconds_per_request = ttft / prefill_avg_batch_size
  decode_seconds_per_request  = tpot × decode_generated_tokens_per_request / decode_avg_batch_size
  avg_cost_per_request_usd          = (prefill_seconds_per_request + decode_seconds_per_request) × effective_$/s
  avg_cost_per_1M_output_tokens_usd = (tpot / decode_avg_batch_size) × 1e6 × effective_$/s
else fallback:
  avg_cost_per_request_usd          = e2e_s      × effective_$/s
  avg_cost_per_1M_output_tokens_usd = tpot × 1e6 × effective_$/s
```

For continuous-batched MoE runs, `tpot` is per decode step, not exclusive per-output-token GPU time. The default path therefore divides by `batch_token_profile.decode_avg_batch_size`. This prevents default-batch runs on slower/older GPUs from being overcharged by ignoring simultaneous token generation across active requests.

The cost block also emits an audit-friendly decomposition:

```text
effective_output_tokens_per_s = decode_avg_batch_size / tpot
cost_per_1M_output_tokens_usd = price_per_second_usd / effective_output_tokens_per_s × 1e6
```

This is the key sanity check for hardware comparisons. A newer accelerator can have a higher hourly price but lower cost/token when its measured effective decode throughput improves more than its price premium. Therefore reports should present `price_per_gpu_hour_usd`, `num_gpus`, `total_hourly_rate_usd`, `tpot`, `decode_avg_batch_size`, `effective_output_tokens_per_s`, and `avg_cost_per_1M_output_tokens_usd` together rather than showing only the final cost.

`scale_other_capital` (default **1.2**, from MoE-CAP) inflates the GPU+CPU bill-of-materials to approximate motherboard + DRAM + SSD overhead.

`base_lifetime_hours` defaults by hardware tier: **43,800** (5 years) for datacentre hardware and **26,280** (3 years) for workstation hardware.

`utilisation` also defaults by tier: **0.9** for datacentre hardware and **0.4** for workstation hardware. The script uses `effective_lifetime_hours = base_lifetime_hours × utilisation`; an explicit CLI value applies across tiers.

`electricity_$_per_kWh` default **0.15** (MoE-CAP default).

---

## 3. CLI

```
python3 compute_cost.py --root <dir> [rent flags] [buy flags] [--dry-run]
```

### Common

| Flag | Default | Meaning |
|---|---|---|
| `--root` | script dir | Root tree to scan for metrics files |
| `--dry-run` | off | Print would-be output paths; do not write |

### Rent

| Flag | Format | Notes |
|---|---|---|
| `--rent-price` | `gpu=usd` | Per-GPU $/h, repeatable: `--rent-price b200=4.26` |
| `--rent-prices-json` | path | JSON `{"b200": 4.26, "h200": 3.75}` |
| `--rent-price-source` | `gpu=url` | Override the URL written into `rent.price_source`. Default: `https://vast.ai/pricing` |
| `--rent-price-quote-time` | ISO time | When you sampled the price. Default: now (UTC) |

Missing prices for some GPU types → that GPU's `rent` block is omitted; `buy` still emitted.

### Buy (per-GPU + per-CPU overrides)

| Flag | Format | Notes |
|---|---|---|
| `--buy-gpu-price` | `gpu=usd` | Override default GPU purchase price |
| `--buy-gpu-prices-json` | path | JSON `{"b200": 35000}` or `{"b200": {"price_per_unit_usd": 35000, "price_source": "..."}}` |
| `--buy-gpu-tdp` | `gpu=W` | Override default GPU TDP |
| `--buy-cpu-for` | `gpu=cpu_key` | Change which CPU is paired with a GPU |
| `--buy-num-cpus` | `gpu=N` | Change CPU count per platform (default 2) |
| `--buy-cpu-price` | `cpu_key=usd` | Override a CPU's purchase price |
| `--buy-cpu-prices-json` | path | JSON `{"xeon-8468": 7214}` or `{"xeon-8468": {"price_per_unit_usd": 7214, "price_source": "..."}}` |
| `--buy-cpu-tdp` | `cpu_key=W` | Override a CPU's TDP |
| `--buy-lifetime-hours` | float | Override calendar lifetime hours across tiers; default 43800 datacentre / 26280 workstation |
| `--utilisation` / `--utilization` | float | Override average utilisation across tiers in `(0, 1]`; default 0.9 datacentre / 0.4 workstation |
| `--buy-electricity-usd-per-kwh` | float | Default 0.15 |
| `--buy-scale-other-capital` | float | Default 1.2 (MoE-CAP) |

---

## 4. Built-in defaults

### GPUs

Purchase prices below are curated current/recent-average market estimates for TEASBench, not a claim that one source is authoritative for every release. The linked price source is an audit trail / anchor for the estimate; override with `--buy-gpu-prices-json` for a release-specific curated table.

| key | price (USD/unit) | TDP (W) | Default host CPU × qty | Price source | TDP source |
|---|---:|---:|---|---|---|
| `a100`   | 18,000 | 400  | Intel Xeon Platinum 8468 × 2 | [Modal A100 article](https://modal.com/blog/nvidia-a100-price-article) | [Lenovo lp1734](https://lenovopress.lenovo.com/lp1734-thinksystem-nvidia-a100-pcie-40-gpu) |
| `h100`   | 25,000 | 700  | Intel Xeon Platinum 8468 × 2 | [Modal H100 article](https://modal.com/blog/nvidia-h100-price-article) | [Lenovo lp1732](https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu) |
| `h200`   | 30,000 | 700  | Intel Xeon Platinum 8468 × 2 | [Modal H200 article](https://modal.com/blog/nvidia-h200-price-article) | [Lenovo lp1944](https://lenovopress.lenovo.com/lp1944-nvidia-h200-141gb-gpu) |
| `b200`   | 35,000 | 1000 | Intel Xeon Platinum 8468 × 2 | [Modal B200 blog](https://modal.com/blog/nvidia-b200-pricing) | [NVIDIA HGX B200 PCF](https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf) |
| `b300`   | 42,500 | 1400 | Intel Xeon Platinum 8558 × 2 | [tech-insider Blackwell](https://tech-insider.org/nvidia-blackwell-gpu-pricing/) | [NVIDIA Blackwell Ultra datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-data-sheet) |
| `mi355x` | 30,000 | 1400 | AMD EPYC 7713P × 2 | [FitMyLLM MI355X](https://www.fitmyllm.com/gpu/radeon-instinct-mi355x) | [AMD MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) |

### CPUs

CPU prices are likewise curated estimates with linked public references; override with `--buy-cpu-prices-json` when a release uses a different price table.

| key | model | price (USD) | TDP (W) | Source |
|---|---|---:|---:|---|
| `xeon-8468`   | Intel Xeon Platinum 8468 | 7,214 | 350 | [Intel ARK 231735](https://www.intel.com/content/www/us/en/products/sku/231735/intel-xeon-platinum-8468-processor-105m-cache-2-10-ghz/specifications.html) |
| `xeon-8558`   | Intel Xeon Platinum 8558 | 5,208 | 330 | [Intel ARK 237255](https://www.intel.com/content/www/us/en/products/sku/237255/intel-xeon-platinum-8558-processor-260m-cache-2-10-ghz/specifications.html) |
| `epyc-7713p`  | AMD EPYC 7713P            | 5,010 | 225 | [AMD EPYC 7713P](https://www.amd.com/en/products/processors/server/epyc/7003-series/amd-epyc-7713p.html) |

Every URL above returns HTTP 200; the price/TDP value appears in the page's static HTML (Intel ARK pages anti-bot to curl, verified via web search instead). B300 TDP source is NVIDIA's official Blackwell Ultra datasheet landing page; the 1,400W value is in the (gated) PDF and also confirmed by Lenovo / Supermicro GB300 docs.

---

## 5. Output schema

One file per metrics file, written to the same directory. Naming:

| Input | Output |
|---|---|
| `metrics.json` | `cost.json` |
| `metrics_<suffix>.json` | `cost_<suffix>.json` |

```jsonc
{
  "recorded_at": "2026-06-07T19:33:00Z",
  "run": {
    "location": "amd",
    "framework": "sglang",
    "framework_version": "0.5.9",
    "model": "gpt-oss-120b",
    "dataset": "gsm8k_256samples",
    "gpu_key": "mi355x",
    "num_gpus": 1
  },
  "performance": {
    "e2e_s": 1.84,
    "ttft_s": 0.024846,
    "tpot_s": 0.005945
  },
  "rent": {
    "price_quote_time": "2026-06-07T19:33:00Z",
    "price_per_gpu_hour_usd": 2.65,
    "total_hourly_rate_usd": 2.65,
    "price_per_second_usd": 0.0007361,
    "price_source": "https://www.vultr.com/products/cloud-gpu/amd-mi355x/",
    "cost": {
      "avg_cost_per_request_usd": 0.00003731,
      "avg_cost_per_1M_output_tokens_usd": 0.0684,
      "method": "batch_token_profile",
      "prefill_avg_batch_size": 8.0,
      "decode_avg_batch_size": 64.0,
      "decode_generated_tokens_per_request": 512.0,
      "effective_output_tokens_per_s": 10765.3,
      "formula": "price_per_second_usd * tpot_s / decode_avg_batch_size * 1e6",
      "breakdown": {
        "pricing": {
          "price_per_hour_usd": 2.65,
          "price_per_second_usd": 0.0007361
        },
        "latency": {
          "e2e_s": 1.84,
          "ttft_s": 0.024846,
          "tpot_s": 0.005945
        },
        "batch_profile": {
          "prefill_avg_batch_size": 8.0,
          "decode_avg_batch_size": 64.0,
          "prefill_tokens_per_request": 1024.0,
          "decode_generated_tokens_per_request": 512.0
        },
        "throughput": {
          "effective_output_tokens_per_s": 10765.3,
          "prefill_tokens_per_s": 329831.8,
          "formula": "decode_avg_batch_size / tpot_s"
        },
        "request_seconds": {
          "prefill_seconds_per_request": 0.00310575,
          "decode_seconds_per_request": 0.04756,
          "total_seconds_per_request": 0.05067
        },
        "request_cost_usd": {
          "prefill_cost_per_request_usd": 0.00000229,
          "decode_cost_per_request_usd": 0.00003502,
          "total_cost_per_request_usd": 0.00003731
        },
        "output_token_cost": {
          "seconds_per_output_token": 0.00009289,
          "cost_per_output_token_usd": 0.0000000684,
          "cost_per_1M_output_tokens_usd": 0.0684,
          "formula": "price_per_second_usd / effective_output_tokens_per_s * 1e6"
        }
      }
    }
  },
  "buy": {
    "lifetime_hours": 15768,
    "base_lifetime_hours": 26280,
    "utilisation": 0.6,
    "electricity_usd_per_kwh": 0.15,
    "scale_other_capital": 1.2,
    "gpu":  { "key": "mi355x", "num": 1, "price_per_unit_usd": 30000,
              "price_source": "...", "tdp_w": 1400, "tdp_source": "..." },
    "cpu":  { "key": "epyc-7713p", "model": "AMD EPYC 7713P", "num": 2,
              "price_per_unit_usd": 5010, "price_source": "...",
              "tdp_w": 225, "tdp_source": "..." },
    "total_capital_usd": 48024.0,
    "total_power_w": 1850,
    "amortized_capital_usd_per_hour": 1.8274,
    "energy_usd_per_hour": 0.2775,
    "effective_hourly_rate_usd": 2.1049,
    "cost": {
      "avg_cost_per_request_usd": 0.001076,
      "avg_cost_per_1M_output_tokens_usd": 3.476
    }
  }
}
```

If a price is missing for a GPU, the corresponding block (`rent` or `buy`) is simply omitted from the JSON; the script does not abort.

---

## 6. Examples

### Rent + buy, all default specs

```bash
python3 compute_cost.py \
  --root TEAS_Development_Results_Private/moe \
  --rent-price a100=0.5630  \
  --rent-price h100=2.1622  \
  --rent-price h200=3.6864  \
  --rent-price b200=4.3771  \
  --rent-price b300=5.0059  \
  --rent-price mi355x=2.650 \
  --rent-price-source mi355x=https://www.vultr.com/products/cloud-gpu/amd-mi355x/ \
  --rent-price-quote-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

### Use a more realistic buy TCO and utilisation

```bash
python3 compute_cost.py \
  --root TEAS_Development_Results_Private/moe \
  --buy-scale-other-capital 2.5 \
  --buy-lifetime-hours 26280 \
  --utilisation 0.60 \
  --buy-electricity-usd-per-kwh 0.10
```

### Use curated release-specific buy price tables

```bash
cat > gpu_prices.json <<'JSON'
{
  "a100": {"price_per_unit_usd": 18000, "price_source": "TEASBench curated estimate, 2026-Q2"},
  "h100": {"price_per_unit_usd": 25000, "price_source": "TEASBench curated estimate, 2026-Q2"},
  "b200": {"price_per_unit_usd": 35000, "price_source": "TEASBench curated estimate, 2026-Q2"}
}
JSON

cat > cpu_prices.json <<'JSON'
{
  "xeon-8468": {"price_per_unit_usd": 7214, "price_source": "Intel ARK / curated estimate"},
  "xeon-8558": {"price_per_unit_usd": 5208, "price_source": "Intel ARK / curated estimate"}
}
JSON

python3 compute_cost.py \
  --root TEAS_Development_Results_Private/moe \
  --buy-gpu-prices-json gpu_prices.json \
  --buy-cpu-prices-json cpu_prices.json
```

### Override a single GPU spec on the fly

```bash
python3 compute_cost.py \
  --buy-gpu-price b200=40000 \
  --buy-gpu-tdp   b200=1200  \
  --buy-cpu-for   b200=xeon-8558 \
  --buy-num-cpus  b200=2
```

### Dry run

```bash
python3 compute_cost.py ... --dry-run | head
```

---

## 7. Caveats — read before quoting these numbers

The buy figures are a **theoretical lower bound** under the MoE-CAP simplification. They will look optimistic vs. cloud rent if you don't adjust:

1. **Utilisation.** Published defaults assume 90% average utilisation for datacentre hardware and 40% for workstation hardware. Set `--utilisation` explicitly to model a different duty cycle across every tier; capital $/h scales inversely with the selected value.
2. **Capital scaling = 1.2** covers only motherboard / DRAM / SSD. Real DC TCO adds chassis, NICs, switches, PDUs, cooling (PUE ~1.5-2.0), rack space, ops staff, financing. Industry rule-of-thumb: server TCO ≈ 2-3× BoM. Override with `--buy-scale-other-capital 2.5` (or higher).
3. **Electricity $0.15/kWh** is mid-tier. Industrial rates can be $0.05-0.10/kWh, but multiply by PUE.
4. **Lifetime depends on deployment class.** Published datacentre figures use 5 years, while workstation figures use 3 years; `--buy-lifetime-hours` explicitly overrides both.
5. **GPU/CPU prices** are curated current/recent-average market estimates with linked public anchors. Hyperscaler / OEM volume deals can be substantially lower; "list" pricing for new datacenter GPUs is also rarely public (NVIDIA does not publish DC GPU prices on nvidia.com). For each TEASBench release, prefer checking in a release-specific JSON table via `--buy-gpu-prices-json` / `--buy-cpu-prices-json` when the built-ins are stale.
6. **Single-card prices for GB200/GB300/HGX boards are nominal** — those parts are almost always sold as 4-/8-GPU boards or full systems. The per-GPU number is derived from system price ÷ GPU count where indicated.

For an apples-to-apples comparison with the rent quote, match your assumed utilisation to the provider's effective utilisation, or compare against **reserved** cloud rates (1-3 yr) rather than on-demand.
