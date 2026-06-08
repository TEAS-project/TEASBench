# `compute_cost.py` — Rent + Buy cost computation for TEAS_Development MoE runs

Computes two cost metrics per benchmark run, for both **rent** and **buy**:

- `avg_cost_per_request_usd`
- `avg_cost_per_1M_output_tokens_usd`

Following [MoE-CAP, arXiv 2412.07067 v6](https://arxiv.org/html/2412.07067v6) (Eqs. 1-3) for the buy model; rent uses the live per-GPU hourly price you look up on vast.ai (or any other provider).

Required Installation:
```bash
git clone https://github.com/Auto-CAP/MoE-CAP.git
cd MoE-CAP
pip install -e .
```

The script writes one sidecar JSON next to each `metrics_*.json` / `metrics.json` file in the input tree.

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

avg_cost_per_request_usd          = e2e_s          × price_per_second_$/s
avg_cost_per_1M_output_tokens_usd = tpot × 1e6     × price_per_second_$/s
```

### Buy (`buy` block, MoE-CAP Eqs. 1-3)

```
capital_$        = (gpu_$ × N + cpu_$ × M) × scale_other_capital
power_W          =  gpu_W × N + cpu_W × M

amortized_$/h    = capital_$ / lifetime_hours
energy_$/h       = (power_W / 1000) × electricity_$_per_kWh
effective_$/h    = amortized_$/h + energy_$/h
effective_$/s    = effective_$/h / 3600

avg_cost_per_request_usd          = e2e_s          × effective_$/s
avg_cost_per_1M_output_tokens_usd = tpot × 1e6     × effective_$/s
```

`scale_other_capital` (default **1.2**, from MoE-CAP) inflates the GPU+CPU bill-of-materials to approximate motherboard + DRAM + SSD overhead.

`lifetime_hours` default **26,280** = 3 yr × 365 × 24.

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
| `--buy-gpu-tdp` | `gpu=W` | Override default GPU TDP |
| `--buy-cpu-for` | `gpu=cpu_key` | Change which CPU is paired with a GPU |
| `--buy-num-cpus` | `gpu=N` | Change CPU count per platform (default 2) |
| `--buy-cpu-price` | `cpu_key=usd` | Override a CPU's purchase price |
| `--buy-cpu-tdp` | `cpu_key=W` | Override a CPU's TDP |
| `--buy-lifetime-hours` | float | Default 26280 (3 yr) |
| `--buy-electricity-usd-per-kwh` | float | Default 0.15 |
| `--buy-scale-other-capital` | float | Default 1.2 (MoE-CAP) |

---

## 4. Built-in defaults

### GPUs

| key | price (USD/unit) | TDP (W) | Default host CPU × qty | Price source | TDP source |
|---|---:|---:|---|---|---|
| `a100`   | 18,000 | 400  | Intel Xeon Platinum 8468 × 2 | [Modal A100 article](https://modal.com/blog/nvidia-a100-price-article) | [Lenovo lp1734](https://lenovopress.lenovo.com/lp1734-thinksystem-nvidia-a100-pcie-40-gpu) |
| `h100`   | 25,000 | 700  | Intel Xeon Platinum 8468 × 2 | [Modal H100 article](https://modal.com/blog/nvidia-h100-price-article) | [Lenovo lp1732](https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu) |
| `h200`   | 30,000 | 700  | Intel Xeon Platinum 8468 × 2 | [Modal H200 article](https://modal.com/blog/nvidia-h200-price-article) | [Lenovo lp1944](https://lenovopress.lenovo.com/lp1944-nvidia-h200-141gb-gpu) |
| `b200`   | 35,000 | 1000 | Intel Xeon Platinum 8468 × 2 | [Modal B200 blog](https://modal.com/blog/nvidia-b200-pricing) | [NVIDIA HGX B200 PCF](https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf) |
| `b300`   | 42,500 | 1400 | Intel Xeon Platinum 8558 × 2 | [tech-insider Blackwell](https://tech-insider.org/nvidia-blackwell-gpu-pricing/) | [NVIDIA Blackwell Ultra datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-data-sheet) |
| `mi355x` | 30,000 | 1400 | AMD EPYC 7713P × 2 | [FitMyLLM MI355X](https://www.fitmyllm.com/gpu/radeon-instinct-mi355x) | [AMD MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) |

### CPUs

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
      "avg_cost_per_request_usd": 0.001354,
      "avg_cost_per_1M_output_tokens_usd": 4.376
    }
  },
  "buy": {
    "lifetime_hours": 26280,
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

### Use a more realistic buy TCO

```bash
python3 compute_cost.py \
  --root TEAS_Development_Results_Private/moe \
  --buy-scale-other-capital 2.5 \
  --buy-lifetime-hours 15768  \   # 3 yr × 60% utilization
  --buy-electricity-usd-per-kwh 0.10
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

1. **Utilization.** Defaults assume 24×7 use over `lifetime_hours`. Real fleets see 40-70%; at 50% your effective $/h roughly doubles on the capital side.
2. **Capital scaling = 1.2** covers only motherboard / DRAM / SSD. Real DC TCO adds chassis, NICs, switches, PDUs, cooling (PUE ~1.5-2.0), rack space, ops staff, financing. Industry rule-of-thumb: server TCO ≈ 2-3× BoM. Override with `--buy-scale-other-capital 2.5` (or higher).
3. **Electricity $0.15/kWh** is mid-tier. Industrial rates can be $0.05-0.10/kWh, but multiply by PUE.
4. **3-year lifetime** is conservative; hyperscalers depreciate 5-6 yr now.
5. **GPU/CPU prices** are public-market mids. Hyperscaler / OEM volume deals can be substantially lower; "list" pricing for new datacenter GPUs is also rarely public (NVIDIA does not publish DC GPU prices on nvidia.com, hence the third-party `price_source` URLs).
6. **Single-card prices for GB200/GB300/HGX boards are nominal** — those parts are almost always sold as 4-/8-GPU boards or full systems. The per-GPU number is derived from system price ÷ GPU count where indicated.

For an apples-to-apples comparison with the rent quote, match your assumed utilization to the provider's effective utilization, or compare against **reserved** cloud rates (1-3 yr) rather than on-demand.
