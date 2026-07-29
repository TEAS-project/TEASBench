#!/usr/bin/env python3
"""
Compute rent + buy cost for TEAS_Development AGENTIC runs.

Per the user spec:
  cost_per_task              = avg_e2e_latency_s * effective_$/s
  cost_per_1M_output_tokens  = (avg_e2e_latency_s / avg_total_output_tokens)
                               * 1_000_000 * effective_$/s

Costs are amortised over the run's achieved concurrency (total task-seconds /
wall-clock), so a run executing tasks in parallel is not billed for the whole
node once per task in flight. `reserved_worker` below is exempt: it is defined
as an exclusive-node upper bound.

For buy-cost accounting, the script reports two explicit modes instead of
mixing conventions:
  active_resource: charge GPU for estimated LLM-active time and CPU for
                   estimated tool-wait/tool-execution time (default).
  reserved_worker: charge both GPU and CPU for full end-to-end wall time as a
                   single-task/exclusive-worker upper bound.

The time breakdown below is not only reported, it drives `active_resource`:
`llm_active_s` becomes the GPU-billable seconds and `tool_wait_s` the CPU-billable
seconds, so a change to `prefill_s` moves the published buy cost.
  prefill_s    = mean per-task total of the per-turn `prefill_time_s` records,
                 falling back to `ttft * avg_num_requests` where those records
                 are absent or cover only part of a task (`prefill_source`
                 says which). See task_prefill_times().
  llm_active_s = prefill_s + tpot * avg_total_output_tokens
  tool_wait_s  = max(0, avg_e2e_latency_s - llm_active_s)

Output: a sibling JSON (`cost.json` / `cost_<suffix>.json`) is written next
to each `metrics.json` / `metrics_<suffix>.json` in the input tree.

For the buy formula, defaults, and units, see ../moe/compute_cost.md
(the GPU and CPU specs here mirror that script).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Optional


VASTAI_PRICING_URL = "https://vast.ai/pricing"
DEFAULT_RENT_PRICE_SOURCE = VASTAI_PRICING_URL

DEFAULT_LIFETIME_HOURS = 3 * 365 * 24
DEFAULT_ELECTRICITY_USD_PER_KWH = 0.15
DEFAULT_SCALE_OTHER_CAPITAL = 1.2

GPU_SPECS: dict[str, dict] = {
    "a100": {
        "price_per_unit_usd": 15000.0,
        "price_source": "https://www.trgdatacenters.com/resource/h100-vs-a100/",
        "tdp_w": 400,
        "tdp_source": "https://lenovopress.lenovo.com/lp1734-thinksystem-nvidia-a100-pcie-40-gpu",
    },
    "h100": {
        "price_per_unit_usd": 27000.0,
        "price_source": "https://www.trgdatacenters.com/resource/nvidia-h100-price/",
        "tdp_w": 700,
        "tdp_source": "https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu",
    },
    "h200": {
        "price_per_unit_usd": 31000.0,
        "price_source": "https://www.trgdatacenters.com/resource/nvidia-h200-price/",
        "tdp_w": 700,
        "tdp_source": "https://lenovopress.lenovo.com/lp1944-nvidia-h200-141gb-gpu",
    },
    "b200": {
        "price_per_unit_usd": 40000.0,
        "price_source": "https://epoch.ai/blog/how-much-does-it-cost-to-train-frontier-ai-models",
        "tdp_w": 1000,
        "tdp_source": "https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf",
    },
    "b300": {
        "price_per_unit_usd": 37500.0,
        "price_source": "https://tech-insider.org/nvidia-blackwell-gpu-pricing/",
        "tdp_w": 1400,
        "tdp_source": "https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-data-sheet",
    },
    "gb10": {
        "price_per_unit_usd": 3999.0,
        "price_source": "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
        "tdp_w": 140,
        "tdp_source": "https://docs.nvidia.com/dgx/dgx-spark/hardware.html",
    },
    "mi355x": {
        "price_per_unit_usd": 30000.0,
        "price_source": "https://www.fitmyllm.com/gpu/radeon-instinct-mi355x",
        "tdp_w": 1400,
        "tdp_source": "https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html",
    },
}

CPU_SPECS: dict[str, dict] = {
    "gb10-soc": {
        "model": "Arm Cortex-X925/A725 integrated in NVIDIA GB10",
        "price_per_unit_usd": 0.0,
        "price_source": "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
        "tdp_w": 0,
        "tdp_source": "https://docs.nvidia.com/dgx/dgx-spark/hardware.html",
    },
    "epyc-7713p": {
        "model": "AMD EPYC 7713P",
        "price_per_unit_usd": 5010.0,
        "price_source": "https://www.amd.com/en/products/processors/server/epyc/7003-series/amd-epyc-7713p.html",
        "tdp_w": 225,
        "tdp_source": "https://www.amd.com/en/products/processors/server/epyc/7003-series/amd-epyc-7713p.html",
    },
    "xeon-8468": {
        "model": "Intel Xeon Platinum 8468",
        "price_per_unit_usd": 7214.0,
        "price_source": "https://www.intel.com/content/www/us/en/products/sku/231735/intel-xeon-platinum-8468-processor-105m-cache-2-10-ghz/specifications.html",
        "tdp_w": 350,
        "tdp_source": "https://www.intel.com/content/www/us/en/products/sku/231735/intel-xeon-platinum-8468-processor-105m-cache-2-10-ghz/specifications.html",
    },
    "xeon-8558": {
        "model": "Intel Xeon Platinum 8558",
        "price_per_unit_usd": 5208.0,
        "price_source": "https://www.intel.com/content/www/us/en/products/sku/237255/intel-xeon-platinum-8558-processor-260m-cache-2-10-ghz/specifications.html",
        "tdp_w": 330,
        "tdp_source": "https://www.intel.com/content/www/us/en/products/sku/237255/intel-xeon-platinum-8558-processor-260m-cache-2-10-ghz/specifications.html",
    },
}

GPU_HOST_CPU: dict[str, tuple[int, str]] = {
    "a100": (2, "xeon-8468"),
    "h100": (2, "xeon-8468"),
    "h200": (2, "xeon-8468"),
    "b200": (2, "xeon-8468"),
    "b300": (2, "xeon-8558"),
    "gb10": (1, "gb10-soc"),
    "mi355x": (2, "epyc-7713p"),
}


GPU_DIR_RE = re.compile(r"^([a-z][a-z0-9]*?)x(\d+)(?:[_-].*)?$")


def parse_gpu_dir(name: str) -> Optional[tuple[str, int]]:
    m = GPU_DIR_RE.match(name.lower())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def find_metrics_files(root: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in ("metrics_*.json", "metrics.json"):
        for p in root.rglob(pat):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  [warn] cannot read {path}: {e}", file=sys.stderr)
        return None


def describe_run(path: Path, root: Path) -> dict[str, str]:
    rel = path.relative_to(root).parts
    gpu_idx = next(
        (i for i, p in enumerate(rel) if parse_gpu_dir(p) is not None), -1
    )
    return {
        "location": rel[gpu_idx - 4] if gpu_idx >= 4 else "",
        "framework": rel[gpu_idx - 3] if gpu_idx >= 3 else "",
        "model": rel[gpu_idx - 2] if gpu_idx >= 2 else "",
        "dataset": rel[gpu_idx - 1] if gpu_idx >= 1 else "",
        "gpu_dir": rel[gpu_idx] if gpu_idx >= 0 else "",
    }


def _swap_prefix(name: str, old: str, new: str) -> str:
    if name == f"{old}.json":
        return f"{new}.json"
    if name.startswith(f"{old}_"):
        return new + name[len(old):]
    return f"{new}_{name}"


def task_latencies(metrics_path: Path) -> list[float]:
    """Per-task end-to-end latencies (s) from the run's `output-data_*.jsonl`, or []."""
    files = sorted(metrics_path.parent.glob("output-data_*.jsonl"))
    if not files:
        return []
    latencies: list[float] = []
    try:
        with files[0].open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line).get("e2e_latency_s")
                if value is not None:
                    latencies.append(float(value))
    except (OSError, ValueError):
        return []
    return latencies


def task_prefill_times(
    metrics_path: Path,
    avg_num_requests: Optional[float],
    avg_total_output_tokens: Optional[float],
) -> list[float]:
    """Per-task total prefill time (s), summed over a task's turns, or [].

    `performance.ttft` does not mean the same thing across the agentic suite. SWE-bench and
    MCP-Atlas publish a per-turn value, so `avg_num_requests * ttft` estimates the task total.
    The IMO runners accumulate prefill across the turn loop and publish that sum as
    `avg_ttft_ms`, so for those runs `ttft` is itself the task total: measured on the tree,
    `ttft` over the per-task prefill total is 1.000 on every IMO run, against 0.03 (SWE) and
    0.11 (MCP).

    Summing the records answers the question without depending on which convention a run
    follows. Runs also differ in layout -- most write one row per turn, some one row per task
    -- and grouping on `example_index` gives the task total under either.

    What matters is that the rows account for the whole task, which is checked against the
    run's independently recorded output-token total: a row set spanning every turn sums to it,
    while one holding only each task's first turn lands near 1/turns. Returns [] when the rows
    cannot be grouped or fall short, and the caller falls back to `avg_num_requests * ttft`.

    Without an output-token column there is nothing to check against, and the fallback counts
    rows per task instead. That test cannot tell a per-task layout from a truncated per-turn
    one, so it rejects both. No run currently reaches it.
    """
    files = sorted(metrics_path.parent.glob("detailed-results_*.jsonl"))
    if not files:
        return []
    prefill: dict = {}
    out_tokens: dict = {}
    rows = 0
    try:
        with files[0].open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                value = record.get("prefill_time_s")
                if value is None:
                    continue
                key = record.get("example_index")
                if key is None:
                    return []  # ungroupable: every row would collapse onto one task
                rows += 1
                prefill[key] = prefill.get(key, 0.0) + float(value)
                tokens = record.get("output_tokens")
                if tokens is not None:
                    out_tokens[key] = out_tokens.get(key, 0.0) + float(tokens)
    except (OSError, ValueError):
        return []
    if len(prefill) < 2:
        return []
    if avg_total_output_tokens and len(out_tokens) == len(prefill):
        covered = sum(out_tokens.values()) / len(out_tokens)
        return list(prefill.values()) if covered >= 0.8 * avg_total_output_tokens else []
    # No output-token column to check against; fall back to counting rows per task.
    if avg_num_requests and rows / len(prefill) < 0.8 * avg_num_requests:
        return []
    return list(prefill.values())


def _percentile(values: list[float], pct: int) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    rank = -(-pct * len(ordered) // 100)
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def cost_path_for(metrics_path: Path) -> Path:
    return metrics_path.with_name(_swap_prefix(metrics_path.name, "metrics", "cost"))


def metadata_path_for(metrics_path: Path) -> Path:
    return metrics_path.with_name(_swap_prefix(metrics_path.name, "metrics", "metadata"))


def load_framework_version(metrics_path: Path) -> Optional[str]:
    meta_path = metadata_path_for(metrics_path)
    if not meta_path.is_file():
        return None
    meta = load_json(meta_path)
    if not meta:
        return None
    env = meta.get("system_environment") or {}
    v = env.get("inference_engine_version")
    if v:
        return v
    hw = meta.get("hardware") or {}
    for k in ("vllm_version", "sglang_version"):
        if k in hw:
            return hw[k]
    return None


def parse_kv_args(items: list[str], cast=str) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip().lower()] = cast(v)
    return out


def discover_gpu_keys(metrics_files: list[Path], root: Path) -> list[str]:
    keys: set[str] = set()
    for f in metrics_files:
        parsed = parse_gpu_dir(describe_run(f, root)["gpu_dir"])
        if parsed:
            keys.add(parsed[0])
    return sorted(keys)


def need_rent_prices_message(needed: list[str], have: dict[str, float]) -> str:
    missing = [k for k in needed if k not in have]
    lines = [
        "",
        "Per-GPU on-demand RENT hourly prices are required.",
        f"Look them up on vast.ai: {VASTAI_PRICING_URL}",
        "",
        "Re-run with prices, e.g.:",
        "    python compute_agentic_cost.py \\",
    ]
    for k in needed:
        if k in have:
            lines.append(f"        --rent-price {k}={have[k]} \\")
        else:
            lines.append(f"        --rent-price {k}=<USD_PER_GPU_PER_HOUR> \\")
    lines[-1] = lines[-1].rstrip(" \\")
    lines.append("")
    lines.append(f"Missing: {', '.join(missing)}")
    return "\n".join(lines)


def build_buy_pricing(
    gpu_key: str, num_gpus: int,
    *, gpu_specs, cpu_specs, gpu_host_cpu,
    lifetime_hours, electricity_usd_per_kwh, scale_other_capital,
    buy_price_quote_time=None,
) -> Optional[dict]:
    gpu = gpu_specs.get(gpu_key)
    host = gpu_host_cpu.get(gpu_key)
    if gpu is None or host is None:
        return None
    num_cpus, cpu_key = host
    cpu = cpu_specs.get(cpu_key)
    if cpu is None:
        return None

    gpu_capital = gpu["price_per_unit_usd"] * num_gpus * scale_other_capital
    cpu_capital = cpu["price_per_unit_usd"] * num_cpus * scale_other_capital
    gpu_power_w = gpu["tdp_w"] * num_gpus
    cpu_power_w = cpu["tdp_w"] * num_cpus

    gpu_amort_per_h = gpu_capital / lifetime_hours
    cpu_amort_per_h = cpu_capital / lifetime_hours
    gpu_energy_per_h = (gpu_power_w / 1000.0) * electricity_usd_per_kwh
    cpu_energy_per_h = (cpu_power_w / 1000.0) * electricity_usd_per_kwh
    gpu_per_h = gpu_amort_per_h + gpu_energy_per_h
    cpu_per_h = cpu_amort_per_h + cpu_energy_per_h
    return {
        "lifetime_hours": lifetime_hours,
        "electricity_usd_per_kwh": electricity_usd_per_kwh,
        "scale_other_capital": scale_other_capital,
        "gpu": {
            "key": gpu_key, "num": num_gpus,
            "price_per_unit_usd": gpu["price_per_unit_usd"],
            "price_source": gpu.get("price_source", "user-supplied"),
            "price_quote_time": buy_price_quote_time,
            "tdp_w": gpu["tdp_w"],
            "tdp_source": gpu.get("tdp_source", "user-supplied"),
            "capital_usd": gpu_capital,
            "amortized_usd_per_hour": gpu_amort_per_h,
            "energy_usd_per_hour": gpu_energy_per_h,
            "effective_hourly_rate_usd": gpu_per_h,
        },
        "cpu": {
            "key": cpu_key, "model": cpu.get("model", cpu_key), "num": num_cpus,
            "price_per_unit_usd": cpu["price_per_unit_usd"],
            "price_source": cpu.get("price_source", "user-supplied"),
            "price_quote_time": buy_price_quote_time,
            "tdp_w": cpu["tdp_w"],
            "tdp_source": cpu.get("tdp_source", "user-supplied"),
            "capital_usd": cpu_capital,
            "amortized_usd_per_hour": cpu_amort_per_h,
            "energy_usd_per_hour": cpu_energy_per_h,
            "effective_hourly_rate_usd": cpu_per_h,
        },
    }


def _per_token_scale(output_tokens: float) -> float:
    if output_tokens and output_tokens > 0:
        return 1_000_000.0 / output_tokens
    return float("nan")


def achieved_concurrency(perf: dict, latencies: list[float]) -> float:
    """Mean tasks in flight = total task-seconds / wall-clock seconds; 1.0 when not derivable.

    A run that executes tasks concurrently finishes N of them in far less than N x avg_e2e, so
    charging each task the whole node for its own wall time bills the node once per task in flight.
    Costs are therefore amortised over this value, which keeps a concurrent run comparable with a
    serial one."""
    wall_min = perf.get("total_wall_time_min")
    if not wall_min or not latencies:
        return 1.0
    c = sum(latencies) / (wall_min * 60.0)
    return c if c >= 1.0 else 1.0


def compute_costs_lumped(
    e2e_s: float,
    output_tokens: float,
    hourly_rate_usd: float,
    *,
    p50_e2e_s: Optional[float] = None,
    p99_e2e_s: Optional[float] = None,
    concurrency: float = 1.0,
) -> dict[str, float]:
    # Amortise the node over the tasks sharing it: in wall time T at concurrency C the run completes
    # C*T/e2e tasks, so cost per token is e2e*price/(C*tokens). Dividing the rate is equivalent.
    price_per_s = hourly_rate_usd / 3600.0 / max(concurrency, 1.0)
    scale = _per_token_scale(output_tokens)
    out: dict[str, float] = {
        "avg_cost_per_task_usd": e2e_s * price_per_s,
        "avg_cost_per_1M_output_tokens_usd": e2e_s * price_per_s * scale,
    }
    if p50_e2e_s is not None:
        out["p50_cost_per_task_usd"] = p50_e2e_s * price_per_s
        out["p50_cost_per_1M_output_tokens_usd"] = p50_e2e_s * price_per_s * scale
    if p99_e2e_s is not None:
        out["p99_cost_per_task_usd"] = p99_e2e_s * price_per_s
        out["p99_cost_per_1M_output_tokens_usd"] = p99_e2e_s * price_per_s * scale
    return out


def _compute_split_mode_costs(
    gpu_billable_s: float,
    cpu_billable_s: float,
    output_tokens: float,
    gpu_hourly_rate_usd: float,
    cpu_hourly_rate_usd: float,
    *,
    p99_gpu_billable_s: Optional[float] = None,
    p99_cpu_billable_s: Optional[float] = None,
) -> dict[str, float]:
    gpu_per_s = gpu_hourly_rate_usd / 3600.0
    cpu_per_s = cpu_hourly_rate_usd / 3600.0
    scale = _per_token_scale(output_tokens)

    gpu_cost_task = gpu_billable_s * gpu_per_s
    cpu_cost_task = cpu_billable_s * cpu_per_s
    out: dict[str, float] = {
        "gpu_billable_s": gpu_billable_s,
        "cpu_billable_s": cpu_billable_s,
        "gpu_cost_per_task_usd": gpu_cost_task,
        "cpu_cost_per_task_usd": cpu_cost_task,
        "avg_cost_per_task_usd": gpu_cost_task + cpu_cost_task,
        "gpu_cost_per_1M_output_tokens_usd": gpu_cost_task * scale,
        "cpu_cost_per_1M_output_tokens_usd": cpu_cost_task * scale,
        "avg_cost_per_1M_output_tokens_usd": (gpu_cost_task + cpu_cost_task) * scale,
    }
    if p99_gpu_billable_s is not None and p99_cpu_billable_s is not None:
        p99_gpu_task = p99_gpu_billable_s * gpu_per_s
        p99_cpu_task = p99_cpu_billable_s * cpu_per_s
        out["p99_gpu_billable_s"] = p99_gpu_billable_s
        out["p99_cpu_billable_s"] = p99_cpu_billable_s
        out["p99_gpu_cost_per_task_usd"] = p99_gpu_task
        out["p99_cpu_cost_per_task_usd"] = p99_cpu_task
        out["p99_cost_per_task_usd"] = p99_gpu_task + p99_cpu_task
        out["p99_gpu_cost_per_1M_output_tokens_usd"] = p99_gpu_task * scale
        out["p99_cpu_cost_per_1M_output_tokens_usd"] = p99_cpu_task * scale
        out["p99_cost_per_1M_output_tokens_usd"] = (p99_gpu_task + p99_cpu_task) * scale
    return out


def compute_costs_split(
    e2e_s: float,
    llm_active_s: float,
    tool_wait_s: float,
    output_tokens: float,
    gpu_hourly_rate_usd: float,
    cpu_hourly_rate_usd: float,
    *,
    default_mode: str = "active_resource",
    p99_e2e_s: Optional[float] = None,
    p99_llm_active_s: Optional[float] = None,
    p99_tool_wait_s: Optional[float] = None,
    concurrency: float = 1.0,
) -> dict[str, object]:
    if default_mode not in {"active_resource", "reserved_worker"}:
        raise ValueError(f"unknown buy cost mode: {default_mode}")

    # active_resource attributes the node across the tasks actually sharing it. reserved_worker is
    # by definition an exclusive-node upper bound, so it is NOT amortised -- that would remove the
    # property that makes it a bound.
    c = max(concurrency, 1.0)
    active = _compute_split_mode_costs(
        llm_active_s, tool_wait_s, output_tokens,
        gpu_hourly_rate_usd / c, cpu_hourly_rate_usd / c,
        p99_gpu_billable_s=p99_llm_active_s,
        p99_cpu_billable_s=p99_tool_wait_s,
    )
    reserved = _compute_split_mode_costs(
        e2e_s, e2e_s, output_tokens,
        gpu_hourly_rate_usd, cpu_hourly_rate_usd,
        p99_gpu_billable_s=p99_e2e_s,
        p99_cpu_billable_s=p99_e2e_s,
    )

    selected = active if default_mode == "active_resource" else reserved
    out: dict[str, object] = {
        "active_resource": active,
        "reserved_worker": reserved,
    }
    # Backward-friendly aliases: top-level cost fields mirror the selected mode.
    out.update(selected)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--rent-price", action="append", default=[],
                        help='Per-GPU rent $/h, e.g. --rent-price b200=4.26 (repeat)')
    parser.add_argument("--rent-price-source", action="append", default=[],
                        help="Per-GPU rent price source URL, e.g. --rent-price-source "
                             "mi355x=https://www.vultr.com/.../amd-mi355x/")
    parser.add_argument("--rent-prices-json", type=Path, default=None)
    parser.add_argument("--rent-price-quote-time", default=None,
                        help="ISO time when rent prices were quoted (default: now, UTC)")

    parser.add_argument("--buy-gpu-price", action="append", default=[])
    parser.add_argument("--buy-gpu-tdp", action="append", default=[])
    parser.add_argument("--buy-cpu-for", action="append", default=[])
    parser.add_argument("--buy-num-cpus", action="append", default=[])
    parser.add_argument("--buy-cpu-price", action="append", default=[])
    parser.add_argument("--buy-cpu-tdp", action="append", default=[])
    parser.add_argument("--buy-lifetime-hours", type=float, default=DEFAULT_LIFETIME_HOURS)
    parser.add_argument("--buy-electricity-usd-per-kwh", type=float, default=DEFAULT_ELECTRICITY_USD_PER_KWH)
    parser.add_argument("--buy-scale-other-capital", type=float, default=DEFAULT_SCALE_OTHER_CAPITAL)
    parser.add_argument("--buy-price-quote-time", default=None,
                        help="ISO time when buy prices were quoted (default: now, UTC)")
    parser.add_argument("--buy-cost-mode", choices=("active-resource", "reserved-worker"), default="active-resource",
                        help="Which buy-cost accounting mode to mirror at buy.cost top level. "
                             "Both modes are always reported: active-resource charges GPU for LLM-active time "
                             "and CPU for tool-wait time; reserved-worker charges both for full e2e latency.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"error: root {root} not found", file=sys.stderr)
        return 2

    metrics_files = find_metrics_files(root)
    if not metrics_files:
        print(f"error: no metrics files under {root}", file=sys.stderr)
        return 2

    gpu_keys = discover_gpu_keys(metrics_files, root)
    print(f"GPU types found: {gpu_keys}")

    rent_prices: dict[str, float] = {}
    if args.rent_prices_json and args.rent_prices_json.is_file():
        rent_prices.update(
            {k.lower(): float(v) for k, v in json.loads(args.rent_prices_json.read_text()).items()}
        )
    rent_prices.update({k: float(v) for k, v in parse_kv_args(args.rent_price).items()})
    rent_price_sources: dict[str, str] = {
        k: str(v) for k, v in parse_kv_args(args.rent_price_source).items()
    }

    if not all(k in rent_prices for k in gpu_keys):
        print(need_rent_prices_message(gpu_keys, rent_prices), file=sys.stderr)
        missing = [k for k in gpu_keys if k not in rent_prices]
        print(f"\nProceeding; rent figures will be omitted for {missing}.\n", file=sys.stderr)

    gpu_specs = {k: dict(v) for k, v in GPU_SPECS.items()}
    for k, v in parse_kv_args(args.buy_gpu_price, float).items():
        spec = gpu_specs.setdefault(k, {}); spec["price_per_unit_usd"] = float(v); spec["price_source"] = "user-supplied"
    for k, v in parse_kv_args(args.buy_gpu_tdp, float).items():
        spec = gpu_specs.setdefault(k, {}); spec["tdp_w"] = float(v); spec["tdp_source"] = "user-supplied"

    cpu_specs = {k: dict(v) for k, v in CPU_SPECS.items()}
    for k, v in parse_kv_args(args.buy_cpu_price, float).items():
        spec = cpu_specs.setdefault(k, {"model": k}); spec["price_per_unit_usd"] = float(v); spec["price_source"] = "user-supplied"
    for k, v in parse_kv_args(args.buy_cpu_tdp, float).items():
        spec = cpu_specs.setdefault(k, {"model": k}); spec["tdp_w"] = float(v); spec["tdp_source"] = "user-supplied"

    gpu_host_cpu = dict(GPU_HOST_CPU)
    for k, v in parse_kv_args(args.buy_cpu_for).items():
        n, _ = gpu_host_cpu.get(k, (2, str(v)))
        gpu_host_cpu[k] = (n, str(v))
    for k, v in parse_kv_args(args.buy_num_cpus, int).items():
        _, cpu_key = gpu_host_cpu.get(k, (int(v), "xeon-8468"))
        gpu_host_cpu[k] = (int(v), cpu_key)

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    recorded_at = now.isoformat().replace("+00:00", "Z")
    rent_quote_time = args.rent_price_quote_time or recorded_at
    buy_price_quote_time = args.buy_price_quote_time or recorded_at

    written = 0
    skipped = 0
    for f in metrics_files:
        info = describe_run(f, root)
        parsed = parse_gpu_dir(info["gpu_dir"])
        if not parsed:
            skipped += 1
            continue
        gpu_key, num_gpus = parsed

        metrics = load_json(f)
        if not metrics:
            skipped += 1
            continue
        perf = metrics.get("performance") or {}
        ag = metrics.get("agentic") or {}

        avg_e2e = perf.get("avg_e2e_latency_s")
        p50_e2e = perf.get("p50_e2e_latency_s")
        p99_e2e = perf.get("p99_e2e_latency_s")
        ttft = perf.get("ttft")
        p99_ttft = perf.get("p99_ttft")
        tpot = perf.get("tpot")
        p99_tpot = perf.get("p99_tpot")
        num_req = ag.get("avg_num_requests")
        out_tok = ag.get("avg_total_output_tokens")
        tool_calls = ag.get("avg_tool_call_count")

        # Runs that omit the end-to-end summary fields still record each task individually.
        latencies = task_latencies(f)
        concurrency = achieved_concurrency(perf, latencies)
        e2e_source = "metrics"
        if avg_e2e is None:
            if latencies:
                e2e_source = "output-data"
                avg_e2e = sum(latencies) / len(latencies)
                if p50_e2e is None:
                    p50_e2e = _percentile(latencies, 50)
                if p99_e2e is None:
                    p99_e2e = _percentile(latencies, 99)

        if avg_e2e is None or ttft is None or tpot is None or out_tok is None:
            skipped += 1
            continue

        # Prefill comes from the per-turn records where they support it, and falls back to the
        # num_req * ttft estimate otherwise. See task_prefill_times() for why ttft cannot be
        # used directly across the suite.
        prefill_times = task_prefill_times(f, num_req, out_tok)
        if prefill_times:
            prefill_s = sum(prefill_times) / len(prefill_times)
            prefill_source = "measured"
        else:
            prefill_s = (num_req or 0) * ttft
            prefill_source = "derived"

        llm_active_s = prefill_s + out_tok * tpot
        tool_wait_s = max(0.0, avg_e2e - llm_active_s)

        p99_llm_active_s = None
        p99_tool_wait_s = None
        if p99_e2e is not None and p99_tpot is not None:
            # The p99 prefill term comes from the same per-task series as the mean, so the two
            # figures share a basis. Without this the average becomes physically possible while
            # the p99 keeps the double-count.
            if prefill_times:
                p99_prefill_s = _percentile(prefill_times, 99)
            elif p99_ttft is not None:
                p99_prefill_s = (num_req or 0) * p99_ttft
            else:
                p99_prefill_s = None
            if p99_prefill_s is not None:
                p99_llm_active_s = p99_prefill_s + out_tok * p99_tpot
                p99_tool_wait_s = max(0.0, p99_e2e - p99_llm_active_s)

        payload = {
            "recorded_at": recorded_at,
            "run": {
                "location": info["location"],
                "framework": info["framework"],
                "framework_version": load_framework_version(f),
                "model": info["model"],
                "dataset": info["dataset"],
                "gpu_key": gpu_key,
                "num_gpus": num_gpus,
            },
            "performance": {
                "avg_e2e_latency_s": avg_e2e,
                "p50_e2e_latency_s": p50_e2e,
                "p99_e2e_latency_s": p99_e2e,
                "e2e_latency_source": e2e_source,
                "achieved_concurrency": round(concurrency, 3),
                "ttft_s": ttft,
                "p99_ttft_s": p99_ttft,
                "tpot_s": tpot,
                "p99_tpot_s": p99_tpot,
            },
            "agentic": {
                "avg_num_requests": num_req,
                "avg_tool_call_count": tool_calls,
                "avg_total_output_tokens": out_tok,
                "prefill_s": prefill_s,
                "prefill_source": prefill_source,
                "llm_active_s": llm_active_s,
                "tool_wait_s": tool_wait_s,
                "p99_llm_active_s": p99_llm_active_s,
                "p99_tool_wait_s": p99_tool_wait_s,
            },
        }

        if gpu_key in rent_prices:
            rent_hourly = rent_prices[gpu_key] * num_gpus
            payload["rent"] = {
                "price_quote_time": rent_quote_time,
                "price_per_gpu_hour_usd": rent_prices[gpu_key],
                "total_hourly_rate_usd": rent_hourly,
                "price_per_second_usd": rent_hourly / 3600.0,
                "price_source": rent_price_sources.get(gpu_key, DEFAULT_RENT_PRICE_SOURCE),
                "cost": compute_costs_lumped(
                    avg_e2e, out_tok, rent_hourly,
                    p50_e2e_s=p50_e2e, p99_e2e_s=p99_e2e,
                    concurrency=concurrency,
                ),
            }

        buy_pricing = build_buy_pricing(
            gpu_key, num_gpus,
            gpu_specs=gpu_specs, cpu_specs=cpu_specs, gpu_host_cpu=gpu_host_cpu,
            lifetime_hours=args.buy_lifetime_hours,
            electricity_usd_per_kwh=args.buy_electricity_usd_per_kwh,
            scale_other_capital=args.buy_scale_other_capital,
            buy_price_quote_time=buy_price_quote_time,
        )
        if buy_pricing is not None:
            buy = dict(buy_pricing)
            buy_cost_mode = args.buy_cost_mode.replace("-", "_")
            buy["default_cost_mode"] = buy_cost_mode
            buy["accounting_modes"] = {
                "active_resource": {
                    "description": "Default per-resource active-time attribution. GPU is charged for estimated LLM-active time; CPU is charged for estimated tool-wait/tool-execution time. Appropriate for continuous batching/multiplexing where the GPU can serve other requests while this task waits on tools.",
                    "avg_gpu_billable_s": llm_active_s,
                    "avg_cpu_billable_s": tool_wait_s,
                },
                "reserved_worker": {
                    "description": "Single-task/exclusive-worker upper bound. Both GPU and CPU are charged for full end-to-end latency because the whole worker is treated as reserved.",
                    "avg_gpu_billable_s": avg_e2e,
                    "avg_cpu_billable_s": avg_e2e,
                },
            }
            buy["cost"] = compute_costs_split(
                avg_e2e, llm_active_s, tool_wait_s, out_tok,
                buy_pricing["gpu"]["effective_hourly_rate_usd"],
                buy_pricing["cpu"]["effective_hourly_rate_usd"],
                default_mode=buy_cost_mode,
                concurrency=concurrency,
                p99_e2e_s=p99_e2e,
                p99_llm_active_s=p99_llm_active_s,
                p99_tool_wait_s=p99_tool_wait_s,
            )
            payload["buy"] = buy

        out_path = cost_path_for(f)
        if args.dry_run:
            print(f"  [dry-run] would write {out_path.relative_to(root.parent)}")
        else:
            out_path.write_text(json.dumps(payload, indent=2) + "\n")
        written += 1

    print(f"\nWrote {written} cost JSON files (skipped {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
