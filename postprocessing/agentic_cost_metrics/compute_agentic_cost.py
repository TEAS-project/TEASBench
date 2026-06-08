#!/usr/bin/env python3
"""
Compute rent + buy cost for TEAS_Development AGENTIC runs.

Per the user spec:
  cost_per_task              = avg_e2e_latency_s * effective_$/s
  cost_per_1M_output_tokens  = (avg_e2e_latency_s / avg_total_output_tokens)
                               * 1_000_000 * effective_$/s

The per-token figure naturally amortizes GPU-idle-during-tool-call time onto
each generated output token (since avg_e2e_latency_s spans both the LLM
generation and the tool-call wall time the GPU was reserved through).

For transparency the cost JSON also reports an estimated time breakdown:
  llm_active_s_est = ttft * avg_num_requests + tpot * avg_total_output_tokens
  tool_wait_s_est  = max(0, avg_e2e_latency_s - llm_active_s_est)

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
        "price_per_unit_usd": 18000.0,
        "price_source": "https://modal.com/blog/nvidia-a100-price-article",
        "tdp_w": 400,
        "tdp_source": "https://lenovopress.lenovo.com/lp1734-thinksystem-nvidia-a100-pcie-40-gpu",
    },
    "h100": {
        "price_per_unit_usd": 25000.0,
        "price_source": "https://modal.com/blog/nvidia-h100-price-article",
        "tdp_w": 700,
        "tdp_source": "https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu",
    },
    "h200": {
        "price_per_unit_usd": 30000.0,
        "price_source": "https://modal.com/blog/nvidia-h200-price-article",
        "tdp_w": 700,
        "tdp_source": "https://lenovopress.lenovo.com/lp1944-nvidia-h200-141gb-gpu",
    },
    "b200": {
        "price_per_unit_usd": 35000.0,
        "price_source": "https://modal.com/blog/nvidia-b200-pricing",
        "tdp_w": 1000,
        "tdp_source": "https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf",
    },
    "b300": {
        "price_per_unit_usd": 42500.0,
        "price_source": "https://tech-insider.org/nvidia-blackwell-gpu-pricing/",
        "tdp_w": 1400,
        "tdp_source": "https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-data-sheet",
    },
    "mi355x": {
        "price_per_unit_usd": 30000.0,
        "price_source": "https://www.fitmyllm.com/gpu/radeon-instinct-mi355x",
        "tdp_w": 1400,
        "tdp_source": "https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html",
    },
}

CPU_SPECS: dict[str, dict] = {
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


def compute_costs_lumped(
    e2e_s: float,
    output_tokens: float,
    hourly_rate_usd: float,
    *,
    p50_e2e_s: Optional[float] = None,
    p99_e2e_s: Optional[float] = None,
) -> dict[str, float]:
    price_per_s = hourly_rate_usd / 3600.0
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


def compute_costs_split(
    e2e_s: float,
    tool_wait_s: float,
    output_tokens: float,
    gpu_hourly_rate_usd: float,
    cpu_hourly_rate_usd: float,
    *,
    p99_e2e_s: Optional[float] = None,
    p99_tool_wait_s: Optional[float] = None,
) -> dict[str, float]:
    gpu_per_s = gpu_hourly_rate_usd / 3600.0
    cpu_per_s = cpu_hourly_rate_usd / 3600.0
    scale = _per_token_scale(output_tokens)

    gpu_cost_task = e2e_s * gpu_per_s
    cpu_cost_task = tool_wait_s * cpu_per_s
    out: dict[str, float] = {
        "gpu_cost_per_task_usd": gpu_cost_task,
        "cpu_cost_per_task_usd": cpu_cost_task,
        "avg_cost_per_task_usd": gpu_cost_task + cpu_cost_task,
        "gpu_cost_per_1M_output_tokens_usd": gpu_cost_task * scale,
        "cpu_cost_per_1M_output_tokens_usd": cpu_cost_task * scale,
        "avg_cost_per_1M_output_tokens_usd": (gpu_cost_task + cpu_cost_task) * scale,
    }
    if p99_e2e_s is not None and p99_tool_wait_s is not None:
        p99_gpu_task = p99_e2e_s * gpu_per_s
        p99_cpu_task = p99_tool_wait_s * cpu_per_s
        out["p99_gpu_cost_per_task_usd"] = p99_gpu_task
        out["p99_cpu_cost_per_task_usd"] = p99_cpu_task
        out["p99_cost_per_task_usd"] = p99_gpu_task + p99_cpu_task
        out["p99_gpu_cost_per_1M_output_tokens_usd"] = p99_gpu_task * scale
        out["p99_cpu_cost_per_1M_output_tokens_usd"] = p99_cpu_task * scale
        out["p99_cost_per_1M_output_tokens_usd"] = (p99_gpu_task + p99_cpu_task) * scale
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

        if avg_e2e is None or ttft is None or tpot is None or out_tok is None:
            skipped += 1
            continue

        llm_active_s = (num_req or 0) * ttft + out_tok * tpot
        tool_wait_s = max(0.0, avg_e2e - llm_active_s)

        p99_llm_active_s = None
        p99_tool_wait_s = None
        if p99_e2e is not None and p99_ttft is not None and p99_tpot is not None:
            p99_llm_active_s = (num_req or 0) * p99_ttft + out_tok * p99_tpot
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
                "ttft_s": ttft,
                "p99_ttft_s": p99_ttft,
                "tpot_s": tpot,
                "p99_tpot_s": p99_tpot,
            },
            "agentic": {
                "avg_num_requests": num_req,
                "avg_tool_call_count": tool_calls,
                "avg_total_output_tokens": out_tok,
                "llm_active_s_est": llm_active_s,
                "tool_wait_s_est": tool_wait_s,
                "p99_llm_active_s_est": p99_llm_active_s,
                "p99_tool_wait_s_est": p99_tool_wait_s,
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
                ),
            }

        buy_pricing = build_buy_pricing(
            gpu_key, num_gpus,
            gpu_specs=gpu_specs, cpu_specs=cpu_specs, gpu_host_cpu=gpu_host_cpu,
            lifetime_hours=args.buy_lifetime_hours,
            electricity_usd_per_kwh=args.buy_electricity_usd_per_kwh,
            scale_other_capital=args.buy_scale_other_capital,
        )
        if buy_pricing is not None:
            buy = dict(buy_pricing)
            buy["cost"] = compute_costs_split(
                avg_e2e, tool_wait_s, out_tok,
                buy_pricing["gpu"]["effective_hourly_rate_usd"],
                buy_pricing["cpu"]["effective_hourly_rate_usd"],
                p99_e2e_s=p99_e2e, p99_tool_wait_s=p99_tool_wait_s,
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
