#!/usr/bin/env python3
"""
Compute rent + buy cost for MoE-CAP TEAS_Development MoE results.

Two top-level metrics (identical schema for rent and buy):
  - avg_cost_per_request_usd
  - avg_cost_per_1M_output_tokens_usd

Each rent/buy block also carries a `wall` sub-block pricing the whole-run wall clock:
  - node_seconds_per_request = e2e_s (run makespan / N requests)
  - cost_per_request_usd     = price_per_second * node_seconds_per_request
This basis needs no batch profile, so it is present even on concurrent runs whose
profile cannot support the per-token metrics (those runs get a wall-only block).

When batch_token_profile is available, continuous-batching cost is amortized as:
  - prefill_seconds_per_request = TTFT / prefill_avg_batch_size
  - decode_seconds_per_request  = TPOT * decode_tokens_per_request / decode_avg_batch_size
  - effective_output_tokens_per_s = decode_avg_batch_size / TPOT
  - avg_cost_per_1M_output_tokens_usd = effective_$/s / effective_output_tokens_per_s * 1e6

Each cost block also includes a breakdown with pricing, latency, batch profile,
throughput, request-seconds, request-cost, and output-token-cost fields so the
reported cost can be audited and explained.

RENT (vast.ai etc.):
  effective_$/s = (price_per_GPU_per_hour * num_gpus) / 3600
  Prices come from the USER (look them up at https://vast.ai/pricing).

BUY (per https://arxiv.org/html/2412.07067v6, Eqs. 1-3):
  Amortize hardware capital over lifetime, add energy:
    capital_$  = (gpu_$ * num_gpus + cpu_$ * num_cpus) * capital_scale
    power_W    = gpu_W * num_gpus + cpu_W * num_cpus
    amort_$/h  = capital_$ / lifetime_hours
    energy_$/h = (power_W / 1000) * electricity_$_per_kWh
    effective_$/h = amort_$/h + energy_$/h
    effective_$/s = effective_$/h / 3600

  capital_scale normally equals --buy-scale-other-capital; a catalogued complete
  system can set a per-entry scale of 1 to avoid adding its included host twice.
  Defaults use curated current/recent-average market estimates plus vendor
  datasheets for power. Override via --buy-gpu-price <gpu>=<usd>,
  --buy-gpu-prices-json <path>, --buy-gpu-tdp <gpu>=<W>, etc.

Output: a sibling JSON file (cost.json / cost_<name>.json) is written
next to each metrics file (metrics.json / metrics_<name>.json).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Optional

# Hardware specs live in hardware_catalog.py, not here: they are published by this script and
# by the dashboard assembler in the results repo, and a second copy is how a corrected figure
# gets silently reverted in one publisher and not the other. Imported under both invocations —
# as a package module (tests) and as a standalone script (the sync workflow).
if __package__:
    from .hardware_catalog import (  # noqa: F401  (re-exported for existing importers)
        CPU_SPECS,
        GPU_HOST_CPU,
        GPU_SPECS,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hardware_catalog import (  # noqa: F401  (re-exported for existing importers)
        CPU_SPECS,
        GPU_HOST_CPU,
        GPU_SPECS,
    )


VASTAI_PRICING_URL = "https://vast.ai/pricing"
DEFAULT_RENT_PRICE_SOURCE = VASTAI_PRICING_URL

DEFAULT_LIFETIME_HOURS = 3 * 365 * 24
DEFAULT_ELECTRICITY_USD_PER_KWH = 0.15
DEFAULT_SCALE_OTHER_CAPITAL = 1.2

GPU_DIR_RE = re.compile(r"^([a-z][a-z0-9-]*?)x(\d+)(?:[_-].*)?$")  # hyphen allows blackhole-p150b


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


def parse_price_json(path: Optional[Path]) -> dict[str, dict[str, object]]:
    """Load price overrides from JSON.

    Accepted shapes:
      {"b200": 35000}
      {"b200": {"price_per_unit_usd": 35000, "price_source": "..."}}
      {"b200": {"price": 35000, "source": "..."}}
    """
    if path is None:
        return {}
    data = load_json(path)
    if data is None:
        raise SystemExit(f"cannot read price JSON: {path}")
    if not isinstance(data, dict):
        raise SystemExit(f"price JSON must be an object: {path}")
    out: dict[str, dict[str, object]] = {}
    for raw_key, raw_value in data.items():
        key = str(raw_key).strip().lower()
        if isinstance(raw_value, (int, float)):
            out[key] = {
                "price_per_unit_usd": float(raw_value),
                "price_source": "user-supplied-json",
            }
            continue
        if not isinstance(raw_value, dict):
            raise SystemExit(
                f"price JSON entry for {raw_key!r} must be a number or object"
            )
        price = (
            raw_value.get("price_per_unit_usd")
            if "price_per_unit_usd" in raw_value
            else raw_value.get("price")
        )
        if price is None:
            raise SystemExit(
                f"price JSON entry for {raw_key!r} missing price_per_unit_usd"
            )
        out[key] = {
            "price_per_unit_usd": float(price),
            "price_source": str(
                raw_value.get("price_source")
                or raw_value.get("source")
                or "user-supplied-json"
            ),
        }
    return out


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
    return (meta.get("system_environment") or {}).get("inference_engine_version")


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
        "    python compute_cost.py \\",
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



def decode_profile_usable(tpot, batch_profile: dict) -> bool:
    """True when the core DECODE-derived outputs can be computed: the per-1M
    output-token cost and the effective output rate. These need only `tpot` and
    `decode_avg_batch_size` — a missing prefill batch does not affect them, and
    the decode token count is deliberately not required: it feeds only the
    per-request seconds/cost outputs, which report null when it is withheld
    rather than taking the per-1M cost down with them.

    A missing decode batch must still fail: on concurrent runs the single-stream
    fallback overstates cost by roughly the achieved batch, so the caller skips
    those runs rather than falling back.
    """
    decode_bs = batch_profile.get("decode_avg_batch_size")
    return (
        isinstance(tpot, (int, float)) and tpot > 0
        and isinstance(decode_bs, (int, float)) and decode_bs > 0
    )


def prefill_profile_usable(ttft, batch_profile: dict) -> bool:
    """True when the PREFILL-derived outputs can be computed: the prefill seconds,
    the per-request cost that includes them, and the prefill token rate. A run whose
    prefill batch is withheld reports these as null and keeps its decode cost."""
    prefill_bs = batch_profile.get("prefill_avg_batch_size")
    return (
        isinstance(ttft, (int, float))
        and isinstance(prefill_bs, (int, float)) and prefill_bs > 0
    )


def profile_usable(ttft, tpot, batch_profile: dict) -> bool:
    """True when BOTH halves are computable — the full batch_token_profile path."""
    return decode_profile_usable(tpot, batch_profile) and prefill_profile_usable(
        ttft, batch_profile
    )


def build_cost_metrics(
    *,
    price_per_s: float,
    e2e_s: float,
    ttft: Optional[float],
    tpot: float,
    batch_profile: dict,
) -> dict:
    """Return cost metrics plus an auditable throughput/cost breakdown.

    TPOT is a per-decode-step latency. In continuous batching, one decode step
    emits roughly decode_avg_batch_size output tokens across active requests, so
    cost per output token must divide TPOT by that average decode batch size.
    Per-request cost similarly uses the measured prefill/decode batch sizes
    rather than treating each request latency as exclusive accelerator time.

    The returned ``breakdown`` block intentionally duplicates the formula inputs
    and derived values. This makes apparent hardware-cost inversions auditable:
    a higher-priced accelerator can have lower cost/token when its measured
    effective decode throughput (decode_avg_batch_size / TPOT) improves more
    than its price premium.
    """
    decode_avg_batch_size = batch_profile.get("decode_avg_batch_size")
    decode_tokens_per_request = batch_profile.get("decode_generated_tokens_per_request")
    prefill_avg_batch_size = batch_profile.get("prefill_avg_batch_size")
    prefill_tokens_per_request = batch_profile.get("prefill_tokens_per_request")
    price_per_hour = price_per_s * 3600.0

    if decode_profile_usable(tpot, batch_profile):
        # Prefill terms are withheld, not estimated, when the prefill batch is null:
        # they are the only outputs that depend on it, and the per-1M output-token
        # cost below is computed from the decode fields alone.
        have_prefill = prefill_profile_usable(ttft, batch_profile)
        have_decode_tok = isinstance(decode_tokens_per_request, (int, float)) and decode_tokens_per_request >= 0
        prefill_seconds_per_request = (ttft / prefill_avg_batch_size) if have_prefill else None
        decode_seconds_per_request = (
            (tpot * decode_tokens_per_request) / decode_avg_batch_size
            if have_decode_tok else None
        )
        request_seconds = (
            prefill_seconds_per_request + decode_seconds_per_request
            if have_prefill and have_decode_tok else None
        )
        output_token_seconds = tpot / decode_avg_batch_size
        effective_output_tokens_per_s = decode_avg_batch_size / tpot
        avg_cost_per_request_usd = (request_seconds * price_per_s) if request_seconds is not None else None
        avg_cost_per_1m_output_tokens_usd = output_token_seconds * 1_000_000 * price_per_s
        prefill_cost_per_request_usd = (
            prefill_seconds_per_request * price_per_s if have_prefill else None
        )
        decode_cost_per_request_usd = (
            decode_seconds_per_request * price_per_s
            if decode_seconds_per_request is not None else None
        )
        prefill_tokens_per_s = None
        if have_prefill and isinstance(prefill_tokens_per_request, (int, float)) and prefill_tokens_per_request > 0:
            prefill_tokens_per_s = prefill_tokens_per_request * prefill_avg_batch_size / ttft
        return {
            "avg_cost_per_request_usd": avg_cost_per_request_usd,
            "avg_cost_per_1M_output_tokens_usd": avg_cost_per_1m_output_tokens_usd,
            "method": "batch_token_profile",
            "prefill_avg_batch_size": float(prefill_avg_batch_size) if have_prefill else None,
            "decode_avg_batch_size": float(decode_avg_batch_size),
            "decode_generated_tokens_per_request": float(decode_tokens_per_request) if have_decode_tok else None,
            "effective_output_tokens_per_s": effective_output_tokens_per_s,
            "formula": "price_per_second_usd * tpot_s / decode_avg_batch_size * 1e6",
            "breakdown": {
                "pricing": {
                    "price_per_hour_usd": price_per_hour,
                    "price_per_second_usd": price_per_s,
                },
                "latency": {
                    "e2e_s": e2e_s,
                    "ttft_s": ttft,
                    "tpot_s": tpot,
                },
                "batch_profile": {
                    "prefill_avg_batch_size": float(prefill_avg_batch_size) if have_prefill else None,
                    "decode_avg_batch_size": float(decode_avg_batch_size),
                    "prefill_tokens_per_request": (
                        float(prefill_tokens_per_request)
                        if isinstance(prefill_tokens_per_request, (int, float))
                        else None
                    ),
                    "decode_generated_tokens_per_request": float(decode_tokens_per_request) if have_decode_tok else None,
                },
                "throughput": {
                    "effective_output_tokens_per_s": effective_output_tokens_per_s,
                    "prefill_tokens_per_s": prefill_tokens_per_s,
                    "formula": "decode_avg_batch_size / tpot_s",
                },
                "request_seconds": {
                    "prefill_seconds_per_request": prefill_seconds_per_request,
                    "decode_seconds_per_request": decode_seconds_per_request,
                    "total_seconds_per_request": request_seconds,
                },
                "request_cost_usd": {
                    "prefill_cost_per_request_usd": prefill_cost_per_request_usd,
                    "decode_cost_per_request_usd": decode_cost_per_request_usd,
                    "total_cost_per_request_usd": avg_cost_per_request_usd,
                },
                "output_token_cost": {
                    "seconds_per_output_token": output_token_seconds,
                    "cost_per_output_token_usd": output_token_seconds * price_per_s,
                    "cost_per_1M_output_tokens_usd": avg_cost_per_1m_output_tokens_usd,
                    "formula": "price_per_second_usd / effective_output_tokens_per_s * 1e6",
                },
            },
        }

    seconds_per_output_token = tpot if isinstance(tpot, (int, float)) else 0.0
    effective_output_tokens_per_s = (1.0 / tpot) if isinstance(tpot, (int, float)) and tpot > 0 else 0.0
    avg_cost_per_request_usd = e2e_s * price_per_s
    avg_cost_per_1m_output_tokens_usd = seconds_per_output_token * 1_000_000 * price_per_s
    return {
        "avg_cost_per_request_usd": avg_cost_per_request_usd,
        "avg_cost_per_1M_output_tokens_usd": avg_cost_per_1m_output_tokens_usd,
        "method": "latency_tpot_no_batch_profile",
        "effective_output_tokens_per_s": effective_output_tokens_per_s,
        "formula": "price_per_second_usd * tpot_s * 1e6",
        "breakdown": {
            "pricing": {
                "price_per_hour_usd": price_per_hour,
                "price_per_second_usd": price_per_s,
            },
            "latency": {
                "e2e_s": e2e_s,
                "ttft_s": ttft,
                "tpot_s": tpot,
            },
            "throughput": {
                "effective_output_tokens_per_s": effective_output_tokens_per_s,
                "formula": "1 / tpot_s (fallback; no measured batch profile)",
            },
            "request_seconds": {
                "total_seconds_per_request": e2e_s,
            },
            "request_cost_usd": {
                "total_cost_per_request_usd": avg_cost_per_request_usd,
            },
            "output_token_cost": {
                "seconds_per_output_token": seconds_per_output_token,
                "cost_per_output_token_usd": seconds_per_output_token * price_per_s,
                "cost_per_1M_output_tokens_usd": avg_cost_per_1m_output_tokens_usd,
                "formula": "price_per_second_usd / effective_output_tokens_per_s * 1e6",
            },
        },
    }

def build_wall_block(price_per_s: float, e2e_s: float) -> dict:
    """Wall-clock per-request cost: the node's whole run time divided over its requests.

    e2e_s is the run makespan / N, so price * e2e_s charges every second the node spent
    on the run — prefill, queueing and gaps included — to the requests it served. It is
    a different quantity from avg_cost_per_request_usd, which charges engine-attributable
    prefill+decode time only.
    """
    return {
        "node_seconds_per_request": e2e_s,
        "cost_per_request_usd": e2e_s * price_per_s,
        "formula": "price_per_second_usd * node_seconds_per_request",
    }


def build_buy_block(
    gpu_key: str,
    num_gpus: int,
    e2e_s: float,
    ttft: Optional[float],
    tpot: float,
    batch_profile: dict,
    *,
    gpu_specs: dict[str, dict],
    cpu_specs: dict[str, dict],
    gpu_host_cpu: dict[str, tuple[int, str]],
    lifetime_hours: float,
    base_lifetime_hours: float,
    utilisation: float,
    electricity_usd_per_kwh: float,
    scale_other_capital: float,
    buy_price_quote_time: Optional[str] = None,
    include_token_cost: bool = True,
) -> Optional[dict]:
    gpu = gpu_specs.get(gpu_key)
    host = gpu_host_cpu.get(gpu_key)
    if gpu is None or host is None:
        return None
    num_cpus, cpu_key = host
    cpu = cpu_specs.get(cpu_key)
    if cpu is None:
        return None

    gpu_capital = gpu["price_per_unit_usd"] * num_gpus
    cpu_capital = cpu["price_per_unit_usd"] * num_cpus
    capital_scale = gpu.get("capital_scale", scale_other_capital)
    total_capital_usd = (gpu_capital + cpu_capital) * capital_scale
    total_power_w = gpu["tdp_w"] * num_gpus + cpu["tdp_w"] * num_cpus

    amort_per_h = total_capital_usd / lifetime_hours
    energy_per_h = (total_power_w / 1000.0) * electricity_usd_per_kwh
    effective_per_h = amort_per_h + energy_per_h
    effective_per_s = effective_per_h / 3600.0

    return {
        "lifetime_hours": lifetime_hours,
        "base_lifetime_hours": base_lifetime_hours,
        "utilisation": utilisation,
        "electricity_usd_per_kwh": electricity_usd_per_kwh,
        "scale_other_capital": scale_other_capital,
        **({"capital_scale": capital_scale} if "capital_scale" in gpu else {}),
        "gpu": {
            "key": gpu_key,
            "num": num_gpus,
            "price_per_unit_usd": gpu["price_per_unit_usd"],
            "price_source": gpu.get("price_source", "user-supplied"),
            "price_quote_time": buy_price_quote_time,
            "tdp_w": gpu["tdp_w"],
            "tdp_source": gpu.get("tdp_source", "user-supplied"),
        },
        "cpu": {
            "key": cpu_key,
            "model": cpu.get("model", cpu_key),
            "num": num_cpus,
            "price_per_unit_usd": cpu["price_per_unit_usd"],
            "price_source": cpu.get("price_source", "user-supplied"),
            "price_quote_time": buy_price_quote_time,
            "tdp_w": cpu["tdp_w"],
            "tdp_source": cpu.get("tdp_source", "user-supplied"),
        },
        "total_capital_usd": total_capital_usd,
        "total_power_w": total_power_w,
        "amortized_capital_usd_per_hour": amort_per_h,
        "energy_usd_per_hour": energy_per_h,
        "effective_hourly_rate_usd": effective_per_h,
        "wall": build_wall_block(effective_per_s, e2e_s),
        **({"cost": build_cost_metrics(
            price_per_s=effective_per_s,
            e2e_s=e2e_s,
            ttft=ttft,
            tpot=tpot,
            batch_profile=batch_profile,
        )} if include_token_cost else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).parent,
        help="Tree to scan for metrics files",
    )
    parser.add_argument(
        "--rent-price", action="append", default=[],
        help='Per-GPU rent $/h, e.g. --rent-price b200=4.26 (repeat)',
    )
    parser.add_argument(
        "--rent-price-source", action="append", default=[],
        help=f"Per-GPU rent price source URL, e.g. --rent-price-source "
             f"mi355x=https://www.vultr.com/.../amd-mi355x/ "
             f"(default: {DEFAULT_RENT_PRICE_SOURCE})",
    )
    parser.add_argument(
        "--rent-prices-json", type=Path, default=None,
        help='JSON file mapping gpu_key -> $/h',
    )
    parser.add_argument(
        "--rent-price-quote-time", default=None,
        help="ISO time when rent prices were quoted (default: now, UTC)",
    )

    parser.add_argument(
        "--buy-gpu-price", action="append", default=[],
        help="Override GPU purchase price, e.g. --buy-gpu-price b200=35000",
    )
    parser.add_argument(
        "--buy-gpu-prices-json", type=Path, default=None,
        help=(
            "JSON mapping gpu_key -> purchase USD, or gpu_key -> "
            "{price_per_unit_usd, price_source}"
        ),
    )
    parser.add_argument(
        "--buy-gpu-tdp", action="append", default=[],
        help="Override GPU TDP (W), e.g. --buy-gpu-tdp b200=1000",
    )
    parser.add_argument(
        "--buy-cpu-for", action="append", default=[],
        help="Override CPU choice per GPU, e.g. --buy-cpu-for mi355x=epyc-7713p",
    )
    parser.add_argument(
        "--buy-num-cpus", action="append", default=[],
        help="Override CPU count per GPU platform, e.g. --buy-num-cpus b200=2",
    )
    parser.add_argument(
        "--buy-cpu-price", action="append", default=[],
        help="Override CPU price, e.g. --buy-cpu-price xeon-8468=7214",
    )
    parser.add_argument(
        "--buy-cpu-prices-json", type=Path, default=None,
        help=(
            "JSON mapping cpu_key -> purchase USD, or cpu_key -> "
            "{price_per_unit_usd, price_source}"
        ),
    )
    parser.add_argument(
        "--buy-cpu-tdp", action="append", default=[],
        help="Override CPU TDP (W)",
    )
    parser.add_argument(
        "--buy-lifetime-hours", type=float, default=DEFAULT_LIFETIME_HOURS,
        help=f"Server calendar lifetime hours before utilisation (default: {DEFAULT_LIFETIME_HOURS} = 3 yr)",
    )
    parser.add_argument(
        "--utilisation", "--utilization", dest="utilisation", type=float, default=1.0,
        help=(
            "Average hardware utilisation in (0, 1]; effective buy lifetime "
            "hours are --buy-lifetime-hours * utilisation (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--buy-electricity-usd-per-kwh", type=float, default=DEFAULT_ELECTRICITY_USD_PER_KWH,
        help=f"Electricity price (default: {DEFAULT_ELECTRICITY_USD_PER_KWH})",
    )
    parser.add_argument(
        "--buy-scale-other-capital", type=float, default=DEFAULT_SCALE_OTHER_CAPITAL,
        help=f"Capital scaling for motherboard/DRAM/SSD/etc. (default: "
             f"{DEFAULT_SCALE_OTHER_CAPITAL}, from MoE-CAP)",
    )
    parser.add_argument(
        "--buy-price-quote-time", default=None,
        help="ISO time when buy prices were quoted (default: now, UTC)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.utilisation <= 1.0):
        print("error: --utilisation must be in (0, 1]", file=sys.stderr)
        return 2
    effective_lifetime_hours = args.buy_lifetime_hours * args.utilisation

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
    for k, override in parse_price_json(args.buy_gpu_prices_json).items():
        spec = gpu_specs.setdefault(k, {})
        spec["price_per_unit_usd"] = float(override["price_per_unit_usd"])
        spec["price_source"] = str(override["price_source"])
    for k, v in parse_kv_args(args.buy_gpu_price, float).items():
        spec = gpu_specs.setdefault(k, {})
        spec["price_per_unit_usd"] = float(v)
        spec["price_source"] = "user-supplied"
    for k, v in parse_kv_args(args.buy_gpu_tdp, float).items():
        spec = gpu_specs.setdefault(k, {})
        spec["tdp_w"] = float(v)
        spec["tdp_source"] = "user-supplied"

    cpu_specs = {k: dict(v) for k, v in CPU_SPECS.items()}
    for k, override in parse_price_json(args.buy_cpu_prices_json).items():
        spec = cpu_specs.setdefault(k, {"model": k})
        spec["price_per_unit_usd"] = float(override["price_per_unit_usd"])
        spec["price_source"] = str(override["price_source"])
    for k, v in parse_kv_args(args.buy_cpu_price, float).items():
        spec = cpu_specs.setdefault(k, {"model": k})
        spec["price_per_unit_usd"] = float(v)
        spec["price_source"] = "user-supplied"
    for k, v in parse_kv_args(args.buy_cpu_tdp, float).items():
        spec = cpu_specs.setdefault(k, {"model": k})
        spec["tdp_w"] = float(v)
        spec["tdp_source"] = "user-supplied"

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

    print(f"\nRent prices ($/GPU/h):")
    for k in gpu_keys:
        if k in rent_prices:
            src = rent_price_sources.get(k, DEFAULT_RENT_PRICE_SOURCE)
            print(f"  {k}: {rent_prices[k]:.4f}   [{src}]")
        else:
            print(f"  {k}: <missing>")

    print(
        "\nBuy-cost assumptions note: built-in GPU/CPU purchase prices are "
        "curated current/recent-average estimates, not canonical vendor list "
        "prices. For reproducible TEASBench releases, prefer "
        "--buy-gpu-prices-json/--buy-cpu-prices-json and set --utilisation "
        "explicitly."
    )
    print(f"\nBuy specs (lifetime={effective_lifetime_hours}h, "
          f"base_lifetime={args.buy_lifetime_hours}h, utilisation={args.utilisation}, "
          f"electricity=${args.buy_electricity_usd_per_kwh}/kWh, "
          f"scale_other_capital={args.buy_scale_other_capital}):")
    for k in gpu_keys:
        if k in gpu_specs and k in gpu_host_cpu:
            g = gpu_specs[k]
            num_cpu, cpu_key = gpu_host_cpu[k]
            c = cpu_specs.get(cpu_key, {})
            print(f"  {k}: GPU ${g.get('price_per_unit_usd', '?')}/{g.get('tdp_w', '?')}W "
                  f"+ {num_cpu}x {c.get('model', cpu_key)} "
                  f"(${c.get('price_per_unit_usd', '?')}/{c.get('tdp_w', '?')}W)")
        else:
            print(f"  {k}: <no buy spec; supply --buy-gpu-price/-tdp/--buy-cpu-for>")

    written = 0
    skipped = 0
    wall_only = 0
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
        batch_profile = metrics.get("batch_token_profile") or {}
        e2e_s = perf.get("e2e_s")
        tpot = perf.get("tpot")
        ttft = perf.get("ttft")
        if e2e_s is None:
            # Newer collection schema omits e2e_s; it is exactly 1 / request rate.
            request_rate = perf.get("request/s")
            if request_rate:
                e2e_s = 1.0 / request_rate
        if e2e_s is None:
            skipped += 1
            continue

        # batch-size-1 pins concurrency server-side, so both average batch sizes are 1.0 by
        # construction. A profile violating that bound is an accounting artefact, not a
        # measurement — discard it rather than price on it. Costing then takes the
        # single-stream fallback, whose arithmetic is exact at batch 1 (cost/1M = tpot x price)
        # and whose per-request cost rests on e2e rather than a token count from the same
        # suspect block. The violating values stay untouched in metrics.json so the audit can
        # keep flagging them; the sidecar records the drop below.
        single = any(p == "batch-size-1" for p in f.parts)
        profile_invalid = None
        if single and batch_profile:
            off = {
                k: batch_profile[k]
                for k in ("prefill_avg_batch_size", "decode_avg_batch_size")
                if isinstance(batch_profile.get(k), (int, float))
                and abs(batch_profile[k] - 1.0) > 1e-9
            }
            if off:
                profile_invalid = {
                    "reason": "avg batch size violates the single-stream bound of 1.0; "
                              "profile not used for costing",
                    "recorded": off,
                }
                batch_profile = {}

        # Concurrent run whose batch profile cannot price the per-token metrics (absent, or
        # partial — a missing field routes build_cost_metrics to the single-stream fallback,
        # which inflates cost 10-20x for a run that actually batched). Those runs keep their
        # wall-clock per-request cost, which needs no profile, and omit the `cost` block —
        # a blank per-token cost is more honest than a fabricated one. Single-query runs
        # keep the fallback (single-stream is their real cost).
        concurrent = any(p.startswith("batch-size-default") for p in f.parts)
        token_cost_ok = tpot is not None and not (concurrent and not decode_profile_usable(tpot, batch_profile))
        if not token_cost_ok:
            wall_only += 1

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
                "e2e_s": e2e_s,
                "ttft_s": ttft,
                "tpot_s": tpot,
            },
        }
        if profile_invalid:
            payload["batch_profile_invalid"] = profile_invalid

        if gpu_key in rent_prices:
            price_per_gpu_h = rent_prices[gpu_key]
            hourly_rate = price_per_gpu_h * num_gpus
            price_per_s = hourly_rate / 3600.0
            payload["rent"] = {
                "price_quote_time": rent_quote_time,
                "price_per_gpu_hour_usd": price_per_gpu_h,
                "total_hourly_rate_usd": hourly_rate,
                "price_per_second_usd": price_per_s,
                "price_source": rent_price_sources.get(gpu_key, DEFAULT_RENT_PRICE_SOURCE),
                "wall": build_wall_block(price_per_s, e2e_s),
            }
            if token_cost_ok:
                payload["rent"]["cost"] = build_cost_metrics(
                    price_per_s=price_per_s,
                    e2e_s=e2e_s,
                    ttft=ttft,
                    tpot=tpot,
                    batch_profile=batch_profile,
                )

        buy = build_buy_block(
            gpu_key, num_gpus, e2e_s, ttft, tpot, batch_profile,
            gpu_specs=gpu_specs, cpu_specs=cpu_specs, gpu_host_cpu=gpu_host_cpu,
            lifetime_hours=effective_lifetime_hours,
            base_lifetime_hours=args.buy_lifetime_hours,
            utilisation=args.utilisation,
            electricity_usd_per_kwh=args.buy_electricity_usd_per_kwh,
            scale_other_capital=args.buy_scale_other_capital,
            buy_price_quote_time=buy_price_quote_time,
            include_token_cost=token_cost_ok,
        )
        if buy is not None:
            payload["buy"] = buy

        out_path = cost_path_for(f)
        if args.dry_run:
            print(f"  [dry-run] would write {out_path.relative_to(root.parent)}")
        else:
            out_path.write_text(json.dumps(payload, indent=2) + "\n")
        written += 1

    print(f"\nWrote {written} cost JSON files (skipped {skipped}; "
          f"{wall_only} concurrent runs with no batch profile carry wall-clock cost only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
