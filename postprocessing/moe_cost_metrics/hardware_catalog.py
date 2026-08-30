#!/usr/bin/env python3
"""Hardware specifications, owned here and imported by everything that publishes them.

One table per axis, in one file, so a card cannot be published at two different figures.
Three bandwidth catalogs once coexisted in the sparsity producer and the one that governed
the published numbers was a fallback nobody was editing; a fourth copy then sat in the
dashboard assembler, which kept reading 8.0 TB/s for Blackwell after this catalog moved to
the HGX per-GPU column at 7.7. Consumers import from here rather than restating a value.

Two keyings coexist and both belong to the same catalog:
  - the canonical device key ('NVIDIA-B200-183GB'), which the per-run metrics resolve to and
    which the bandwidth and peak-FLOPS tables are keyed on;
  - the short directory key ('b200'), which the result-tree layout uses and which the
    purchase/TDP tables are keyed on.
`GPU_KEY_FALLBACK` is the bridge between them.

Stdlib only, and no side effects on import: this is read by scripts run standalone from a
checkout as well as by the dashboard assembler in another repository.
"""

from __future__ import annotations

# Bandwidth is owned here for the same reason as peak FLOPS below: vendors quote it on
# different bases, and a second catalog patched over a first is how a corrected figure gets
# silently reverted. One table per axis, nothing overriding it later in the module.
#
# Sources are the vendor datasheets listed in the peak-FLOPS block below, plus the two edge
# parts those datacentre datasheets do not cover:
#   GB10 (DGX Spark)          273 GB/s LPDDR5X   https://www.nvidia.com/en-us/products/workstations/dgx-spark/
#   Blackhole p150b           512 GB/s GDDR6     https://docs.tenstorrent.com/aibs/blackhole/specifications.html
#
# The Blackwell entries are the HGX per-GPU column, 7.7 TB/s. Reading the NVL72 column of the
# same table instead gives 8.0, which is what these held before, and the measured part is an
# air-cooled SXM module rather than an NVL72 tray.
MEM_BW_DICT = {
    "NVIDIA-A100-SXM4-80GB": 2.039e12,
    "NVIDIA-H100-HBM3-80GB": 3.35e12,
    "NVIDIA-H200-141GB": 4.8e12,
    "NVIDIA-B200-183GB": 7.7e12,
    "NVIDIA-B300-269GB": 7.7e12,
    "AMD-Instinct-MI355X-288GB": 8.0e12,
    "NVIDIA-GB10": 273e9,
    "Tenstorrent-Blackhole-P150b": 512e9,
}


def get_peak_bw(gpu_key):
    return MEM_BW_DICT.get(gpu_key, 0)


# Memory capacity is owned here for the same reason as bandwidth, and joins it late: it was the
# one figure in the published hardware table with no sourcing record, living as a literal in the
# dashboard script while every other column in that table resolved through this catalog.
#
# These are NAMEPLATE capacities, not what a driver reports. The two differ by vendor and by
# part: AMD reports its nameplate while NVIDIA reports post-carve-out usable, so publishing
# device figures would put two bases in one column and make an MI355X read larger than a B300
# that is physically its equal. Device figures are also not stable per part -- this catalog's
# own key strings carry two of them for the B200 and two for the H200 -- which is what makes
# them identifiers rather than specs.
#
# Sources are the per-GPU columns of the datasheets already named in the peak-FLOPS block.
# NOTE: these entries carry the source but not yet the cell-by-cell datasheet confirmation the
# bandwidth and FLOPS rows have. They are the values the dashboard was already publishing,
# moved here so that they can be checked in the same place as everything else.
MEM_GB_DICT = {
    "NVIDIA-A100-SXM4-80GB": 80,
    "NVIDIA-H100-HBM3-80GB": 80,
    "NVIDIA-H200-141GB": 141,
    "NVIDIA-B200-183GB": 192,
    "NVIDIA-B300-269GB": 288,         # B300_KEY, spelled out: that name is bound further down
    "AMD-Instinct-MI355X-288GB": 288,
    "NVIDIA-GB10": 128,               # unified LPDDR5X, not discrete HBM
    "Tenstorrent-Blackhole-P150b": 32,
}


def mem_capacity_gb(gpu_key: str) -> int:
    """Nameplate memory capacity in GB for one device, by short key.

    Raises on a short key with no canonical key or no capacity entry rather than reading as a
    zero-capacity part, so a card added to a consumer without being catalogued here stops the
    run instead of publishing a blank spec.
    """
    return MEM_GB_DICT[GPU_KEY_FALLBACK[gpu_key]]


# Peak FLOPS is owned here for the same reason, because vendors quote it on
# different bases and mixing them silently biases S-MFU across vendors.
#
# Every entry is DENSE and per GPU, and S-MFU divides by the value as written. Each names its
# with-sparsity counterpart inline where the vendor publishes one. The bases differ by
# vendor and part:
#   - H100/H200 datasheets lead with the with-sparsity figure; dense is half.
#   - A100's leads with dense (312) and footnotes the sparse one (624).
#   - Blackwell's per-GPU rows are with-sparsity and halve, except FP4, which prints
#     `sparse | dense` outright and is taken as printed. Its board totals disagree with its
#     own per-GPU rows there: HGX B300 reads 18 | 14 PFLOPS per GPU while the total on the
#     same page reads 144 | 108, and 108/8 is 13.5. A per-GPU catalog follows the per-GPU row.
#   - AMD names dense and with-sparsity as separate rows, and its sparse figures are 2.02x
#     dense rather than 2x: OCP-FP8 reads 10.1 PFLOPS sparse against 5.0 dense, INT8 10.1
#     POPS against 5.0. Halving the sparse row is what put FP8 and INT8 at 5050 here.
#     MXFP4/6/8 have no sparsity row at all.
#
# The Blackwell figures are the HGX per-GPU columns, not the NVL72 ones: the measured part
# reports `NVIDIA-B300-SXM6-AC-269GB`, an air-cooled SXM module. NVL72 trays run higher clocks
# and are quoted above HGX on several rows — FP4 15 PFLOPS dense against 14, INT8 330 TOPS
# against 307, bandwidth 8.0 TB/s against 7.7.
#
# Sources (vendor primary, per-GPU dense figures):
#   A100        https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
#   H100/H200   https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306
#   B200        https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
#   B300        https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet
#   MI355X      https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
#   GB10        https://www.nvidia.com/en-us/products/workstations/dgx-spark/ — NVIDIA
#               publishes one figure, 1 PFLOP FP4 with sparsity. The dense FP4 entry halves
#               it (the NVIDIA convention above) and each wider precision halves again,
#               mirroring the B200 ladder. Derived, not datasheet, below FP4.
#   Blackhole   deliberately absent: Tenstorrent publishes only a BLOCKFP8 figure (664
#               TFLOPS at the 120-core spec), no FP16/BF16 rate, so bf16 runs have no
#               defensible denominator and publish a null S-MFU.
PEAK_FLOPS_BASIS = "dense"
PEAK_FLOPS_DICT = {
    "bfloat16": {
        "NVIDIA-A100-SXM4-80GB": 312e12,      # 624 w/ sparsity
        "NVIDIA-H100-HBM3-80GB": 989.5e12,    # 1979 w/ sparsity
        "NVIDIA-H200-141GB": 989.5e12,        # 1979 w/ sparsity
        "NVIDIA-B200-183GB": 2250e12,         # 4500 w/ sparsity
        "NVIDIA-B300-269GB": 2250e12,         # 4500 w/ sparsity
        "AMD-Instinct-MI355X-288GB": 2500e12,
        "NVIDIA-GB10": 125e12,
    },
    "float16": {
        "NVIDIA-A100-SXM4-80GB": 312e12,
        "NVIDIA-H100-HBM3-80GB": 989.5e12,
        "NVIDIA-H200-141GB": 989.5e12,
        "NVIDIA-B200-183GB": 2250e12,
        "NVIDIA-B300-269GB": 2250e12,
        "AMD-Instinct-MI355X-288GB": 2500e12,
        "NVIDIA-GB10": 125e12,
    },
    "fp8": {
        "NVIDIA-A100-SXM4-80GB": 312e12,      # A100 has no FP8 tensor cores -> upcasts to bf16
        "NVIDIA-H100-HBM3-80GB": 1979e12,     # 3958 w/ sparsity
        "NVIDIA-H200-141GB": 1979e12,
        "NVIDIA-B200-183GB": 4500e12,         # 9000 w/ sparsity
        "NVIDIA-B300-269GB": 4500e12,         # 9000 w/ sparsity
        "AMD-Instinct-MI355X-288GB": 5000e12, # 10100 w/ sparsity; the published dense row
        "NVIDIA-GB10": 250e12,
    },
    "int8": {
        "NVIDIA-A100-SXM4-80GB": 624e12,      # 1248 w/ sparsity
        "NVIDIA-H100-HBM3-80GB": 1979e12,
        "NVIDIA-H200-141GB": 1979e12,
        "NVIDIA-B200-183GB": 4500e12,          # 9000 w/ sparsity
        # Blackwell Ultra's INT8 tensor path is far narrower than B200's: 307 TOPS per GPU
        # against 9 POPS, both with-sparsity figures from the same table.
        "NVIDIA-B300-269GB": 153.5e12,         # 307 w/ sparsity
        "AMD-Instinct-MI355X-288GB": 5000e12,  # 10100 w/ sparsity; the published dense row
        "NVIDIA-GB10": 250e12,                 # the fp8 rate, as on every card but A100/B300
    },
    "fp4": {
        "NVIDIA-A100-SXM4-80GB": 312e12,      # A100 has no FP4 tensor cores -> mxfp4 upcasts to bf16
        "NVIDIA-H100-HBM3-80GB": 1979e12,     # no FP4 tensor cores -> FP8 path
        "NVIDIA-H200-141GB": 1979e12,
        "NVIDIA-B200-183GB": 9000e12,         # 18000 w/ sparsity
        # The one row NVIDIA prints as `sparse | dense` rather than sparse alone: HGX B300
        # reads 18 | 14 PFLOPS per GPU, so dense is taken as printed rather than halved, and
        # sparse is 1.29x dense here. The board total on the same page implies 13.5 (108
        # PFLOPS across 8 GPUs), which is where this entry sat before.
        "NVIDIA-B300-269GB": 14000e12,        # 18000 w/ sparsity
        "AMD-Instinct-MI355X-288GB": 10100e12, # MXFP4; AMD publishes no sparsity row for it
        "NVIDIA-GB10": 500e12,
    },
    # `int4` is reached by weight-only 4-bit checkpoints (compressed-tensors
    # `num_bits=4 type=int`), which dequantise before the matmul rather than using a native
    # INT4 tensor path; only A100 has one. Each entry mirrors the card's fp4 rate. That is a
    # modelling assumption rather than a datasheet figure: the exact denominator is whatever
    # precision the kernel computes in, which the run does not record.
    "int4": {
        "NVIDIA-A100-SXM4-80GB": 1248e12,     # 2496 w/ sparsity; Ampere has a real INT4 path
        "NVIDIA-H100-HBM3-80GB": 1979e12,
        "NVIDIA-H200-141GB": 1979e12,
        "NVIDIA-B200-183GB": 9000e12,
        "NVIDIA-B300-269GB": 14000e12,
        "AMD-Instinct-MI355X-288GB": 10100e12,
        "NVIDIA-GB10": 500e12,
    },
}


def get_peak_flops(gpu_key, precision="bfloat16"):
    return PEAK_FLOPS_DICT.get((precision or "").lower(), {}).get(gpu_key, 0)


B300_KEY = "NVIDIA-B300-269GB"


# Device strings as the runners record them, mapped onto the canonical keys above. The same
# part reaches this table under more than one capacity: B200 as both 180GB and 183GB, H200 as
# both 140GB and 141GB. The number in a key is part of an identifier, not a capacity reading —
# nothing here parses it.
GPU_TYPE_MAP = {
    "AMD-Instinct-MI355X-288GB": "AMD-Instinct-MI355X-288GB",
    "AMD-Instinct-MI355X": "AMD-Instinct-MI355X-288GB",
    "AMD--288GB": "AMD-Instinct-MI355X-288GB",
    "NVIDIA-A100-SXM4-80GB": "NVIDIA-A100-SXM4-80GB",
    "NVIDIA-B200-180GB": "NVIDIA-B200-183GB",
    "NVIDIA-B200-183GB": "NVIDIA-B200-183GB",
    "NVIDIA-B300-SXM6-AC-269GB": B300_KEY,
    "NVIDIA-H100-HBM3-80GB": "NVIDIA-H100-HBM3-80GB",
    "NVIDIA-H200-140GB": "NVIDIA-H200-141GB",
    "NVIDIA-H200-141GB": "NVIDIA-H200-141GB",
    "NVIDIA-GB10": "NVIDIA-GB10",
    "Tenstorrent-Blackhole-P150b": "Tenstorrent-Blackhole-P150b",
}

# Short result-tree key -> canonical device key. Used as a fallback when a run records no
# usable gpu_type, and as the bridge between the two keyings for anything reading a
# canonical-keyed table off a short key.
GPU_KEY_FALLBACK = {
    "a100": "NVIDIA-A100-SXM4-80GB",
    "h100": "NVIDIA-H100-HBM3-80GB",
    "h200": "NVIDIA-H200-141GB",
    "b200": "NVIDIA-B200-183GB",
    "b300": B300_KEY,
    "mi355x": "AMD-Instinct-MI355X-288GB",
    "gb10": "NVIDIA-GB10",
    "blackhole-p150b": "Tenstorrent-Blackhole-P150b",
}


# Deployment tier is an owned-hardware property, so it lives beside the price/power catalog
# consumed by both cost producers.  A missing entry is not silently treated as datacentre:
# choosing the wrong lifetime/utilisation pair changes every buy metric for that accelerator.
GPU_TIER = {
    "a100": "datacentre",
    "h100": "datacentre",
    "h200": "datacentre",
    "b200": "datacentre",
    "b300": "datacentre",
    "mi355x": "datacentre",
    "gb10": "workstation",
    "blackhole-p150b": "workstation",
    "cs3": "datacentre",
}

# Published buy-TCO defaults.  `base_lifetime_hours` is the calendar lifetime before the
# utilisation discount; the effective amortisation window is their product.  CLI overrides are
# resolved per run by resolve_buy_tco_assumptions, after the accelerator key is known.
BUY_TCO_DEFAULTS_BY_TIER = {
    "datacentre": {"base_lifetime_hours": 5 * 365 * 24, "utilisation": 0.9},
    "workstation": {"base_lifetime_hours": 3 * 365 * 24, "utilisation": 0.4},
}


def resolve_buy_tco_assumptions(
    gpu_key: str,
    *,
    base_lifetime_hours: float | None = None,
    utilisation: float | None = None,
) -> dict[str, float | str]:
    """Resolve tier defaults plus independent explicit CLI overrides for one accelerator."""
    tier = GPU_TIER[gpu_key]
    defaults = BUY_TCO_DEFAULTS_BY_TIER[tier]
    base = defaults["base_lifetime_hours"] if base_lifetime_hours is None else base_lifetime_hours
    util = defaults["utilisation"] if utilisation is None else utilisation
    if not isinstance(base, (int, float)) or isinstance(base, bool) or base <= 0:
        raise ValueError("buy lifetime hours must be greater than zero")
    if not isinstance(util, (int, float)) or isinstance(util, bool) or not (0.0 < util <= 1.0):
        raise ValueError("utilisation must be in (0, 1]")
    return {
        "hardware_tier": tier,
        "base_lifetime_hours": float(base),
        "utilisation": float(util),
        "lifetime_hours": float(base) * float(util),
    }


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
    "blackhole-p150b": {
        "price_per_unit_usd": 1399.0,
        "price_source": "https://tenstorrent.com/en/hardware/cards",
        "tdp_w": 300,
        "tdp_source": "https://docs.tenstorrent.com/aibs/blackhole/",
    },
    # Cerebras publishes no CS-3 list price. The $1.2M per-system figure was privately
    # communicated by the hardware developer (Cerebras); there is no public source, so
    # price_source carries a provenance label rather than a URL. Both it and the 23 kW
    # figure cover one complete integrated CS-3 system, so the normal 1.2x host/chassis
    # uplift must not be added a second time.
    "cs3": {
        "price_per_unit_usd": 1200000.0,
        "price_source": "privately communicated by the hardware developer (Cerebras); no public source",
        "tdp_w": 23000,
        "tdp_source": "https://www.cerebras.ai/blog/cerebras-cs-3-vs-nvidia-b200-2024-ai-accelerators-compared",
        "capital_scale": 1.0,
    },
}

CPU_SPECS: dict[str, dict] = {
    # Accounting-only host for an integrated system. CPU, MemoryX, SwarmX, management
    # hardware and the rest of the deployed boundary are already inside the CS-3 price
    # and power entries above; zeroes prevent compute_cost from adding them twice.
    "cs3-integrated-host": {
        "model": "Included in complete Cerebras CS-3 system",
        "price_per_unit_usd": 0.0,
        "price_source": "privately communicated by the hardware developer (Cerebras); no public source",
        "tdp_w": 0,
        "tdp_source": "https://www.cerebras.ai/blog/cerebras-cs-3-vs-nvidia-b200-2024-ai-accelerators-compared",
    },
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
    "ryzen-9700x": {
        "model": "AMD Ryzen 7 9700X",
        "price_per_unit_usd": 359.0,
        "price_source": "https://shop-us-en.amd.com/amd-ryzen-7-9700x-processor/",
        "tdp_w": 65,
        "tdp_source": "https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-7-9700x.html",
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
    "blackhole-p150b": (1, "ryzen-9700x"),  # PCIe dev cards in a workstation host; the CPU class Tenstorrent ships in its own Blackhole workstation (TT-QuietBox 2)
    "cs3": (1, "cs3-integrated-host"),
}


def gpu_tdp_w(gpu_key: str) -> int:
    """Accelerator nameplate TDP in watts, for one device, by short key.

    Raises on an unknown key rather than returning a default: a caller composing node power
    from an uncatalogued card would otherwise publish it at zero watts.
    """
    return GPU_SPECS[gpu_key]["tdp_w"]


def host_cpu_power_w(gpu_key: str) -> int:
    """Host-CPU nameplate power for one node of `gpu_key`, in watts, by short key.

    Zero for a catalogued part with no host entry: the CS-3's TDP above is a whole-system
    figure with no separate host to add. An uncatalogued key raises, so a card that reaches a
    consumer without reaching this file fails rather than reading as an unpowered host.
    """
    if gpu_key not in GPU_SPECS:
        raise KeyError(f"{gpu_key} is not in the hardware catalog")
    host = GPU_HOST_CPU.get(gpu_key)
    if host is None:
        return 0
    num_cpus, cpu_key = host
    return num_cpus * CPU_SPECS[cpu_key]["tdp_w"]


def peak_bw_gb_s(gpu_key: str):
    """Peak memory bandwidth in GB/s for a short key, for consumers publishing GB/s.

    Whole figures come back as `int`, which every catalogued part currently is, so a consumer
    serialising this emits the same literal a hand-written GB/s table did rather than moving
    a published `2039` to `2039.0`. A fractional figure is returned as a float.

    Raises on a short key with no canonical key or no bandwidth entry, rather than reading as
    a zero-bandwidth part. `get_peak_bw` keeps the zero-default behaviour the per-run sparsity
    path relies on to skip a card it cannot denominate.
    """
    gb_s = MEM_BW_DICT[GPU_KEY_FALLBACK[gpu_key]] / 1e9
    return int(gb_s) if gb_s.is_integer() else gb_s
