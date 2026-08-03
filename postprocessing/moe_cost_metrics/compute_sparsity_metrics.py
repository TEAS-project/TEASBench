#!/usr/bin/env python3
"""
Compute S-MBU and S-MFU per the MoE-CAP paper (arXiv 2412.07067 v6, Eqs. 4-5)
for every metrics file in the moe/ tree. S-MFU is always computed; S-MBU needs
the run's own expert_activation trace and is null without one — except on dense
models, where there are no experts to trace and the bytes term is fully
determined by the config, so S-MBU reduces to plain MBU and is always computed.

Per the paper:
  S-MBU = B_achieved / B_peak,  B_achieved = (S_activated + S_KV) / TPOT
  S-MFU = (T_token * S_F_token) / F_peak,
          S_F_token = F_attn + 2*N_router + 2*k_expert*N_expert

Where S_activated counts only the experts actually loaded
  (avg_expert_activation_{prefill,decode} from the runner's trace).

Parameter accounting is local to this script: MLA and GQA attention are
counted separately, as are shared and routed experts, and the hardware
peaks come from the catalogs below. KV-cache and attention-compute
terms are set to 0 because the aggregated metrics here do not carry
per-request prefill/output lengths — the reported S-MBU is therefore a
lower bound (weights-only) and S-MFU is a lower bound (no attention
score term). Add `--avg-prefill-len` / `--avg-decode-len` to override.

Output: sibling JSON `sparsity_<suffix>.json` next to each metrics file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Bandwidth is owned here for the same reason as peak FLOPS below: vendors quote it on
# different bases, and a second catalog patched over a first is how a corrected figure gets
# silently reverted. One table per axis, nothing overriding it later in the module.
#
# Sources are the vendor datasheets listed in the peak-FLOPS block below, plus the two edge
# parts those datacentre datasheets do not cover:
#   GB10 (DGX Spark)          273 GB/s LPDDR5X   https://www.nvidia.com/en-us/products/workstations/dgx-spark/
#   Blackhole p150b           512 GB/s GDDR6     https://docs.tenstorrent.com/aibs/blackhole/specifications.html
MEM_BW_DICT = {
    "NVIDIA-A100-SXM4-80GB": 2.04e12,
    "NVIDIA-H100-HBM3-80GB": 3.35e12,
    "NVIDIA-H200-141GB": 4.8e12,
    "NVIDIA-B200-183GB": 8.0e12,
    "NVIDIA-B300-269GB": 8.0e12,
    "AMD-Instinct-MI355X-288GB": 8.0e12,
    "NVIDIA-GB10": 273e9,
    "Tenstorrent-Blackhole-P150b": 512e9,
}


def get_peak_bw(gpu_key):
    return MEM_BW_DICT.get(gpu_key, 0)


# Peak FLOPS is owned here for the same reason, because vendors quote it on
# different bases and mixing them silently biases S-MFU across vendors.
#
# Every entry is DENSE, and S-MFU divides by the value as written. Each names its
# with-sparsity counterpart inline where the vendor publishes one. The bases differ by
# vendor and part:
#   - H100/H200 datasheets lead with the with-sparsity figure; dense is half.
#   - A100's leads with dense (312) and footnotes the sparse one (624).
#   - Blackwell prints both. B300 NVFP4 is the one row where sparse is 1.33x dense rather
#     than 2x: the uplift over B200 is on the dense figure.
#   - AMD's main column is dense. It publishes 2x sparse rates for FP16/BF16/INT8 and
#     OCP-FP8 only; MXFP4/6/8 are marked N/A for sparsity.
#
# Blackwell figures are HGX 8-GPU board specs divided by 8 — the form factor the measured
# parts report (`NVIDIA-B300-SXM6-AC-269GB`, air-cooled SXM), and one basis for both cards.
#
# Sources (vendor primary, dense figures):
#   A100        https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
#   H100/H200   https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306
#   B200/B300   https://www.nvidia.com/en-us/data-center/hgx/
#   MI355X      AMD Instinct MI355X GPU datasheet (amd.com)
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
        "AMD-Instinct-MI355X-288GB": 5050e12,
        "NVIDIA-GB10": 250e12,
    },
    "int8": {
        "NVIDIA-A100-SXM4-80GB": 624e12,      # 1248 w/ sparsity
        "NVIDIA-H100-HBM3-80GB": 1979e12,
        "NVIDIA-H200-141GB": 1979e12,
        "NVIDIA-B200-183GB": 4500e12,          # 9000 w/ sparsity
        # Blackwell Ultra's INT8 tensor path is far narrower than B200's: 3 POPS per HGX
        # board against 72.
        "NVIDIA-B300-269GB": 187.5e12,         # 375 w/ sparsity
        "AMD-Instinct-MI355X-288GB": 5050e12,
        "NVIDIA-GB10": 250e12,                 # the fp8 rate, as on every card but A100/B300
    },
    "fp4": {
        "NVIDIA-A100-SXM4-80GB": 312e12,      # A100 has no FP4 tensor cores -> mxfp4 upcasts to bf16
        "NVIDIA-H100-HBM3-80GB": 1979e12,     # no FP4 tensor cores -> FP8 path
        "NVIDIA-H200-141GB": 1979e12,
        "NVIDIA-B200-183GB": 9000e12,         # 18000 w/ sparsity
        # The HGX board is rated 144 | 108 PFLOPS NVFP4 (with-sparsity | dense), so
        # sparse is 1.33x dense here. The GB300 NVL72 tray is quoted at 15 PF dense and
        # runs higher clocks than the air-cooled SXM modules these runs use.
        "NVIDIA-B300-269GB": 13500e12,        # 18000 w/ sparsity
        "AMD-Instinct-MI355X-288GB": 10100e12,
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
        "NVIDIA-B300-269GB": 13500e12,
        "AMD-Instinct-MI355X-288GB": 10100e12,
        "NVIDIA-GB10": 500e12,
    },
}


def get_peak_flops(gpu_key, precision="bfloat16"):
    return PEAK_FLOPS_DICT.get((precision or "").lower(), {}).get(gpu_key, 0)


DATASET_TOKEN_PROFILE = {
    "gsm8k":        {"avg_input": 60,     "avg_output": 300},
    "arena-hard":   {"avg_input": 110,    "avg_output": 940},
    "longbench_v1": {"avg_input": 10000,  "avg_output": 220},
    "longbench_v2": {"avg_input": 10000,  "avg_output": 220},
}


def dataset_token_profile(dataset_path_part: str) -> Optional[dict]:
    base = dataset_path_part.split("_")[0]
    head = "_".join(dataset_path_part.split("_")[:2])
    return DATASET_TOKEN_PROFILE.get(head) or DATASET_TOKEN_PROFILE.get(base)


def is_mla(cfg_d: dict) -> bool:
    return all(
        cfg_d.get(k) is not None
        for k in ("qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim", "kv_lora_rank")
    )


def attention_params_per_layer(cfg_d: dict, d_model: int, n_heads: int, n_kv_heads: int, d_head: int) -> int:
    if is_mla(cfg_d):
        kv_lora = cfg_d["kv_lora_rank"]
        qk_rope = cfg_d["qk_rope_head_dim"]
        qk_nope = cfg_d["qk_nope_head_dim"]
        v_head = cfg_d["v_head_dim"]
        q_head_dim = qk_rope + qk_nope
        base = (
            d_model * (kv_lora + qk_rope)
            + kv_lora * n_heads * (q_head_dim - qk_rope + v_head)
            + v_head * n_heads * d_model
        )
        q_lora = cfg_d.get("q_lora_rank")
        if q_lora:
            q = d_model * q_lora + q_lora * n_heads * q_head_dim
        else:
            q = d_model * n_heads * q_head_dim
        return base + q
    return d_model * (n_heads * d_head + 2 * n_kv_heads * d_head) + n_heads * d_head * d_model


def kv_per_token_entries(cfg_d: dict, n_layers: int, d_head: int, n_kv_heads: int) -> int:
    if is_mla(cfg_d):
        return n_layers * (cfg_d["kv_lora_rank"] + cfg_d["qk_rope_head_dim"])
    return 2 * n_layers * d_head * n_kv_heads


B300_KEY = "NVIDIA-B300-269GB"


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

MODEL_REPO_MAP = {
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "unsloth/gpt-oss-120b": "openai/gpt-oss-120b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "unsloth/gpt-oss-20b": "openai/gpt-oss-20b",
    "moonshotai/Kimi-K2.5": "moonshotai/Kimi-K2.5",
    "deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2-Exp",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
}

PRECISION_BYTES = {
    "float32": 4.0, "fp32": 4.0,
    "float16": 2.0, "fp16": 2.0,
    "bfloat16": 2.0, "bf16": 2.0,
    "int8": 1.0, "fp8": 1.0,
    "int4": 0.5, "fp4": 0.5, "mxfp4": 0.5, "awq": 0.5, "gptq": 0.5,
}

QUANT_METHOD_TO_PRECISION = {
    "fp8": "fp8",
    "mxfp4": "fp4",
    "fp4": "fp4",
    "awq": "int4",
    "gptq": "int4",
}


def resolve_precision(metadata_precision: str, cfg_d: dict) -> tuple[str, str]:
    qc = cfg_d.get("quantization_config") or {}
    quant = (qc.get("quant_method") or "").lower()
    if quant in QUANT_METHOD_TO_PRECISION:
        return QUANT_METHOD_TO_PRECISION[quant], f"quantization_config.quant_method={quant}"
    if quant == "compressed-tensors":
        groups = qc.get("config_groups") or {}
        for g in groups.values():
            w = (g or {}).get("weights") or {}
            nb = w.get("num_bits")
            wtype = (w.get("type") or "").lower()
            if nb is None:
                continue
            kind = "int" if wtype.startswith("int") else "fp"
            if nb == 4:
                return f"{kind}4", f"quantization_config.compressed-tensors num_bits=4 type={wtype or 'float'}"
            if nb == 8:
                return f"{kind}8", f"quantization_config.compressed-tensors num_bits=8 type={wtype or 'float'}"
    return (metadata_precision or "bfloat16"), "metadata.model_config.precision"


GPU_DIR_RE = re.compile(r"^([a-z][a-z0-9-]*?)x(\d+)(?:[_-].*)?$")  # hyphen allows blackhole-p150b


def parse_gpu_dir(name: str) -> Optional[tuple[str, int]]:
    m = GPU_DIR_RE.match(name.lower())
    if not m:
        return None
    return m.group(1), int(m.group(2))


@lru_cache(maxsize=None)
def load_hf_config(repo_id: str):
    import urllib.request

    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=30).read())
    except Exception as e:
        print(f"  [warn] cannot fetch config for {repo_id}: {e}", file=sys.stderr)
        return None

    if (
        isinstance(data.get("text_config"), dict)
        and data["text_config"].get("num_hidden_layers")
    ):
        data = data["text_config"]

    class _Cfg:
        def __init__(self, d):
            self._d = d
            for k, v in d.items():
                setattr(self, k, v)

        def __getattr__(self, k):
            return self._d.get(k)

    return _Cfg(data)


def describe_run(path: Path, root: Path) -> dict:
    rel = path.relative_to(root).parts
    gpu_idx = next(
        (i for i, p in enumerate(rel) if parse_gpu_dir(p) is not None), -1
    )
    parsed = parse_gpu_dir(rel[gpu_idx]) if gpu_idx >= 0 else None
    return {
        "location": rel[gpu_idx - 4] if gpu_idx >= 4 else "",
        "framework": rel[gpu_idx - 3] if gpu_idx >= 3 else "",
        "model": rel[gpu_idx - 2] if gpu_idx >= 2 else "",
        "dataset": rel[gpu_idx - 1] if gpu_idx >= 1 else "",
        "gpu_key": parsed[0] if parsed else "",
        "num_gpus_path": parsed[1] if parsed else 0,
        "batch_size_dir": rel[gpu_idx + 1] if gpu_idx + 1 < len(rel) - 1 else "",
    }


def _swap_prefix(name: str, old: str, new: str) -> str:
    if name == f"{old}.json":
        return f"{new}.json"
    if name.startswith(f"{old}_"):
        return new + name[len(old):]
    return f"{new}_{name}"


def metadata_path_for(metrics_path: Path) -> Path:
    return metrics_path.with_name(
        _swap_prefix(metrics_path.name, "metrics", "metadata")
    )


def sparsity_path_for(metrics_path: Path) -> Path:
    return metrics_path.with_name(
        _swap_prefix(metrics_path.name, "metrics", "sparsity")
    )


def find_metrics_files(root: Path) -> list[Path]:
    seen, out = set(), []
    for pat in ("metrics_*.json", "metrics.json"):
        for p in root.rglob(pat):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


def resolve_gpu_key(meta_gpu: str, fallback_key: str) -> Optional[str]:
    if meta_gpu in GPU_TYPE_MAP:
        return GPU_TYPE_MAP[meta_gpu]
    if meta_gpu and meta_gpu != "Unknown":
        for k in GPU_TYPE_MAP:
            if meta_gpu in k or k in meta_gpu:
                return GPU_TYPE_MAP[k]
    return GPU_KEY_FALLBACK.get(fallback_key)


def precision_bytes(prec: str) -> float:
    return PRECISION_BYTES.get((prec or "").lower(), 2.0)


def _layer_partition(cfg_d: dict, n_layers: int) -> tuple[int, int]:
    """Return (num_moe_layers, num_dense_layers) following MoE-CAP conventions."""
    mt = cfg_d.get("model_type") or ""
    deepseek_family = mt in ("deepseek_v3", "deepseek_v32", "kimi_k2") or "DeepSeek" in (
        cfg_d.get("_name_or_path") or ""
    )
    if deepseek_family:
        first_dense = cfg_d.get("first_k_dense_replace", 0) or 0
        return n_layers - first_dense, first_dense
    if mt == "qwen3_moe":
        mlp_only = set(cfg_d.get("mlp_only_layers") or [])
        step = cfg_d.get("decoder_sparse_step", 1) or 1
        n_moe = sum(
            1 for li in range(n_layers)
            if (li not in mlp_only) and ((li + 1) % step == 0)
        )
        return n_moe, n_layers - n_moe
    # A config with no expert count is a dense model: every FFN layer is loaded on
    # every step, so all layers are dense and none carries experts or a router.
    if not any(cfg_d.get(k) for k in (
        "num_local_experts", "num_experts", "n_routed_experts", "num_experts_per_tok"
    )):
        return 0, n_layers
    return n_layers, 0


def compute_for_run(
    metrics: dict, hf_cfg, gpu_key: str, num_gpus: int, prec_str: str,
    avg_prefill_len: float, avg_decode_ctx_len: float,
    force_batch_size_one: bool = False,
    concurrent: bool = False,
    checkpoint_dtype_served: bool = True,
) -> dict:
    perf = metrics.get("performance") or {}
    ea = metrics.get("expert_activation") or {}
    # Expert activation is used only by S-MBU, and only the run's own trace is used. It counts the
    # experts a step actually loaded, which depends on the realised batch, so it is a property of
    # this engine on this accelerator and does not carry across from another run: within one model,
    # workload and batch regime it varies several-fold across accelerators. Only SGLang's runner
    # records it, and not on every run; a run without one publishes a null S-MBU, not an imputed one.
    # Dense models are the exception: with no experts to activate, every FFN is loaded on every
    # step and S-MBU is computed from the config alone (see is_dense below).
    # S-MFU is unaffected -- it uses the architectural top_k, not the measured activation.
    prefill_act = ea.get("avg_expert_activation_prefill") or 0
    decode_act = ea.get("avg_expert_activation_decode") or 0
    has_activation = prefill_act > 0 or decode_act > 0
    ttft = perf.get("ttft")
    tpot = perf.get("tpot")
    batch_profile = metrics.get("batch_token_profile") or {}
    prefill_tps = perf.get("prefill_tokens_per_s") or 0
    output_tps = perf.get("output_tokens_per_s") or 0

    profile_prefill_len = batch_profile.get("prefill_tokens_per_request")
    profile_prefill_bs = batch_profile.get("prefill_avg_batch_size")
    profile_decode_tokens = batch_profile.get("decode_generated_tokens_per_request")
    profile_decode_bs = batch_profile.get("decode_avg_batch_size")

    if force_batch_size_one:
        profile_prefill_bs = 1.0
        profile_decode_bs = 1.0

    if isinstance(profile_prefill_len, (int, float)) and profile_prefill_len > 0:
        avg_prefill_len = float(profile_prefill_len)
    if (
        isinstance(profile_prefill_len, (int, float)) and profile_prefill_len > 0
        and isinstance(profile_decode_tokens, (int, float)) and profile_decode_tokens > 0
    ):
        avg_decode_ctx_len = float(profile_prefill_len) + float(profile_decode_tokens) / 2.0

    if ttft is None or tpot is None or not ttft or not tpot:
        return {"skipped": "ttft/tpot missing"}

    model_name = hf_cfg._d.get("_name_or_path", "")
    cfg_d = hf_cfg._d
    resolved_prec, prec_source = resolve_precision(prec_str, cfg_d)
    prec_str = resolved_prec
    # The recorded precision is the checkpoint's dtype, which is what vLLM/SGLang serve. An
    # engine with its own load-time quantization may serve something narrower the profiler
    # cannot see, so on those engines a checkpoint-inherited precision is not evidence of the
    # bytes actually moved: withhold the precision-dependent metrics until the run attests
    # its serving precision (a quantization_config on the checkpoint still counts — the
    # engine loads those weights as stored).
    precision_evidenced = (
        checkpoint_dtype_served or prec_source != "metadata.model_config.precision"
    )

    n_layers = cfg_d.get("num_hidden_layers")
    d_model = cfg_d.get("hidden_size") or cfg_d.get("d_model")
    n_attn_heads = cfg_d.get("num_attention_heads") or cfg_d.get("n_heads")
    n_kv_heads = cfg_d.get("num_key_value_heads") or n_attn_heads
    d_head = cfg_d.get("head_dim") or (d_model // n_attn_heads if n_attn_heads else None)
    d_ff_moe = cfg_d.get("moe_intermediate_size") or cfg_d.get("intermediate_size")
    d_ff_dense = cfg_d.get("intermediate_size") or d_ff_moe
    top_k = (
        cfg_d.get("num_experts_per_tok")
        or cfg_d.get("moe_top_k")
        or cfg_d.get("router_topk")
        or cfg_d.get("n_routed_experts")
        or 1
    )
    n_shared = cfg_d.get("n_shared_experts") or cfg_d.get("num_shared_experts") or 0
    n_experts_total = (
        cfg_d.get("num_local_experts")
        or cfg_d.get("num_experts")
        or cfg_d.get("n_routed_experts")
        or 1
    )

    prec_bytes = precision_bytes(prec_str)

    attn_params_layer = attention_params_per_layer(
        cfg_d, d_model, n_attn_heads, n_kv_heads, d_head
    )
    attn_size_per_token_TB = attn_params_layer / 1e12

    expert_params = d_ff_moe * 3 * d_model
    shared_expert_params = n_shared * d_ff_moe * 3 * d_model
    dense_ffn_params = d_ff_dense * 3 * d_model
    router_params = d_model * n_experts_total
    expert_size_TB = expert_params / 1e12
    shared_size_TB = shared_expert_params / 1e12
    dense_size_TB = dense_ffn_params / 1e12
    router_size_TB = router_params / 1e12

    num_moe_layers, num_dense_layers = _layer_partition(cfg_d, n_layers)

    per_token_kv = kv_per_token_entries(cfg_d, n_layers, d_head, n_kv_heads)
    kv_size_prefill_TB = (avg_prefill_len * per_token_kv) / 1e12 if avg_prefill_len else 0.0
    kv_size_decode_TB = (avg_decode_ctx_len * per_token_kv) / 1e12 if avg_decode_ctx_len else 0.0

    peak_bw = get_peak_bw(gpu_key)
    peak_flops = get_peak_flops(gpu_key, precision=prec_str.lower())
    # Bandwidth gates the whole sidecar; a missing FLOPS peak nulls S-MFU alone, so a card
    # with a published bandwidth but no per-precision compute figure still gets S-MBU.
    if peak_bw <= 0:
        return {
            "skipped": f"no hardware spec for {gpu_key} @ {prec_str}",
            "gpu_key": gpu_key,
        }

    def smbu(activation: float, time_s: float, kv_TB: float) -> float:
        # `activation` is the recorder's mean of unique routed experts over ALL
        # layers (dense layers contribute zeros), so total routed-expert loads
        # are activation x n_layers; shared experts and routers exist per MoE
        # layer only. On all-MoE models the two forms coincide.
        bytes_loaded_TB = (
            n_layers * attn_size_per_token_TB
            + n_layers * activation * expert_size_TB
            + num_moe_layers * (shared_size_TB + router_size_TB)
            + num_dense_layers * dense_size_TB
            + kv_TB
        )
        bytes_per_s = bytes_loaded_TB * 1e12 * prec_bytes / time_s
        return bytes_per_s / (num_gpus * peak_bw)

    def smfu(throughput_tok_s: float) -> float:
        params_per_token_TB = (
            n_layers * attn_size_per_token_TB
            + num_moe_layers * (top_k * expert_size_TB + shared_size_TB + router_size_TB)
            + num_dense_layers * dense_size_TB
        )
        flops_per_token = params_per_token_TB * 1e12 * 2
        flops_per_s = flops_per_token * throughput_tok_s
        # PEAK_FLOPS_DICT is dense for every vendor, so the peak divides in as written.
        return flops_per_s / (num_gpus * peak_flops)

    if (
        isinstance(profile_prefill_len, (int, float)) and profile_prefill_len > 0
        and isinstance(profile_prefill_bs, (int, float)) and profile_prefill_bs > 0
    ):
        # per-request prefill rate: len / ttft. NOT x prefill_bs — ttft is one request's
        # first-token latency, so the batch's prefills are not all realised within it (that
        # over-counted throughput -> S-MFU > 100%). Matches the else-branch's avg_prefill_len/ttft.
        prefill_tp = float(profile_prefill_len) / ttft
    else:
        prefill_tp = prefill_tps or (avg_prefill_len / ttft if avg_prefill_len else 0)
    # S-MBU wants the duration of ONE forward pass, not one request's time to first token: it
    # divides the bytes a pass moves by the time that pass took. Runs from before the harness split
    # the two report the per-pass mean under `ttft`, so fall back to it and keep them comparable.
    pass_s = perf.get("prefill_pass_latency_s") or ttft
    # A dense model needs no activation trace: with zero MoE layers the bytes term is
    # attention + full FFN + KV, all from the config, so S-MBU reduces to plain MBU.
    is_dense = num_moe_layers == 0
    prefill_smbu = (
        smbu(prefill_act, pass_s, kv_size_prefill_TB)
        if (prefill_act > 0 or is_dense) and precision_evidenced else None
    )
    prefill_smfu = (
        smfu(prefill_tp)
        if prefill_tp > 0 and peak_flops > 0 and precision_evidenced else None
    )

    if isinstance(profile_decode_bs, (int, float)) and profile_decode_bs > 0:
        decoding_tp = float(profile_decode_bs) / tpot
    elif concurrent:
        # A concurrent run without a measured decode batch has no node-level rate:
        # 1/tpot is a single-stream rate, several-fold under what the node served.
        # Publishing it would silently misprice everything downstream (energy divides
        # node power by this rate), so withhold the rate instead.
        decoding_tp = None
    else:
        decoding_tp = output_tps or (1.0 / tpot)
    # The decode KV term needs the run's own context length; without a usable token
    # count avg_decode_ctx_len is 0 and S-MBU would silently publish a KV-less value
    # that understates long-context runs. Withhold it instead.
    decode_smbu = (
        smbu(decode_act, tpot, kv_size_decode_TB)
        if (decode_act > 0 or is_dense) and avg_decode_ctx_len > 0
        and precision_evidenced else None
    )
    decode_smfu = (
        smfu(decoding_tp) if decoding_tp and peak_flops > 0 and precision_evidenced else None
    )

    # Backstop: utilisation > 100% is physically impossible (bad clock, cache-inflated
    # prefill accounting, or a mislabeled precision peak). Null it rather than emit an
    # impossible number the radar/methodology would show as real.
    def _cap(x):
        return None if (x is not None and x > 1.0) else x
    prefill_smbu, prefill_smfu = _cap(prefill_smbu), _cap(prefill_smfu)
    decode_smbu, decode_smfu = _cap(decode_smbu), _cap(decode_smfu)

    hardware_specs = {
        "peak_bandwidth_tb": peak_bw / 1e12,
        "peak_flops_tf": peak_flops / 1e12,
    }

    return {
        "inputs": {
            "model": model_name,
            "model_type": cfg_d.get("model_type"),
            "n_layers": n_layers,
            "num_moe_layers": num_moe_layers,
            "num_dense_layers": num_dense_layers,
            "hidden_size": d_model,
            "n_attn_heads": n_attn_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": d_head,
            "moe_ffn": d_ff_moe,
            "dense_ffn": d_ff_dense,
            "top_k": top_k,
            "n_shared_experts": n_shared,
            "n_experts_total": n_experts_total,
            "precision": prec_str,
            "precision_bytes": prec_bytes,
            "precision_source": prec_source,
            "gpu_key": gpu_key,
            "num_gpus": num_gpus,
            "peak_bandwidth_tb_s": hardware_specs["peak_bandwidth_tb"],
            "peak_flops_tf_s": hardware_specs["peak_flops_tf"],
            "peak_flops_basis": PEAK_FLOPS_BASIS,
        },
        "activation": {
            "avg_expert_activation_prefill": prefill_act if has_activation else None,
            "avg_expert_activation_decode": decode_act if has_activation else None,
            "activation_source": (
                "measured" if has_activation
                else "not-applicable-dense" if is_dense
                else "unavailable"
            ),
        },
        "prefill": {
            "ttft_s": ttft,
            "prefill_tokens_per_s": prefill_tp,
            "S_MBU": prefill_smbu,
            "S_MFU": prefill_smfu,
        },
        "decode": {
            "tpot_s": tpot,
            "output_tokens_per_s": decoding_tp,
            "S_MBU": decode_smbu,
            "S_MFU": decode_smfu,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).parent,
        help="Tree to scan (default: this script's parent)",
    )
    parser.add_argument(
        "--avg-prefill-len", type=float, default=0.0,
        help="Avg prefill tokens per request (for KV-cache term). 0 = skip.",
    )
    parser.add_argument(
        "--avg-decode-ctx-len", type=float, default=0.0,
        help="Avg context length during decode (for KV-cache term). 0 = skip.",
    )
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


    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    recorded_at = now.isoformat().replace("+00:00", "Z")

    written = 0
    skipped = 0
    skip_reasons: dict[str, int] = {}

    for f in metrics_files:
        info = describe_run(f, root)
        try:
            metrics = json.loads(f.read_text())
        except Exception as e:
            print(f"  [warn] cannot read {f}: {e}", file=sys.stderr)
            skipped += 1
            continue

        meta_path = metadata_path_for(f)
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass

        hw = meta.get("hardware") or {}
        mc = meta.get("model_config") or {}
        meta_gpu = hw.get("gpu_type", "")
        # TEAS result paths encode the benchmark hardware shape (for example
        # h200x8). Some compact metadata files were backfilled with
        # hardware.num_gpus=1; do not let stale metadata collapse xN runs to
        # single-GPU peak bandwidth/FLOP denominators.
        meta_num_gpus = info["num_gpus_path"] or hw.get("num_gpus") or 1
        prec_str = mc.get("precision") or "bfloat16"
        model_name_meta = mc.get("model_name") or ""

        gpu_key = resolve_gpu_key(meta_gpu, info["gpu_key"])
        if not gpu_key:
            skipped += 1
            skip_reasons["no_gpu_key"] = skip_reasons.get("no_gpu_key", 0) + 1
            continue

        repo = MODEL_REPO_MAP.get(model_name_meta) or model_name_meta
        if not repo:
            skipped += 1
            skip_reasons["no_model"] = skip_reasons.get("no_model", 0) + 1
            continue

        hf_cfg = load_hf_config(repo)
        if hf_cfg is None or not getattr(hf_cfg, "num_hidden_layers", None):
            skipped += 1
            skip_reasons["no_hf_config"] = skip_reasons.get("no_hf_config", 0) + 1
            continue

        profile = dataset_token_profile(info["dataset"])
        if args.avg_prefill_len > 0:
            prefill_len = args.avg_prefill_len
        elif profile:
            prefill_len = float(profile["avg_input"])
        else:
            prefill_len = 0.0
        if args.avg_decode_ctx_len > 0:
            decode_ctx_len = args.avg_decode_ctx_len
        elif profile:
            decode_ctx_len = profile["avg_input"] + profile["avg_output"] / 2.0
        else:
            decode_ctx_len = 0.0

        result = compute_for_run(
            metrics, hf_cfg, gpu_key, meta_num_gpus, prec_str,
            prefill_len, decode_ctx_len,
            force_batch_size_one=info["batch_size_dir"].startswith("batch-size-1"),
            concurrent=info["batch_size_dir"].startswith("batch-size-default"),
            # kai (KernelAgentIR on Tenstorrent) quantizes at load time, so the checkpoint
            # dtype the profiler records is not what the engine necessarily served.
            checkpoint_dtype_served=info["framework"] != "kai",
        )
        if "skipped" in result:
            skipped += 1
            reason = str(result["skipped"]).split(",")[0]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            # A withheld ttft/tpot invalidates any sidecar computed from the run's earlier
            # values; remove it so the skip is effective, not shadowed by a stale file.
            # Environmental skips (missing hardware spec, and the config/model skips above)
            # keep the existing sidecar — a transient lookup failure must not delete it.
            if reason == "ttft/tpot missing":
                stale = sparsity_path_for(f)
                if stale.exists() and not args.dry_run:
                    stale.unlink()
            continue

        payload = {
            "recorded_at": recorded_at,
            "run": {
                "location": info["location"],
                "framework": info["framework"],
                "model": info["model"],
                "model_repo_used": repo,
                "dataset": info["dataset"],
                "gpu_metadata_string": meta_gpu,
            },
            "sparsity": result,
        }

        out_path = sparsity_path_for(f)
        if args.dry_run:
            print(f"  [dry-run] would write {out_path.relative_to(root.parent)}")
        else:
            out_path.write_text(json.dumps(payload, indent=2) + "\n")
        written += 1

    print(f"\nWrote {written} sparsity JSON files (skipped {skipped})")
    for k, v in sorted(skip_reasons.items()):
        print(f"  skip[{k}] = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
