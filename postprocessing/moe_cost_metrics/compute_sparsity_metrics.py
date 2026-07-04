#!/usr/bin/env python3
"""
Compute S-MBU and S-MFU per the MoE-CAP paper (arXiv 2412.07067 v6, Eqs. 4-5)
for every metrics file in the moe/ tree that has non-zero expert_activation.

Per the paper:
  S-MBU = B_achieved / B_peak,  B_achieved = (S_activated + S_KV) / TPOT
  S-MFU = (T_token * S_F_token) / F_peak,
          S_F_token = F_attn + 2*N_router + 2*k_expert*N_expert

Where S_activated counts only the experts actually loaded
  (avg_expert_activation_{prefill,decode} from the runner's trace).

This script delegates to MoE-CAP's own per-model calculators
(qwen_utils, qwen3_utils, deepseek_utils, default for gpt-oss / Mixtral)
so the answers match `moe_cap` exactly. KV-cache and attention-compute
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

sys.path.insert(0, "/home/sicheng/MoE-CAP")

from moe_cap.utils.hardware_utils import (  # noqa: E402
    MEM_BW_DICT,
    PEAK_FLOPS_DICT,
    get_peak_bw,
    get_peak_flops,
)


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


B200_KEY = "NVIDIA-B200-183GB"
B300_KEY = "NVIDIA-B300-269GB"

MEM_BW_DICT.setdefault(B300_KEY, 8000e9)
MEM_BW_DICT[B200_KEY] = 8000e9

for _p in PEAK_FLOPS_DICT:
    PEAK_FLOPS_DICT[_p].setdefault(
        B300_KEY, PEAK_FLOPS_DICT[_p].get(B200_KEY, 0)
    )

PEAK_FLOPS_DICT["bfloat16"][B200_KEY] = 4500e12
PEAK_FLOPS_DICT["float16"][B200_KEY] = 4500e12
PEAK_FLOPS_DICT["fp8"][B200_KEY] = 9000e12
PEAK_FLOPS_DICT["int8"][B200_KEY] = 9000e12
PEAK_FLOPS_DICT["fp4"][B200_KEY] = 18000e12
PEAK_FLOPS_DICT["int4"][B200_KEY] = 18000e12

PEAK_FLOPS_DICT["bfloat16"][B300_KEY] = 5000e12
PEAK_FLOPS_DICT["float16"][B300_KEY] = 5000e12
PEAK_FLOPS_DICT["fp8"][B300_KEY] = 10000e12
PEAK_FLOPS_DICT["int8"][B300_KEY] = 10000e12
PEAK_FLOPS_DICT["fp4"][B300_KEY] = 20000e12
PEAK_FLOPS_DICT["int4"][B300_KEY] = 20000e12


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
}

GPU_KEY_FALLBACK = {
    "a100": "NVIDIA-A100-SXM4-80GB",
    "h100": "NVIDIA-H100-HBM3-80GB",
    "h200": "NVIDIA-H200-141GB",
    "b200": "NVIDIA-B200-183GB",
    "b300": B300_KEY,
    "mi355x": "AMD-Instinct-MI355X-288GB",
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


GPU_DIR_RE = re.compile(r"^([a-z][a-z0-9]*?)x(\d+)(?:[_-].*)?$")


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


def build_activation_lookup(metrics_files: list[Path], root: Path) -> dict[tuple, tuple[float, float, str]]:
    table: dict[tuple, tuple[float, float, str]] = {}
    for f in metrics_files:
        info = describe_run(f, root)
        try:
            metrics = json.loads(f.read_text())
        except Exception:
            continue
        ea = metrics.get("expert_activation") or {}
        p = ea.get("avg_expert_activation_prefill") or 0
        de = ea.get("avg_expert_activation_decode") or 0
        if p <= 0 and de <= 0:
            continue
        key = (info["model"], info["dataset"], info["batch_size_dir"])
        if key not in table:
            table[key] = (p, de, f"{info['location']}/{info['framework']}/{Path(f).parent.relative_to(root)}")
    return table


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
    return n_layers, 0


def compute_for_run(
    metrics: dict, hf_cfg, gpu_key: str, num_gpus: int, prec_str: str,
    avg_prefill_len: float, avg_decode_ctx_len: float,
    borrowed_activation: Optional[tuple[float, float, str]] = None,
) -> dict:
    perf = metrics.get("performance") or {}
    ea = metrics.get("expert_activation") or {}
    prefill_act = ea.get("avg_expert_activation_prefill") or 0
    decode_act = ea.get("avg_expert_activation_decode") or 0
    if prefill_act <= 0 and decode_act <= 0 and borrowed_activation is not None:
        prefill_act, decode_act, _activation_source = borrowed_activation
    ttft = perf.get("ttft")
    tpot = perf.get("tpot")
    prefill_tps = perf.get("prefill_tokens_per_s") or 0
    output_tps = perf.get("output_tokens_per_s") or 0

    if prefill_act <= 0 and decode_act <= 0:
        return {"skipped": "expert_activation missing or zero"}
    if ttft is None or tpot is None or not ttft or not tpot:
        return {"skipped": "ttft/tpot missing"}

    model_name = hf_cfg._d.get("_name_or_path", "")
    cfg_d = hf_cfg._d
    resolved_prec, prec_source = resolve_precision(prec_str, cfg_d)
    prec_str = resolved_prec

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
    if peak_bw <= 0 or peak_flops <= 0:
        return {
            "skipped": f"no hardware spec for {gpu_key} @ {prec_str}",
            "gpu_key": gpu_key,
        }

    def smbu(activation: float, time_s: float, kv_TB: float) -> float:
        bytes_loaded_TB = (
            n_layers * attn_size_per_token_TB
            + num_moe_layers * (activation * expert_size_TB + shared_size_TB + router_size_TB)
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
        return flops_per_s / (num_gpus * peak_flops / 2)

    prefill_tp = prefill_tps or (avg_prefill_len / ttft if avg_prefill_len else 0)
    prefill_smbu = smbu(prefill_act, ttft, kv_size_prefill_TB) if prefill_act > 0 else None
    prefill_smfu = smfu(prefill_tp) if prefill_tp > 0 else None

    decoding_tp = output_tps or (1.0 / tpot)
    decode_smbu = smbu(decode_act, tpot, kv_size_decode_TB) if decode_act > 0 else None
    decode_smfu = smfu(decoding_tp) if decoding_tp > 0 else None

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
        },
        "activation": {
            "avg_expert_activation_prefill": prefill_act,
            "avg_expert_activation_decode": decode_act,
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

    activation_lookup = build_activation_lookup(metrics_files, root)
    print(f"Activation lookup built from {len(activation_lookup)} (model, dataset, batch) keys.", file=sys.stderr)

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

        borrowed = activation_lookup.get(
            (info["model"], info["dataset"], info["batch_size_dir"])
        )
        result = compute_for_run(
            metrics, hf_cfg, gpu_key, meta_num_gpus, prec_str,
            prefill_len, decode_ctx_len,
            borrowed_activation=borrowed,
        )
        if "skipped" in result:
            skipped += 1
            reason = str(result["skipped"]).split(",")[0]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
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
