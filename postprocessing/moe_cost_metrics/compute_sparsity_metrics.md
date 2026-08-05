# `compute_sparsity_metrics.py` — S-MBU and S-MFU per MoE-CAP

Writes one `sparsity_*.json` sidecar next to every `metrics_*.json` (or `metrics.json`) the run tree holds. S-MBU needs the run's own `expert_activation` trace and is null without it; S-MFU does not and is always published. Formulas follow [MoE-CAP, arXiv 2412.07067 v6](https://arxiv.org/html/2412.07067v6), Eqs. 4-5.

## 1. Formulas (paper convention)

```
S-MBU = B_achieved / B_peak
      = ((bytes loaded per step) / time_per_step) / (num_gpus × peak_bandwidth)

bytes_per_step =
    n_layers × attn_params_per_token × precision_bytes
  + n_moe_layers   × (activation × expert_params + shared × expert_params + router_params) × precision_bytes
  + n_dense_layers × dense_ffn_params × precision_bytes
  + S_KV × precision_bytes              # KV-cache reads at context length

S-MFU = (T_token × F_token) / F_peak_dense
F_token = 2 × (
    n_layers × attn_params_per_token
  + n_moe_layers × (top_k × expert_params + shared × expert_params + router_params)
  + n_dense_layers × dense_ffn_params
)
F_peak_dense = peak_flops                # table is already dense; see §4
```

`activation` is the **number of distinct experts touched per layer** for that step (from the runner's `avg_expert_activation_{prefill,decode}` trace). `top_k` is the **per-token** activated experts (architecture constant).

## 2. Verified model architectures

Read live from each model's HF `config.json` (cached per repo). Aliases applied:

| Metric `model_name` | HF repo used | Notes |
|---|---|---|
| `openai/gpt-oss-120b`, `unsloth/gpt-oss-120b` | `openai/gpt-oss-120b` | |
| `openai/gpt-oss-20b`, `unsloth/gpt-oss-20b` | `openai/gpt-oss-20b` | |
| `moonshotai/Kimi-K2.5` | `moonshotai/Kimi-K2.5` | arch nested under `text_config` (auto-unwrapped); `kimi_k2` backbone |
| `deepseek-ai/DeepSeek-R1` | `deepseek-ai/DeepSeek-R1` | |
| `deepseek-ai/DeepSeek-V3.2` | `deepseek-ai/DeepSeek-V3.2-Exp` | |
| `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | |

Derived per-model parameters (verified against the live config):

| model | type | layers | (MoE / dense) | hidden | attn (Q/KV heads, head_dim) | top-k + shared | n_experts | d_ff_moe | d_ff_dense | MLA |
|---|---|---:|---|---:|---|---|---:|---:|---:|---|
| gpt-oss-120b | `gpt_oss` | 36 | 36 / 0 | 2880 | 64/8, 64 | 4 + 0 | 128 | 2880 | — | no (GQA) |
| gpt-oss-20b | `gpt_oss` | 24 | 24 / 0 | 2880 | 64/8, 64 | 4 + 0 | 32 | 2880 | — | no (GQA) |
| kimi-k2.5 | `kimi_k2` | 61 | 60 / 1 | 7168 | 64/64, — | 8 + 1 | 384 | 2048 | 18432 | yes (q_lora=1536, kv_lora=512, qk_rope=64, qk_nope=128, v_head=128) |
| deepseek-r1 | `deepseek_v3` | 61 | 58 / 3 | 7168 | 128/128, — | 8 + 1 | 256 | 2048 | 18432 | yes (q_lora=1536, kv_lora=512, qk_rope=64, qk_nope=128, v_head=128) |
| qwen3-235b-A22B | `qwen3_moe` | 94 | 94 / 0 | 4096 | 64/4, 128 | 8 + 0 | 128 | 1536 | 12288 | no (GQA) |

Layer partition rule:
- `deepseek_v3` / `deepseek_v32` / `kimi_k2` → `(n_layers − first_k_dense_replace, first_k_dense_replace)`
- `qwen3_moe` → walk `mlp_only_layers` × `decoder_sparse_step`
- everything else → all MoE, no dense

## 3. Verified precision dispatch

Reads HF `config.json` `quantization_config` (overrides runner metadata, which reports activation dtype not weight storage).

| Model | quant_method | resolved precision | bytes/param |
|---|---|---|---:|
| DeepSeek-R1 | `fp8` (e4m3, block 128×128) | **fp8** | 1.0 |
| Qwen3-235B-A22B-Instruct-2507-FP8 | `fp8` (e4m3, norms+gates excluded) | **fp8** | 1.0 |
| Kimi-K2.5 | `compressed-tensors`, `num_bits=4`, `type=int`, group=32 | **int4** | 0.5 |
| gpt-oss-120b / 20b | `mxfp4` (experts only; self_attn + router kept BF16) | **fp4** | 0.5 |
| anything else | runner metadata `model_config.precision` | bf16 typically | 2.0 |

Provenance recorded in `inputs.precision_source`.

## 4. Verified hardware peak (table = **dense**, no `/2` in the denominator)

TEASBench owns this table rather than importing MoE-CAP's, because vendors quote peak on
different bases and mixing them biases S-MFU across vendors. Every figure below is dense and
per GPU.

| GPU key | Memory⁶ | HBM BW | BF16 | FP8 | INT8 | FP4 | INT4 | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `NVIDIA-A100-SXM4-80GB` | 80 GB | 2.039 TB/s | 312 | 312¹ | 624 | 312¹ | 1248 | A100 datasheet (leads dense; 2× sparse footnoted) |
| `NVIDIA-H100-HBM3-80GB` | 80 GB | 3.35 TB/s | 989.5 | 1979 | 1979 | 1979² | 1979 | H100 datasheet (leads with-sparsity) |
| `NVIDIA-H200-141GB` | 141 GB | 4.80 TB/s | 989.5 | 1979 | 1979 | 1979² | 1979 | H200 datasheet (same die as H100) |
| `NVIDIA-B200-183GB` | 192 GB | 7.70 TB/s | 2250 | 4500 | 4500 | 9000 | 9000³ | Blackwell datasheet, HGX B200 per-GPU column |
| `NVIDIA-B300-269GB` | 288 GB | 7.70 TB/s | 2250 | 4500 | **153.5** | **14000** | 14000³ | Blackwell Ultra datasheet, HGX B300 per-GPU column |
| `AMD-Instinct-MI355X-288GB` | 288 GB | 8.00 TB/s | 2500 | 5000 | 5000 | 10100 | 10100³ | MI355X product page, dense rows as published |
| `NVIDIA-GB10` | 128 GB | 0.273 TB/s | 125⁴ | 250⁴ | 250⁴ | 500⁴ | 500³ | DGX Spark page (one published figure) |
| `Tenstorrent-Blackhole-P150b` | 32 GB | 0.512 TB/s | — | — | — | — | — | Tenstorrent specifications⁵ |

¹ No FP8/FP4 tensor path on Ampere; falls back to the BF16 rate.
² No FP4 tensor path on Hopper; falls back to the FP8 rate.
³ Weight-only 4-bit checkpoints dequantise before the matmul, so this mirrors the card's FP4
rate as a modelling assumption rather than a datasheet figure. Only A100 has a native INT4
path. See the `int4` comment in the script.
⁴ Derived, not datasheet. NVIDIA publishes a single GB10 figure — 1 PFLOP FP4 with sparsity.
Dense FP4 halves it, and each wider precision halves again, mirroring the B200 ladder.
⁵ No dense FLOPS figure published: Tenstorrent quotes only a BLOCKFP8 rate (664 TFLOPS at the
120-core spec). Blackhole runs therefore publish S-MBU with a null S-MFU rather than divide by
a denominator we cannot defend.
⁶ Nameplate memory capacity per device, from the same vendor pages the Source column names.
These cells carry that source but **not** the cell-by-cell datasheet confirmation the bandwidth
and FLOPS columns have — they are the figures the dashboard was already publishing, moved into
the catalog so that they are checkable in one place rather than restated in the assembler. No
metric on this page divides by them; they are published as a hardware spec.

**Two B300 rows sit apart from the pattern.** Every other Blackwell row is published with
sparsity and halved here; FP4 is the one NVIDIA prints as `sparse | dense` outright, `18 | 14`
PFLOPS per GPU, so its dense figure is taken as printed. Sparse is therefore 1.29× dense
rather than 2×, and the uplift over B200 is on the dense figure. And INT8 is 307 TOPS per GPU
with sparsity against B200's 9 POPS — a ~29× narrower tensor path, so 153.5 TFLOPS dense.

**The Blackwell figures are the per-GPU HGX columns.** The measured parts report
`NVIDIA-B300-SXM6-AC-269GB` — air-cooled SXM modules, the HGX form factor. The NVL72 trays run
higher clocks and are quoted above HGX on several rows (FP4 15 PFLOPS dense, INT8 330 TOPS,
bandwidth 8.0 TB/s), so the HGX column is the one that matches the hardware and keeps B200 and
B300 comparable. Where a datasheet's board total disagrees with its own per-GPU row — HGX B300
FP4 totals `144 | 108` PFLOPS across 8 GPUs, implying 13.5 dense against the 14 printed
per GPU — the per-GPU row governs, because S-MFU divides by `num_gpus × peak`.

FLOPS columns are TFLOPS per GPU; the other two columns carry their units in the cell. Each precision dict in the script is a complete literal with nothing patched over it after the table is built, so the dispatch either finds a documented value or returns zero — as it does for Blackhole, em-dashed in every FLOPS column above, whose runs publish a null S-MFU rather than a guess.

**A capacity in a key is not a capacity reading.** Several keys carry a device-reported figure
that differs from the nameplate in the Memory column — B200 keyed `183GB` against 192, B300
keyed `269GB` against 288, and H200 reaching the map as both `140GB` and `141GB`. The number in
a key is part of an identifier: keys are resolved by lookup and containment against
`GPU_TYPE_MAP`, and nothing in the producers or the assembler parses a capacity out of one. That
is what keeps a nameplate table and a device-reported figure from ending up in one computation.

Metadata `gpu_type` strings normalized to the keys above; raw variants seen and mapped:
`AMD-Instinct-MI355X`, `AMD-Instinct-MI355X-288GB`, `AMD--288GB`, `NVIDIA-A100-SXM4-80GB`, `NVIDIA-B200-180GB`, `NVIDIA-B200-183GB`, `NVIDIA-B300-SXM6-AC-269GB`, `NVIDIA-H100-HBM3-80GB`, `NVIDIA-H200-140GB`, `NVIDIA-H200-141GB`. `Unknown` falls back to the path prefix (`a100|h100|h200|b200|b300|mi355x`).

## 5. Per-dataset token defaults

Used as fallbacks to materialize the KV-cache term (S-MBU) and derive throughput when `batch_token_profile` is absent. When present, `batch_token_profile` is authoritative for default batching: `prefill_tokens_per_s = prefill_tokens_per_request × prefill_avg_batch_size / ttft`, and `decode output_tokens_per_s = decode_avg_batch_size / tpot`. For `batch-size-1` and `batch-size-1_input..._output...` result directories, the effective prefill/decode batch size is forced to `1` even if a historical profile block contains larger averages. Override fallback lengths globally with `--avg-prefill-len` / `--avg-decode-ctx-len`.

| dataset prefix | avg prefill | avg decode ctx = prefill + output/2 |
|---|---:|---:|
| `gsm8k`        |     60 |    210 |
| `arena-hard`   |    110 |    580 |
| `longbench_v1` | 10,000 | 10,110 |
| `longbench_v2` | 10,000 | 10,110 |

## 6. Activation is per-run; S-MBU is null without it

Expert activation counts the experts a decode step actually loaded. That depends on the realised
batch, so it is a property of *this* engine on *this* accelerator and is not transferable between
runs. Only sglang's runner records it, and not on every run; no vllm run currently carries one.

A run without its own trace publishes `activation_source: "unavailable"`, null activation values and
a **null S-MBU**, rather than an imputed number. `inputs.precision_source` and this field are the two
provenance markers on the sidecar.

**S-MFU is unaffected.** It uses the architectural `top_k`, not the measured activation, so every run
publishes it — including the ones with no trace, which previously produced no sidecar at all.

Within a single `(model, dataset, batch_size_dir)` group the measured activations vary several-fold
across accelerators, which is why one run's value cannot stand in for another's.

## 7. Output schema (`sparsity_*.json`)

```jsonc
{
  "recorded_at": "<UTC ISO>",
  "run": {
    "location": "amd | eidf | vastai",
    "framework": "vllm | sglang",
    "model": "<dir name>",
    "model_repo_used": "openai/gpt-oss-120b",
    "dataset": "gsm8k_256samples",
    "gpu_metadata_string": "AMD-Instinct-MI355X-288GB"
  },
  "sparsity": {
    "inputs": {
      "model_type": "gpt_oss | kimi_k2 | deepseek_v3 | qwen3_moe | ...",
      "n_layers": 36,
      "num_moe_layers": 36,
      "num_dense_layers": 0,
      "hidden_size": 2880,
      "n_attn_heads": 64,
      "n_kv_heads": 8,
      "head_dim": 64,
      "moe_ffn": 2880,
      "dense_ffn": 2880,
      "top_k": 4,
      "n_shared_experts": 0,
      "n_experts_total": 128,
      "precision": "fp4",
      "precision_bytes": 0.5,
      "precision_source": "quantization_config.quant_method=mxfp4",
      "gpu_key": "NVIDIA-B300-269GB",
      "num_gpus": 1,
      "peak_bandwidth_tb_s": 7.7,
      "peak_flops_tf_s": 14000.0,
      "peak_flops_basis": "dense"
    },
    "activation": {
      "activation_source": "measured",
      "avg_expert_activation_prefill": 75.39,
      "avg_expert_activation_decode": 4.00
    },
    "prefill": {
      "ttft_s": 0.079,
      "prefill_tokens_per_s": 126428.15,
      "S_MBU": 0.0565,
      "S_MFU": 0.1110
    },
    "decode": {
      "tpot_s": 0.00518,
      "output_tokens_per_s": 193.11,
      "S_MBU": 0.0617,
      "S_MFU": 0.00020
    }
  }
}
```

## 8. Verified counts (current run)

The script prints its own tallies on each run — metrics files found, sidecars written,
`measured` against `unavailable`, and the skip reasons. Read them from the run rather than
from here, so this section cannot go stale.

Only sglang's runner records the trace, and not on every run; no vllm run currently carries
one. Runs whose `gpu_key` does not resolve are skipped before a sidecar is written, so the
`measured` count sits slightly below the number of metrics files holding a trace.

## 9. Reproduction

```bash
# MoE-CAP is imported as a library (hardware peak tables, attention-size helpers)
cd /home/sicheng/MoE-CAP
python3 /home/sicheng/TEAS_Development_Results_Private/moe/compute_sparsity_metrics.py \
  --root /home/sicheng/TEAS_Development_Results_Private/moe

# Override per-dataset token counts globally:
python3 .../compute_sparsity_metrics.py \
  --avg-prefill-len 8000 --avg-decode-ctx-len 8500
```

Outputs are sidecar files: `sparsity_<suffix>.json` (or `sparsity.json`) next to each `metrics_<suffix>.json`.

## 10. Caveats

1. **`attention_score = 0`** — Q·K^T and (scores·V) compute scales with context length; not subtracted from the metrics file in aggregate form. S-MFU undercounts attention compute slightly (negligible at small context, ~10% at 10K ctx).
2. **Peak FLOPS is dense throughout**, so the denominator is the table value as written. The table is owned here rather than imported from MoE-CAP, which keeps one basis across vendors, and `inputs.peak_flops_basis` records it per sidecar.
3. **gpt-oss-120b mxfp4** has `self_attn` + `mlp.router` kept at BF16; we apply fp4 uniformly. Attention params ≈ 0.8% of total weights → error is negligible.
4. **Activation is never borrowed.** It depends on the batch a run realised, so it is a property of that engine on that accelerator, varying several-fold across accelerators within one model, workload and batch regime. A run without its own trace publishes a null S-MBU, and `activation_source` records which case applies.
