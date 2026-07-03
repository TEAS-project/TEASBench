import sys
import types
import unittest

hw = types.ModuleType("moe_cap.utils.hardware_utils")
hw.MEM_BW_DICT = {"NVIDIA-A100-SXM4-80GB": 2_000e9, "NVIDIA-B200-183GB": 8_000e9}
hw.PEAK_FLOPS_DICT = {k: {"NVIDIA-A100-SXM4-80GB": 312e12, "NVIDIA-B200-183GB": 9000e12} for k in ["bfloat16", "float16", "fp8", "int8", "fp4", "int4"]}
hw.get_peak_bw = lambda gpu_key: hw.MEM_BW_DICT[gpu_key]
hw.get_peak_flops = lambda gpu_key, precision="bfloat16": hw.PEAK_FLOPS_DICT[precision][gpu_key]
sys.modules.setdefault("moe_cap", types.ModuleType("moe_cap"))
sys.modules.setdefault("moe_cap.utils", types.ModuleType("moe_cap.utils"))
sys.modules["moe_cap.utils.hardware_utils"] = hw

from postprocessing.moe_cost_metrics.compute_sparsity_metrics import compute_for_run


class _Cfg:
    def __init__(self, d):
        self._d = d
        for k, v in d.items():
            setattr(self, k, v)


class MoeSparsityMetricsTests(unittest.TestCase):
    def test_smfu_throughput_uses_batch_profile_not_single_token_tpot(self):
        cfg = _Cfg({
            "_name_or_path": "unit-test-moe",
            "model_type": "unit",
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 32,
            "moe_intermediate_size": 32,
            "num_experts_per_tok": 2,
            "num_local_experts": 4,
        })
        metrics = {
            "performance": {"ttft": 0.5, "tpot": 0.01},
            "expert_activation": {
                "avg_expert_activation_prefill": 2.0,
                "avg_expert_activation_decode": 2.0,
            },
            "batch_token_profile": {
                "prefill_tokens_per_request": 128.0,
                "decode_avg_batch_size": 16.0,
            },
        }
        result = compute_for_run(
            metrics, cfg, "NVIDIA-A100-SXM4-80GB", 1, "bfloat16",
            avg_prefill_len=128.0,
            avg_decode_ctx_len=200.0,
            decode_batch_size=16.0,
        )
        self.assertEqual(result["prefill"]["prefill_tokens_per_s"], 256.0)
        self.assertEqual(result["decode"]["output_tokens_per_s"], 1600.0)


if __name__ == "__main__":
    unittest.main()
