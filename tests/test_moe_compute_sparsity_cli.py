import unittest

from postprocessing.moe_cost_metrics.compute_sparsity_metrics import compute_for_run


class DummyCfg:
    def __init__(self, d):
        self._d = d
        for k, v in d.items():
            setattr(self, k, v)


class MoeComputeSparsityTests(unittest.TestCase):

    def test_prefill_smfu_throughput_uses_prefill_batch_size(self):
        metrics = {
            "performance": {"ttft": 0.5, "tpot": 0.01},
            "expert_activation": {
                "avg_expert_activation_prefill": 1.0,
                "avg_expert_activation_decode": 1.0,
            },
            "batch_token_profile": {
                "prefill_tokens_per_request": 20.0,
                "prefill_avg_batch_size": 10.0,
                "decode_generated_tokens_per_request": 50.0,
                "decode_avg_batch_size": 16.0,
            },
        }
        cfg = DummyCfg({
            "_name_or_path": "test/moe",
            "model_type": "gpt_oss",
            "num_hidden_layers": 1,
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 32,
            "intermediate_size": 256,
            "num_experts_per_tok": 1,
            "num_local_experts": 4,
        })

        result = compute_for_run(
            metrics, cfg, "NVIDIA-B200-183GB", 1, "bfloat16",
            avg_prefill_len=60.0, avg_decode_ctx_len=210.0,
        )

        self.assertEqual(result["prefill"]["prefill_tokens_per_s"], 400.0)
        self.assertEqual(result["decode"]["output_tokens_per_s"], 1600.0)
        self.assertGreater(result["prefill"]["S_MFU"], 0.0)
        self.assertGreater(result["decode"]["S_MFU"], 0.0)


if __name__ == "__main__":
    unittest.main()
