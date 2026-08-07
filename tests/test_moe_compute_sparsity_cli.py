import unittest

from postprocessing.moe_cost_metrics.compute_sparsity_metrics import compute_for_run
from postprocessing.moe_cost_metrics.prefill_rate import resolve_prefill_rate


class DummyCfg:
    def __init__(self, d):
        self._d = d
        for k, v in d.items():
            setattr(self, k, v)


def moe_metrics(**performance_extra):
    return {
        "performance": {"ttft": 0.5, "tpot": 0.01, **performance_extra},
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


def moe_cfg():
    return DummyCfg({
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


class MoeComputeSparsityTests(unittest.TestCase):

    def test_prefill_rate_and_provenance_come_from_the_shared_resolver(self):
        """The published prefill rate is the resolver's node-aggregate value.

        A concurrent short-prompt run estimates tokens x batch / pass latency
        (hybrid rung 1), and the sidecar carries the resolver's provenance
        beside the value, so a reader can tell an estimate from a measurement.
        """
        metrics = moe_metrics(prefill_pass_latency_s=0.4)
        resolved = resolve_prefill_rate(metrics, batch_regime="batch-size-default")

        result = compute_for_run(
            metrics, moe_cfg(), "NVIDIA-B200-183GB", 1, "bfloat16",
            avg_prefill_len=60.0, avg_decode_ctx_len=210.0,
            concurrent=True,
            prefill_rate=resolved,
        )

        # 20 tokens/request x batch 10 / 0.4 s pass latency.
        self.assertEqual(result["prefill"]["prefill_tokens_per_s"], 500.0)
        self.assertEqual(result["prefill"]["basis"], "estimated")
        self.assertEqual(result["prefill"]["method"], "hybrid-rung1")
        self.assertEqual(result["prefill"]["token_basis"], "nominal-attempted")
        self.assertIsNone(result["prefill"]["reason"])
        # Decode DOES scale with its batch: 16 sequences / 0.01 s per step.
        self.assertEqual(result["decode"]["output_tokens_per_s"], 1600.0)
        self.assertGreater(result["prefill"]["S_MFU"], 0.0)
        self.assertGreater(result["decode"]["S_MFU"], 0.0)

    def test_unresolvable_prefill_rate_publishes_null_with_reason(self):
        """No resolvable evidence -> null rate and null S-MFU, never a fallback.

        The dataset token constants and the harness-computed rate are no longer
        reachable from this path; the sidecar states what was missing instead.
        """
        metrics = moe_metrics()  # short prompt, no pass latency recorded
        resolved = resolve_prefill_rate(metrics, batch_regime="batch-size-default")

        result = compute_for_run(
            metrics, moe_cfg(), "NVIDIA-B200-183GB", 1, "bfloat16",
            avg_prefill_len=60.0, avg_decode_ctx_len=210.0,
            concurrent=True,
            prefill_rate=resolved,
        )

        self.assertIsNone(result["prefill"]["prefill_tokens_per_s"])
        self.assertIsNone(result["prefill"]["S_MFU"])
        self.assertIsNone(result["prefill"]["basis"])
        self.assertEqual(result["prefill"]["reason"], "no-latency-evidence")
        # The decode side is untouched by the prefill disposition.
        self.assertEqual(result["decode"]["output_tokens_per_s"], 1600.0)

    def test_batch_size_one_regime_resolves_the_aggregation_identity(self):
        """batch-size-1 pins prefill batch 1: per-request rate IS the node rate.

        The regime name wins over a recorded average above 1 (an accounting
        artefact on a single-stream run), and decode still forces batch 1.
        """
        metrics = moe_metrics()
        resolved = resolve_prefill_rate(metrics, batch_regime="batch-size-1")

        result = compute_for_run(
            metrics, moe_cfg(), "NVIDIA-B200-183GB", 1, "bfloat16",
            avg_prefill_len=60.0, avg_decode_ctx_len=210.0,
            force_batch_size_one=True,
            prefill_rate=resolved,
        )

        # 20 prefill tokens per request / 0.5 s TTFT, aggregation-exact.
        self.assertEqual(result["prefill"]["prefill_tokens_per_s"], 40.0)
        self.assertEqual(result["prefill"]["basis"], "measured")
        self.assertEqual(result["prefill"]["method"], "identity-bs1")
        self.assertEqual(result["decode"]["output_tokens_per_s"], 100.0)

    def test_null_ttft_keeps_decode_metrics_and_nulls_only_prefill(self):
        """A null ttft no longer skips the run: decode needs only tpot.

        Prefill S-MBU nulls for want of a pass latency, but decode S-MBU/S-MFU
        and the decode rate publish from the run's own tpot and batch profile.
        """
        metrics = moe_metrics(ttft=None)
        resolved = resolve_prefill_rate(metrics, batch_regime="batch-size-default")

        result = compute_for_run(
            metrics, moe_cfg(), "NVIDIA-B200-183GB", 1, "bfloat16",
            avg_prefill_len=60.0, avg_decode_ctx_len=210.0,
            concurrent=True,
            prefill_rate=resolved,
        )

        self.assertNotIn("skipped", result)
        self.assertIsNone(result["prefill"]["ttft_s"])
        self.assertIsNone(result["prefill"]["S_MBU"])
        self.assertEqual(result["decode"]["output_tokens_per_s"], 1600.0)
        self.assertGreater(result["decode"]["S_MBU"], 0.0)
        self.assertGreater(result["decode"]["S_MFU"], 0.0)

    def test_null_tpot_nulls_decode_metrics_without_skipping(self):
        """The reverse gap: no tpot nulls every decode metric, prefill stands."""
        metrics = moe_metrics(tpot=0, prefill_pass_latency_s=0.4)
        resolved = resolve_prefill_rate(metrics, batch_regime="batch-size-default")

        result = compute_for_run(
            metrics, moe_cfg(), "NVIDIA-B200-183GB", 1, "bfloat16",
            avg_prefill_len=60.0, avg_decode_ctx_len=210.0,
            concurrent=True,
            prefill_rate=resolved,
        )

        self.assertNotIn("skipped", result)
        self.assertIsNone(result["decode"]["tpot_s"])
        self.assertIsNone(result["decode"]["output_tokens_per_s"])
        self.assertIsNone(result["decode"]["S_MBU"])
        self.assertIsNone(result["decode"]["S_MFU"])
        self.assertEqual(result["prefill"]["prefill_tokens_per_s"], 500.0)
        self.assertGreater(result["prefill"]["S_MBU"], 0.0)

    def test_fixed_length_batch_size_one_dir_is_detected(self):
        from pathlib import Path
        from postprocessing.moe_cost_metrics.compute_sparsity_metrics import describe_run

        root = Path("/tmp/root/moe")
        metrics = root / "vastai" / "vllm" / "gpt-oss-120b" / "gsm8k_256samples" / "b200x1" / "batch-size-1_input1024_output1024" / "20260101" / "metrics.json"
        info = describe_run(metrics, root)

        self.assertTrue(info["batch_size_dir"].startswith("batch-size-1"))


if __name__ == "__main__":
    unittest.main()
