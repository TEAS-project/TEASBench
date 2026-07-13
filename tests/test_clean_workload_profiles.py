import unittest

from postprocessing.moe_cost_metrics.clean_workload_profiles import compact_profile


class CleanWorkloadProfileTests(unittest.TestCase):
    def test_compact_profile_removes_provenance_and_keeps_totals(self):
        data = {
            "schema_version": 2,
            "run_type": "moe",
            "run_path": "moe/x/run",
            "workload_profile": {
                "prefill_tokens": {"total": 123, "source": {"method": "old"}},
                "decode_tokens": {"total": 456, "estimated_total": 999, "confidence": "low"},
                "tool_calls": None,
                "relative_to_default_batch": {"decode_tokens_ratio": 1.0},
                "decode_attention_seq_lens_sum": 789,
            },
            "source": {"metrics_json": "metrics.json"},
        }

        compact = compact_profile(data)

        self.assertEqual(compact, {
            "schema_version": 3,
            "run_type": "moe",
            "run_path": "moe/x/run",
            "workload_profile": {
                "prefill_tokens_total": 123,
                "decode_tokens_total": 456,
            },
        })

    def test_compact_profile_accepts_metric_style_numeric_totals(self):
        data = {
            "schema_version": 2,
            "run_type": "moe",
            "run_path": "moe/x/run",
            "workload_profile": {
                "prefill_tokens": {"total": 123.0},
                "decode_tokens": {"total": 456.0},
            },
        }

        compact = compact_profile(data)

        self.assertEqual(compact["workload_profile"]["prefill_tokens_total"], 123)
        self.assertEqual(compact["workload_profile"]["decode_tokens_total"], 456)

    def test_compact_profile_is_idempotent_for_v3_schema(self):
        data = {
            "schema_version": 3,
            "run_type": "moe",
            "run_path": "moe/x/run",
            "workload_profile": {
                "prefill_tokens_total": 123,
                "decode_tokens_total": 456,
            },
        }

        self.assertEqual(compact_profile(data), data)


if __name__ == "__main__":
    unittest.main()
