import unittest

import pandas as pd

from postprocessing.aggregate_results import apply_explicit_performance_fallbacks


class ExplicitPerformanceFallbackTests(unittest.TestCase):
    def test_prefill_throughput_fills_only_missing_sparsity_values(self):
        df = pd.DataFrame({
            "sparsity.prefill.prefill_tokens_per_s": [100.0, None],
            "performance.prefill_tokens_per_s": [200.0, 24454.12831904223],
        })

        result = apply_explicit_performance_fallbacks(df)

        self.assertEqual(
            result["sparsity.prefill.prefill_tokens_per_s"].tolist(),
            [100.0, 24454.12831904223],
        )

    def test_prefill_throughput_creates_display_column_without_sparsity(self):
        df = pd.DataFrame({
            "performance.prefill_tokens_per_s": [24454.12831904223],
        })

        result = apply_explicit_performance_fallbacks(df)

        self.assertEqual(
            result.loc[0, "sparsity.prefill.prefill_tokens_per_s"],
            24454.12831904223,
        )


if __name__ == "__main__":
    unittest.main()