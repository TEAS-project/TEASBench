import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "postprocessing" / "moe_cost_metrics" / "compute_cost.py"


def make_run(root: Path) -> Path:
    run = root / "moe" / "vastai" / "sglang" / "gpt-oss-120b" / "gsm8k_256samples" / "b200x1" / "batch-size-1" / "20260101-0000"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"performance": {"e2e_s": 10.0, "ttft": 0.1, "tpot": 0.002}}))
    (run / "metadata.json").write_text(json.dumps({"system_environment": {"inference_engine_version": "test"}}))
    return run


class MoeComputeCostCliTests(unittest.TestCase):
    def test_buy_price_json_files_override_gpu_and_cpu_specs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = make_run(root)
            gpu_prices = root / "gpu_prices.json"
            cpu_prices = root / "cpu_prices.json"
            gpu_prices.write_text(json.dumps({"b200": 12345.0}))
            cpu_prices.write_text(json.dumps({"xeon-8468": 678.0}))

            result = subprocess.run([
                sys.executable, str(SCRIPT),
                "--root", str(root / "moe"),
                "--buy-gpu-prices-json", str(gpu_prices),
                "--buy-cpu-prices-json", str(cpu_prices),
            ], check=True, cwd=REPO, text=True, capture_output=True)

            self.assertIn("Buy-cost assumptions note", result.stdout)
            cost = json.loads((run / "cost.json").read_text())
            self.assertEqual(cost["buy"]["gpu"]["price_per_unit_usd"], 12345.0)
            self.assertEqual(cost["buy"]["gpu"]["price_source"], "user-supplied-json")
            self.assertEqual(cost["buy"]["cpu"]["price_per_unit_usd"], 678.0)
            self.assertEqual(cost["buy"]["cpu"]["price_source"], "user-supplied-json")


    def test_default_buy_gpu_prices_order_a100_below_blackwell(self):
        from postprocessing.moe_cost_metrics.compute_cost import GPU_SPECS

        self.assertLess(
            GPU_SPECS["a100"]["price_per_unit_usd"],
            GPU_SPECS["b200"]["price_per_unit_usd"],
        )
        self.assertLess(
            GPU_SPECS["a100"]["price_per_unit_usd"],
            GPU_SPECS["b300"]["price_per_unit_usd"],
        )

    def test_utilisation_scales_effective_lifetime_hours(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = make_run(root)

            result = subprocess.run([
                sys.executable, str(SCRIPT),
                "--root", str(root / "moe"),
                "--buy-lifetime-hours", "1000",
                "--utilisation", "0.25",
            ], check=True, cwd=REPO, text=True, capture_output=True)

            self.assertIn("Buy-cost assumptions note", result.stdout)
            cost = json.loads((run / "cost.json").read_text())
            self.assertEqual(cost["buy"]["lifetime_hours"], 250.0)
            self.assertEqual(cost["buy"]["base_lifetime_hours"], 1000.0)
            self.assertEqual(cost["buy"]["utilisation"], 0.25)

    def test_invalid_utilisation_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_run(root)

            result = subprocess.run([
                sys.executable, str(SCRIPT),
                "--root", str(root / "moe"),
                "--utilisation", "0",
            ], cwd=REPO, text=True, capture_output=True)

            self.assertEqual(result.returncode, 2)
            self.assertIn("--utilisation must be in (0, 1]", result.stderr)


if __name__ == "__main__":
    unittest.main()
