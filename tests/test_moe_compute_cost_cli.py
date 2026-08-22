import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "postprocessing" / "moe_cost_metrics" / "compute_cost.py"


def make_run(root: Path, gpu: str = "b200", timestamp: str = "20260101-0000") -> Path:
    run = root / "moe" / "vastai" / "sglang" / "gpt-oss-120b" / "gsm8k_256samples" / f"{gpu}x1" / "batch-size-1" / timestamp
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"performance": {"e2e_s": 10.0, "ttft": 0.1, "tpot": 0.002}}))
    (run / "metadata.json").write_text(json.dumps({"system_environment": {"inference_engine_version": "test"}}))
    return run


def make_batched_run(root: Path) -> Path:
    run = root / "moe" / "vastai" / "sglang" / "gpt-oss-120b" / "gsm8k_256samples" / "b200x1" / "batch-size-default" / "20260101-0000"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({
        "performance": {"e2e_s": 10.0, "ttft": 0.2, "tpot": 0.01,
                        "prefill_pass_latency_s": 0.2},
        "batch_token_profile": {
            "prefill_tokens": 2000,
            "prefill_tokens_per_request": 20.0,
            "prefill_avg_batch_size": 10.0,
            "decode_generated_tokens": 5000,
            "decode_generated_tokens_per_request": 50.0,
            "decode_avg_batch_size": 20.0,
        },
    }))
    (run / "metadata.json").write_text(json.dumps({"system_environment": {"inference_engine_version": "test"}}))
    return run


class MoeComputeCostCliTests(unittest.TestCase):

    def test_buy_defaults_resolve_per_hardware_tier_in_one_invocation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dc_run = make_run(root, "b200")
            ws_run = make_run(root, "gb10")

            result = subprocess.run([
                sys.executable, str(SCRIPT), "--root", str(root / "moe"),
            ], check=True, cwd=REPO, text=True, capture_output=True)

            dc = json.loads((dc_run / "cost.json").read_text())["buy"]
            ws = json.loads((ws_run / "cost.json").read_text())["buy"]
            self.assertEqual((dc["base_lifetime_hours"], dc["utilisation"], dc["lifetime_hours"]), (43800.0, 0.9, 39420.0))
            self.assertEqual((ws["base_lifetime_hours"], ws["utilisation"], ws["lifetime_hours"]), (26280.0, 0.4, 10512.0))
            self.assertIn("b200 [datacentre]", result.stdout)
            self.assertIn("gb10 [workstation]", result.stdout)

    def test_partial_cli_override_keeps_each_tier_lifetime_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dc_run = make_run(root, "b200")
            ws_run = make_run(root, "gb10")

            subprocess.run([
                sys.executable, str(SCRIPT), "--root", str(root / "moe"), "--utilisation", "0.25",
            ], check=True, cwd=REPO, text=True, capture_output=True)

            dc = json.loads((dc_run / "cost.json").read_text())["buy"]
            ws = json.loads((ws_run / "cost.json").read_text())["buy"]
            self.assertEqual((dc["base_lifetime_hours"], dc["utilisation"], dc["lifetime_hours"]), (43800.0, 0.25, 10950.0))
            self.assertEqual((ws["base_lifetime_hours"], ws["utilisation"], ws["lifetime_hours"]), (26280.0, 0.25, 6570.0))

    def test_cost_per_output_token_uses_decode_batch_size(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = make_batched_run(root)

            subprocess.run([
                sys.executable, str(SCRIPT),
                "--root", str(root / "moe"),
                "--rent-price", "b200=3.6",
            ], check=True, cwd=REPO, text=True, capture_output=True)

            cost = json.loads((run / "cost.json").read_text())
            rent_cost = cost["rent"]["cost"]
            self.assertEqual(rent_cost["avg_cost_per_1M_output_tokens_usd"], 0.5)
            self.assertEqual(rent_cost["decode_avg_batch_size"], 20.0)
            self.assertAlmostEqual(rent_cost["avg_cost_per_request_usd"], 0.000045)
            self.assertEqual(rent_cost["method"], "batch_token_profile")
            self.assertEqual(rent_cost["effective_output_tokens_per_s"], 2000.0)
            self.assertEqual(
                rent_cost["formula"],
                "price_per_second_usd * tpot_s / decode_avg_batch_size * 1e6",
            )
            self.assertEqual(rent_cost["breakdown"]["pricing"]["price_per_hour_usd"], 3.6)
            self.assertEqual(rent_cost["breakdown"]["throughput"]["effective_output_tokens_per_s"], 2000.0)
            # Node-aggregate prefill rate from the shared resolver: short-prompt
            # concurrent run -> tokens x batch / pass latency, labelled as the
            # estimate it is. 20 x 10 / 0.2 s.
            throughput = rent_cost["breakdown"]["throughput"]
            self.assertEqual(throughput["prefill_tokens_per_s"], 1000.0)
            self.assertEqual(throughput["prefill_basis"], "estimated")
            self.assertEqual(throughput["prefill_method"], "hybrid-rung1")
            self.assertEqual(throughput["prefill_token_basis"], "nominal-attempted")
            self.assertIsNone(throughput["prefill_reason"])
            self.assertEqual(rent_cost["breakdown"]["request_seconds"]["prefill_seconds_per_request"], 0.02)
            self.assertEqual(rent_cost["breakdown"]["request_seconds"]["decode_seconds_per_request"], 0.025)
            self.assertEqual(rent_cost["breakdown"]["output_token_cost"]["cost_per_1M_output_tokens_usd"], 0.5)

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

    def test_wall_block_present_on_rent_and_buy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = make_batched_run(root)

            subprocess.run([
                sys.executable, str(SCRIPT),
                "--root", str(root / "moe"),
                "--rent-price", "b200=3.6",
            ], check=True, cwd=REPO, text=True, capture_output=True)

            cost = json.loads((run / "cost.json").read_text())
            rent_wall = cost["rent"]["wall"]
            self.assertEqual(rent_wall["node_seconds_per_request"], 10.0)
            self.assertAlmostEqual(rent_wall["cost_per_request_usd"], 3.6 / 3600 * 10.0)
            buy_wall = cost["buy"]["wall"]
            self.assertEqual(buy_wall["node_seconds_per_request"], 10.0)
            self.assertAlmostEqual(
                buy_wall["cost_per_request_usd"],
                cost["buy"]["effective_hourly_rate_usd"] / 3600 * 10.0,
            )

    def test_cs3_buy_uses_one_complete_system_and_withholds_token_dependent_cost(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "moe" / "cerebras" / "waferengine" / "qwen3-4b" / "gsm8k_256samples" / "cs3x1" / "batch-size-1" / "20260804-011335"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text(json.dumps({
                "performance": {"e2e_s": 2.0, "ttft": 0.1, "tpot": 0.001},
                "batch_token_profile": {
                    "prefill_tokens_per_request": 100.0,
                    "prefill_avg_batch_size": 1.0,
                    "decode_generated_tokens": None,
                    "decode_generated_tokens_per_request": None,
                    "decode_avg_batch_size": 1.0,
                },
            }))
            (run / "metadata.json").write_text(json.dumps({
                "system_environment": {"inference_engine_version": None},
            }))

            subprocess.run([
                sys.executable, str(SCRIPT), "--root", str(root / "moe"),
            ], check=True, cwd=REPO, text=True, capture_output=True)

            cost = json.loads((run / "cost.json").read_text())
            buy = cost["buy"]
            expected_rate_h = 1_200_000 / (5 * 365 * 24 * 0.9) + 23 * 0.15
            self.assertEqual(buy["total_capital_usd"], 1_200_000)
            self.assertEqual(buy["total_power_w"], 23_000)
            self.assertEqual(buy["scale_other_capital"], 1.2)
            self.assertEqual(buy["capital_scale"], 1.0)
            self.assertEqual(buy["cpu"]["price_per_unit_usd"], 0.0)
            self.assertEqual(buy["cpu"]["tdp_w"], 0)
            self.assertAlmostEqual(buy["effective_hourly_rate_usd"], expected_rate_h)
            self.assertAlmostEqual(buy["wall"]["cost_per_request_usd"], expected_rate_h / 3600 * 2.0)
            self.assertAlmostEqual(
                buy["cost"]["avg_cost_per_1M_output_tokens_usd"],
                expected_rate_h / 3600 * 0.001 * 1_000_000,
            )
            self.assertIsNone(buy["cost"]["avg_cost_per_request_usd"])
            self.assertIsNone(buy["cost"]["decode_generated_tokens_per_request"])

    def test_concurrent_run_without_profile_gets_wall_only_block(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "moe" / "vastai" / "sglang" / "gpt-oss-120b" / "gsm8k_256samples" / "b200x1" / "batch-size-default" / "20260101-0000"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text(json.dumps({
                "performance": {"e2e_s": 10.0, "ttft": 0.2, "tpot": 0.01},
            }))
            (run / "metadata.json").write_text(json.dumps({"system_environment": {"inference_engine_version": "test"}}))

            subprocess.run([
                sys.executable, str(SCRIPT),
                "--root", str(root / "moe"),
                "--rent-price", "b200=3.6",
            ], check=True, cwd=REPO, text=True, capture_output=True)

            cost = json.loads((run / "cost.json").read_text())
            self.assertNotIn("cost", cost["rent"])
            self.assertNotIn("cost", cost["buy"])
            self.assertAlmostEqual(cost["rent"]["wall"]["cost_per_request_usd"], 0.01)
            self.assertIn("wall", cost["buy"])

    def test_single_stream_run_keeps_token_cost_beside_wall(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = make_run(root)  # batch-size-1, no batch profile

            subprocess.run([
                sys.executable, str(SCRIPT),
                "--root", str(root / "moe"),
                "--rent-price", "b200=3.6",
            ], check=True, cwd=REPO, text=True, capture_output=True)

            cost = json.loads((run / "cost.json").read_text())
            self.assertIn("cost", cost["rent"])
            self.assertEqual(cost["rent"]["cost"]["method"], "latency_tpot_no_batch_profile")
            self.assertAlmostEqual(cost["rent"]["wall"]["cost_per_request_usd"], 0.01)

    def test_newer_schema_recovers_e2e_from_request_rate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "moe" / "vastai" / "sglang" / "gpt-oss-120b" / "gsm8k_256samples" / "b200x1" / "batch-size-1" / "20260101-0000"
            run.mkdir(parents=True)
            (run / "metrics.json").write_text(json.dumps({
                "performance": {"request/s": 0.1, "ttft": 0.1, "tpot": 0.002},
            }))
            (run / "metadata.json").write_text(json.dumps({"system_environment": {"inference_engine_version": "test"}}))

            subprocess.run([
                sys.executable, str(SCRIPT),
                "--root", str(root / "moe"),
                "--rent-price", "b200=3.6",
            ], check=True, cwd=REPO, text=True, capture_output=True)

            cost = json.loads((run / "cost.json").read_text())
            self.assertEqual(cost["performance"]["e2e_s"], 10.0)
            self.assertEqual(cost["rent"]["wall"]["node_seconds_per_request"], 10.0)
            self.assertAlmostEqual(cost["rent"]["wall"]["cost_per_request_usd"], 0.01)

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

    def test_skipped_leaf_removes_its_stale_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = make_run(root)
            (run / "metrics.json").write_text(json.dumps({"performance": {}}))
            stale = run / "cost.json"
            stale.write_text(json.dumps({"sentinel": "stale"}))

            subprocess.run([
                sys.executable, str(SCRIPT), "--root", str(root / "moe"),
            ], check=True, cwd=REPO, text=True, capture_output=True)

            self.assertFalse(stale.exists())

if __name__ == "__main__":
    unittest.main()
