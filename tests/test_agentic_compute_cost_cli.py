import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "postprocessing" / "agentic_cost_metrics" / "compute_agentic_cost.py"


def make_agentic_run(root: Path) -> Path:
    run = root / "agentic_results" / "vastai" / "sglang" / "gpt-oss-120b" / "swe-bench-lite" / "b200x1" / "batch-size-default" / "20260101-0000"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({
        "performance": {
            "avg_e2e_latency_s": 10.0,
            "p50_e2e_latency_s": 8.0,
            "p99_e2e_latency_s": 20.0,
            "ttft": 1.0,
            "p99_ttft": 2.0,
            "tpot": 0.1,
            "p99_tpot": 0.2,
        },
        "agentic": {
            "avg_num_requests": 2,
            "avg_total_output_tokens": 30,
            "avg_tool_call_count": 1,
        },
    }))
    (run / "metadata.json").write_text(json.dumps({"system_environment": {"inference_engine_version": "test"}}))
    return run


class AgenticComputeCostCliTests(unittest.TestCase):
    def test_buy_cost_reports_active_resource_and_reserved_worker_modes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = make_agentic_run(root)

            subprocess.run([
                sys.executable, str(SCRIPT),
                "--root", str(root / "agentic_results"),
                "--rent-price", "b200=4.0",
                "--buy-gpu-price", "b200=3600",
                "--buy-gpu-tdp", "b200=0",
                "--buy-cpu-price", "xeon-8468=7200",
                "--buy-cpu-tdp", "xeon-8468=0",
                "--buy-lifetime-hours", "3600",
                "--buy-scale-other-capital", "1",
            ], check=True, cwd=REPO, text=True, capture_output=True)

            cost = json.loads((run / "cost.json").read_text())
            self.assertEqual(cost["agentic"]["llm_active_s"], 5.0)
            self.assertEqual(cost["agentic"]["tool_wait_s"], 5.0)

            active = cost["buy"]["cost"]["active_resource"]
            reserved = cost["buy"]["cost"]["reserved_worker"]
            # Rates: gpu=1 $/h, cpu=4 $/h because CPU count defaults to 2.
            self.assertEqual(active["gpu_billable_s"], 5.0)
            self.assertEqual(active["cpu_billable_s"], 5.0)
            self.assertAlmostEqual(active["avg_cost_per_task_usd"], (5.0 * 1.0 + 5.0 * 4.0) / 3600.0)
            self.assertEqual(reserved["gpu_billable_s"], 10.0)
            self.assertEqual(reserved["cpu_billable_s"], 10.0)
            self.assertAlmostEqual(reserved["avg_cost_per_task_usd"], (10.0 * 1.0 + 10.0 * 4.0) / 3600.0)
            self.assertIn("accounting_modes", cost["buy"])
            self.assertEqual(cost["buy"]["default_cost_mode"], "active_resource")

    def test_buy_cost_mode_flag_selects_reported_default_cost(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = make_agentic_run(root)

            subprocess.run([
                sys.executable, str(SCRIPT),
                "--root", str(root / "agentic_results"),
                "--rent-price", "b200=4.0",
                "--buy-cost-mode", "reserved-worker",
                "--buy-gpu-price", "b200=3600",
                "--buy-gpu-tdp", "b200=0",
                "--buy-cpu-price", "xeon-8468=7200",
                "--buy-cpu-tdp", "xeon-8468=0",
                "--buy-lifetime-hours", "3600",
                "--buy-scale-other-capital", "1",
            ], check=True, cwd=REPO, text=True, capture_output=True)

            cost = json.loads((run / "cost.json").read_text())
            self.assertEqual(cost["buy"]["default_cost_mode"], "reserved_worker")
            self.assertEqual(
                cost["buy"]["cost"]["avg_cost_per_task_usd"],
                cost["buy"]["cost"]["reserved_worker"]["avg_cost_per_task_usd"],
            )


if __name__ == "__main__":
    unittest.main()
