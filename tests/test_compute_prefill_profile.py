import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "postprocessing" / "moe_cost_metrics" / "compute_prefill_profile.py"


def prefill_step(index, reqs, ttft):
    return {"forward_mode": "prefill", "index": index, "ttft": ttft,
            "per_req_info": reqs}


def req(req_id, extend_len, last=True):
    return {"req_id": req_id, "extend_len": extend_len, "is_last_chunk": last}


def write_run(root: Path, *, dp=None, trace_rows=None, metadata=None) -> Path:
    run = (root / "moe" / "prov" / "eng" / "model" / "ds_2samples" / "gpux1"
           / "batch-size-default" / "20990101-0000")
    run.mkdir(parents=True)
    if metadata is None:
        metadata = {
            "system_environment": {
                "inference_engine": "eng", "inference_engine_version": "1.0",
                "tensor_parallel_size": 1,
                **({"data_parallel_size": dp} if dp else {}),
            },
            "hardware": {"num_gpus": 1},
        }
    (run / "metadata.json").write_text(json.dumps(metadata))
    (run / "output_data.jsonl").write_text("".join(
        json.dumps({"index": i, "success": i != 2}) + "\n" for i in range(3)
    ))
    if trace_rows is None:
        # One warm-up probe step, then the two successful client requests:
        # r1 in two chunks (the second sharing a mixed step's full latency
        # convention is irrelevant here), r2 in one.
        trace_rows = [
            prefill_step(0, [req("warm", 5)], 0.5),
            prefill_step(1, [req("r1", 100, last=False)], 1.0),
            prefill_step(2, [req("r1", 20), req("r2", 30)], 0.25),
        ]
    (run / "detailed_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in trace_rows))
    (run / "metrics.json").write_text(json.dumps({
        "batch_token_profile": {"prefill_tokens": 200}}))
    return run


def run_script(root: Path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root / "moe")],
        check=True, cwd=REPO, text=True, capture_output=True)


class ComputePrefillProfileTests(unittest.TestCase):

    def test_profile_sums_forwarded_tokens_and_full_step_time(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = write_run(root)
            run_script(root)
            profile = json.loads((run / "prefill_profile.json").read_text())
            self.assertEqual(profile["schema"], "prefill-profile/1")
            self.assertEqual(profile["prefill_forwarded_tokens"], 150)
            self.assertEqual(profile["prefill_step_elapsed_s"], 1.25)
            self.assertEqual(profile["prefill_nominal_attempted_tokens"], 200)
            self.assertEqual(profile["prefill_physical_steps"], 2)
            self.assertEqual(profile["excluded_leading_steps"], 1)
            self.assertEqual(profile["cohort"], {
                "attempts": 3, "successes": 2, "failures": 1,
                "trace_request_ids": 2,
                "trace_request_ids_sha256": profile["cohort"]["trace_request_ids_sha256"],
            })
            # Deterministic: a second run rewrites nothing.
            before = (run / "prefill_profile.json").read_bytes()
            run_script(root)
            self.assertEqual((run / "prefill_profile.json").read_bytes(), before)

    def test_gate_failure_writes_no_profile_and_removes_a_stale_one(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = write_run(root, dp=2)  # rank rule unprovable under DP
            stale = run / "prefill_profile.json"
            stale.write_text("{}")
            result = run_script(root)
            self.assertIn("rank-rule-unprovable-dp", result.stdout)
            self.assertFalse(stale.exists())

    def test_absent_rank_layout_evidence_fails_closed(self):
        # A metadata file recording neither tensor_parallel_size nor
        # hardware.num_gpus proves nothing about the rank layout; None == None
        # must not read as a proof and earn a trace-exact profile.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = write_run(root, metadata={
                "system_environment": {"inference_engine": "eng"},
                "hardware": {},
            })
            result = run_script(root)
            self.assertIn("rank-rule-unprovable-no-layout-evidence", result.stdout)
            self.assertFalse((run / "prefill_profile.json").exists())

    def test_one_sided_rank_layout_evidence_fails_closed(self):
        # Only one side recorded is still no proof of a TP-only layout.
        for metadata in (
            {"system_environment": {"inference_engine": "eng",
                                    "tensor_parallel_size": 1}, "hardware": {}},
            {"system_environment": {"inference_engine": "eng"},
             "hardware": {"num_gpus": 1}},
        ):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                run = write_run(root, metadata=metadata)
                result = run_script(root)
                self.assertIn("rank-rule-unprovable-no-layout-evidence", result.stdout)
                self.assertFalse((run / "prefill_profile.json").exists())

    def test_cohort_mismatch_is_gated(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # r2 never completes: no window can match the 2-success cohort.
            run = write_run(root, trace_rows=[
                prefill_step(0, [req("r1", 100)], 1.0),
                prefill_step(1, [req("r2", 30, last=False)], 0.25),
            ])
            result = run_script(root)
            self.assertIn("no-exact-client-cohort-window", result.stdout)
            self.assertFalse((run / "prefill_profile.json").exists())

    def test_run_without_trace_keeps_an_existing_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = write_run(root)
            (run / "detailed_results.jsonl").unlink()
            keeper = run / "prefill_profile.json"
            keeper.write_text('{"kept": true}')
            run_script(root)
            self.assertTrue(keeper.exists())
            self.assertEqual(json.loads(keeper.read_text()), {"kept": True})


if __name__ == "__main__":
    unittest.main()
