import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
STAGER = REPO / "pipeline" / "scripts" / "stage_agentic_evidence.py"
DRIVER = REPO / "pipeline" / "templates" / "agentic-driver.sh"
PROVIDERS = REPO / "pipeline" / "k8s" / "lib" / "k8s_pod_providers" / "providers.py"
DEVELOPMENT = REPO.parent / "TEAS_Development_Results_Private"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stager = load_module("stage_agentic_evidence_test", STAGER)
providers = load_module("agentic_release_providers_test", PROVIDERS)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_evidence(root):
    suffix = "swe-bench-lite_20990101-0000"
    results = []
    outputs = []
    detailed = []
    for index in range(2):
        task_id = f"task-{index}"
        (root / f"task_{task_id}").mkdir()
        input_tokens = 10 + index
        output_tokens = 2 + index
        results.append({"task_id": task_id, "total_usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "requests": 1}})
        outputs.append({"index": index, "task_id": task_id, "input_tokens": input_tokens, "output_tokens": output_tokens, "num_requests": 1})
        detailed.append({"example_index": index, "request_index": 0, "input_tokens": input_tokens, "output_tokens": output_tokens})
    write_jsonl(root / "results.jsonl", results)
    write_jsonl(root / f"detailed-results_{suffix}.jsonl", detailed)
    write_jsonl(root / f"output-data_{suffix}.jsonl", outputs)
    hardware = {"gpu_type": "NVIDIA A100", "num_gpus": 2}
    environment = {"dataset": "swe-bench-lite", "num_examples": 2, "sweagent_call_limit": 200, "sweagent_task_timeout_s": 3600, "inference_engine": "sglang", "inference_engine_version": "0.5.12.post1", "tensor_parallel_size": 2, "concurrency": 2, "observed_max_concurrency": 2}
    (root / f"metadata_{suffix}.json").write_text(json.dumps({"hardware": hardware, "model_config": {"model_name": "unsloth/gpt-oss-120b"}, "system_environment": environment}), encoding="utf-8")
    (root / f"metrics_{suffix}.json").write_text(json.dumps({"hardware": {**hardware, "sglang_version": "0.5.12.post1"}, "quality": {"total_examples": 2}}), encoding="utf-8")


def validate_evidence(root):
    return stager.validate_source(root, expected_tasks=2, call_limit=200, task_timeout_s=3600, expected_engine="sglang", expected_engine_version="0.5.12.post1", expected_model_name="unsloth/gpt-oss-120b", expected_gpu_type="NVIDIA A100", expected_num_gpus=2, concurrency=2)


class EvidenceStagerTests(unittest.TestCase):
    def test_three_families_and_execution_policy_are_required(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_evidence(root)
            sources = validate_evidence(root)
            self.assertEqual({path.name for path in sources}, {"results.jsonl", "detailed-results_swe-bench-lite_20990101-0000.jsonl", "output-data_swe-bench-lite_20990101-0000.jsonl"})
            metadata = root / "metadata_swe-bench-lite_20990101-0000.json"
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            payload["system_environment"]["observed_max_concurrency"] = 1
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(stager.EvidenceError):
                validate_evidence(root)
            payload["system_environment"]["observed_max_concurrency"] = 2
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            (root / "results.jsonl").unlink()
            with self.assertRaises(stager.EvidenceError):
                validate_evidence(root)

    def test_results_completion_order_does_not_change_task_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_evidence(root)
            path = root / "results.jsonl"
            write_jsonl(path, list(reversed(stager.load_jsonl(path))))
            self.assertEqual(len(validate_evidence(root)), 3)

    def test_metadata_identity_and_destination_slugs_are_exact(self):
        relative = "agentic/eidf/sglang/gpt-oss-120b/swe-bench-lite/a100x2/batch-size-default/20990101-0000"
        path = stager.validate_destination(relative, expected_engine="sglang", expected_model_name="unsloth/gpt-oss-120b", expected_gpu_type="NVIDIA A100", expected_num_gpus=2)
        self.assertEqual(path.as_posix(), relative)
        with self.assertRaises(stager.EvidenceError):
            stager.validate_destination(relative.replace("a100x2", "h100x2"), expected_engine="sglang", expected_model_name="unsloth/gpt-oss-120b", expected_gpu_type="NVIDIA A100", expected_num_gpus=2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_evidence(root)
            metadata = root / "metadata_swe-bench-lite_20990101-0000.json"
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            payload["system_environment"]["inference_engine_version"] = "wrong"
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(stager.EvidenceError):
                validate_evidence(root)

    def test_exact_lfs_pointer_parser_rejects_normal_jsonl(self):
        digest = "a" * 64
        self.assertEqual(stager.parse_pointer(f"version {stager.POINTER_VERSION}\noid sha256:{digest}\nsize 7\n"), (digest, 7))
        with self.assertRaises(stager.EvidenceError):
            stager.parse_pointer('{"task_id":"not-a-pointer"}\n')

    def test_real_development_policy_dry_run_is_non_mutating(self):
        if not (DEVELOPMENT / ".git").exists():
            self.skipTest("canonical Development checkout is not present")
        before = subprocess.run(["git", "-C", str(DEVELOPMENT), "status", "--short"], text=True, capture_output=True, check=True).stdout
        with tempfile.TemporaryDirectory() as td:
            source = Path(td)
            make_evidence(source)
            proc = subprocess.run([sys.executable, str(STAGER), "--source-run-dir", str(source), "--repo", str(DEVELOPMENT), "--destination-relative", "agentic/eidf/sglang/gpt-oss-120b/swe-bench-lite/a100x2/batch-size-default/20990101-0000", "--expected-tasks", "2", "--sweagent-call-limit", "200", "--sweagent-task-timeout-s", "3600", "--expected-engine", "sglang", "--expected-engine-version", "0.5.12.post1", "--expected-model-name", "unsloth/gpt-oss-120b", "--expected-gpu-type", "NVIDIA A100", "--expected-num-gpus", "2", "--concurrency", "2", "--dry-run"], text=True, capture_output=True)
        after = subprocess.run(["git", "-C", str(DEVELOPMENT), "status", "--short"], text=True, capture_output=True, check=True).stdout
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["mode"], "dry-run")
        self.assertEqual(after, before)


class LauncherWiringTests(unittest.TestCase):
    def test_eidf_runtime_observation_is_stamped_and_fail_closed(self):
        body = DRIVER.read_text(encoding="utf-8")
        section = body.index('echo "[4] Recording dependency versions"')
        script_start = body.index("<<'PYEOF'\n", section) + len("<<'PYEOF'\n")
        script = body[script_start:body.index("\nPYEOF", script_start)]
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            metadata = run_dir / "metadata_swe-bench-lite_test.json"
            metadata.write_text(json.dumps({"hardware": {"gpu_type": "NVIDIA A100", "num_gpus": 2}, "model_config": {"model_name": "unsloth/gpt-oss-120b"}, "system_environment": {"inference_engine": "sglang", "inference_engine_version": "0.5.12.post1", "tensor_parallel_size": 2, "concurrency": 2, "observed_max_concurrency": 0}}), encoding="utf-8")
            observation = run_dir / "runtime-observation.json"
            observation.write_text(json.dumps({"schema_version": 1, "publishable": False, "requested_task_concurrency": 2, "observed_max_task_concurrency": 2}), encoding="utf-8")
            versions = run_dir / "versions.json"
            versions.write_text("{}", encoding="utf-8")
            command = [sys.executable, "-", str(run_dir), str(versions), "200", "3600", "sglang", "0.5.12.post1", "unsloth/gpt-oss-120b", "NVIDIA A100", "2", "2", "1", "50", "1"]
            proc = subprocess.run(command, input=script, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            environment = json.loads(metadata.read_text(encoding="utf-8"))["system_environment"]
            self.assertEqual(environment["concurrency"], 2)
            self.assertEqual(environment["observed_max_concurrency"], 2)
            self.assertEqual(environment["inference_engine_version"], "0.5.12.post1")
            bad = json.loads(observation.read_text(encoding="utf-8"))
            bad["observed_max_task_concurrency"] = 1
            observation.write_text(json.dumps(bad), encoding="utf-8")
            proc = subprocess.run(command, input=script, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("did not reach requested concurrency", proc.stderr)

    def test_generated_driver_is_a_no_publication_wiring_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "generated"
            proc = subprocess.run([sys.executable, "generate.py", "--csv_file", str(REPO / "experiments" / "swe-bench-lite-eidf.csv"), "--target_dir", str(target)], cwd=REPO / "pipeline", text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            drivers = list(target.glob("*.sh"))
            self.assertTrue(drivers)
            for driver in drivers:
                syntax = subprocess.run(["bash", "-n", str(driver)], text=True, capture_output=True)
                self.assertEqual(syntax.returncode, 0, syntax.stderr)
                body = driver.read_text(encoding="utf-8")
                self.assertIn('SWEAGENT_TASK_TIMEOUT="${SWEAGENT_TASK_TIMEOUT:-3600}"', body)
                self.assertIn('export TEAS_CONCURRENCY="', body)
                expected_version = "0.5.12.post1" if 'export TEAS_ENGINE="sglang"' in body else "0.21.0"
                self.assertIn(f'export TEAS_ENGINE_VERSION="{expected_version}"', body)
                self.assertIn("stage_agentic_evidence.py", body)
                self.assertIn(f'--expected-engine-version "{expected_version}"', body)
                self.assertIn('--expected-model-name "unsloth/gpt-oss-120b"', body)
                self.assertIn('--expected-gpu-type "NVIDIA ', body)
                self.assertIn("runtime-observation.json", body)
                self.assertIn("http.https://github.com/.extraheader", body)
                self.assertNotIn("oauth2:${GIT_TOKEN}", body)
                self.assertNotIn('git -c "http.https://github.com/.extraheader', body)
                self.assertIn("GIT_CONFIG_VALUE_0=\"$GITHUB_EXTRAHEADER\" git", body)

    def test_sandbox_cleanup_is_scoped_to_the_current_run(self):
        with patch.dict(os.environ, {"TEASBENCH_RUN_ID": "run-123"}, clear=False):
            job = providers._sandbox_job_spec("ns", "queue", "image", "token", 9999)
        self.assertEqual(job["metadata"]["labels"]["teasbench.run/id"], "run-123")
        self.assertEqual(job["spec"]["template"]["metadata"]["labels"]["teasbench.run/id"], "run-123")
        body = DRIVER.read_text(encoding="utf-8")
        self.assertNotIn("delete jobs -l app=teasbench-sandbox", body)
        self.assertGreaterEqual(body.count('delete jobs -l "$SANDBOX_SELECTOR"'), 2)

if __name__ == "__main__":
    unittest.main()
