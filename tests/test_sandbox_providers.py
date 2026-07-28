"""Tests for teasbench.sandbox.k8s - no cluster, no kubectl binary, no
network. `subprocess.run`/`subprocess.Popen` and `urllib.request.urlopen`
are monkeypatched with in-process fakes that never touch a real process
or socket beyond the OS-local free-port lookup used by
PortForwardK8sProvider (a loopback bind, not a network call)."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from teasbench.sandbox.k8s import (
    InClusterK8sProvider,
    PortForwardK8sProvider,
    SandboxEndpoint,
    _sandbox_job_spec,
)


class FakeKubectl:
    """Stand-in for `subprocess.run(["kubectl", "-n", ns, ...])`. Scripts a
    fixed pod phase/name/IP for every `get pods` poll and a fresh job name
    for every `create`, and records every invocation so tests can assert
    on exactly what was sent to (fake) kubectl."""

    def __init__(self, phase="Running", pod_name="swe-rex-podabc", pod_ip="10.1.2.3"):
        self.phase = phase
        self.pod_name = pod_name
        self.pod_ip = pod_ip
        self.calls = []
        self.job_names = []
        self._job_counter = 0

    def run(self, cmd, *args, **kwargs):
        self.calls.append({"cmd": list(cmd), "kwargs": dict(kwargs)})
        assert cmd[0] == "kubectl"
        sub = cmd[3]
        if sub == "create":
            self._job_counter += 1
            job_name = f"fake-job-{self._job_counter:05d}"
            self.job_names.append(job_name)
            return subprocess.CompletedProcess(cmd, 0, stdout=job_name + "\n", stderr="")
        if sub == "get":
            jsonpath = cmd[-1]
            if "podIP" in jsonpath:
                stdout = f"{self.phase}|{self.pod_name}|{self.pod_ip}"
            else:
                stdout = f"{self.phase}|{self.pod_name}"
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if sub == "delete":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if sub == "cp":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if sub == "exec":
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        raise AssertionError(f"unexpected kubectl subcommand in {cmd}")


class FakePopen:
    """Stand-in for a `kubectl port-forward` subprocess.Popen handle.
    `.poll()` returns None (still running) until `.kill()` is called."""

    def __init__(self, cmd):
        self.args = list(cmd)
        self._returncode = None

    def poll(self):
        return self._returncode

    def kill(self):
        self._returncode = -9

    def wait(self, timeout=None):
        return self._returncode


class FakeHTTPResponse:
    """Stand-in for the context manager `urllib.request.urlopen` returns."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return b"{}"


def fake_urlopen_ok(req, timeout=None):
    return FakeHTTPResponse()


class SandboxJobSpecTests(unittest.TestCase):
    """B2/B3: the Job spec shared by both sandbox providers."""

    def test_job_spec_has_queue_label_image_token_and_resource_limits(self):
        with patch.dict(os.environ, {
            "TEASBENCH_SANDBOX_CPU_REQUEST": "3",
            "TEASBENCH_SANDBOX_CPU_LIMIT": "7",
            "TEASBENCH_SANDBOX_MEM_REQUEST": "5Gi",
            "TEASBENCH_SANDBOX_MEM_LIMIT": "25Gi",
        }):
            job = _sandbox_job_spec(
                "eidf230ns", "eidf230ns-user-queue",
                "docker.io/swebench/sweb.eval.x86_64.foo:latest",
                "test-token-123", 9999)

        self.assertEqual(
            job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
            "eidf230ns-user-queue")
        self.assertEqual(job["metadata"]["labels"]["app"], "teasbench-sandbox")
        container = job["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["image"],
                         "docker.io/swebench/sweb.eval.x86_64.foo:latest")
        self.assertIn("test-token-123", container["args"][0])
        self.assertIn("--port 9999", container["args"][0])
        self.assertEqual(container["resources"], {
            "requests": {"cpu": "3", "memory": "5Gi"},
            "limits": {"cpu": "7", "memory": "25Gi"},
        })


class InClusterK8sProviderTests(unittest.TestCase):
    """B2: the default in-cluster provider."""

    def test_acquire_returns_pod_ip_host_and_token_differs_between_calls(self):
        kubectl = FakeKubectl(phase="Running", pod_name="swe-rex-abcde", pod_ip="10.1.2.3")
        with patch("subprocess.run", side_effect=kubectl.run), \
             patch("urllib.request.urlopen", side_effect=fake_urlopen_ok):
            provider = InClusterK8sProvider(namespace="eidf230ns", queue="eidf230ns-user-queue")
            ep1 = provider.acquire("some/image:latest", label="task-1")
            ep2 = provider.acquire("some/image:latest", label="task-2")

        self.assertEqual(ep1.host, "http://10.1.2.3")
        self.assertEqual(ep1.port, 9999)
        self.assertTrue(ep1.auth_token)
        self.assertNotEqual(ep1.auth_token, ep2.auth_token)

        create_calls = [c for c in kubectl.calls if c["cmd"][3] == "create"]
        self.assertEqual(len(create_calls), 2)
        body1 = json.loads(create_calls[0]["kwargs"]["input"])
        self.assertIn(ep1.auth_token,
                      body1["spec"]["template"]["spec"]["containers"][0]["args"][0])

    def test_pod_failed_phase_raises_and_deletes_job(self):
        kubectl = FakeKubectl(phase="Failed", pod_name="", pod_ip="")
        with patch("subprocess.run", side_effect=kubectl.run):
            provider = InClusterK8sProvider(namespace="eidf230ns")
            with self.assertRaises(RuntimeError):
                provider.acquire("some/image:latest", label="task-failed")

        delete_calls = [c for c in kubectl.calls if c["cmd"][3] == "delete"]
        self.assertEqual(len(delete_calls), 1)
        self.assertIn(kubectl.job_names[0], delete_calls[0]["cmd"])

    def test_pod_timeout_raises_and_deletes_job(self):
        kubectl = FakeKubectl(phase="Pending", pod_name="", pod_ip="")
        with patch("subprocess.run", side_effect=kubectl.run), \
             patch.dict(os.environ, {"TEASBENCH_SANDBOX_POD_TIMEOUT": "0"}):
            provider = InClusterK8sProvider(namespace="eidf230ns")
            with self.assertRaises(RuntimeError):
                provider.acquire("some/image:latest", label="task-timeout")

        # a zero-second timeout means the poll loop must never have run
        get_calls = [c for c in kubectl.calls if c["cmd"][3] == "get"]
        self.assertEqual(len(get_calls), 0)
        delete_calls = [c for c in kubectl.calls if c["cmd"][3] == "delete"]
        self.assertEqual(len(delete_calls), 1)
        self.assertIn(kubectl.job_names[0], delete_calls[0]["cmd"])

    def test_release_issues_kubectl_delete_job_with_handle(self):
        kubectl = FakeKubectl()
        provider = InClusterK8sProvider(namespace="eidf230ns")
        endpoint = SandboxEndpoint(host="http://10.1.2.3", port=9999,
                                   auth_token="tok", handle="fake-job-00099")
        with patch("subprocess.run", side_effect=kubectl.run):
            provider.release(endpoint)

        self.assertEqual(len(kubectl.calls), 1)
        cmd = kubectl.calls[0]["cmd"]
        self.assertIn("delete", cmd)
        self.assertIn("job", cmd)
        self.assertIn("fake-job-00099", cmd)
        self.assertIn("--wait=false", cmd)
        self.assertIn("--ignore-not-found=true", cmd)

    def test_release_is_a_no_op_for_none(self):
        provider = InClusterK8sProvider(namespace="eidf230ns")
        with patch("subprocess.run") as mock_run:
            provider.release(None)
        mock_run.assert_not_called()


class PortForwardK8sProviderTests(unittest.TestCase):
    """B3: the login-node fallback provider."""

    def test_acquire_allocates_local_port_and_starts_kubectl_port_forward(self):
        kubectl = FakeKubectl(phase="Running", pod_name="swe-rex-pf001", pod_ip="10.9.9.9")
        popen_calls = []

        def fake_popen(cmd, *a, **kw):
            popen_calls.append(list(cmd))
            return FakePopen(cmd)

        provider = PortForwardK8sProvider(namespace="eidf230ns", queue="eidf230ns-user-queue")
        endpoint = None
        try:
            with patch("subprocess.run", side_effect=kubectl.run), \
                 patch("subprocess.Popen", side_effect=fake_popen), \
                 patch("urllib.request.urlopen", side_effect=fake_urlopen_ok):
                endpoint = provider.acquire("some/image:latest", label="task-pf")

            self.assertEqual(endpoint.host, "http://127.0.0.1")
            self.assertIsInstance(endpoint.port, int)
            self.assertGreater(endpoint.port, 0)
            self.assertNotEqual(endpoint.port, 9999)  # the OS-assigned LOCAL port, not the pod's

            self.assertTrue(popen_calls, "expected kubectl port-forward to be started")
            pf_cmd = popen_calls[0]
            self.assertIn("port-forward", pf_cmd)
            self.assertIn("pod/swe-rex-pf001", pf_cmd)
            self.assertIn(f"{endpoint.port}:9999", pf_cmd)
        finally:
            if endpoint is not None:
                provider.release(endpoint)

    def test_pod_failed_phase_raises_and_deletes_job_without_port_forwarding(self):
        kubectl = FakeKubectl(phase="Failed", pod_name="", pod_ip="")
        provider = PortForwardK8sProvider(namespace="eidf230ns")
        with patch("subprocess.run", side_effect=kubectl.run), \
             patch("subprocess.Popen") as mock_popen:
            with self.assertRaises(RuntimeError):
                provider.acquire("some/image:latest", label="task-pf-failed")

        mock_popen.assert_not_called()
        delete_calls = [c for c in kubectl.calls if c["cmd"][3] == "delete"]
        self.assertEqual(len(delete_calls), 1)


class SharedExecSupportTests(unittest.TestCase):
    """B4: acquire_exec/release_exec are identical on both providers
    (shared base class), and K8sExecHandle drives cp/exec via kubectl."""

    def test_both_providers_share_the_same_acquire_exec_and_release_exec(self):
        self.assertIs(InClusterK8sProvider.acquire_exec, PortForwardK8sProvider.acquire_exec)
        self.assertIs(InClusterK8sProvider.release_exec, PortForwardK8sProvider.release_exec)

    def test_acquire_exec_starts_pod_and_handle_drives_cp_and_exec(self):
        kubectl = FakeKubectl(phase="Running", pod_name="swe-eval-77", pod_ip="")
        provider = InClusterK8sProvider(namespace="eidf230ns", queue="eidf230ns-user-queue")

        with patch("subprocess.run", side_effect=kubectl.run):
            handle = provider.acquire_exec(
                "docker.io/swebench/sweb.eval.x86_64.foo:latest", label="task-eval")
            self.assertEqual(handle.pod_name, "swe-eval-77")

            handle.cp("/local/patch.diff", "/testbed/patch.diff")
            result = handle.exec("git apply /testbed/patch.diff")
            provider.release_exec(handle)

        create_calls = [c for c in kubectl.calls if c["cmd"][3] == "create"]
        self.assertEqual(len(create_calls), 1)
        body = json.loads(create_calls[0]["kwargs"]["input"])
        self.assertEqual(body["metadata"]["labels"]["app"], "teasbench-eval")
        self.assertEqual(body["metadata"]["generateName"], "swe-eval-")
        self.assertEqual(
            body["spec"]["template"]["spec"]["containers"][0]["command"],
            ["sleep", "10800"])
        self.assertEqual(body["spec"]["activeDeadlineSeconds"], 3 * 3600)

        cp_calls = [c for c in kubectl.calls if c["cmd"][3] == "cp"]
        self.assertEqual(len(cp_calls), 1)
        self.assertIn("swe-eval-77:/testbed/patch.diff", cp_calls[0]["cmd"])

        exec_calls = [c for c in kubectl.calls if c["cmd"][3] == "exec"]
        self.assertEqual(len(exec_calls), 1)
        self.assertIn("swe-eval-77", exec_calls[0]["cmd"])
        self.assertEqual(result.stdout, "ok")

        delete_calls = [c for c in kubectl.calls if c["cmd"][3] == "delete"]
        self.assertEqual(len(delete_calls), 1)


if __name__ == "__main__":
    unittest.main()
