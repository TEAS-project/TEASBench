"""Tests for k8s_pod_providers - no cluster, no kubectl binary, no
network. `subprocess.run`/`subprocess.Popen` and `urllib.request.urlopen`
are monkeypatched with in-process fakes that never touch a real process
or socket beyond the OS-local free-port lookup used by
PortForwardK8sProvider (a loopback bind, not a network call)."""

import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline" / "k8s" / "lib"))

# From the module, not the package: SandboxEndpoint and the private
# _sandbox_job_spec are implementation detail that __init__ deliberately does
# not re-export (only the two provider classes are public API).
_PATCH_SCRIPT = (Path(__file__).resolve().parents[1] / "pipeline" / "k8s"
                 / "setup" / "patch_swerex_retries.py")
_spec = importlib.util.spec_from_file_location("patch_swerex_retries", _PATCH_SCRIPT)
patch_swerex_retries = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch_swerex_retries)
# The fixture below must be text the patch actually recognises, so take it from
# the patch itself rather than keeping a copy that can quietly drift.
OLD_MIDDLEWARE_TEXT = patch_swerex_retries.OLD_MIDDLEWARE

from k8s_pod_providers.providers import (
    _PortForwardSandbox,
    InClusterK8sProvider,
    PortForwardK8sProvider,
    SandboxEndpoint,
    _journal,
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
    `.poll()` returns None (still running) until `.kill()` is called, OR
    until `dies_after_polls` scripted deaths have been observed - this is
    what lets a babysitter test express "the process itself exits after
    N health checks" (the OLD, `proc.poll()`-only failure signal) as
    opposed to "the process stays alive but the tunnel stops forwarding"
    (the NEW failure mode `_keep_pf_alive` was rewritten to catch, which a
    test expresses by leaving `dies_after_polls=None` and instead
    scripting `urlopen` to fail - see `ScriptedUrlopen` below)."""

    _next_pid = 1000

    def __init__(self, cmd, dies_after_polls=None):
        self.args = list(cmd)
        self._returncode = None
        self._poll_count = 0
        self._dies_after_polls = dies_after_polls
        self.pid = FakePopen._next_pid
        FakePopen._next_pid += 1

    def poll(self):
        self._poll_count += 1
        if (self._returncode is None and self._dies_after_polls is not None
                and self._poll_count >= self._dies_after_polls):
            self._returncode = 1  # scripted death, not a kill()
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


class ScriptedProbe:
    """Stand-in for `_PortForwardSandbox._probe_tunnel`, scriptable per-call.

    The babysitter's probe is a TCP connect to the tunnel's local port, not an
    HTTP request, so it cannot be steered through ScriptedUrlopen -- that one
    still drives the `/is_alive` readiness check `start()` makes before handing
    the sandbox over. Splitting them is the point: the two probes answer
    different questions (is the server up / is the tunnel up), and conflating
    them is what made the babysitter restart healthy tunnels.

    Same semantics as ScriptedUrlopen: `outcomes` is consumed one per call and
    the last entry repeats forever, so a test can flip `.outcomes` mid-run
    without predicting how many probes will happen.
    """

    def __init__(self, outcomes=(True,)):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, *_a, **_kw):
        outcomes = self.outcomes  # snapshot in case a test reassigns mid-call
        idx = min(self.calls, len(outcomes) - 1)
        self.calls += 1
        return outcomes[idx]


class ScriptedUrlopen:
    """Stand-in for `urllib.request.urlopen`, scriptable per-call so a
    test can control exactly which `/is_alive` probes succeed and which
    fail. `outcomes` is a list of booleans consumed one per call, in
    order; once exhausted the last entry repeats forever (so a test can
    do e.g. `outcomes = [True]` for a clean start() and then mutate
    `.outcomes` afterwards to `[False]` to make every subsequent
    babysitter probe fail, without needing to know in advance exactly
    how many probes will occur). Every call is recorded in `.calls` for
    assertions and so the object is safe to mutate concurrently from the
    test thread while the babysitter thread is reading it (list
    reads/writes are individually atomic under the GIL, which is all
    these tests need)."""

    def __init__(self, outcomes=(True,)):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, req, timeout=None):
        outcomes = self.outcomes  # snapshot in case a test reassigns mid-call
        idx = min(len(self.calls), len(outcomes) - 1)
        ok = outcomes[idx]
        self.calls.append(req.full_url)
        if ok:
            return FakeHTTPResponse()
        raise OSError("simulated /is_alive probe failure")


@contextmanager
def _env_without(*names):
    """Guarantee the given env vars are UNSET for the duration of the
    block, restoring whatever was there afterwards. Unlike
    `patch.dict(os.environ, {...})`, which only ever *adds*/overrides
    keys, this is needed for the "new env vars unset" tests: a
    developer's shell could happen to export a TEASBENCH_PF_* var for
    unrelated reasons, and the test must not silently pass or fail based
    on that."""
    saved = {name: os.environ.pop(name, None) for name in names}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def _wait_until(condition, timeout=3.0, interval=0.01):
    """Poll `condition` (a zero-arg callable) until truthy or `timeout`
    seconds elapse, sleeping in small slices in between. Used throughout
    the babysitter tests instead of a fixed `time.sleep(N)` so each test
    only waits as long as the background thread actually needs (fast
    machine: near-instant; loaded CI box: still correct, just slower) -
    the `timeout` is a backstop, not the expected duration. Returns the
    final condition() value (truthy on success, falsy on timeout) so
    callers can assert on it directly."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return bool(condition())


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

    def test_pod_applies_the_swerex_server_patch_before_serving(self):
        """The retries the client now performs are only safe because the server
        awaits an in-flight duplicate instead of running it twice, and the
        server lives in the pod, not on the login node. If this drops out of
        the pod command, retries silently become able to put two commands on
        one shell."""
        args = _sandbox_job_spec("ns", "q", "img", "tok", 9999)[
            "spec"]["template"]["spec"]["containers"][0]["args"][0]
        self.assertIn("TEASBENCH_SWEREX_INFLIGHT_DEDUPE_PATCH_APPLIED", args)
        self.assertIn("TEASBENCH_SWEREX_NONBLOCKING_PATCH_APPLIED", args)
        # Ordering is load-bearing in both directions: patch after the install
        # (nothing to patch before it) and before the server starts (patching a
        # running server does nothing).
        self.assertLess(args.index("pip install"),
                        args.index("TEASBENCH_SWEREX_PATCH_EOF_0"))
        self.assertLess(args.rindex("TEASBENCH_SWEREX_PATCH_EOF_"),
                        args.index("exec python3 -m swerex"))
        # Each script gets its own delimiter; a shared one would end the first
        # heredoc at the wrong place and feed the rest to the shell.
        delimiters = sorted(set(re.findall(r"TEASBENCH_SWEREX_PATCH_EOF_\d+", args)))
        self.assertEqual(delimiters, ["TEASBENCH_SWEREX_PATCH_EOF_0",
                                      "TEASBENCH_SWEREX_PATCH_EOF_1"])

    def test_pod_command_is_valid_shell(self):
        """The patch source is embedded in a heredoc; a quoting slip would only
        show up as a pod that dies at startup, mid-run."""
        args = _sandbox_job_spec("ns", "q", "img", "tok", 9999)[
            "spec"]["template"]["spec"]["containers"][0]["args"][0]
        proc = subprocess.run(["bash", "-n"], input=args, text=True,
                              capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_embedded_patch_source_is_the_real_script(self):
        """Read from disk rather than duplicated, so the pod's server half and
        the login node's client half cannot drift apart."""
        args = _sandbox_job_spec("ns", "q", "img", "tok", 9999)[
            "spec"]["template"]["spec"]["containers"][0]["args"][0]
        setup = Path(__file__).resolve().parents[1] / "pipeline" / "k8s" / "setup"
        for name in ("patch_swerex_retries.py", "patch_swerex_nonblocking.py"):
            with self.subTest(script=name):
                self.assertIn((setup / name).read_text(), args)

    def test_pod_patch_survives_a_client_that_cannot_import(self):
        """A sandbox pod has no aiohttp -- it is not a swe-rex dependency, and
        only the login node installs it -- so importing the *client* module
        there raises ModuleNotFoundError. The patch must treat that as "not
        applicable", not as failure: under the pod's `set -e` a non-zero exit
        kills the container before it serves, and every task in the run then
        dies with `swerex not alive`."""
        script = (Path(__file__).resolve().parents[1] / "pipeline" / "k8s"
                  / "setup" / "patch_swerex_retries.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pkg = root / "swerex"
            (pkg / "runtime").mkdir(parents=True)
            (pkg / "__init__.py").write_text("")
            (pkg / "runtime" / "__init__.py").write_text("")
            # Stands in for the real client module, which drags in aiohttp.
            (pkg / "runtime" / "remote.py").write_text(
                "import aiohttp_not_installed_in_pods\n")
            # Minimal but faithful copy of the two spots the server half edits.
            (pkg / "server.py").write_text(
                "class ResponseManager:\n"
                "    def __init__(self):\n"
                "        self.last_processed_request_id = None\n"
                "        self.last_processed_response = None\n"
                "\n"
                "    def get_response(self, request_id):\n"
                "        if request_id == self.last_processed_request_id:\n"
                "            return self.last_processed_response\n"
                "        return None\n"
                "\n"
                "    def set_response(self, request_id, response):\n"
                "        self.last_processed_request_id = request_id\n"
                "        self.last_processed_response = response\n"
                "\n"
                "\n"
                "async def handle_request_id(request, call_next):\n"
                + OLD_MIDDLEWARE_TEXT + "\n")
            env = dict(os.environ, PYTHONPATH=str(root))
            proc = subprocess.run(
                [sys.executable, str(script), "--require", "server"],
                capture_output=True, text=True, env=env, cwd=td)
            self.assertEqual(proc.returncode, 0,
                             f"pod patch failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertIn("client: not applicable here", proc.stdout)
            self.assertIn("wait_for_in_flight", (pkg / "server.py").read_text())

    def test_require_client_still_fails_when_the_client_is_missing(self):
        """The flip side: skipping must not become a silent pass on the login
        node, where the client half is the one that matters."""
        script = (Path(__file__).resolve().parents[1] / "pipeline" / "k8s"
                  / "setup" / "patch_swerex_retries.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pkg = root / "swerex"
            (pkg / "runtime").mkdir(parents=True)
            (pkg / "__init__.py").write_text("")
            (pkg / "runtime" / "__init__.py").write_text("")
            (pkg / "runtime" / "remote.py").write_text(
                "import aiohttp_not_installed_in_pods\n")
            (pkg / "server.py").write_text("")
            env = dict(os.environ, PYTHONPATH=str(root))
            proc = subprocess.run(
                [sys.executable, str(script), "--require", "client"],
                capture_output=True, text=True, env=env, cwd=td)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("--require client", proc.stderr)

    def test_nonblocking_patch_leaves_interrupt_alone(self):
        """`BashSession.run` awaits `self.interrupt`. Wrapping both against one
        non-reentrant lock deadlocks: run holds it, interrupt is dispatched to
        another thread and waits for it forever. Offloading run already moves
        interrupt off the loop, and the server has no route to it directly."""
        src = (Path(__file__).resolve().parents[1] / "pipeline" / "k8s" / "setup"
               / "patch_swerex_nonblocking.py").read_text()
        wrapped = re.findall(r'\((BashSession|LocalRuntime), "(\w+)"\)', src)
        self.assertEqual(sorted(wrapped),
                         [("BashSession", "close"), ("BashSession", "run"),
                          ("BashSession", "start"), ("LocalRuntime", "execute")])
        self.assertNotIn(("BashSession", "interrupt"), wrapped)

    def test_nonblocking_patch_applies_and_is_idempotent(self):
        """Applied against a stand-in for swerex.runtime.local: the patch binds
        to class/method names at import time, so a layout change has to be a
        refusal rather than a file that imports and silently does nothing."""
        script = (Path(__file__).resolve().parents[1] / "pipeline" / "k8s" / "setup"
                  / "patch_swerex_nonblocking.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pkg = root / "swerex" / "runtime"
            pkg.mkdir(parents=True)
            (root / "swerex" / "__init__.py").write_text("")
            (pkg / "__init__.py").write_text("")
            (pkg / "local.py").write_text(
                "class BashSession:\n"
                "    async def start(self): return 'start'\n"
                "    async def run(self, a): return a\n"
                "    async def interrupt(self, a): return a\n"
                "    async def close(self): return 'close'\n"
                "\n"
                "class LocalRuntime:\n"
                "    async def execute(self, c): return c\n")
            env = dict(os.environ, PYTHONPATH=str(root))
            first = subprocess.run([sys.executable, str(script)],
                                   capture_output=True, text=True, env=env, cwd=td)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run([sys.executable, str(script)],
                                    capture_output=True, text=True, env=env, cwd=td)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already patched", second.stdout)

            # And the patched module must still import, with run offloaded and
            # interrupt untouched.
            check = subprocess.run(
                [sys.executable, "-c",
                 "import swerex.runtime.local as l;"
                 "print(hasattr(l.BashSession.run,'_tb_wrapped'),"
                 "      hasattr(l.BashSession.interrupt,'_tb_wrapped'))"],
                capture_output=True, text=True, env=env, cwd=td)
            self.assertEqual(check.returncode, 0, check.stderr)
            self.assertEqual(check.stdout.strip(), "True False")

    def test_nonblocking_patch_refuses_an_unrecognised_layout(self):
        script = (Path(__file__).resolve().parents[1] / "pipeline" / "k8s" / "setup"
                  / "patch_swerex_nonblocking.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pkg = root / "swerex" / "runtime"
            pkg.mkdir(parents=True)
            (root / "swerex" / "__init__.py").write_text("")
            (pkg / "__init__.py").write_text("")
            (pkg / "local.py").write_text("class SomethingElse:\n    pass\n")
            env = dict(os.environ, PYTHONPATH=str(root))
            proc = subprocess.run([sys.executable, str(script)],
                                  capture_output=True, text=True, env=env, cwd=td)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("layout has changed", proc.stderr)


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


def _fast_pod_timeouts():
    """Env overrides shared by every babysitter test: bound the pod/swerex
    wait loops tightly so a test that (by construction) never satisfies
    them fails fast instead of hanging for the real 1200s/600s defaults."""
    return {
        "TEASBENCH_SANDBOX_POD_TIMEOUT": "5",
        "TEASBENCH_SWEREX_TIMEOUT": "5",
    }


class PortForwardBabysitterTests(unittest.TestCase):
    """`_keep_pf_alive` (rewritten to HTTP-probe /is_alive through the
    tunnel instead of only polling proc.poll()) and `stop()`. No cluster,
    no network, no real port-forward - `subprocess.Popen` and
    `urllib.request.urlopen` are the in-process fakes above, and the
    thread this exercises is the real daemon thread `start()` launches,
    driven with tiny TEASBENCH_PF_* intervals via patch.dict so the tests
    stay fast. Every test releases the endpoint in `finally` - these start
    a real thread and a leak would pollute later tests."""

    def test_probe_failure_with_live_process_triggers_restart_on_same_local_port(self):
        kubectl = FakeKubectl(phase="Running", pod_name="swe-rex-babysit1", pod_ip="10.5.5.5")
        popen_instances = []

        def fake_popen(cmd, *a, **kw):
            p = FakePopen(cmd)  # never exits on its own - only the probe fails
            popen_instances.append(p)
            return p

        urlopen = ScriptedUrlopen(outcomes=[True])  # start() readiness probe
        probe = ScriptedProbe(outcomes=[True])      # babysitter tunnel probe

        provider = PortForwardK8sProvider(namespace="eidf230ns")
        endpoint = None
        env = dict(_fast_pod_timeouts(), **{
            "TEASBENCH_PF_PROBE_INTERVAL": "0.02",
            "TEASBENCH_PF_PROBE_TIMEOUT": "1",
            "TEASBENCH_PF_PROBE_FAILURES": "2",
            "TEASBENCH_PF_MAX_RESTARTS": "5",
            "TEASBENCH_PF_BACKOFF_MAX": "1",
        })
        try:
            with tempfile.TemporaryDirectory() as tmp:
                journal_path = os.path.join(tmp, "portforward-events.jsonl")
                env["TEASBENCH_PF_EVENTS"] = journal_path
                with patch("subprocess.run", side_effect=kubectl.run), \
                     patch("subprocess.Popen", side_effect=fake_popen), \
                     patch("urllib.request.urlopen", side_effect=urlopen), \
                     patch.object(_PortForwardSandbox, "_probe_tunnel",
                                  lambda self, timeout: probe()), \
                     patch.dict(os.environ, env):
                    endpoint = provider.acquire("some/image:latest", label="task-restart")
                    self.assertEqual(len(popen_instances), 1)
                    first_port_arg = popen_instances[0].args[-1]

                    # Flip every subsequent probe to fail - the babysitter
                    # (not start(), which already returned) must notice.
                    probe.outcomes = [False]

                    self.assertTrue(
                        _wait_until(lambda: len(popen_instances) >= 2, timeout=5.0),
                        "babysitter never restarted the tunnel after repeated probe failures")

                    second_port_arg = popen_instances[1].args[-1]
                    # SWE-agent was handed the original local_port URL at
                    # launch and cannot learn a new one - the restart MUST
                    # reuse it (CONTRACT.md, class docstring).
                    self.assertEqual(first_port_arg, second_port_arg)
                    self.assertEqual(endpoint.port, int(first_port_arg.split(":")[0]))

                provider.release(endpoint)

                events = [json.loads(l) for l in Path(journal_path).read_text().splitlines()]
            drops = [e for e in events if e["event"] == "pf_drop"]
            running_drops = [e for e in drops if e["phase"] == "running"]
            self.assertTrue(running_drops, f"expected a running-phase pf_drop, got {events}")
            self.assertEqual(running_drops[0]["reason"], "probe_failed")
            restarts = [e for e in events if e["event"] == "pf_restart"]
            self.assertTrue(restarts)
            self.assertEqual(restarts[-1]["local_port"], endpoint.port)
        finally:
            if endpoint is not None:
                provider.release(endpoint)

    def test_single_probe_failure_does_not_restart(self):
        kubectl = FakeKubectl(phase="Running", pod_name="swe-rex-babysit2", pod_ip="10.5.5.6")
        popen_instances = []

        def fake_popen(cmd, *a, **kw):
            p = FakePopen(cmd)
            popen_instances.append(p)
            return p

        # start() succeeds; the babysitter's 1st tunnel probe fails and every
        # later one succeeds - never two CONSECUTIVE failures.
        urlopen = ScriptedUrlopen(outcomes=[True])
        probe = ScriptedProbe(outcomes=[False, True])

        provider = PortForwardK8sProvider(namespace="eidf230ns")
        endpoint = None
        env = dict(_fast_pod_timeouts(), **{
            "TEASBENCH_PF_PROBE_INTERVAL": "0.02",
            "TEASBENCH_PF_PROBE_TIMEOUT": "1",
            "TEASBENCH_PF_PROBE_FAILURES": "2",
            "TEASBENCH_PF_MAX_RESTARTS": "5",
            "TEASBENCH_PF_BACKOFF_MAX": "1",
        })
        try:
            with tempfile.TemporaryDirectory() as tmp:
                journal_path = os.path.join(tmp, "portforward-events.jsonl")
                env["TEASBENCH_PF_EVENTS"] = journal_path
                with patch("subprocess.run", side_effect=kubectl.run), \
                     patch("subprocess.Popen", side_effect=fake_popen), \
                     patch("urllib.request.urlopen", side_effect=urlopen), \
                     patch.object(_PortForwardSandbox, "_probe_tunnel",
                                  lambda self, timeout: probe()), \
                     patch.dict(os.environ, env):
                    endpoint = provider.acquire("some/image:latest", label="task-no-restart")

                    # Give the babysitter several probe_interval windows to
                    # run through the scripted single blip and settle back
                    # to healthy - a short, bounded real wait, not a
                    # guessed exact duration.
                    time.sleep(0.02 * 8)

                    self.assertEqual(len(popen_instances), 1,
                                      "a single probe failure below the threshold must not restart")
                provider.release(endpoint)

                events = [json.loads(l) for l in Path(journal_path).read_text().splitlines()]
            running_drops = [e for e in events if e["event"] == "pf_drop" and e["phase"] == "running"]
            self.assertEqual(running_drops, [])
        finally:
            if endpoint is not None:
                provider.release(endpoint)

    def test_pod_gone_stops_respawning_instead_of_hot_looping(self):
        kubectl = FakeKubectl(phase="Running", pod_name="swe-rex-babysit3", pod_ip="10.5.5.7")
        popen_instances = []

        def fake_popen(cmd, *a, **kw):
            p = FakePopen(cmd)
            popen_instances.append(p)
            return p

        urlopen = ScriptedUrlopen(outcomes=[True])  # start() readiness probe
        probe = ScriptedProbe(outcomes=[True])      # babysitter tunnel probe

        provider = PortForwardK8sProvider(namespace="eidf230ns")
        endpoint = None
        env = dict(_fast_pod_timeouts(), **{
            "TEASBENCH_PF_PROBE_INTERVAL": "0.02",
            "TEASBENCH_PF_PROBE_TIMEOUT": "1",
            "TEASBENCH_PF_PROBE_FAILURES": "1",  # restart-eligible on the first failure
            "TEASBENCH_PF_MAX_RESTARTS": "5",
            "TEASBENCH_PF_BACKOFF_MAX": "1",
        })
        try:
            with tempfile.TemporaryDirectory() as tmp:
                journal_path = os.path.join(tmp, "portforward-events.jsonl")
                env["TEASBENCH_PF_EVENTS"] = journal_path
                with patch("subprocess.run", side_effect=kubectl.run), \
                     patch("subprocess.Popen", side_effect=fake_popen), \
                     patch("urllib.request.urlopen", side_effect=urlopen), \
                     patch.object(_PortForwardSandbox, "_probe_tunnel",
                                  lambda self, timeout: probe()), \
                     patch.dict(os.environ, env):
                    endpoint = provider.acquire("some/image:latest", label="task-pod-gone")
                    sandbox = endpoint.handle

                    # Now the pod is gone AND the tunnel is broken - both
                    # the babysitter's own probe and its _pod_alive() guard
                    # must see this.
                    probe.outcomes = [False]
                    kubectl.phase = "Failed"

                    self.assertTrue(
                        _wait_until(lambda: not sandbox._pf_keeper.is_alive(), timeout=5.0),
                        "babysitter kept running after the pod disappeared")

                    # No hot-loop: the babysitter must have given up without
                    # ever attempting a respawn against a dead pod.
                    self.assertEqual(len(popen_instances), 1)

                provider.release(endpoint)

                events = [json.loads(l) for l in Path(journal_path).read_text().splitlines()]
            pod_gone_drops = [e for e in events
                               if e["event"] == "pf_drop" and e.get("reason") == "pod_gone"]
            self.assertTrue(pod_gone_drops)
            self.assertEqual(pod_gone_drops[0]["phase"], "running")
            unrecoverable = [e for e in events if e["event"] == "pf_unrecoverable"]
            self.assertTrue(unrecoverable)
            self.assertEqual(unrecoverable[0]["reason"], "pod_gone")
            self.assertEqual([e for e in events if e["event"] == "pf_restart"], [])
        finally:
            if endpoint is not None:
                provider.release(endpoint)

    def test_stop_during_restart_leaves_no_orphan_process_and_joins_keeper(self):
        kubectl = FakeKubectl(phase="Running", pod_name="swe-rex-babysit4", pod_ip="10.5.5.8")
        popen_instances = []

        def fake_popen(cmd, *a, **kw):
            p = FakePopen(cmd)
            popen_instances.append(p)
            return p

        urlopen = ScriptedUrlopen(outcomes=[True])  # start() readiness probe
        probe = ScriptedProbe(outcomes=[True])      # babysitter tunnel probe

        provider = PortForwardK8sProvider(namespace="eidf230ns")
        endpoint = None
        env = dict(_fast_pod_timeouts(), **{
            "TEASBENCH_PF_PROBE_INTERVAL": "0.02",
            "TEASBENCH_PF_PROBE_TIMEOUT": "1",
            "TEASBENCH_PF_PROBE_FAILURES": "1",
            "TEASBENCH_PF_MAX_RESTARTS": "5",
            # A long backoff so stop() races the babysitter mid-wait rather
            # than after it has already respawned.
            "TEASBENCH_PF_BACKOFF_MAX": "30",
        })
        try:
            with tempfile.TemporaryDirectory() as tmp:
                journal_path = os.path.join(tmp, "portforward-events.jsonl")
                env["TEASBENCH_PF_EVENTS"] = journal_path
                with patch("subprocess.run", side_effect=kubectl.run), \
                     patch("subprocess.Popen", side_effect=fake_popen), \
                     patch("urllib.request.urlopen", side_effect=urlopen), \
                     patch.object(_PortForwardSandbox, "_probe_tunnel",
                                  lambda self, timeout: probe()), \
                     patch.dict(os.environ, env):
                    endpoint = provider.acquire("some/image:latest", label="task-stop-race")
                    sandbox = endpoint.handle

                    probe.outcomes = [False]  # every subsequent probe fails
                    # Wait for the babysitter to have logged the running-phase
                    # drop (i.e. it is now inside its ~1s pre-restart backoff
                    # sleep) before calling stop() - this is the race window
                    # the class docstring's _lock exists for.
                    self.assertTrue(_wait_until(
                        lambda: Path(journal_path).exists()
                        and any(json.loads(l).get("event") == "pf_drop"
                                for l in Path(journal_path).read_text().splitlines()),
                        timeout=5.0))

                    provider.release(endpoint)
                    endpoint = None  # released; finally must not double-release

                    self.assertFalse(sandbox._pf_keeper.is_alive(),
                                      "keeper thread must be joined by the time stop() returns")
                    for p in popen_instances:
                        self.assertIsNotNone(p.poll(), "orphan port-forward process left running")
        finally:
            if endpoint is not None:
                provider.release(endpoint)

    def test_journal_records_startup_restart_and_running_drop_with_correct_phase(self):
        kubectl = FakeKubectl(phase="Running", pod_name="swe-rex-babysit5", pod_ip="10.5.5.9")
        popen_instances = []
        spawn_count = [0]

        def fake_popen(cmd, *a, **kw):
            spawn_count[0] += 1
            # The FIRST tunnel dies immediately (pod still pip-installing) -
            # start()'s own retry loop must handle this and tag it
            # phase="startup". Every later spawn stays up.
            dies_after = 1 if spawn_count[0] == 1 else None
            p = FakePopen(cmd, dies_after_polls=dies_after)
            popen_instances.append(p)
            return p

        urlopen = ScriptedUrlopen(outcomes=[True])  # start() readiness probe
        probe = ScriptedProbe(outcomes=[True])      # babysitter tunnel probe

        provider = PortForwardK8sProvider(namespace="eidf230ns")
        endpoint = None
        env = dict(_fast_pod_timeouts(), **{
            "TEASBENCH_PF_PROBE_INTERVAL": "0.02",
            "TEASBENCH_PF_PROBE_TIMEOUT": "1",
            "TEASBENCH_PF_PROBE_FAILURES": "1",
            "TEASBENCH_PF_MAX_RESTARTS": "5",
            "TEASBENCH_PF_BACKOFF_MAX": "1",
        })
        try:
            with tempfile.TemporaryDirectory() as tmp:
                journal_path = os.path.join(tmp, "portforward-events.jsonl")
                env["TEASBENCH_PF_EVENTS"] = journal_path
                with patch("subprocess.run", side_effect=kubectl.run), \
                     patch("subprocess.Popen", side_effect=fake_popen), \
                     patch("urllib.request.urlopen", side_effect=urlopen), \
                     patch.object(_PortForwardSandbox, "_probe_tunnel",
                                  lambda self, timeout: probe()), \
                     patch.dict(os.environ, env):
                    # start()'s internal "not ready yet" retry uses a fixed
                    # (non-configurable) 3s pause before re-probing - real
                    # wait, unavoidable without mocking time itself.
                    endpoint = provider.acquire("some/image:latest", label="task-journal-phase")

                    # Now drive a babysitter (post-startup, "running" phase)
                    # drop the same way test 1 does.
                    probe.outcomes = [False]
                    self.assertTrue(_wait_until(
                        lambda: Path(journal_path).exists() and any(
                            json.loads(l).get("event") == "pf_drop"
                            and json.loads(l).get("phase") == "running"
                            for l in Path(journal_path).read_text().splitlines()),
                        timeout=5.0))

                provider.release(endpoint)

                events = [json.loads(l) for l in Path(journal_path).read_text().splitlines()]
            drops = [e for e in events if e["event"] == "pf_drop"]
            startup_drops = [e for e in drops if e["phase"] == "startup"]
            running_drops = [e for e in drops if e["phase"] == "running"]
            self.assertTrue(startup_drops, f"expected a startup-phase pf_drop, got {events}")
            self.assertEqual(startup_drops[0]["reason"], "process_exited")
            self.assertTrue(running_drops, f"expected a running-phase pf_drop, got {events}")
            self.assertEqual(running_drops[0]["reason"], "probe_failed")
            # The retry classifier (swebench_run_audit) only acts on
            # phase == "running" - getting these backwards would either
            # flood the retry list with every ordinary pod-startup churn
            # event, or silently drop real infrastructure evidence.
            self.assertNotEqual(startup_drops[0]["phase"], running_drops[0]["phase"])
        finally:
            if endpoint is not None:
                provider.release(endpoint)

    def test_journal_is_noop_when_pf_events_unset(self):
        with _env_without("TEASBENCH_PF_EVENTS"):
            with patch("builtins.open") as mock_open:
                _journal("task-x", "pf_drop", phase="running", reason="probe_failed")
            mock_open.assert_not_called()

    def test_babysitter_env_vars_unset_is_inert(self):
        """With none of the TEASBENCH_PF_* knobs set (the state every
        non-driver caller of this provider is in today), acquire/release
        must behave exactly as before this rewrite: no crash, and no
        journal file appears anywhere - this is what protects every
        existing caller that never heard of this feature."""
        kubectl = FakeKubectl(phase="Running", pod_name="swe-rex-inert", pod_ip="10.5.5.10")

        def fake_popen(cmd, *a, **kw):
            return FakePopen(cmd)

        pf_vars = ("TEASBENCH_PF_EVENTS", "TEASBENCH_PF_LOG_DIR",
                   "TEASBENCH_PF_PROBE_INTERVAL", "TEASBENCH_PF_PROBE_TIMEOUT",
                   "TEASBENCH_PF_PROBE_FAILURES", "TEASBENCH_PF_MAX_RESTARTS",
                   "TEASBENCH_PF_BACKOFF_MAX")
        provider = PortForwardK8sProvider(namespace="eidf230ns")
        endpoint = None
        try:
            with tempfile.TemporaryDirectory() as tmp, _env_without(*pf_vars):
                cwd = os.getcwd()
                os.chdir(tmp)
                try:
                    with patch("subprocess.run", side_effect=kubectl.run), \
                         patch("subprocess.Popen", side_effect=fake_popen), \
                         patch("urllib.request.urlopen", side_effect=fake_urlopen_ok), \
                         patch.dict(os.environ, _fast_pod_timeouts()):
                        endpoint = provider.acquire("some/image:latest", label="task-inert")
                        # Give the (real, but default 15s-interval) babysitter
                        # thread a moment to run its startup checks; it must
                        # not crash or write anything even though it's alive.
                        time.sleep(0.05)
                        provider.release(endpoint)
                        endpoint = None
                finally:
                    os.chdir(cwd)
                self.assertEqual(list(Path(tmp).glob("*.jsonl")), [])
        finally:
            if endpoint is not None:
                provider.release(endpoint)


class TunnelProbeTests(unittest.TestCase):
    """_probe_tunnel must answer "is the tunnel up", not "is the server idle".

    swe-rex blocks its own event loop for the duration of every command, so an
    HTTP probe reports a healthy tunnel as dead whenever the agent is doing
    real work -- and the restart that follows kills the request in flight.
    """

    def _sandbox(self, port):
        sb = _PortForwardSandbox.__new__(_PortForwardSandbox)
        sb.local_port = port
        sb.token = "tok"
        return sb

    @contextmanager
    def _listener(self, mode):
        """A stand-in for the local end of `kubectl port-forward`.

        mode "hold":  accept and keep the connection open, saying nothing.
                      This is a working tunnel to a server that is busy.
        mode "close": accept and immediately close, which is what kubectl does
                      when it cannot forward ("pod is not running").
        """
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        held = []
        stop = threading.Event()

        def serve():
            srv.settimeout(0.2)
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                except (socket.timeout, OSError):
                    continue
                if mode == "close":
                    conn.close()
                else:
                    held.append(conn)
        t = threading.Thread(target=serve, daemon=True)
        t.start()
        try:
            yield srv.getsockname()[1]
        finally:
            stop.set()
            t.join(2)
            for c in held:
                c.close()
            srv.close()

    def test_busy_server_with_a_live_tunnel_is_healthy(self):
        """The regression that matters: an established, silent connection is
        a working tunnel. Probing this with HTTP times out and looks dead."""
        sb_port_ctx = self._listener("hold")
        with sb_port_ctx as port:
            sb = self._sandbox(port)
            self.assertTrue(sb._probe_tunnel(5))
            # And the point of the change: the old HTTP probe calls this exact
            # same healthy tunnel dead, because nothing answers it.
            self.assertFalse(sb._probe_server(2))

    def test_forward_that_closes_immediately_is_dead(self):
        with self._listener("close") as port:
            self.assertFalse(self._sandbox(port)._probe_tunnel(5))

    def test_nothing_listening_is_dead(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.assertFalse(self._sandbox(port)._probe_tunnel(1))

    def test_healthy_probe_returns_promptly(self):
        """It waits for an EOF that never comes, so the healthy path costs the
        settle window -- a fifth of the timeout, capped at 1s -- not the whole
        timeout. A babysitter that stalls 30s per probe is its own problem."""
        with self._listener("hold") as port:
            t0 = time.time()
            self._sandbox(port)._probe_tunnel(30)
            self.assertLess(time.time() - t0, 3.0)


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
