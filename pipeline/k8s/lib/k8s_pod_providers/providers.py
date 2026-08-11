"""K8s sandbox + exec-container providers for SWE-bench-style benchmarks.

This is the deployment-scenario side of the AgentCAP<->TEAS interface.
AgentCAP consumes only endpoints - a `SandboxEndpoint` {host, port,
auth_token} speaking the swe-rex protocol, and an exec-container handle
(upload file / run command) - and loads the provider classes below by
dotted path (`k8s_pod_providers:InClusterK8sProvider` etc). AgentCAP
duck-types this interface, so nothing in this module imports agent_cap.

Nothing here is specific to one cluster: everything a site varies is read
from the environment (see the variable list below), which the pipeline
sets from the site profile in `pipeline/configs/sites/`.

Two providers, same capability, different connection strategy:

- InClusterK8sProvider (the default): the caller runs *inside* the
  cluster (e.g. a Job pod), so sandbox pod IPs are directly routable.
  No port-forward, no OS port allocation, no tunnel-babysitter thread.
  Needs the cluster to grant pods RBAC for jobs/pods.
- PortForwardK8sProvider (fallback): the caller runs from a login node
  outside the cluster, so it reaches the sandbox pod through
  `kubectl port-forward` on an OS-assigned local port instead. This is
  what a cluster that refuses pod RBAC needs (EIDF, today). It is a
  faithful port of AgentCAP's original `_K8sSidecar` / K8sSandboxProvider
  (agent_cap/agents/sandbox_providers.py), including the comments
  documenting the production failures that shaped it.

Both providers also support exec containers (K8sExecHandle) for the
harness evaluator, via a shared base class - kubectl cp/exec talk to the
k8s API server rather than the pod IP, so that part needs no connection
strategy at all.

Environment variables (all optional, read fresh at call time so tests
can monkeypatch them per-case):
    TEASBENCH_K8S_NAMESPACE        default "eidf230ns"
    TEASBENCH_K8S_QUEUE            default "<namespace>-user-queue"
    TEASBENCH_SANDBOX_PORT         default "9999"
    TEASBENCH_SANDBOX_CPU_REQUEST  default "2"
    TEASBENCH_SANDBOX_CPU_LIMIT    default "6"
    TEASBENCH_SANDBOX_MEM_REQUEST  default "4Gi"
    TEASBENCH_SANDBOX_MEM_LIMIT    default "24Gi"
    TEASBENCH_SWEREX_SPEC          default "swe-rex>=1.4.0"
    TEASBENCH_SANDBOX_POD_TIMEOUT  default "1200"
    TEASBENCH_SWEREX_TIMEOUT       default "600"
"""

import json
import os
import secrets
import socket
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass
class SandboxEndpoint:
    """A running swe-rex server the agent can attach to (deployment-agnostic).

    AgentCAP only ever reads `.host`, `.port`, `.auth_token` off this, and
    passes the whole object back to `release()` - `handle` is provider-
    private (a Job name, or a `_PortForwardSandbox` instance).
    """
    host: str
    port: int
    auth_token: str
    handle: object = field(default=None, repr=False)


def _env_int(name, default):
    """Read an integer env var, falling back to `default` (int or str)."""
    return int(os.environ.get(name, default))


def _kubectl(namespace, *args, input_text=None, timeout=120):
    return subprocess.run(
        ["kubectl", "-n", namespace, *args],
        input=input_text, capture_output=True, text=True, timeout=timeout,
    )


def _resource_limits():
    return {
        "requests": {
            "cpu": os.environ.get("TEASBENCH_SANDBOX_CPU_REQUEST", "2"),
            "memory": os.environ.get("TEASBENCH_SANDBOX_MEM_REQUEST", "4Gi"),
        },
        "limits": {
            "cpu": os.environ.get("TEASBENCH_SANDBOX_CPU_LIMIT", "6"),
            "memory": os.environ.get("TEASBENCH_SANDBOX_MEM_LIMIT", "24Gi"),
        },
    }


def _sandbox_job_spec(namespace, queue, image, token, port):
    """Job spec for a swe-rex sandbox pod. Shared by InClusterK8sProvider
    and PortForwardK8sProvider - only how they *connect* to the resulting
    pod differs, not how the pod is provisioned."""
    swerex_spec = os.environ.get("TEASBENCH_SWEREX_SPEC", "swe-rex>=1.4.0")
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "generateName": "swe-rex-",
            "namespace": namespace,
            "labels": {"app": "teasbench-sandbox",
                       "kueue.x-k8s.io/queue-name": queue},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 600,
            "activeDeadlineSeconds": 6 * 3600,
            "template": {
                "metadata": {"labels": {"app": "teasbench-sandbox"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "swebench",
                        "image": image,
                        "command": ["/bin/bash", "-c"],
                        "args": [
                            "set -e; "
                            "git config --global --add safe.directory '*'; "
                            f"python3 -m pip install --quiet --no-input '{swerex_spec}' && "
                            f"exec python3 -m swerex --port {port} --auth-token {token}"
                        ],
                        "ports": [{"containerPort": port}],
                        "env": [{"name": "PIP_BREAK_SYSTEM_PACKAGES", "value": "1"}],
                        "resources": _resource_limits(),
                    }],
                },
            },
        },
    }


def _exec_job_spec(namespace, queue, image):
    """Job spec for an exec container (harness eval: upload file / run
    command in the official instance image). No swe-rex, no port - the
    container just sleeps and is driven via `kubectl cp`/`kubectl exec`."""
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "generateName": "swe-eval-",
            "namespace": namespace,
            "labels": {"app": "teasbench-eval",
                       "kueue.x-k8s.io/queue-name": queue},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 600,
            "activeDeadlineSeconds": 3 * 3600,
            "template": {
                "metadata": {"labels": {"app": "teasbench-eval"}},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "eval",
                        "image": image,
                        "command": ["sleep", "10800"],
                        "resources": _resource_limits(),
                    }],
                },
            },
        },
    }


class K8sExecHandle:
    """A pod from an official SWE-bench instance image supporting cp/exec,
    for harness evaluation. Ported from AgentCAP's `K8sExecContainer`
    (agent_cap/agents/sandbox_providers.py) - `kubectl cp`/`kubectl exec`
    talk to the k8s API server, not the pod IP, so this needs no
    port-forward and works identically whether the caller runs in-cluster
    or from a login node.
    """

    def __init__(self, namespace, queue, image, label=""):
        self.namespace = namespace
        self.queue = queue
        self.image = image
        self.label = label
        self.job_name = ""
        self.pod_name = ""

    def start(self):
        pod_timeout_s = _env_int("TEASBENCH_SANDBOX_POD_TIMEOUT", 1200)
        job = _exec_job_spec(self.namespace, self.queue, self.image)
        r = _kubectl(self.namespace, "create", "-f", "-",
                     "-o", "jsonpath={.metadata.name}", input_text=json.dumps(job))
        if r.returncode != 0:
            raise RuntimeError(f"exec job create failed: {r.stderr[:300]}")
        self.job_name = r.stdout.strip()

        deadline = time.time() + pod_timeout_s
        while time.time() < deadline:
            r = _kubectl(self.namespace, "get", "pods", f"-l=job-name={self.job_name}",
                         "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}")
            parts = r.stdout.strip().split("|")
            phase, pod = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
            if phase == "Running" and pod:
                self.pod_name = pod
                return
            if phase == "Failed":
                raise RuntimeError(f"exec pod failed ({self.label or self.job_name})")
            time.sleep(5)
        raise RuntimeError(f"exec pod not Running after {pod_timeout_s}s "
                           f"({self.label or self.job_name})")

    def cp(self, local_path, remote_path, timeout=300):
        _kubectl(self.namespace, "cp", local_path,
                 f"{self.pod_name}:{remote_path}", timeout=timeout)

    def exec(self, command, timeout=300):
        return _kubectl(self.namespace, "exec", self.pod_name, "--",
                        "bash", "-c", command, timeout=timeout)

    def stop(self):
        if self.job_name:
            try:
                _kubectl(self.namespace, "delete", "job", self.job_name,
                         "--wait=false", "--ignore-not-found=true")
            except Exception:
                pass


class _BaseK8sProvider:
    """Namespace/queue resolution + exec-container support shared by both
    sandbox providers. Exec containers don't need a connection strategy,
    so acquire_exec/release_exec are identical for both."""

    def __init__(self, namespace=None, queue=None):
        self.namespace = namespace or os.environ.get("TEASBENCH_K8S_NAMESPACE", "eidf230ns")
        self.queue = queue or os.environ.get("TEASBENCH_K8S_QUEUE", f"{self.namespace}-user-queue")

    def acquire_exec(self, image, label=""):
        handle = K8sExecHandle(self.namespace, self.queue, image, label)
        try:
            handle.start()
        except Exception:
            handle.stop()
            raise
        return handle

    def release_exec(self, handle):
        if handle is not None:
            handle.stop()


class InClusterK8sProvider(_BaseK8sProvider):
    """Default sandbox provider. Runs from *inside* the cluster (e.g.
    a TEASBench Job pod), where sandbox pod IPs are directly routable -
    so, unlike PortForwardK8sProvider, this needs no port-forward, no OS
    port allocation, no `start_new_session`, and no tunnel-babysitter
    thread: the agent just talks to the pod IP over the cluster network.
    """

    def acquire(self, image, label=""):
        token = secrets.token_hex(16)  # per-sandbox random: this is a shared cluster
        port = _env_int("TEASBENCH_SANDBOX_PORT", 9999)
        job_name = None
        try:
            job = _sandbox_job_spec(self.namespace, self.queue, image, token, port)
            r = _kubectl(self.namespace, "create", "-f", "-",
                         "-o", "jsonpath={.metadata.name}", input_text=json.dumps(job))
            if r.returncode != 0:
                raise RuntimeError(f"sandbox job create failed: {r.stderr[:300]}")
            job_name = r.stdout.strip()
            pod_ip = self._wait_for_pod(job_name, label)
            self._wait_for_swerex(pod_ip, port, token, label)
        except Exception:
            if job_name:
                self._delete_job(job_name)
            raise
        return SandboxEndpoint(host=f"http://{pod_ip}", port=port,
                               auth_token=token, handle=job_name)

    def release(self, endpoint):
        if endpoint is None:
            return
        self._delete_job(endpoint.handle)

    def _wait_for_pod(self, job_name, label=""):
        pod_timeout_s = _env_int("TEASBENCH_SANDBOX_POD_TIMEOUT", 1200)
        deadline = time.time() + pod_timeout_s
        while time.time() < deadline:
            r = _kubectl(
                self.namespace, "get", "pods", f"-l=job-name={job_name}",
                "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}"
                      "|{.items[0].status.podIP}")
            parts = r.stdout.strip().split("|")
            phase = parts[0] if len(parts) > 0 else ""
            pod_ip = parts[2] if len(parts) > 2 else ""
            if phase == "Running" and pod_ip:
                return pod_ip
            if phase == "Failed":
                raise RuntimeError(f"sandbox pod failed ({label or job_name})")
            time.sleep(5)
        raise RuntimeError(f"sandbox pod not Running after {pod_timeout_s}s "
                           f"({label or job_name})")

    def _wait_for_swerex(self, pod_ip, port, token, label=""):
        # swe-rex is still pip-installing for the first ~minute, so the
        # server isn't reachable immediately even once the pod is Running.
        swerex_timeout_s = _env_int("TEASBENCH_SWEREX_TIMEOUT", 600)
        deadline = time.time() + swerex_timeout_s
        url = f"http://{pod_ip}:{port}/is_alive"
        while time.time() < deadline:
            try:
                req = urllib.request.Request(url, headers={"X-API-Key": token})
                urllib.request.urlopen(req, timeout=5)
                return
            except Exception:
                time.sleep(3)
        raise RuntimeError(f"swerex not alive after {swerex_timeout_s}s "
                           f"({label or pod_ip})")

    def _delete_job(self, job_name):
        if not job_name:
            return
        try:
            _kubectl(self.namespace, "delete", "job", job_name,
                     "--wait=false", "--ignore-not-found=true")
        except Exception:
            pass


_PORT_LOCK = threading.Lock()


def _free_local_port():
    """OS-assigned free port. A fixed counter base collides with stale
    kubectl port-forwards left by a previous crashed run - SWE-agent then
    talks to the OLD sidecar and dies with SessionExistsError."""
    with _PORT_LOCK:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


class _PortForwardSandbox:
    """One swe-rex sandbox pod, reached via `kubectl port-forward` on an
    OS-assigned local port. Faithful port of AgentCAP's `_K8sSidecar`
    (agent_cap/agents/sandbox_providers.py) - see PortForwardK8sProvider
    for when this connection strategy is needed instead of
    InClusterK8sProvider.
    """

    def __init__(self, namespace, queue, image, token, port, label=""):
        self.namespace = namespace
        self.queue = queue
        self.image = image
        self.token = token
        self.remote_port = port
        self.label = label
        self.job_name = None
        self.pod_name = None
        self.local_port = None
        self.pf_proc = None
        self._stopped = False
        self._pf_keeper = None

    def start(self):
        pod_timeout_s = _env_int("TEASBENCH_SANDBOX_POD_TIMEOUT", 1200)
        swerex_timeout_s = _env_int("TEASBENCH_SWEREX_TIMEOUT", 600)

        job = _sandbox_job_spec(self.namespace, self.queue, self.image,
                                self.token, self.remote_port)
        r = _kubectl(self.namespace, "create", "-f", "-",
                     "-o", "jsonpath={.metadata.name}", input_text=json.dumps(job))
        if r.returncode != 0:
            raise RuntimeError(f"sandbox job create failed: {r.stderr[:300]}")
        self.job_name = r.stdout.strip()

        deadline = time.time() + pod_timeout_s
        while time.time() < deadline:
            r = _kubectl(self.namespace, "get", "pods", f"-l=job-name={self.job_name}",
                         "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}")
            parts = r.stdout.strip().split("|")
            phase, pod = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
            if phase == "Running" and pod:
                self.pod_name = pod
                break
            if phase == "Failed":
                raise RuntimeError(f"sandbox pod failed for {self.label}")
            time.sleep(5)
        else:
            raise RuntimeError(f"sandbox pod not Running after {pod_timeout_s}s "
                               f"({self.label})")

        self.local_port = _free_local_port()
        # start_new_session: detach the tunnel from the caller's process
        # group so a terminal/session teardown can't kill it mid-task
        # (observed: session flap killed in-flight sidecar tunnels ->
        # "Cannot connect to host 127.0.0.1:<port>" -> task rc=1).
        self.pf_proc = subprocess.Popen(
            ["kubectl", "-n", self.namespace, "port-forward",
             f"pod/{self.pod_name}", f"{self.local_port}:{self.remote_port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.time() + swerex_timeout_s
        while time.time() < deadline:
            if self.pf_proc.poll() is not None:
                # port-forward died (e.g. pod still pip-installing) - restart it
                self.pf_proc = subprocess.Popen(
                    ["kubectl", "-n", self.namespace, "port-forward",
                     f"pod/{self.pod_name}", f"{self.local_port}:{self.remote_port}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(3)
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.local_port}/is_alive",
                    headers={"X-API-Key": self.token},
                )
                urllib.request.urlopen(req, timeout=5)
                # Babysit the tunnel for the task's whole lifetime - kubectl
                # port-forward occasionally drops mid-task, which otherwise
                # kills the agent with "Cannot connect to 127.0.0.1:<port>".
                self._pf_keeper = threading.Thread(
                    target=self._keep_pf_alive, daemon=True)
                self._pf_keeper.start()
                return
            except Exception:
                time.sleep(3)
        raise RuntimeError(f"swerex not alive after {swerex_timeout_s}s ({self.label})")

    def _keep_pf_alive(self):
        while not self._stopped:
            proc = self.pf_proc
            if proc is not None and proc.poll() is not None and not self._stopped:
                try:
                    self.pf_proc = subprocess.Popen(
                        ["kubectl", "-n", self.namespace, "port-forward",
                         f"pod/{self.pod_name}", f"{self.local_port}:{self.remote_port}"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                except Exception:
                    pass
            time.sleep(2)

    def stop(self):
        self._stopped = True
        if self.pf_proc is not None:
            self.pf_proc.kill()
            try:
                self.pf_proc.wait(timeout=5)
            except Exception:
                pass
        if self.job_name:
            try:
                _kubectl(self.namespace, "delete", "job", self.job_name,
                         "--wait=false", "--ignore-not-found=true")
            except Exception:
                pass


class PortForwardK8sProvider(_BaseK8sProvider):
    """Fallback sandbox provider for driving a run from a login node
    (outside the cluster), where sandbox pod IPs are not directly
    routable. Faithful port of AgentCAP's `_K8sSidecar` +
    `K8sSandboxProvider` (agent_cap/agents/sandbox_providers.py): OS-
    assigned local port, `kubectl port-forward`, restart-on-death during
    startup, and a babysitter thread for the sandbox's whole lifetime.

    Prefer InClusterK8sProvider when the caller runs inside the cluster
    (e.g. from a TEASBench Job pod) - it needs none of this machinery.
    """

    def acquire(self, image, label=""):
        token = secrets.token_hex(16)  # per-sandbox random: this is a shared cluster
        port = _env_int("TEASBENCH_SANDBOX_PORT", 9999)
        sandbox = _PortForwardSandbox(self.namespace, self.queue, image, token, port, label)
        try:
            sandbox.start()
        except Exception:
            sandbox.stop()
            raise
        return SandboxEndpoint(
            host="http://127.0.0.1",
            port=sandbox.local_port,
            auth_token=token,
            handle=sandbox,
        )

    def release(self, endpoint):
        if endpoint is not None and isinstance(endpoint.handle, _PortForwardSandbox):
            endpoint.handle.stop()
