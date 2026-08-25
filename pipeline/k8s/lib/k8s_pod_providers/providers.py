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
    TEASBENCH_RUN_ID               unset = no run label; else a Kubernetes
                                    label value used to scope cleanup

PortForwardK8sProvider-only (unused by InClusterK8sProvider, which has no
tunnel to babysit or journal):
    TEASBENCH_PF_EVENTS            unset = journalling off; else path to
                                    the port-forward drop journal (JSONL,
                                    one object per line - see _journal())
    TEASBENCH_PF_LOG_DIR           unset = kubectl port-forward stderr to
                                    DEVNULL as before; else a directory to
                                    append per-sandbox port-forward
                                    stdout+stderr into
    TEASBENCH_PF_PROBE_INTERVAL    default "15" (seconds between tunnel
                                    probes in the babysitter)
    TEASBENCH_PF_PROBE_TIMEOUT     default "5" (per-probe connect timeout;
                                    a fifth of it, capped to 1s, is how
                                    long _probe_tunnel waits for the EOF
                                    that means the forward failed)
    TEASBENCH_PF_PROBE_FAILURES    default "3" (consecutive probe
                                    failures before the babysitter
                                    restarts the tunnel)
    TEASBENCH_PF_MAX_RESTARTS      default "20" (cap on babysitter
                                    restarts per sandbox before giving up)
    TEASBENCH_PF_BACKOFF_MAX       default "30" (cap, in seconds, on the
                                    exponential backoff between restarts)
"""

import json
import os
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


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


def _env_float(name, default):
    """Read a float env var, falling back to `default` (float or str)."""
    return float(os.environ.get(name, default))


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


# Patches the sandbox pod applies to its freshly pip-installed swe-rex, as
# (script, argv) pairs. Both fix server-side behaviour, so both have to run in
# the pod rather than on the login node:
#
#   patch_swerex_retries      a retried request must not execute twice
#   patch_swerex_nonblocking  a running command must not stop the server
#                             answering /is_alive, or the babysitter kills the
#                             tunnel out from under that very command
#
# Read from the same files the login node uses, so the two ends cannot drift.
_SWEREX_POD_PATCHES = (
    ("patch_swerex_retries.py", "--require server"),
    ("patch_swerex_nonblocking.py", ""),
)


def _swerex_pod_patch_steps():
    """Shell to apply every pod-side swe-rex patch, one heredoc per script.

    The delimiter is quoted, so each script passes through the shell untouched.
    """
    setup_dir = Path(__file__).resolve().parents[2] / "setup"
    steps = []
    for i, (name, argv) in enumerate(_SWEREX_POD_PATCHES):
        eof = f"TEASBENCH_SWEREX_PATCH_EOF_{i}"
        source = (setup_dir / name).read_text(encoding="utf-8")
        steps.append(f"python3 - {argv} <<'{eof}'\n{source}\n{eof}\n")
    return "".join(steps)


def _sandbox_job_spec(namespace, queue, image, token, port):
    """Job spec for a swe-rex sandbox pod. Shared by InClusterK8sProvider
    and PortForwardK8sProvider - only how they *connect* to the resulting
    pod differs, not how the pod is provisioned."""
    swerex_spec = os.environ.get("TEASBENCH_SWEREX_SPEC", "swe-rex>=1.4.0")
    job_labels = {"app": "teasbench-sandbox", "kueue.x-k8s.io/queue-name": queue}
    pod_labels = {"app": "teasbench-sandbox"}
    run_id = os.environ.get("TEASBENCH_RUN_ID", "")
    if run_id:
        if len(run_id) > 63 or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,61}[A-Za-z0-9])?", run_id):
            raise ValueError("TEASBENCH_RUN_ID is not a valid Kubernetes label value")
        job_labels["teasbench.run/id"] = run_id
        pod_labels["teasbench.run/id"] = run_id
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "generateName": "swe-rex-",
            "namespace": namespace,
            "labels": job_labels,
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 600,
            "activeDeadlineSeconds": 6 * 3600,
            "template": {
                "metadata": {"labels": pod_labels},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "swebench",
                        "image": image,
                        "command": ["/bin/bash", "-c"],
                        "args": [
                            # `set -e` means a failed patch fails the pod at
                            # startup rather than silently leaving the server
                            # in a state the driver assumes it is not -- a loud
                            # failure before any task runs, which the
                            # real-image preflight gate also checks for.
                            "set -e; "
                            "git config --global --add safe.directory '*'; "
                            f"python3 -m pip install --quiet --no-input '{swerex_spec}'\n"
                            f"{_swerex_pod_patch_steps()}"
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


_JOURNAL_LOCK = threading.Lock()

# Non-filesystem-safe characters in a sandbox `label` (task ids can contain
# "/", e.g. "django__django-14787" is fine but some benchmarks use paths)
# get collapsed to "_" when building a log file name - see _PortForwardSandbox
# ._spawn_pf().
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_for_filename(value):
    return _UNSAFE_FILENAME_CHARS.sub("_", value) if value else "sandbox"


def _journal(label, event, **fields):
    """Append one JSON line to $TEASBENCH_PF_EVENTS, the port-forward drop
    journal that swebench_run_audit's retry classifier reads. No-op when
    the var is unset (the default), and never raises: a journalling
    failure (e.g. a full disk) must not be allowed to break a run that
    would otherwise have succeeded - the whole point of this journal is
    to make failures *more* visible, not to introduce a new one.
    """
    path = os.environ.get("TEASBENCH_PF_EVENTS")
    if not path:
        return
    record = {"ts": time.time(), "label": label, "event": event}
    for key, value in fields.items():
        if value is not None:
            record[key] = value
    try:
        with _JOURNAL_LOCK:
            with open(path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
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

    The babysitter thread (`_keep_pf_alive`) used to only check
    `proc.poll()`, i.e. whether the `kubectl port-forward` process had
    exited. In production that missed the failure mode that actually
    killed tasks: the tunnel's stream breaks (e.g. the apiserver side
    resets it) while the local `kubectl` process keeps running - `poll()`
    stays None forever, so the babysitter saw perpetual health while
    every request through the tunnel failed. Eight SWE-bench tasks died
    this way in one evidence run, in two tight clusters (4 within 15s, 4
    within 3s), each with an `aiohttp ServerDisconnectedError` raised out
    of `swerex/runtime/remote.py`, and nothing noticed. `_keep_pf_alive`
    now actively probes `/is_alive` through the tunnel instead of only
    watching the process, and everything it does is journalled to
    `TEASBENCH_PF_EVENTS` so a dropped-and-recovered tunnel leaves a
    record even when the task itself survives.
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
        # Guards the "is this still alive -> if not, spawn a replacement"
        # sequence in _keep_pf_alive against stop(): without it, stop()'s
        # kill() and the babysitter's respawn aren't atomic, so a tunnel
        # spawned in that window is never killed - and start_new_session
        # (below) means it survives the driver's whole process group as
        # an orphan.
        self._lock = threading.Lock()

    def start(self):
        pod_timeout_s = _env_int("TEASBENCH_SANDBOX_POD_TIMEOUT", 1200)
        swerex_timeout_s = _env_int("TEASBENCH_SWEREX_TIMEOUT", 600)
        probe_timeout_s = _env_float("TEASBENCH_PF_PROBE_TIMEOUT", 5)

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
        self.pf_proc = self._spawn_pf()
        deadline = time.time() + swerex_timeout_s
        while time.time() < deadline:
            if self.pf_proc.poll() is not None:
                # port-forward died (e.g. pod still pip-installing) - restart
                # it. This is the ordinary "not ready yet" case, not the
                # production failure this class was rewritten for (a tunnel
                # that stays alive but stops forwarding, see _keep_pf_alive
                # below) - tag it phase="startup" so swebench_run_audit's
                # retry classifier, which only acts on phase="running"
                # drops, ignores it.
                old_pid = getattr(self.pf_proc, "pid", None)
                _journal(self.label, "pf_drop", phase="startup", reason="process_exited",
                         job=self.job_name, pod=self.pod_name,
                         local_port=self.local_port, pid=old_pid)
                self.pf_proc = self._spawn_pf()
                _journal(self.label, "pf_restart", job=self.job_name, pod=self.pod_name,
                         local_port=self.local_port,
                         pid=getattr(self.pf_proc, "pid", None))
                time.sleep(3)
            if self._probe_server(probe_timeout_s):
                _journal(self.label, "acquire", job=self.job_name, pod=self.pod_name,
                         local_port=self.local_port)
                _journal(self.label, "pf_start", job=self.job_name, pod=self.pod_name,
                         local_port=self.local_port, pid=getattr(self.pf_proc, "pid", None))
                # Babysit the tunnel for the task's whole lifetime - kubectl
                # port-forward occasionally drops mid-task, which otherwise
                # kills the agent with "Cannot connect to 127.0.0.1:<port>".
                self._pf_keeper = threading.Thread(
                    target=self._keep_pf_alive, daemon=True)
                self._pf_keeper.start()
                return
            time.sleep(3)
        raise RuntimeError(f"swerex not alive after {swerex_timeout_s}s ({self.label})")

    def _spawn_pf(self):
        """Start (or restart) `kubectl port-forward` for this sandbox -
        single call site for the argv/start_new_session/stderr handling
        used by both spawn points in start() and by the babysitter's
        respawn. All three call sites used to send stderr to DEVNULL,
        which is why the production tunnel drops left no forensic record;
        TEASBENCH_PF_LOG_DIR now optionally captures it instead."""
        cmd = ["kubectl", "-n", self.namespace, "port-forward",
               f"pod/{self.pod_name}", f"{self.local_port}:{self.remote_port}"]
        log_dir = os.environ.get("TEASBENCH_PF_LOG_DIR")
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                log_name = (f"{_sanitize_for_filename(self.label)}-"
                            f"{self.job_name or 'nojob'}.log")
                with open(os.path.join(log_dir, log_name), "a") as fh:
                    return subprocess.Popen(
                        cmd, stdout=fh, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            except OSError:
                pass  # a broken log dir must not take down the sandbox
        return subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _probe_server(self, timeout):
        """Is the swe-rex *server* up and answering, through the tunnel?

        Used once, at startup, to decide when the sandbox is ready to hand to
        SWE-agent -- which needs the server itself, not merely a tunnel to a
        pod that is still pip-installing. Do NOT use this for liveness while a
        task is running: see _probe_tunnel.
        """
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.local_port}/is_alive",
                headers={"X-API-Key": self.token},
            )
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except Exception:
            return False

    def _probe_tunnel(self, timeout):
        """Is the *tunnel* up? Deliberately not an HTTP request.

        swe-rex runs its shell synchronously inside its own event loop --
        pexpect's blocking .expect() in run_in_session, subprocess.run() in
        execute, neither offloaded to a thread. So for as long as the agent's
        command runs, which for a test suite is minutes, the server answers
        nothing at all: /is_alive does not time out because the tunnel is
        broken, it times out because the server is busy doing what we asked.

        Probing with HTTP therefore reports a healthy tunnel as dead, and the
        restart that follows tears down the connection carrying that very
        command -- the babysitter manufacturing the drop it exists to detect.
        The signature is a floor on the drop interval at exactly
        probe_interval x probe_failures: real faults do not arrive on a
        schedule, and none of those drops was ever unrecoverable.

        A TCP connect asks the question this thread is actually responsible
        for. kubectl accepts locally first and only then opens the stream out
        through the API server and kubelet into the pod's netns; if the pod is
        gone that fails and kubectl closes the connection straight back at us.
        So connect, then look for an immediate EOF:

          cannot connect      -> kubectl is not listening: dead
          connects, then EOF  -> kubectl could not forward: dead
          connects, stays up  -> the relay is established and idle: healthy

        Silence is the healthy answer, which is why this waits for a read that
        should never arrive rather than for one that should.
        """
        settle = min(1.0, max(0.2, float(timeout) / 5.0))
        try:
            with socket.create_connection(("127.0.0.1", self.local_port),
                                          timeout=timeout) as sock:
                sock.settimeout(settle)
                try:
                    # b"" means the peer closed: the forward failed. Any bytes
                    # at all mean something is talking, which is also fine.
                    return bool(sock.recv(1))
                except socket.timeout:
                    return True
                except OSError:
                    return False
        except OSError:
            return False

    def _pod_alive(self):
        """Same `get pods -l=job-name=...` shape start() uses (lines
        above) - called by the babysitter before respawning so a sandbox
        pod that has actually died doesn't get kubectl port-forward
        relaunched against it every restart-interval for the rest of the
        task's life."""
        try:
            r = _kubectl(self.namespace, "get", "pods", f"-l=job-name={self.job_name}",
                         "-o", "jsonpath={.items[0].status.phase}|{.items[0].metadata.name}")
            phase = r.stdout.strip().split("|")[0]
            return phase == "Running"
        except Exception:
            return False

    def _sleep_responsive(self, seconds):
        """Sleep in short slices instead of one long time.sleep(), so
        stop() isn't kept waiting for a full probe interval / backoff
        window before the babysitter notices _stopped and exits."""
        deadline = time.time() + seconds
        while not self._stopped:
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            time.sleep(min(0.5, remaining))

    def _keep_pf_alive(self):
        probe_interval = _env_float("TEASBENCH_PF_PROBE_INTERVAL", 15)
        probe_timeout = _env_float("TEASBENCH_PF_PROBE_TIMEOUT", 5)
        max_failures = _env_int("TEASBENCH_PF_PROBE_FAILURES", 3)
        max_restarts = _env_int("TEASBENCH_PF_MAX_RESTARTS", 20)
        backoff_max = _env_float("TEASBENCH_PF_BACKOFF_MAX", 30)

        consecutive_failures = 0
        restarts = 0
        # Clamped, not just seeded: a site that lowers TEASBENCH_PF_BACKOFF_MAX
        # below 1s means it for the first restart too, not only for later ones.
        backoff = min(1.0, backoff_max)

        while not self._stopped:
            self._sleep_responsive(probe_interval)
            if self._stopped:
                break

            proc = self.pf_proc
            exited = proc is None or proc.poll() is not None
            if exited:
                reason = "process_exited"
            else:
                if self._probe_tunnel(probe_timeout):
                    consecutive_failures = 0
                    continue
                consecutive_failures += 1
                if consecutive_failures < max_failures:
                    # Tolerate one blip - a busy sandbox mid-request
                    # shouldn't get its tunnel torn down needlessly.
                    continue
                reason = "probe_failed"

            old_pid = getattr(proc, "pid", None)
            _journal(self.label, "pf_drop", phase="running", reason=reason,
                     job=self.job_name, pod=self.pod_name,
                     local_port=self.local_port, pid=old_pid)
            consecutive_failures = 0

            if not self._pod_alive():
                _journal(self.label, "pf_drop", phase="running", reason="pod_gone",
                         job=self.job_name, pod=self.pod_name, local_port=self.local_port)
                _journal(self.label, "pf_unrecoverable", reason="pod_gone",
                         job=self.job_name, pod=self.pod_name, local_port=self.local_port)
                break

            if restarts >= max_restarts:
                _journal(self.label, "pf_unrecoverable", reason="restart_exhausted",
                         job=self.job_name, pod=self.pod_name, local_port=self.local_port)
                break

            self._sleep_responsive(backoff)
            if self._stopped:
                break
            backoff = min(backoff * 2, backoff_max)

            with self._lock:
                if self._stopped:
                    break
                if self.pf_proc is not None:
                    try:
                        self.pf_proc.kill()  # may still hold the listen socket
                    except Exception:
                        pass
                try:
                    # Same local_port: SWE-agent was handed this URL at
                    # launch via --env.deployment.port and has no way to
                    # learn a new one.
                    self.pf_proc = self._spawn_pf()
                    restarts += 1
                    _journal(self.label, "pf_restart", job=self.job_name, pod=self.pod_name,
                             local_port=self.local_port,
                             pid=getattr(self.pf_proc, "pid", None))
                except Exception:
                    pass

    def stop(self):
        with self._lock:
            self._stopped = True
            proc = self.pf_proc
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        if self._pf_keeper is not None:
            self._pf_keeper.join(timeout=5)
        _journal(self.label, "release", job=self.job_name, pod=self.pod_name,
                 local_port=self.local_port)
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
