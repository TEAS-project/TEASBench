#!/usr/bin/env python3
"""Preflight for PortForwardK8sProvider -- the login-node fallback.

Run this ON AN EIDF LOGIN NODE (or anywhere your kubectl is configured for the
project namespace). Unlike the in-cluster preflight, this path uses *your own*
kubectl credentials and needs no ServiceAccount and no RBAC manifest, which is
exactly why it is the fallback when in-cluster mode is unavailable.

    python3 pipeline/k8s/preflight/preflight_portforward.py
    python3 pipeline/k8s/preflight/preflight_portforward.py --namespace eidf230ns

What it exercises: the REAL k8s_pod_providers.PortForwardK8sProvider --
its OS port allocation, `kubectl port-forward` spawn, restart-on-death during
startup, readiness poll, tunnel-babysitter thread, and cleanup on release.

The only thing substituted is the sandbox container's payload. The real spec
pip-installs swe-rex into a multi-GB per-instance SWE-bench image, which is slow
and unnecessary here: what is under test is the *tunnel mechanism*, not swe-rex.
A busybox httpd serving a file named `is_alive` answers the provider's readiness
probe identically, in seconds, with no GPU. The exec-container path keeps the
real spec (which just runs `sleep`) but uses a debian image, because the provider
execs `bash -c` and busybox only ships `sh`.

Exit code 0 and "ALL CHECKS PASSED" means the fallback works on this cluster.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

# pipeline/k8s/lib is the import root for k8s_pod_providers -- the same single
# directory the driver puts on PYTHONPATH, so this exercises the real path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

FAILED = 0
CHECK_IMAGE = os.environ.get("TEASBENCH_PREFLIGHT_IMAGE", "busybox:1.36")
# The exec path needs a DIFFERENT image from the sandbox path. K8sExecHandle.exec
# runs `kubectl exec -- bash -c ...`, and busybox ships `sh`, not `bash`, so it
# fails with "executable file not found". That is a property of the probe image,
# not of the provider: the real per-instance SWE-bench images
# (docker.io/swebench/sweb.eval.x86_64.*) are Debian-based and do have bash,
# which is why the provider hardcodes it. debian:stable-slim also has the `tar`
# that `kubectl cp` needs.
EXEC_IMAGE = os.environ.get("TEASBENCH_PREFLIGHT_EXEC_IMAGE", "debian:stable-slim")


def swebench_image(instance_id):
    """Official per-instance image name. The `_1776_` substitution is how the
    SWE-bench images are published on Docker Hub -- see AgentCAP's
    _swebench_image(); it is registry naming, not anything k8s-specific."""
    iid = instance_id.lower().replace("/", "__").replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest"


def ok(msg):
    print(f"  PASS  {msg}")


def bad(msg, hint=""):
    global FAILED
    FAILED = 1
    print(f"  FAIL  {msg}")
    if hint:
        for line in hint.strip().splitlines():
            print(f"        {line}")


def run(*args, timeout=60):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def check_kubectl(ns):
    print("\n[1] kubectl and namespace access (your own credentials)")
    if not run("which", "kubectl").returncode == 0:
        bad("kubectl not found on PATH", "This path runs kubectl locally; install it or load the module.")
        return False
    ctx = run("kubectl", "config", "current-context").stdout.strip()
    print(f"      context: {ctx or '(none)'}")
    r = run("kubectl", "-n", ns, "get", "pods")
    if r.returncode != 0:
        bad(f"cannot list pods in {ns}", r.stderr.strip()[:300])
        return False
    ok(f"kubectl can list pods in {ns}")
    return True


def check_verbs(ns):
    print("\n[2] Permissions this path needs")
    # pods/portforward is the one unique to this provider -- the in-cluster
    # provider never needs it, because it talks to pod IPs directly.
    needed = [("create", "jobs"), ("get", "jobs"), ("delete", "jobs"),
              ("get", "pods"), ("list", "pods"),
              ("create", "pods/portforward"), ("create", "pods/exec")]
    for verb, res in needed:
        r = run("kubectl", "-n", ns, "auth", "can-i", verb, res)
        if r.stdout.strip() == "yes":
            ok(f"can {verb} {res}")
        else:
            hint = ""
            if res == "pods/portforward":
                hint = "Without this the tunnel cannot be established at all."
            bad(f"cannot {verb} {res}", hint)


def patch_sandbox_payload(k8s):
    """Swap the swe-rex container for a busybox that answers /is_alive.

    Keeps every other field the real provider sets (labels, queue, backoffLimit,
    ttl, resources), so what is exercised is the real Job shape.
    """
    real = k8s._sandbox_job_spec

    def light(namespace, queue, image, token, port):
        spec = real(namespace, queue, image, token, port)
        c = spec["spec"]["template"]["spec"]["containers"][0]
        c["image"] = CHECK_IMAGE
        c["command"] = ["/bin/sh", "-c"]
        c["args"] = [
            f"mkdir -p /www && echo teasbench-ok > /www/is_alive && "
            f"httpd -f -p {port} -h /www"
        ]
        return spec

    k8s._sandbox_job_spec = light
    return real


def check_babysitter_recovery(ep, journal_path):
    """Kill `kubectl port-forward` out from under the babysitter and check
    that TWO separate things recover, reported separately so a failure
    says which half broke:

      1. the tunnel itself, back up on the SAME local port -- SWE-agent is
         handed that port once at launch (--env.deployment.port) and has
         no way to learn a new one, which is why _keep_pf_alive respawns
         onto the same local_port instead of reallocating;
      2. a `pf_drop`/phase="running" row in the journal for this drop --
         swebench_run_audit's retry classifier only retries a task on that
         exact row, so a tunnel that quietly heals with nothing journalled
         is still a failure from the driver's point of view: a task that
         hit this same failure for real, but didn't survive it, would
         never get flagged for retry.

    Killing the `kubectl port-forward` process is the closest fault we can
    inject from outside the cluster. It is not quite the production
    failure that motivated the babysitter rewrite (an apiserver-side SPDY
    stream reset that leaves `kubectl` running but silently stops
    forwarding -- see the _PortForwardSandbox docstring in providers.py):
    that failure mode can't be induced without cluster-side access. A hard
    kill at least exercises the same recovery path (_keep_pf_alive sees
    the process gone, restarts it on the same port, journals both sides),
    which used to be entirely unexercised outside of production.
    """
    print("\n[3b] fault injection: killing kubectl port-forward mid-task")
    sandbox = ep.handle
    proc = sandbox.pf_proc
    if proc is None:
        bad("no pf_proc on the sandbox handle",
            "cannot exercise the babysitter without a live tunnel process to kill.")
        return
    old_pid = proc.pid
    local_port = sandbox.local_port
    proc.kill()
    try:
        proc.wait(timeout=10)
    except Exception:
        pass  # already gone, or refusing to reap -- either way we move on
    print(f"      killed kubectl port-forward pid {old_pid} (local port {local_port})")

    # Worst case, from _keep_pf_alive in providers.py:
    #   probe_interval (default 15s)  -- we may have killed it just as the
    #     babysitter started its sleep, so up to a full interval passes
    #     before it even looks again;
    # + _pod_alive() kubectl round trip (a `get pods`, a few seconds);
    # + first restart backoff (starts at 1s);
    # + time for the freshly spawned kubectl port-forward to establish
    #   before /is_alive answers again (a few more seconds).
    # That sums to well under a minute on a healthy cluster. 90s leaves
    # generous headroom for a slow apiserver without masking a real hang.
    timeout_s = 90
    poll_every = 3
    print(f"      waiting up to {timeout_s}s for the babysitter to notice and "
          f"respawn the tunnel on the SAME local port ({local_port})...")
    t_kill = time.time()
    deadline = t_kill + timeout_s
    recovered = False
    last_status_at = t_kill
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{ep.host}:{local_port}/is_alive",
                headers={"X-API-Key": ep.auth_token})
            urllib.request.urlopen(req, timeout=5).read()
            recovered = True
            break
        except Exception:
            now = time.time()
            if now - last_status_at >= 15:
                print(f"      ...still down after {now - t_kill:.0f}s, still waiting")
                last_status_at = now
            time.sleep(poll_every)
    elapsed = time.time() - t_kill

    if recovered:
        ok(f"tunnel recovered on the same local port ({local_port}) after {elapsed:.0f}s")
    else:
        bad(f"tunnel did not recover within {timeout_s}s",
            "kubectl port-forward is not being respawned by the babysitter after a "
            "kill; a long SWE-bench task would die permanently on any tunnel drop, "
            "not just degrade.")

    # Second assertion: the journal row, independent of whether the tunnel
    # itself came back. TEASBENCH_PF_EVENTS was pointed at journal_path
    # before acquire() (the provider reads the var fresh at call time), so
    # every event this sandbox's babysitter wrote landed there.
    journal_ok = False
    read_error = ""
    try:
        with open(journal_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (row.get("label") == sandbox.label
                        and row.get("event") == "pf_drop"
                        and row.get("phase") == "running"):
                    journal_ok = True
                    break
    except OSError as exc:
        read_error = f"could not read journal file {journal_path}: {exc}"

    if journal_ok:
        ok("babysitter journalled a pf_drop/running row for this drop")
    else:
        bad("no pf_drop/running row in the journal for this drop",
            read_error or
            "swebench_run_audit's retry classifier keys on exactly this row; "
            "without it a task that survives this failure today would not be "
            "flagged for retry if the same drop killed it for real next time.")


def check_sandbox(ns, queue, instance_id=None, fault_injection=True):
    print("\n[3] PortForwardK8sProvider.acquire() / release()")
    from k8s_pod_providers import providers as k8s

    if instance_id:
        # Full fidelity: the real per-instance image, the real swe-rex install,
        # no substitution whatsoever. Slower (multi-GB pull) but this is the
        # only check that proves swe-rex actually installs and runs inside the
        # instance image -- an old conda env where a dependency clash is
        # plausible. Still no GPU.
        image = swebench_image(instance_id)
        print(f"      mode: REAL image ({instance_id})")
        print(f"      {image}")
        print("      expect several minutes for the pull + pip install")
    else:
        image = CHECK_IMAGE
        patch_sandbox_payload(k8s)
        print(f"      mode: fast probe ({CHECK_IMAGE}); use --real-image for full fidelity")
    provider = k8s.PortForwardK8sProvider(namespace=ns, queue=queue)

    # _journal() (providers.py) reads TEASBENCH_PF_EVENTS fresh on every call,
    # not once at import time, so pointing it at a scratch file here -- before
    # acquire() ever starts a tunnel -- is enough to capture every event this
    # sandbox's babysitter writes, including the pf_drop row the fault
    # injection below expects. Restored/unset in the finally block so it
    # doesn't leak into check_exec() (section 4), which must see journalling
    # off exactly like a caller that never set the var.
    journal_fd, journal_path = tempfile.mkstemp(
        prefix="teasbench-preflight-pf-events-", suffix=".jsonl")
    os.close(journal_fd)
    prior_pf_events = os.environ.get("TEASBENCH_PF_EVENTS")
    os.environ["TEASBENCH_PF_EVENTS"] = journal_path

    ep = None
    try:
        t0 = time.time()
        try:
            ep = provider.acquire(image, "preflight")
        except Exception as exc:
            bad(f"acquire() raised: {exc}",
                "A 'not alive' timeout has three usual causes, in order of likelihood:\n"
                "  1. no `create pods/portforward` rights -- see section 2 above;\n"
                "  2. the pod never left Pending (queue/quota), not a tunnel problem\n"
                "     at all -- check `kubectl -n %s get pods`;\n"
                "  3. the tunnel is being dropped by the cluster.\n"
                "The provider cleans up its own Job on this path, so nothing leaks."
                % ns)
            return
        ok(f"acquire() returned {ep.host}:{ep.port} in {time.time() - t0:.0f}s")

        # The provider only returns once its own readiness poll succeeded, so a
        # second independent fetch here confirms the tunnel is genuinely usable.
        try:
            req = urllib.request.Request(f"{ep.host}:{ep.port}/is_alive",
                                         headers={"X-API-Key": ep.auth_token})
            body = urllib.request.urlopen(req, timeout=10).read().decode().strip()
            if instance_id:
                # Real swe-rex answers /is_alive with its own payload; any 200
                # means it is installed, running and accepting the auth token.
                ok(f"swe-rex is alive in the real instance image ({body[:40]})")
            elif body == "teasbench-ok":
                ok("tunnel carries HTTP through to the sandbox pod")
            else:
                bad(f"unexpected body through the tunnel: {body!r}")
        except Exception as exc:
            bad(f"could not reach the sandbox through the tunnel: {exc}")

        if fault_injection:
            # A passive wait only proves the tunnel didn't spontaneously die
            # in N seconds; it says nothing about whether the babysitter
            # (just rewritten: HTTP /is_alive probing, pod-existence checks,
            # bounded backoff, the drop journal) actually recovers one that
            # does. This is the only place any of that gets exercised against
            # a real cluster, so make it earn its keep.
            check_babysitter_recovery(ep, journal_path)
        else:
            print("      skipping fault injection (--no-fault-injection)")
    finally:
        if ep is not None:
            job = getattr(ep.handle, "job_name", None)
            provider.release(ep)
            ok("release() returned")
            if job:
                time.sleep(5)
                r = run("kubectl", "-n", ns, "get", "job", job)
                if r.returncode != 0:
                    ok(f"sandbox job {job} deleted")
                else:
                    bad(f"sandbox job {job} still present after release()",
                        "Orphaned jobs waste namespace quota; delete it by hand.")
        if prior_pf_events is None:
            os.environ.pop("TEASBENCH_PF_EVENTS", None)
        else:
            os.environ["TEASBENCH_PF_EVENTS"] = prior_pf_events
        try:
            os.unlink(journal_path)
        except OSError:
            pass


def check_exec(ns, queue, instance_id=None):
    print("\n[4] Exec-container path (used by the SWE-bench evaluator)")
    exec_image = swebench_image(instance_id) if instance_id else EXEC_IMAGE
    print(f"      image: {exec_image}"
          + ("" if instance_id else "  (needs bash + tar, as the real instance images have)"))
    from k8s_pod_providers import providers as k8s

    # No substitution needed: the real exec spec just runs `sleep`.
    provider = k8s.PortForwardK8sProvider(namespace=ns, queue=queue)
    handle = None
    try:
        try:
            handle = provider.acquire_exec(exec_image, "preflight")
        except Exception as exc:
            bad(f"acquire_exec() raised: {exc}")
            return
        ok(f"acquire_exec() started pod {getattr(handle, 'pod_name', '?')}")

        probe = Path("/tmp/teasbench_preflight_probe.txt")
        probe.write_text("teasbench-cp-ok\n")
        try:
            handle.cp(str(probe), "/tmp/probe.txt")
            r = handle.exec("cat /tmp/probe.txt")
            if "teasbench-cp-ok" in (r.stdout or ""):
                ok("kubectl cp + kubectl exec both work")
            else:
                err = r.stderr or ""
                hint = err[:300]
                if "bash" in err and ("not found" in err or "no such file" in err.lower()):
                    hint = ("The probe image has no `bash`, but K8sExecHandle.exec runs\n"
                            "`bash -c`. This is a property of the probe image, not the\n"
                            "provider -- the real SWE-bench instance images do have bash.\n"
                            "Set TEASBENCH_PREFLIGHT_EXEC_IMAGE to an image that does.")
                bad(f"exec did not read back the copied file (stdout={r.stdout!r})", hint)
        except Exception as exc:
            bad(f"cp/exec failed: {exc}")
        finally:
            probe.unlink(missing_ok=True)
    finally:
        if handle is not None:
            provider.release_exec(handle)
            ok("release_exec() returned")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace",
                    default=os.environ.get("TEASBENCH_K8S_NAMESPACE", "eidf230ns"))
    ap.add_argument("--queue", default=None,
                    help="Kueue queue name (default <namespace>-user-queue)")
    ap.add_argument("--skip-exec", action="store_true",
                    help="Skip the exec-container check (section 4)")
    ap.add_argument("--no-fault-injection", action="store_true",
                    help="Skip killing kubectl port-forward mid-check (section 3b). "
                         "Fault injection runs by default -- it's the only place the "
                         "babysitter's recovery path gets exercised against a real "
                         "cluster -- but it adds up to ~90s to a preflight run.")
    ap.add_argument("--real-image", nargs="?", const="astropy__astropy-12907",
                    metavar="INSTANCE_ID", default=None,
                    help="Use a real SWE-bench instance image instead of the fast "
                         "busybox probe. Proves the multi-GB pull and that swe-rex "
                         "installs and runs inside the instance image. Still no GPU, "
                         "but takes minutes. Default instance: astropy__astropy-12907.")
    args = ap.parse_args()
    ns = args.namespace
    queue = args.queue or f"{ns}-user-queue"

    print("=" * 62)
    print(f"PortForwardK8sProvider preflight (namespace {ns})")
    # Name the image actually used. Printing CHECK_IMAGE unconditionally said
    # "busybox" even under --real-image, i.e. the exact opposite of what the
    # run had just proved -- and the substitution this banner was hiding is
    # what lets a broken sandbox pod command pass a preflight unnoticed.
    if args.real_image:
        print(f"target image: {swebench_image(args.real_image)}")
        print(f"              (REAL instance image: {args.real_image}, no GPU requested)")
    else:
        print(f"target image: {CHECK_IMAGE}   (no GPU requested)")
        print("              substitutes for the sandbox container: the real")
        print("              swe-rex install is NOT exercised -- use --real-image")
    print("=" * 62)

    if not check_kubectl(ns):
        print("\nAborting: kubectl is not usable, nothing else can be checked.")
        return 1
    check_verbs(ns)
    check_sandbox(ns, queue, args.real_image, fault_injection=not args.no_fault_injection)
    if not args.skip_exec:
        check_exec(ns, queue, args.real_image)

    print("\n" + "=" * 62)
    if FAILED == 0:
        print("ALL CHECKS PASSED -- PortForwardK8sProvider works on this cluster.")
        print("Use it by pointing --sandbox-provider / --exec-provider at")
        print("  k8s_pod_providers:PortForwardK8sProvider")
        print("and driving the run from this node (see docs/USER_GUIDE.md 7.3).")
    else:
        print("SOME CHECKS FAILED -- see FAIL lines above.")
    print("=" * 62)
    return FAILED


if __name__ == "__main__":
    sys.exit(main())
