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
import os
import subprocess
import sys
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


def check_sandbox(ns, queue, instance_id=None):
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

        # The babysitter thread is what keeps a long SWE-bench task alive; a
        # tunnel that dies quietly mid-task is the failure it exists to prevent.
        print("      checking the tunnel stays up (10s)...")
        time.sleep(10)
        try:
            req = urllib.request.Request(f"{ep.host}:{ep.port}/is_alive",
                                         headers={"X-API-Key": ep.auth_token})
            urllib.request.urlopen(req, timeout=10).read()
            ok("tunnel still alive after 10s (babysitter working)")
        except Exception as exc:
            bad(f"tunnel died within 10s: {exc}",
                "kubectl port-forward is dropping; long tasks would fail mid-run.")
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
    print(f"target image: {CHECK_IMAGE}   (no GPU requested)")
    print("=" * 62)

    if not check_kubectl(ns):
        print("\nAborting: kubectl is not usable, nothing else can be checked.")
        return 1
    check_verbs(ns)
    check_sandbox(ns, queue, args.real_image)
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
