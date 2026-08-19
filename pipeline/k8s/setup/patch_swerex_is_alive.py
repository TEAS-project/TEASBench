#!/usr/bin/env python3
"""Give swe-rex's RemoteDeployment.is_alive the `timeout` kwarg its own base
class promises.

WHY THIS EXISTS
---------------
swe-rex declares the kwarg on the abstract deployment and implements it on the
Docker deployment and on RemoteRuntime:

    AbstractDeployment.is_alive(self, *, timeout: float | None = None)
    DockerDeployment.is_alive(self, *, timeout: float | None = None)
    RemoteRuntime.is_alive(self, *, timeout: float | None = None)

but RemoteDeployment overrides it without the kwarg, and drops the timeout on
the floor when delegating:

    async def is_alive(self) -> IsAliveResponse:
        return await self.runtime.is_alive()

SWE-agent calls `self._env.deployment.is_alive(timeout=10)` from
`attempt_autosubmission_after_error` (sweagent/agent/agents.py:831) -- the
recovery path that salvages a partial patch when a task errors mid-run. On the
Docker deployment that call is fine, which is why upstream has not noticed. On
the *remote* deployment -- the one TEASBench's PortForwardK8sProvider uses for
every SWE-bench task on EIDF -- it raises

    TypeError: RemoteDeployment.is_alive() got an unexpected keyword argument 'timeout'

turning every recoverable task error into a total loss of that task's work. In
run 20260807-0001 it cost 46 of 55 failed tasks their partial patch; in
20260817-0728 it accounted for all 46 `sweagent rc=1` failures out of 100 tasks.

No swe-rex release fixes this: 1.4.0 is the latest published version and the
current upstream main still has the un-parameterised override, so raising
SWEREX_SPEC cannot help. Hence a patch, applied and verified the same way
setup_swebench_env.sh already applies and verifies AgentCAP's streaming patch.

The patch restores base-class conformance *and* actually plumbs the timeout
through to the runtime, which is what the caller asked for.

Idempotent: re-running is a no-op, and if a future swe-rex fixes this upstream
the script detects the working signature and leaves the file alone.

Usage:  python patch_swerex_is_alive.py
Exit:   0 patched or already fine, 1 could not patch (message on stderr).
"""

import inspect
import re
import sys
from pathlib import Path

MARKER = "TEASBENCH_SWEREX_IS_ALIVE_TIMEOUT_PATCH_APPLIED"

# Deliberately unannotated: the annotation would have to match the file's own
# `from __future__` state and Python version to import cleanly, and it buys
# nothing here -- inspect.signature() reports the parameter either way, and the
# runtime behaviour is identical.
NEW_DEF = (
    "    async def is_alive(self, *, timeout=None) -> IsAliveResponse:"
    f"  # {MARKER}"
)
OLD_DEF_RE = re.compile(
    r"^[ \t]*async def is_alive\(self\)[ \t]*->[ \t]*IsAliveResponse:.*$",
    re.MULTILINE,
)
OLD_DELEGATE_RE = re.compile(
    r"^([ \t]*)return await self\.runtime\.is_alive\(\)[ \t]*$",
    re.MULTILINE,
)


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main():
    try:
        from swerex.deployment.remote import RemoteDeployment
    except Exception as exc:
        return fail(f"cannot import swerex.deployment.remote: {exc}")

    # Already correct (patched earlier, or fixed upstream)? Do nothing.
    try:
        if "timeout" in inspect.signature(RemoteDeployment.is_alive).parameters:
            print("RemoteDeployment.is_alive already accepts timeout=; nothing to do")
            return 0
    except (TypeError, ValueError) as exc:
        return fail(f"cannot inspect RemoteDeployment.is_alive: {exc}")

    try:
        path = Path(inspect.getsourcefile(RemoteDeployment))
    except Exception as exc:
        return fail(f"cannot locate source file for RemoteDeployment: {exc}")
    if not path.is_file():
        return fail(f"source file for RemoteDeployment does not exist: {path}")

    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        # Marker present but the signature check above said no timeout: the
        # installed bytecode disagrees with the source on disk. Say so plainly
        # rather than patching a second time.
        return fail(
            f"{path} carries the patch marker but the imported class still has no "
            "'timeout' parameter -- a stale .pyc or a shadowed swerex install. "
            "Clear __pycache__ / check `python -c \"import swerex; print(swerex.__file__)\"`."
        )

    n_def = len(OLD_DEF_RE.findall(src))
    n_delegate = len(OLD_DELEGATE_RE.findall(src))
    if n_def != 1:
        return fail(
            f"expected exactly one `async def is_alive(self) -> IsAliveResponse:` in "
            f"{path}, found {n_def}. swe-rex's layout has changed; re-check whether "
            "this patch is still needed before forcing it."
        )
    if n_delegate != 1:
        return fail(
            f"expected exactly one `return await self.runtime.is_alive()` in {path}, "
            f"found {n_delegate}. swe-rex's layout has changed; re-check this patch."
        )

    patched = OLD_DEF_RE.sub(lambda m: NEW_DEF, src, count=1)
    patched = OLD_DELEGATE_RE.sub(
        lambda m: f"{m.group(1)}return await self.runtime.is_alive(timeout=timeout)",
        patched,
        count=1,
    )

    try:
        path.write_text(patched, encoding="utf-8")
    except OSError as exc:
        return fail(f"cannot write {path}: {exc}")

    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
