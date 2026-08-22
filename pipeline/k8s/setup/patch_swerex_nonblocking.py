#!/usr/bin/env python3
"""Stop the swe-rex server blocking its own event loop for the whole of every
command.

WHY THIS EXISTS
---------------
swe-rex runs the sandbox shell synchronously inside an async server:

    swerex/runtime/local.py  BashSession.run    -> pexpect  shell.expect(...)
    swerex/runtime/local.py  LocalRuntime.execute -> subprocess.run(...)

Neither is offloaded to a thread -- there is no asyncio.to_thread or
run_in_executor anywhere in the package. So for as long as an agent command
runs, and a SWE-bench test suite runs for minutes, the server's event loop is
blocked and it answers no other request at all. `/is_alive` included.

That turns a health check into a liveness lie. Anything supervising the link to
the pod -- here, the port-forward babysitter -- probes /is_alive, gets nothing,
concludes the tunnel is dead and restarts it, tearing down the connection
carrying the very command it was waiting on. The agent sees
ServerDisconnectedError, then "Runtime is no longer alive", and the task is
lost along with everything it had already done. The supervisor manufactures the
failure it exists to detect, and it does so on a schedule: any command longer
than probe_interval x probe_failures is guaranteed to trigger it.

TEASBench also fixes the probe itself (PortForwardK8sProvider._probe_tunnel
tests the tunnel rather than the application), which breaks the loop from the
other end. This patch removes the cause: a busy server should still be able to
say it is alive.

HOW
---
Rather than rewriting a dozen call sites inside an installed wheel, each
blocking coroutine is run to completion in a worker thread on its own event
loop. The server's loop stays free.

A single module-wide lock serialises the wrapped methods. That is deliberately
conservative: today the blocking itself serialises everything, and preserving
exactly that removes any chance of two threads interleaving reads on one pty.
The only intended behaviour change is that the loop is now free to answer other
requests while a command runs.

`interrupt` is NOT wrapped, on purpose: `run` calls it, and wrapping both
against one non-reentrant lock would deadlock. Offloading `run` already moves
it off the loop, and it has no route of its own -- the server reaches it only
through run_in_session.

Idempotent: re-running is a no-op, and if a future swe-rex offloads this itself
the script says so and leaves the file alone.

Usage:  python patch_swerex_nonblocking.py
Exit:   0 patched, already patched, or not applicable here; 1 could not patch.
"""

import sys
from pathlib import Path

MARKER = "TEASBENCH_SWEREX_NONBLOCKING_PATCH_APPLIED"

PATCH = '''

# --- ''' + MARKER + ''' ---
# Appended by TEASBench. See pipeline/k8s/setup/patch_swerex_nonblocking.py for
# why. Short version: every method below blocks on pexpect or subprocess inside
# an async server, so while a command runs the server cannot answer /is_alive,
# and whatever is supervising the connection kills it mid-command.
import asyncio as _tb_asyncio
import threading as _tb_threading

# One lock for all of them. The blocking was already serialising every request;
# keeping that exactly means unblocking the loop cannot introduce a pty race.
_tb_serial = _tb_threading.Lock()


def _tb_offload(cls, name):
    original = getattr(cls, name)
    if getattr(original, "_tb_wrapped", None) is not None:
        return

    async def _wrapper(self, *args, **kwargs):
        def _work():
            with _tb_serial:
                # Its own loop, in its own thread: the blocking call blocks
                # only this thread, never the server's loop.
                return _tb_asyncio.run(original(self, *args, **kwargs))

        return await _tb_asyncio.to_thread(_work)

    _wrapper.__name__ = getattr(original, "__name__", name)
    _wrapper.__qualname__ = getattr(original, "__qualname__", name)
    _wrapper.__doc__ = getattr(original, "__doc__", None)
    _wrapper._tb_wrapped = original
    setattr(cls, name, _wrapper)


# Not `interrupt`: `run` awaits it, and one non-reentrant lock across both
# would deadlock. Not LocalRuntime.close either -- it closes sessions, whose
# own close() is wrapped.
for _tb_cls, _tb_name in ((BashSession, "start"),
                          (BashSession, "run"),
                          (BashSession, "close"),
                          (LocalRuntime, "execute")):
    _tb_offload(_tb_cls, _tb_name)
'''


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main():
    try:
        from swerex.runtime import local as local_mod
    except ImportError as exc:
        # A client-only environment may not have what local.py needs. The
        # server half is the one that matters, and it lives in the pod.
        print(f"not applicable here ({exc})")
        return 0
    except Exception as exc:
        return fail(f"cannot import swerex.runtime.local: {exc}")

    path = Path(local_mod.__file__)
    src = path.read_text(encoding="utf-8")

    if MARKER in src:
        print(f"already patched ({path})")
        return 0
    if "to_thread" in src or "run_in_executor" in src:
        print(f"{path} already offloads blocking work; leaving it alone")
        return 0

    # Only append if the two class names the patch binds to are actually there.
    for cls_name in ("BashSession", "LocalRuntime"):
        if not hasattr(local_mod, cls_name):
            return fail(
                f"swerex.runtime.local has no {cls_name}; the layout has changed. "
                "Re-check this patch before forcing it."
            )
    for cls_name, meth in (("BashSession", "start"), ("BashSession", "run"),
                           ("BashSession", "close"), ("LocalRuntime", "execute")):
        if not hasattr(getattr(local_mod, cls_name), meth):
            return fail(
                f"{cls_name}.{meth} is gone; swe-rex's layout has changed. "
                "Re-check this patch before forcing it."
            )

    try:
        path.write_text(src + PATCH, encoding="utf-8")
    except OSError as exc:
        return fail(f"cannot write {path}: {exc}")
    print(f"patched {path} (shell commands no longer block the server's event loop)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
