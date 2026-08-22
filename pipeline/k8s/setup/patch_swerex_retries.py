#!/usr/bin/env python3
"""Turn on swe-rex's transport-level request retries, safely.

WHY THIS EXISTS
---------------
swe-rex already ships the whole retry mechanism and never uses it.

`RemoteRuntime._request` (swerex/runtime/remote.py) has a retry loop with
exponential backoff, and stamps every request with an `X-Request-ID` the server
uses as an idempotency key. But the parameter that drives it defaults to zero:

    async def _request(self, endpoint, payload, output_class, num_retries: int = 0):

and not one of the seven callers -- create_session, run_in_session,
close_session, execute, read_file, write_file, close -- ever passes it. So the
first transport-level failure on any request is fatal.

That is the dominant loss mechanism for SWE-bench on EIDF. A `kubectl
port-forward` tunnel breaks, the in-flight POST dies with

    aiohttp.client_exceptions.ServerDisconnectedError: Server disconnected

and SWE-agent turns that into "Exit due to unknown error" -> "Runtime is no
longer alive" -> no patch, losing every step the agent had already done. The
tunnel babysitter in k8s_pod_providers/providers.py restarts the tunnel in a
millisecond or two, but that cannot help a request that has already raised:
nothing re-sends it. The signature is unambiguous in the evidence -- tasks lost
this way all carry a running-phase tunnel drop in the babysitter's journal,
while tasks that failed for genuine agent reasons carry none.

WHY THE SERVER NEEDS PATCHING TOO
---------------------------------
Enabling client retries alone is not safe. The server's ResponseManager
(swerex/server.py) records a response only *after* the handler returns, so a
retry that arrives while the original is still executing misses the cache and
runs the action a second time. For `run_in_session` that puts two commands on
one shell at once. This script therefore also registers a request as in-flight
*before* it runs, so a duplicate awaits the original instead of racing it.

Both halves are needed, and they live in different places: the client half
matters on the login node, the server half inside the sandbox pod. The same
script runs in both, and each half is skipped where it does not apply.

That split is not cosmetic. `aiohttp` is NOT a swe-rex dependency -- only the
client needs it, and only the login node installs it (SWE-agent pulls it in).
A sandbox pod runs `python3 -m swerex`, which imports cleanly without it. So
importing the client module inside a pod raises ModuleNotFoundError, and a
script that treats that as failure kills every pod at startup under `set -e`.
Hence: an un-importable half is "not applicable here", not an error -- and
`--require` lets each caller name the half it actually depends on, so a real
failure still fails loudly where it matters.

The client half reads SWEREX_NUM_RETRIES from the environment (default 3)
rather than adding a config field. A field is the right shape upstream, and is
what the SWE-ReX PR does; a patch against a released wheel should not be
changing a pydantic model that SWE-agent constructs.

Idempotent: re-running is a no-op, and if a future swe-rex fixes either half
upstream the script detects it and leaves that file alone.

Usage:  python patch_swerex_retries.py [--require client|server|both]
        --require names the half that MUST be applied here; the other is
        applied opportunistically and skipped if its module will not import.
        The sandbox pod passes `server`, the login node passes `client`.
Exit:   0 patched or already fine, 1 could not patch (message on stderr).
"""

import argparse
import re
import sys
from pathlib import Path

# Return values for the two halves.
APPLIED = "applied"     # patched now, or already correct
SKIPPED = "skipped"     # not applicable in this environment
FAILED = "failed"       # importable, but could not be patched

CLIENT_MARKER = "TEASBENCH_SWEREX_REQUEST_RETRIES_PATCH_APPLIED"
SERVER_MARKER = "TEASBENCH_SWEREX_INFLIGHT_DEDUPE_PATCH_APPLIED"

DEFAULT_NUM_RETRIES = 3

# --------------------------------------------------------------------------
# client half: swerex/runtime/remote.py
# --------------------------------------------------------------------------

OLD_REQUEST_SIG_RE = re.compile(
    r"^([ \t]*)async def _request\(self, endpoint: str, payload: BaseModel \| None, "
    r"output_class: Any, num_retries: int = 0\):[ \t]*$",
    re.MULTILINE,
)

NEW_REQUEST_SIG = '''{i}async def _request(self, endpoint: str, payload: BaseModel | None, output_class: Any, num_retries=None):  # {marker}
{i}    if num_retries is None:
{i}        import os as _os
{i}        try:
{i}            num_retries = int(_os.environ.get("SWEREX_NUM_RETRIES", "{default}"))
{i}        except ValueError:
{i}            num_retries = {default}'''


def patch_client():
    try:
        from swerex.runtime import remote as remote_mod
    except ImportError as exc:
        # Almost always `aiohttp` missing, which means this is a sandbox pod:
        # it runs the server only and has no client to patch. Not an error.
        print(f"client: not applicable here ({exc})")
        return SKIPPED
    except Exception as exc:
        fail(f"cannot import swerex.runtime.remote: {exc}")
        return FAILED

    path = Path(remote_mod.__file__)
    src = path.read_text(encoding="utf-8")

    if CLIENT_MARKER in src:
        print(f"client: already patched ({path})")
        return APPLIED

    matches = OLD_REQUEST_SIG_RE.findall(src)
    if len(matches) != 1:
        # Either upstream changed the signature (possibly fixing this) or the
        # layout moved. Say which, rather than forcing a substitution.
        if re.search(r"async def _request\(", src) and "num_retries: int = 0" not in src:
            print(
                f"client: `_request` no longer defaults num_retries to 0 in {path}; "
                "assuming upstream fixed this and leaving it alone"
            )
            return APPLIED
        return _failed(
            f"expected exactly one `_request(..., num_retries: int = 0)` definition in "
            f"{path}, found {len(matches)}. swe-rex's layout has changed; re-check "
            "whether this patch is still needed before forcing it."
        )

    indent = matches[0]
    patched = OLD_REQUEST_SIG_RE.sub(
        lambda m: NEW_REQUEST_SIG.format(
            i=indent, marker=CLIENT_MARKER, default=DEFAULT_NUM_RETRIES
        ),
        src,
        count=1,
    )
    try:
        path.write_text(patched, encoding="utf-8")
    except OSError as exc:
        return _failed(f"cannot write {path}: {exc}")
    print(f"client: patched {path} (default {DEFAULT_NUM_RETRIES} retries, "
          "override with SWEREX_NUM_RETRIES)")
    return APPLIED


# --------------------------------------------------------------------------
# server half: swerex/server.py
# --------------------------------------------------------------------------

OLD_MANAGER_INIT_RE = re.compile(
    r"^([ \t]*)def __init__\(self\):\n"
    r"[ \t]*self\.last_processed_request_id = None\n"
    r"[ \t]*self\.last_processed_response = None[ \t]*$",
    re.MULTILINE,
)

NEW_MANAGER_INIT = '''{i}def __init__(self):  # {marker}
{i}    self.last_processed_request_id = None
{i}    self.last_processed_response = None
{i}    # Keyed by request id, not a single slot: two different requests in
{i}    # flight at once would otherwise overwrite each other's entry, and the
{i}    # first one's waiter would then never be woken.
{i}    self._in_flight = {{}}

{i}def mark_in_flight(self, request_id):
{i}    import asyncio as _asyncio
{i}    self._in_flight[request_id] = _asyncio.Event()

{i}def clear_in_flight(self, request_id):
{i}    event = self._in_flight.pop(request_id, None)
{i}    if event is not None:
{i}        event.set()

{i}async def wait_for_in_flight(self, request_id):
{i}    """Await the in-flight request with this id and return its response.

{i}    None if it is not in flight, or if it finished without recording a
{i}    response (it raised) -- the caller should then execute it.
{i}    """
{i}    event = self._in_flight.get(request_id)
{i}    if event is None:
{i}        return None
{i}    await event.wait()
{i}    return self.get_response(request_id)'''

OLD_MIDDLEWARE = '''    request_id = request.headers.get("X-Request-ID")
    if request_id:
        response = response_manager.get_response(request_id)
        if response:
            return response

    response = await call_next(request)

    body_content = b""
    async for chunk in response.body_iterator:
        body_content += chunk

    new_response = Response(
        content=body_content,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

    if request_id:
        response_manager.set_response(request_id, new_response)

    return new_response'''

NEW_MIDDLEWARE = '''    request_id = request.headers.get("X-Request-ID")
    if request_id:
        response = response_manager.get_response(request_id)
        if response:
            return response
        # A retry that arrived while the original is still executing: wait for
        # it rather than running the same action a second time.
        response = await response_manager.wait_for_in_flight(request_id)
        if response:
            return response
        response_manager.mark_in_flight(request_id)

    try:
        response = await call_next(request)

        body_content = b""
        async for chunk in response.body_iterator:
            body_content += chunk

        new_response = Response(
            content=body_content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        if request_id:
            response_manager.set_response(request_id, new_response)

        return new_response
    finally:
        # Must run even if the handler raised or was cancelled, or a waiting
        # retry would block until its own client timeout.
        response_manager.clear_in_flight(request_id)'''


def patch_server():
    try:
        from swerex import server as server_mod
    except ImportError as exc:
        print(f"server: not applicable here ({exc})")
        return SKIPPED
    except Exception as exc:
        fail(f"cannot import swerex.server: {exc}")
        return FAILED

    path = Path(server_mod.__file__)
    src = path.read_text(encoding="utf-8")

    if SERVER_MARKER in src:
        print(f"server: already patched ({path})")
        return APPLIED
    if "wait_for_in_flight" in src:
        print(f"server: {path} already handles in-flight duplicates; leaving it alone")
        return APPLIED

    n_init = len(OLD_MANAGER_INIT_RE.findall(src))
    if n_init != 1:
        return _failed(
            f"expected exactly one ResponseManager.__init__ of the known shape in "
            f"{path}, found {n_init}. swe-rex's layout has changed; re-check this patch."
        )
    if src.count(OLD_MIDDLEWARE) != 1:
        return _failed(
            f"the handle_request_id middleware in {path} does not match the expected "
            "text. swe-rex's layout has changed; re-check this patch."
        )

    indent = OLD_MANAGER_INIT_RE.findall(src)[0]
    patched = OLD_MANAGER_INIT_RE.sub(
        lambda m: NEW_MANAGER_INIT.format(i=indent, marker=SERVER_MARKER), src, count=1
    )
    patched = patched.replace(OLD_MIDDLEWARE, NEW_MIDDLEWARE, 1)

    try:
        path.write_text(patched, encoding="utf-8")
    except OSError as exc:
        return _failed(f"cannot write {path}: {exc}")
    print(f"server: patched {path} (in-flight duplicate requests now await the original)")
    return APPLIED


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def _failed(msg):
    fail(msg)
    return FAILED


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--require", choices=("client", "server", "both"), default="both",
        help="the half that must be applied here; the other is opportunistic. "
             "A sandbox pod has no client to patch (no aiohttp) and a "
             "client-only environment may have no server, so requiring both "
             "is right only when you know both are installed.")
    args = ap.parse_args(argv)

    results = {"client": patch_client(), "server": patch_server()}

    for half, status in results.items():
        required = args.require in (half, "both")
        if status == FAILED:
            # Always fatal: the module is there and we could not patch it.
            return 1
        if status == SKIPPED and required:
            return fail(
                f"--require {args.require}: the {half} half could not be applied "
                "because its module would not import here. Retries are only safe "
                "with both halves in place, so this is not being skipped silently."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
