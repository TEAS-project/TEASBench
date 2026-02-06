# model_serving/stateful_python_tool.py
"""
Stateful Python tool for Harmony: keeps variables/functions across tool calls
WITHIN one generate_with_python_tool_single() run, but resets by creating a new
tool instance per run.

Requires: jupyter_client (and usually ipykernel).
"""

from __future__ import annotations
import os
import queue
import threading
from typing import AsyncIterator, Optional, List, Any
from abc import ABC, abstractmethod
from uuid import UUID, uuid4
import time
import queue
from loguru import logger


from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)


def add_libs(code: str) -> str:
    """Add common math libraries to code."""
    return "import math\nimport numpy as np\nimport sympy as sp\nfrom sympy import *\n" + code



def ensure_last_print(code: str) -> str:
    """Ensure the last expression is printed."""
    lines = code.strip().split("\n")
    if lines and "print(" not in lines[-1] and "import" not in lines[-1]:
        if "#" in lines[-1]:
            lines[-1] = lines[-1].split("#")[0]
        lines[-1] = "print(" + lines[-1] + ")"
    return "\n".join(lines)



class LocalJupyterSession_2:
    # ... keep your existing __init__ ...

    def _interrupt_best_effort(self) -> None:
        """Best-effort interrupt; keeps kernel state if possible."""
        # Prefer KernelManager if we own the kernel
        if self._km is not None:
            try:
                self._km.interrupt_kernel()
                return
            except Exception:
                pass

        # Fallback: some KernelClient implementations provide interrupt_kernel()
        try:
            interrupt = getattr(self._client, "interrupt_kernel", None)
            if callable(interrupt):
                interrupt()
        except Exception:
            pass

    def execute(self, code: str, *, timeout: float | None = None) -> str:
        """Execute code and return combined stdout/stderr. Enforces an overall wall-clock timeout."""
        client = self._client
        effective_timeout = float(timeout or self._default_timeout)
        deadline = time.monotonic() + effective_timeout

        msg_id = client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        idle_seen = False
        reply_seen = False

        def remaining() -> float:
            return deadline - time.monotonic()

        def try_get_shell_reply(non_blocking: bool = False) -> None:
            nonlocal reply_seen
            if reply_seen:
                return
            t = 0.0 if non_blocking else max(0.0, min(0.25, remaining()))
            try:
                reply = client.get_shell_msg(timeout=t)
            except queue.Empty:
                return

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                return

            reply_seen = True
            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error":
                tb = reply_content.get("traceback")
                if tb:
                    stderr_parts.append("\n".join(tb))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())

        # Main loop: poll iopub in small slices; also poll shell for execute_reply
        while not (idle_seen and reply_seen):
            if remaining() <= 0:
                # Interrupt so the kernel doesn't stay stuck busy
                self._interrupt_best_effort()
                raise TimeoutError(
                    f"Timed out waiting for kernel output (>{effective_timeout:.1f}s). "
                    "Execution was interrupted."
                )

            poll = max(0.01, min(0.25, remaining()))
            try:
                msg = client.get_iopub_msg(timeout=poll)
            except queue.Empty:
                # no iopub traffic in this slice; maybe the reply is already on shell
                try_get_shell_reply(non_blocking=True)
                continue

            # Filter to our execution where possible; BUT be tolerant of status messages
            parent = msg.get("parent_header", {}) or {}
            parent_id = parent.get("msg_id")
            msg_type = msg.get("msg_type")
            content = msg.get("content", {}) or {}

            is_ours = (parent_id == msg_id)

            if msg_type == "status":
                # Some environments occasionally omit/garble parent headers on status;
                # accept idle if we've already seen our execute_reply.
                state = content.get("execution_state")
                if is_ours or (reply_seen and state == "idle"):
                    if state == "idle":
                        idle_seen = True
                continue

            if not is_ours:
                continue

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)

            elif msg_type == "error":
                tb = content.get("traceback")
                if tb:
                    stderr_parts.append("\n".join(tb))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())

            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {}) or {}
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")

            # keep shell moving too (don’t wait until idle)
            try_get_shell_reply(non_blocking=True)

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            stdout = f"{stdout.rstrip()}\n{stderr}" if stdout else stderr

        if not stdout.strip():
            stdout = "[WARN] No output. Use print() to see results."

        return stdout
    



class LocalJupyterSession:
    """Stateful Jupyter kernel session for code execution."""

    # Class-level lock and port counter to avoid port conflicts
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        """Get next available ports for kernel connection."""
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, connection_file: str | None = None, *, timeout: float = 10.0):
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:
            raise RuntimeError("jupyter_client package required") from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client: BlockingKernelClient
        self._km: KernelManager | None = None

        if connection_file:
            from pathlib import Path
            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(f"Connection file not found: {connection_path}")
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
            # Allocate unique ports to avoid conflicts when running multiple kernels
            ports = self._get_next_ports(5)
            km = KernelManager()
            km.shell_port = ports[0]
            km.iopub_port = ports[1]
            km.stdin_port = ports[2]
            km.hb_port = ports[3]
            km.control_port = ports[4]
            km.start_kernel()
            client = km.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
            self._km = km
            self._owns_kernel = True


    def execute(self, code: str, *, timeout: float | None = None) -> str:
        """Execute code and return combined stdout/stderr."""
        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        while True:
            try:
                msg = client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for kernel output.") from exc

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Drain shell channel
        while True:
            try:
                reply = client.get_shell_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError("Timed out waiting for execution reply.") from exc

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            stdout = f"{stdout.rstrip()}\n{stderr}" if stdout else stderr

        if not stdout.strip():
            stdout = "[WARN] No output. Use print() to see results."

        return stdout

    def close(self):
        import contextlib, inspect, asyncio

        with contextlib.suppress(Exception):
            self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                res = self._km.shutdown_kernel(now=True)

                if inspect.isawaitable(res):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        asyncio.run(res)
                    else:
                        loop.create_task(res)

        self._km = None
        self._client = None
        self._owns_kernel = False

    def __del__(self):
        self.close()


class PythonTool:
    """Python execution tool using Jupyter kernel."""

    def __init__(self, execution_backend: str | None = None, local_jupyter_timeout: float = 10.0):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._execution_lock = threading.Lock()
        self._jupyter_session: LocalJupyterSession | None = None
        # Lazy initialization to avoid port conflicts during object creation
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        """Lazily initialize the Jupyter session."""
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = LocalJupyterSession(timeout=self._local_jupyter_timeout)

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """Use this tool to execute Python code. The code runs in a stateful Jupyter notebook. Use print() to see output."""

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name=self.get_tool_name())
        message = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message: Message) -> list[Message]:
        """Execute code from message using Jupyter kernel."""
        self._ensure_session()
        script = message.content[0].text
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(script, timeout=self._local_jupyter_timeout)
            except TimeoutError as exc:
                output = f"[ERROR] (TimeoutError) {exc}"
                logger.error(script[:2000] + ("\n...[truncated]..." if len(script) > 2000 else ""))
                try:
                    self._jupyter_session.close()
                finally:
                    self._jupyter_session = None

        return [self._make_response(output, channel=message.channel)]

    def close(self):
        if self._jupyter_session is not None:
            self._jupyter_session.close()
            self._jupyter_session = None

    def __del__(self):
        self.close()
