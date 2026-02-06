import threading
import contextlib
import time
import queue
import re
import os
from jupyter_client import KernelManager
from loguru import logger

from openai_harmony import (
    HarmonyEncodingName, 
    load_harmony_encoding, 
    SystemContent, 
    ReasoningEffort, 
    ToolNamespaceConfig, 
    Author, 
    Message, 
    Role, 
    TextContent, 
    Conversation
)

class AIMO3Template:

    def __init__(self):

        pass

    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:

        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        tool_config: ToolNamespaceConfig
    ) -> list[Message]:

        system_content = self.get_system_content(system_prompt, tool_config)        
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)

        user_message = Message.from_role_and_content(Role.USER, user_prompt)

        return [system_message, user_message]
    
    

class AIMO3Sandbox:

    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:

        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count

            return ports

    def __init__(self,
                 timeout: float,
                 preload: str = "minimal"
                 ):

        self._default_timeout = timeout
        self._preload = preload
        self._owns_kernel = False
        self._client = None
        self._km = None
        
        # ports = self._get_next_ports(5)


        self._env = os.environ.copy()
        self._env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        self._env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        self._env["JUPYTER_PLATFORM_DIRS"] = "1"
        self._env["PYTHONWARNINGS"] = "ignore"
        self._env["MPLBACKEND"] = "Agg"

        self._extra_args = [
            "--Application.log_level=CRITICAL",
            "--HistoryManager.enabled=False",
        ]

        self._start_kernel()
        self._preload_modules()

    def _start_kernel(self) -> None:
        self._km = KernelManager()
        self._km.start_kernel(env=self._env, extra_arguments=self._extra_args)

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

    def _preload_modules(self) -> None:
        if self._preload == "minimal":
            self.execute("import math\nimport mpmath\nmpmath.mp.dps = 64\n")
        elif self._preload == "full":
            self.execute("import math\nimport numpy\nimport sympy\nimport itertools\nimport collections\nimport mpmath\nmpmath.mp.dps = 64\n")
        elif self._preload == "none":
            pass
        else:
            raise ValueError(f"Unknown preload={self._preload!r}")

    def restart(self) -> None:
        """Hard restart to guarantee memory is released."""
        try:
            self.close()
        except Exception:
            logger.exception("close() failed during restart; continuing.")
        self._client = None
        self._km = None
        self._owns_kernel = False
        self._start_kernel()
        self._preload_modules()

    def _format_error(self, traceback: list[str]) -> str:

        clean_lines = []

        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)

            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue

            clean_lines.append(clean_frame)

        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:

        client = self._client
        effective_timeout = timeout or self._default_timeout
        
        msg_id = client.execute(
            code, 
            store_history=False, 
            allow_stdin=False, 
            stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > effective_timeout:
                logger.info(code)
                logger.info(f'[ERROR] Execution timed out after {effective_timeout} seconds')
                with contextlib.suppress(Exception):
                    self.restart()
                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)

            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')

                if content.get('name') == 'stdout':
                    stdout_parts.append(text)

                else:
                    stderr_parts.append(text)

            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])

                stderr_parts.append(self._format_error(traceback_list))

            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')

                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')

            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):
        km = self._km
        client = self._client

        # Clear refs early to reduce accidental reuse and help GC
        self._client = None
        self._km = None
        self._owns_kernel = False

        with contextlib.suppress(Exception):
            if client:
                client.stop_channels()

        if km is not None:
            with contextlib.suppress(Exception):
                km.shutdown_kernel(now=True)

            # Extra safety: ensure process is dead (implementation-dependent)
            with contextlib.suppress(Exception):
                km.cleanup_resources()

    def reset(self):
        base = (
            "%reset -f\n"
            "%xdel _\n"
            "try:\n"
            "  get_ipython().history_manager.reset()\n"
            "except Exception:\n"
            "  pass\n"
        )
        if self._preload == "minimal":
            base += "import math, mpmath\nmpmath.mp.dps = 64\n"
        elif self._preload == "full":
            base += "import math, numpy, sympy, itertools, collections, mpmath\nmpmath.mp.dps = 64\n"
        self.execute(base, timeout=self._default_timeout)
        
    def __del__(self):

        self.close()


class AIMO3Tool:

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):

        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        
        self._owns_session = sandbox is None
        
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):

        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:

        lines = code.strip().split('\n')

        if not lines:
            return code

        last_line = lines[-1].strip()

        if 'print' in last_line or 'import' in last_line:
            return code

        if not last_line:
            return code

        if last_line.startswith('#'):
            return code

        lines[-1] = 'print(' + last_line + ')'

        return '\n'.join(lines)

    @property
    def instruction(self) -> str:

        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:

        return ToolNamespaceConfig(
            name='python', 
            description=self.instruction, 
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message: # this method is specific to gpt-oss-20b and 120b

        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')

        if channel:
            message = message.with_channel(channel)

        return message

    def process_sync_plus(self, message: Message) -> list[Message]:

        self._ensure_session()
        logger.info('executing python code...')
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script, timeout=self._local_jupyter_timeout)

            except TimeoutError as exc:
                output = f'[ERROR] {exc}'

        return [self._make_response(output, channel=message.channel)]