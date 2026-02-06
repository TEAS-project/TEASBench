import os
import sys
import time
import signal
import subprocess
import json
import collections
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Any, Dict, Optional, Tuple, List
import requests
from openai import OpenAI
from loguru import logger
from model_serving.inference_helpers import infer_reasoning_parser, scan_for_answer, extract_last_boxed_content # , extract_answer_int, extract_boxed_text
from model_serving.AIMO3_gptoss_python_tool import AIMO3Tool, AIMO3Sandbox  # adjust import path
from math_verify import parse, verify
# from model_serving.stateful_python_tool import PythonTool
import argparse
import uuid
import tiktoken
import datetime
import queue
from transformers import AutoTokenizer
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort,
    TextContent,
    RenderConversationConfig
)

SYSTEM_PROMPT = (
    "You are an olympiad-level math problem solver. "
    "Reason step by step and put the final answer in \\boxed{}."
)

def _safe_get(obj: Any, path: List[str], default=None):
    cur = obj
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = getattr(cur, p, None)
    return cur if cur is not None else default

def is_equiv(str1, str2, verbose=False):
    if '$' not in str1:
        str1 = '$' + str1 + '$'
    if '$' not in str2:
        str2 = '$' + str2 + '$'

    gold = parse(str2)
    pred = parse(str1)
    return verify(gold, pred)

class SGLangServer:
    """
    Minimal wrapper for an OpenAI-compatible SGLang server.
    Starts exactly like:

    python3 -m sglang.launch_server --model-path ... --host ... --port ...
      --log-level ... --trust-remote-code --tool-call-parser ... --reasoning-parser ...
    """

    _SERVER_PROCESS: Optional[subprocess.Popen] = None


    def __init__(
        self,
        model_path: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        host: str = "0.0.0.0",
        port: int = 5000,
        log_level: str = "warning",
        # api_key: str | None = None,
        served_model_name: str = "sglang_model",
        dtype: str | None = None,
        kv_cache_dtype: str | None = None,
        context_length: int | None = None,
        mem_fraction_static: float | None = 0.96, # e.g. 0.80
        chunked_prefill_size: int | None = None,     # NEW
        enable_torch_compile: bool | None = None,    # NEW
        allow_auto_truncate: bool = True,
        tp_size: int = 1,
        dp_size: int = 1,
        trust_remote_code: bool = True,
        tool_call_parser: str | None = None,
        reasoning_parser: str | None = None,
        # max_running_requests: int | None = None,
        # max_queued_requests: int | None = None,
        # max_total_tokens: int | None = None,
        enable_metrics: bool | None = None,
        log_requests: bool | None = None,
        log_requests_level: int | None = None,
        timeout_s: float = 360.0,
        temperature: float = 1.0,
        random_seed: int = 2026010799,
        top_p: float = 1.0,
        reasoning_effort: str = 'high',

        # NEW:
        launch_server: bool = True,
        base_url_override: str | None = None,  # e.g. "http://10.0.0.12:5000"
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.log_level = log_level
        # self.api_key = api_key
        self.served_model_name = served_model_name
        self.dtype = dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.context_length = context_length
        self.mem_fraction_static = mem_fraction_static
        self.chunked_prefill_size = chunked_prefill_size         # NEW
        self.enable_torch_compile = enable_torch_compile         # NEW
        self.allow_auto_truncate = allow_auto_truncate           # NEW
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.trust_remote_code = trust_remote_code
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.enable_metrics = enable_metrics
        self.log_requests = log_requests
        self.log_requests_level = log_requests_level
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.random_seed = random_seed
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.estimate_reasoning = 'medium'
        self._launched_here = launch_server

        if launch_server:
            self.ensure_server_running(
                model_path=self.model_path,
                host=self.host,
                port=self.port,
                log_level=self.log_level,
                # api_key=self.api_key, # usuallly none
                served_model_name=self.served_model_name,
                dtype=self.dtype,
                kv_cache_dtype=self.kv_cache_dtype,
                context_length=self.context_length,
                mem_fraction_static=self.mem_fraction_static,
                chunked_prefill_size=self.chunked_prefill_size,      # NEW
                enable_torch_compile=self.enable_torch_compile,      # NEW
                allow_auto_truncate=self.allow_auto_truncate,        # NEW
                tp_size=self.tp_size,
                dp_size=self.dp_size,
                trust_remote_code=self.trust_remote_code,
                tool_call_parser=self.tool_call_parser,
                reasoning_parser=self.reasoning_parser,
                enable_metrics=self.enable_metrics,
                log_requests=self.log_requests,
                log_requests_level=self.log_requests_level,
                random_seed = self.random_seed
            )

        if base_url_override:
            # expect "http://HOST:PORT"
            self.native_base_url = base_url_override.rstrip("/")
            self.openai_base_url = f"{self.native_base_url}/v1"
        else:
            client_host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host

            self.openai_base_url = f"http://{client_host}:{self.port}/v1"
            self.native_base_url = f"http://{client_host}:{self.port}"

        # os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        # os.environ.setdefault("OPENAI_API_KEY", "sk-local")

        # Initialize client
        self.client = OpenAI(base_url=self.openai_base_url, api_key="sk-local", timeout=self.timeout_s)
        self.sandbox_workers = 8
        self.sandbox_timeout_s = 5.0  # how long generate() waits for a sandbox from the pool

        # use python_tool_timeout as default kernel timeout unless overridden later
        self.default_jupyter_timeout_s = 600.0

        self.tool_prompt = (
        'Use this tool to execute Python code for:\n'
        '- Complex calculations that would be error-prone by hand\n'
        '- Numerical verification of analytical results\n'
        '- Generating examples or testing conjectures\n'
        '- Visualizing problem structure when helpful\n'
        '- Brute-force verification for small cases\n\n'
        
        'The environment is a stateful Jupyter notebook. Code persists between executions.\n'
        'Always use print() to display results. Write clear, well-commented code.\n\n'
        
        'Remember: Code should support your mathematical reasoning, not replace it. '
        'Explain what you\'re computing and why before running code.'
    )

        self.sandbox_pool: queue.Queue[AIMO3Sandbox] = queue.Queue(maxsize=self.sandbox_workers)
        # Create 16 kernels, but only 4 in parallel at a time
        self._initialize_kernels(
            workers=self.sandbox_workers,
            jupyter_timeout=self.default_jupyter_timeout_s,
        )


    def __repr__(self) -> str:
        return (
            "SGLangServer("
            f"model_path={self.model_path!r}, host={self.host!r}, port={self.port!r}, "
            f"log_level={self.log_level!r}, trust_remote_code={self.trust_remote_code!r}, "
            f"tool_call_parser={self.tool_call_parser!r}, reasoning_parser={self.reasoning_parser!r}"
            ")"
        )


    def _initialize_kernels_full_high_memory(self, workers: int, jupyter_timeout: float) -> None:
        logger.info(f"Initializing {workers} persistent Jupyter kernels...")
        t0 = time.time()

        def _create_sandbox() -> AIMO3Sandbox:
            return AIMO3Sandbox(timeout=jupyter_timeout)

        # Create in parallel
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_create_sandbox) for _ in range(workers)]
            for fut in as_completed(futures):
                sb = fut.result()  # if kernel creation fails, this raises (good: fail early)
                self.sandbox_pool.put(sb)
        logger.info(f"Kernels initialized in {time.time() - t0:.2f}s")

    def _initialize_kernels(self, workers: int, jupyter_timeout: float) -> None:
        logger.info(f"Initializing {workers} persistent Jupyter kernels...")
        t0 = time.time()

        def _create_sandbox() -> AIMO3Sandbox:
            return AIMO3Sandbox(timeout=jupyter_timeout)

        create_parallelism = min(workers, 4)  # <= actually do what comment says
        with ThreadPoolExecutor(max_workers=create_parallelism) as ex:
            futures = [ex.submit(_create_sandbox) for _ in range(workers)]
            for fut in as_completed(futures):
                self.sandbox_pool.put(fut.result())

        logger.info(f"Kernels initialized in {time.time() - t0:.2f}s")

        

    def _acquire_sandbox(self, timeout_s: float) -> AIMO3Sandbox:
        try:
            return self.sandbox_pool.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(
                f"No sandbox available within {timeout_s}s. "
                f"pool_size={self.sandbox_pool.qsize()}/{self.sandbox_workers}"
            )
        

    def _release_sandbox(self, sandbox: AIMO3Sandbox) -> None:
        try:
            sandbox.close()
        except Exception:
            logger.exception("Sandbox close failed; continuing with replacement.")

        # Always replace with a fresh one so pool size stays stable
        try:
            self.sandbox_pool.put(AIMO3Sandbox(timeout=self.default_jupyter_timeout_s, preload="minimal"))
        except Exception:
            logger.exception("Failed to create replacement sandbox.")
            # If you want, you can re-raise or just drop it.
            raise

    @classmethod
    def start_server(
        cls,
        *,
        model_path: str,
        host: str,
        port: int,
        log_level: str = "warning",
        trust_remote_code: bool = True,
        # auth / identity
        # api_key: str | None = None,
        served_model_name: str | None = None,
        # perf / safety
        dtype: str | None = None,                 # e.g. "auto", "half", "bfloat16"
        kv_cache_dtype: str | None = None,        # e.g. "auto", "bf16", "fp8_e4m3"
        context_length: int | None = None,
        mem_fraction_static: float | None = None, # e.g. 0.80
        chunked_prefill_size: int | None = None,   # NEW
        enable_torch_compile: bool | None = None,  # NEW
        allow_auto_truncate: bool = True,          # NEW
        random_seed: int | None = None,
        # multi-gpu / placement
        tp_size: int | None = None,
        dp_size: int | None = None,
        base_gpu_id: int | None = None,
        gpu_id_step: int | None = None,
        device: str | None = None,                # e.g. "cuda"
        # reproducibility / IO
        download_dir: str | None = None,
        revision: str | None = None,
        tokenizer_path: str | None = None,
        tokenizer_mode: str | None = None,        # "auto" or "slow"
        # startup behavior
        skip_server_warmup: bool = False,
        warmups: str | None = None,               # comma-separated warmup names
        # tool/reasoning parsing (optional; only append if not None)
        tool_call_parser: str | None = None,
        reasoning_parser: str | None = None,
        tool_server: str | None = None,
        # misc controls (optional)
        max_running_requests: int | None = None, # not passed in
        max_queued_requests: int | None = None,  # not passed in
        max_total_tokens: int | None = None,     # not passed in
        enable_metrics: bool | None = None,
        log_requests: bool | None = None,
        log_requests_level: int | None = None,
        # escape hatch
        extra_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.Popen:
        """
        Build & launch:
        python3 -m sglang.launch_server ...

        Only appends flags when their corresponding arg is not None / True.
        """

        # Prefer passing env to subprocess instead of mutating global os.environ
        env = os.environ.copy()
        env["TRANSFORMERS_NO_TF"] = "1"
        env["TRANSFORMERS_NO_FLAX"] = "1"
        if extra_env:
            env.update(extra_env)

        cmd: list[str] = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            log_level,
        ]
        if allow_auto_truncate:
            cmd.append("--allow-auto-truncate")

        # auth / identity
        # if api_key:
        #     cmd += ["--api-key", api_key]
        if served_model_name:
            cmd += ["--served-model-name", served_model_name]

        # perf / safety
        if dtype:
            cmd += ["--dtype", dtype]
        if kv_cache_dtype:
            cmd += ["--kv-cache-dtype", kv_cache_dtype]
        if context_length is not None:
            cmd += ["--context-length", str(int(context_length))]
        if mem_fraction_static is not None:
            cmd += ["--mem-fraction-static", str(float(mem_fraction_static))]
        if chunked_prefill_size is not None:
            cmd += ["--chunked-prefill-size", str(int(chunked_prefill_size))]
        if enable_torch_compile is True:
            cmd.append("--enable-torch-compile")
        if random_seed is not None:
            cmd += ["--random-seed", str(int(random_seed))]

        # multi-gpu / placement
        if device:
            cmd += ["--device", device]
        if tp_size is not None:
            cmd += ["--tensor-parallel-size", str(int(tp_size))]
        if dp_size is not None:
            cmd += ["--data-parallel-size", str(int(dp_size))]
        if base_gpu_id is not None:
            cmd += ["--base-gpu-id", str(int(base_gpu_id))]
        if gpu_id_step is not None:
            cmd += ["--gpu-id-step", str(int(gpu_id_step))]

        # reproducibility / IO
        if download_dir:
            cmd += ["--download-dir", download_dir]
        if revision:
            cmd += ["--revision", revision]
        if tokenizer_path:
            cmd += ["--tokenizer-path", tokenizer_path]
        if tokenizer_mode:
            cmd += ["--tokenizer-mode", tokenizer_mode]

        # startup behavior
        if skip_server_warmup:
            cmd.append("--skip-server-warmup")
        if warmups:
            cmd += ["--warmups", warmups]

        # tool/reasoning parsing
        if tool_call_parser:
            cmd += ["--tool-call-parser", tool_call_parser]
        if reasoning_parser:
            cmd += ["--reasoning-parser", reasoning_parser]
        if tool_server:
            cmd += ["--tool-server", tool_server]

        # misc controls
        if max_running_requests is not None:
            cmd += ["--max-running-requests", str(int(max_running_requests))]
        if max_queued_requests is not None:
            cmd += ["--max-queued-requests", str(int(max_queued_requests))]
        if max_total_tokens is not None:
            cmd += ["--max-total-tokens", str(int(max_total_tokens))]
        if enable_metrics is True:
            cmd.append("--enable-metrics")
        if log_requests is True:
            cmd.append("--log-requests")
        if log_requests_level is not None:
            cmd += ["--log-requests-level", str(int(log_requests_level))]

        if trust_remote_code:
            cmd.append("--trust-remote-code")

        if extra_args:
            cmd += list(extra_args)

        logger.info("Starting SGLang server: {}", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
            start_new_session=True,
            env=env,
        )
        cls._SERVER_PROCESS = proc
        return proc

    @classmethod
    def ensure_server_running(
        cls,
        **start_kwargs,
    ) -> subprocess.Popen:
        # Reuse running process if alive
        if cls._SERVER_PROCESS is not None and cls._SERVER_PROCESS.poll() is None:
            return cls._SERVER_PROCESS

        proc = cls.start_server(**start_kwargs)

        port = int(start_kwargs["port"])
        health_url = f"http://127.0.0.1:{port}/v1/models"

        # Wait for readiness
        time.sleep(2.0)
        for attempt in range(1200):
            if proc.poll() is not None:
                raise RuntimeError(
                    "SGLang server process exited unexpectedly. "
                    "Check the console output for error messages."
                )
            try:
                resp = requests.get(health_url, timeout=1.0)
                if resp.status_code < 500:
                    print(f"[Model] SGLang server is up (attempt {attempt + 1}).")
                    break
            except requests.RequestException:
                pass
            time.sleep(1.0)
        else:
            raise RuntimeError(f"Timed out waiting for server to become ready at {health_url}.")

        cls._SERVER_PROCESS = proc
        return proc

    @classmethod
    def stop_server(cls) -> None:
        proc = cls._SERVER_PROCESS
        if proc is None:
            return

        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        cls._SERVER_PROCESS = None



    def gptoss_get_estimate_reasoning_effort_enum(self, estimate_reasoning) -> ReasoningEffort:
        """Map self.estimate_reasoning ('low'|'medium'|'high') to ReasoningEffort enum."""
        r = (estimate_reasoning or "").lower()
        if r == "low":
            return ReasoningEffort.LOW
        if r == "high":
            return ReasoningEffort.HIGH
        # default / unknown → medium
        return ReasoningEffort.MEDIUM
        
    def gptoss_estimate_problem_difficulty(
            self,
            problem: str,
            estimate_reasoning='medium'
    ):
        
        _HARMONY_TOK = tiktoken.get_encoding("o200k_harmony")
        _HARMONY_ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = _HARMONY_ENC.stop_tokens_for_assistant_actions()
        TIME_BUDGET_S = 10.0
        DEFAULT = "default_medium"
        DIFFICULTY_CLASSIFY_PROMPT_TEMPLATE = """You are an expert at estimating math olympiad problem difficulty.
Classify this problem on a scale:
- easy (basic algebra/counting, solvable quickly)
- medium (AIME-level, moderate insight needed)
- hard (national olympiad, deep reasoning/tools)
- hardest (IMO-level, very tricky/proofs)

Respond ONLY with one word: easy, medium, hard, or hardest.

Problem: {problem}


DO NOT think too hard about the problem. DO NOT attempt to come up with a complete solution.
Remember that your task is to classify the difficulty, not to solve the problem. Respond with only ONE word."""
        prompt = DIFFICULTY_CLASSIFY_PROMPT_TEMPLATE.format(problem=problem)
        client = self.client.with_options(timeout=TIME_BUDGET_S, max_retries=0)

        system_content = (
            SystemContent.new()
            .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
            # .with_model_identity(
            #     "You are an olympiad-level math problem solver. Before attempting the problem, you always estimate the difficulty of the problem."
            # )
            .with_reasoning_effort(reasoning_effort=self.gptoss_get_estimate_reasoning_effort_enum(estimate_reasoning=estimate_reasoning))
        )

        # rubric inspiration from https://web.evanchen.cc/upload/MOHS-hardness.pdf
        developer_content = (
            DeveloperContent.new().with_instructions(
                "You are an expert at estimating math olympiad problem difficulty. "
                "Classify problems according to the following scale: "
                "- easy (Entry level. Basic algebra, counting, or diagram drawing. Solvable quickly.) "
                "- medium (AIME-level, one or two insights needed which are not obvious from the problem statement. Some non-trivial algebraic manipulation and casework. Geometry problems at this level may require drawing additional helper lines. ) "
                "- hard (National Olympiad Level. Requires knowledge of obscure mathematical lemmas, or requires reasoning through multiple distinct cases. Geometry problems at this level may require additional, intricate constructions. )"
                "- hardest (Questions at the level of the International Math Olympiad. Either very tricky, often requiring multiple insights that are not obvious from the problem statement. Solutions might be very long with multiple cases that need to be examined. Geometry problems at this level might require additional, complicated constructions and special properties involving them. )"
            )
        )

        messages = [
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.DEVELOPER, developer_content),
            Message.from_role_and_content(Role.USER, prompt),
            
        ]

        prompt_ids = _HARMONY_ENC.render_conversation_for_completion(
            Conversation.from_messages(messages), Role.ASSISTANT
        )

        t0 = time.perf_counter()
        try:
            response = client.completions.create(
                model=self.served_model_name,
                prompt=prompt_ids,
                max_tokens=32768,
                temperature=0.1,
                top_p=self.top_p,
                seed=self.random_seed,
                stream=False,
                extra_body=dict(
                    stop_token_ids=stop_token_ids,
                    return_token_ids=True,
                ),
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.warning(f"[difficulty_timeout_or_error] elapsed={elapsed:.2f}s err={type(e).__name__}: {e}")
            return DEFAULT

        choice = response.choices[0]
        response_token_ids = getattr(choice, "token_ids", None)

        # Get token_ids

        if response_token_ids is None and getattr(choice, "model_extra", None):
            response_token_ids = choice.model_extra.get("token_ids")
        if not response_token_ids:
            raise RuntimeError(
                f"No token_ids returned from server. "
                f"choice.model_extra={getattr(choice, 'model_extra', None)!r}"
            )
        
        new_messages = _HARMONY_ENC.parse_messages_from_completion_tokens(
            response_token_ids,
            Role.ASSISTANT
        )


        last_message = new_messages[-1]
        def _message_text(msg) -> str:
            # Most Harmony messages store text in msg.content as a list of Content objects
            if not msg.content:
                return ""
            return "".join(
                c.text for c in msg.content
                if isinstance(c, TextContent) and getattr(c, "text", None) is not None
            )
        
        if last_message.channel == 'final':

            
            final_text = _message_text(last_message)
            text = final_text.strip().lower()
            logger.info(text)
            logger.info(f"[difficulty_final_channel] {text!r}")
            import re
            m = re.search(r"\b(easy|medium|hardest|hard)\b", text)
            return m.group(1) if m else DEFAULT
        
        else:
            return DEFAULT

    def gptoss_generate_with_python_tool_single_text_early_return(
        self,
        prompt: str,
        stop_event: Optional[threading.Event] = None,
        reasoning_budget: int = 125000,          # interpret as reasoning tokens (best-effort)
        python_tool_timeout: float = 5.0,
        reasoning_time: Optional[float] = None,  # wall-clock seconds
        max_new_tokens: Optional[int] = 8192,
    ) -> Tuple[str, Dict[str, Any]]:
        
        _HARMONY_TOK = tiktoken.get_encoding("o200k_harmony")
        _HARMONY_ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = _HARMONY_ENC.stop_tokens_for_assistant_actions()
        # tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_path,
        #     trust_remote_code=True,
        # )

        def _append_stop_reasoning_and_answer_tokens(
                prompt_ids: List[int]
                ) -> List[int]:
            
            """
            Returns a new prompt_ids list with tokens appended that makes the model
            think that it has reached the reasoning budget, and force the model to output final_text.

            Note: token IDs are hard-coded, tokenizer/encoding dependent
            tokenizer: o200k_harmony
            models: gpt_oss_20b | gpt_oss_120b.
            """

            ASSISTANT_START_TOKENS = [200006, 173781]  # <|start|>assistant
            STOP_REASONING_AND_ANSWER_TOKENS = [
                200005, 35644, 200008, 200007,  # <|channel|>analysis<|message|><|end|>
                200006, 173781,                 # <|start|>assistant
            ]

            # These are my observed special token IDs
            CHANNEL = 200005
            MESSAGE = 200008
            END = 200007
            START = 200006
            ASSISTANT = 173781
            ANALYSIS = 35644

            stop_text = (
                "Reasoning budget reached. STOP.\n"
                "Next assistant message MUST match this EXACT schema (2 lines only):\n"
                "(1) FINAL_ANSWER: \\boxed{}\n"
                '(2) SUMMARY_JSON: {"summary":"...","key_points":["...","...","..."],"sanity_check":"..."}\n'
                "Rules:\n"
                "- Output exactly 2 lines, nothing else.\n"
                "- SUMMARY_JSON must be valid JSON on a single line with exactly these keys: summary, key_points, sanity_check.\n"
                "- key_points must be an array of 3 to 6 short strings explaining the reasoning so far.\n"
                "- Do NOT omit SUMMARY_JSON.\n"
                "- Do NOT call tools. Do NOT include step-by-step reasoning."
            )
            # Convert the text into token IDs (raw string → ids)
            # You need a tokenizer instance somewhere; adapt this to how you already load it.
            # e.g. self.tokenizer = AutoTokenizer.from_pretrained(..., trust_remote_code=True)

            # text_ids = self.tokenizer.encode(stop_text, add_special_tokens=False)

            # hardcoded to make it run faster, loading AutoTokenizer is slow

            # boxed answer only
            # text_ids = [30377, 289, 9946, 15237, 13, 357, 738, 5666, 57927, 1954, 13, 730, 922, 2613, 3176, 357, 738, 4733, 43234, 290, 1721, 6052, 306, 2381, 172278, 108872, 3004, 6623, 57927, 13, 3004, 4584, 11666, 13]
            # summary with boxed answer
            text_ids = [30377, 289, 9946, 15237, 13, 82926, 558, 7695, 29186, 3176, 52178, 3981, 495, 195286, 19581, 350, 17, 8698, 1606, 1883, 7, 16, 8, 110437, 62, 160761, 25, 2381, 172278, 34494, 7, 17, 8, 119240, 43356, 25, 10494, 3861, 7534, 1008, 4294, 1898, 30070, 95067, 1008, 4294, 1008, 4294, 1008, 17695, 1, 33972, 536, 15847, 7534, 1008, 31085, 28744, 734, 12, 18315, 9707, 220, 17, 8698, 11, 6939, 1203, 558, 12, 119240, 43356, 2804, 413, 4529, 8205, 402, 261, 4590, 2543, 483, 9707, 1879, 12994, 25, 18522, 11, 2140, 30070, 11, 94610, 15847, 558, 12, 2140, 30070, 2804, 413, 448, 2644, 328, 220, 18, 316, 220, 21, 4022, 18279, 45379, 290, 57927, 813, 4150, 558, 12, 3756, 7116, 113296, 119240, 43356, 558, 12, 3756, 7116, 2421, 8437, 13, 3756, 7116, 3931, 5983, 23541, 41570, 57927, 13]

            out = list(prompt_ids)
            logger.warning(f"Current conversation token length: {len(out)} | Reasoning budget reached. Forcing answer now.")

            out.extend([
                CHANNEL, ANALYSIS, MESSAGE,
                *text_ids,
                END,
                START, ASSISTANT,
            ])

            return out

        def _get_reasoning_effort_enum() -> ReasoningEffort:
            """Map self.reasoning_effort ('low'|'medium'|'high') to ReasoningEffort enum."""
            r = (self.reasoning_effort or "").lower()
            if r == "low":
                return ReasoningEffort.LOW
            if r == "high":
                return ReasoningEffort.HIGH
            # default / unknown → medium
            return ReasoningEffort.MEDIUM
        
        def _apply_chat_template(prompt: str, python_tool) -> list[Message]:
            """Create Harmony messages with system prompt, developer instructions, and tools."""
        
            system_content = (
                SystemContent.new()
                .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
                .with_model_identity(
                    'You are a world-class International Mathematical Olympiad (IMO) competitor. '
                    'The final answer must be a non-negative integer between 0 and 99999. '
                    'You must place the final integer answer inside \\boxed{}.'
                )
                .with_reasoning_effort(reasoning_effort=_get_reasoning_effort_enum())
                .with_tools(python_tool.tool_config)
            )
        
            developer_content = (
                DeveloperContent.new().with_instructions(
                    "You may use the Python tool to run SMALL computations, check arithmetic, and inspect variables step by step. "
                    "The python tool is STATEFUL within this problem: variables and functions persist across python tool calls. "
                    "Do NOT use Python for heavy brute force, large-grid searches, or high-complexity algorithms "
                    "(e.g., nested loops over large ranges, exponential/backtracking, or anything with billions of iterations). "
                    "HARD CAP: avoid helpfully writing or running Python code with more than ~1e6 total loop iterations "
                    "(across all loops in a call); if an approach would exceed this, switch to analytic reasoning, closed-form "
                    "derivations, precomputation, memoization, or drastically reduce the search space first. "
                    "When using Python, keep loops tiny and make progress-visible prints when helpful. "
                    "Always put the final answer in \\boxed{} and also include a short summary of the reasoning."
                )
            )
        
            return [
                Message.from_role_and_content(Role.SYSTEM, system_content),
                Message.from_role_and_content(Role.DEVELOPER, developer_content),
                Message.from_role_and_content(Role.USER, prompt),
            ]

        def _message_text(msg) -> str:
            # Most Harmony messages store text in msg.content as a list of Content objects
            if not msg.content:
                return ""
            return "".join(
                c.text for c in msg.content
                if isinstance(c, TextContent) and getattr(c, "text", None) is not None
            )
        
        def _should_stop() -> bool:
            return (stop_event is not None and stop_event.is_set())
        
        def _choice_text_any(choice) -> str | None:
            # Chat API shape
            msg = getattr(choice, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if content:
                    return content

            # Legacy completions shape
            text = getattr(choice, "text", None)
            if text:
                return text

            # Sometimes servers stuff things into model_extra
            me = getattr(choice, "model_extra", None)
            if isinstance(me, dict):
                return me.get("text") or me.get("content")

            return None

        def _get_token_ids_fallback(choice, harmony_tok) -> list[int]:
            token_ids = getattr(choice, "token_ids", None)
            if token_ids is None and getattr(choice, "model_extra", None):
                token_ids = choice.model_extra.get("token_ids")
            if token_ids:
                return list(token_ids)

            text = _choice_text_any(choice)
            if not text:
                raise RuntimeError(
                    f"No token_ids and no text/content. model_extra={getattr(choice,'model_extra',None)!r}"
                )

            # IMPORTANT for Harmony markers like <|start|>assistant ...
            ids = harmony_tok.encode(text, allowed_special="all")

            me = getattr(choice, "model_extra", None)
            if isinstance(me, dict) and me.get("matched_stop") is not None:
                ids.append(int(me["matched_stop"]))

            return ids
        
        def _sglang_generate_with_ids(
            base_url: str,
            prompt_ids: list[int],
            *,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            stop_token_ids: list[int],
            timeout_s: float = 360.0,
        ):
            payload = {
                "input_ids": prompt_ids,
                "rid": str(uuid.uuid4()),
                "sampling_params": {
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "stop_token_ids": list(map(int, stop_token_ids)),
                    # CRITICAL: keep Harmony markers in text / decoding
                    "skip_special_tokens": False,
                    # optional, but often helpful for exact Harmony strings:
                    "spaces_between_special_tokens": False,
                    # optional: don’t trim stop tokens from decoded text
                    "no_stop_trim": True,
                },
                "stream": False,
            }

            r = requests.post(f"{base_url}/generate", json=payload, timeout=timeout_s)
            if r.status_code == 400:
                # This usually contains the real error message
                logger.error(f"400 body: {r.text}")
                raise RuntimeError(
                    "SGLang /generate returned 400.\n"
                    f"Response text:\n{r.text}\n\n"
                    f"payload keys={list(payload.keys())}\n"
                    f"len(input_ids)={len(payload.get('input_ids', []))}\n"
                    f"sampling_params={payload.get('sampling_params')}"
                )
            r.raise_for_status()
            data = r.json()

            text = data.get("text", "")
            output_ids = data.get("output_ids", None)   # <- this is what you want
            meta = data.get("meta_info", {})

            if not output_ids:
                raise RuntimeError(f"/generate returned no output_ids. keys={list(data.keys())} meta={meta!r}")

            return text, list(map(int, output_ids)), meta
        
        # ---------------------------
        # Stats accumulators (9 items)
        # ---------------------------
        tool_uses = 0
        tool_time_total = 0.0
        tool_timeouts = 0
        model_calls = 0
        model_time_total = 0.0  # time spent inside client.completions.create (incl. network wait)
        total_prefill_tokens = 0
        total_generated_tokens = 0
        total_tokens_processed = 0  # prefill + generated
        max_prefill_tokens_single_call = 0
        max_decode_tokens_single_call = 0
        max_prefill_iter = None
        max_decode_iter = None

        wall_t0 = time.perf_counter()
        sandbox = None
        python_tool = None
        cancelled_early = False
        try:
            if _should_stop():
                cancelled_early = True
                return "", {
                    "cancelled_early": True,
                    "num_tool_uses": tool_uses,
                    "tool_execution_time_s": tool_time_total,
                    "num_tool_timeouts": tool_timeouts,
                    "num_model_calls": model_calls,
                    "model_wait_time_s": model_time_total,
                    "total_prefill_tokens": total_prefill_tokens,
                    "total_generated_tokens": total_generated_tokens,
                    "total_tokens_processed": total_tokens_processed,
                    "tokens_per_second": (total_tokens_processed / model_time_total) if model_time_total > 0 else None,
                }
            
            else:
                sandbox = self._acquire_sandbox(self.sandbox_timeout_s)
                python_tool = AIMO3Tool(
                    local_jupyter_timeout=python_tool_timeout,
                    tool_prompt=self.tool_prompt,
                    sandbox=sandbox
                    )

                try:
                    messages = _apply_chat_template(prompt, python_tool)

                    for iteration in range(1024):
                        

                        if _should_stop():
                            cancelled_early = True
                            break

                        elapsed_s = time.perf_counter() - wall_t0
                        
                        prompt_ids = _HARMONY_ENC.render_conversation_for_completion(
                            Conversation.from_messages(messages), Role.ASSISTANT
                        )

                        # (2) If prompt length exceeds reasoning budget, force answer now
                        time_exceeded = (reasoning_time is not None and elapsed_s >= reasoning_time)
                        budget_exceeded = (len(prompt_ids) > reasoning_budget)
                        if time_exceeded or budget_exceeded:
                            logger.warning(
                                "reasoning budget or time reached. forcing answer."
                                f"reasoning budget: {reasoning_budget} | current tokens: {len(prompt_ids)} "
                                f"reasoning time: {reasoning_time} | current time: {elapsed_s}"
                            )
                            prompt_ids = _append_stop_reasoning_and_answer_tokens(prompt_ids)

                        SAFETY_MARGIN = 64
                        max_new = self.context_length - len(prompt_ids) - SAFETY_MARGIN
                        max_new = min(max_new, max_new_tokens)

                        if max_new < 1:
                            break

                        # If majority already found, don't start another call
                        if _should_stop():
                            cancelled_early = True
                            break


                        # ---- model call timing + counts ----
                        model_calls += 1
                        prefill_n = len(prompt_ids)
                        total_prefill_tokens += prefill_n

                        # max prefill
                        if prefill_n > max_prefill_tokens_single_call:
                            max_prefill_tokens_single_call = prefill_n
                            max_prefill_iter = iteration

                        t0 = time.perf_counter()

                        # response = self.client.completions.create(
                        #     model=self.served_model_name,
                        #     prompt=prompt_ids,
                        #     max_tokens=max_new,
                        #     temperature=self.temperature,
                        #     top_p=self.top_p,
                        #     seed=self.random_seed,
                        #     stream=False,
                        #     extra_body=dict(
                        #         stop_token_ids=stop_token_ids,
                        #         return_token_ids=True,
                        #     ),
                        #     timeout=360,
                        # )
                        text, response_token_ids, meta = _sglang_generate_with_ids(
                            self.native_base_url,
                            prompt_ids,
                            max_new_tokens=max_new,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            stop_token_ids=stop_token_ids,
                        )
                        model_time_total += (time.perf_counter() - t0)

                        decode_n = len(response_token_ids)   # “decode tokens”
                        total_generated_tokens += decode_n
                        total_tokens_processed += prefill_n + decode_n

                        # max decode
                        if decode_n > max_decode_tokens_single_call:
                            max_decode_tokens_single_call = decode_n
                            max_decode_iter = iteration

                        # text = response.choices[0].message.content

                        # logger.info(text)
                        # Get token_ids

                        # response_token_ids = _get_token_ids_fallback(text, harmony_tok=_HARMONY_TOK)

                        if not response_token_ids:
                            raise RuntimeError(
                                f"No token_ids returned from server. "
                                f"choice.model_extra={getattr(text, 'model_extra', None)!r}"
                            )
                        
                        
                        # Parse completion tokens into messages + append

                        new_messages = _HARMONY_ENC.parse_messages_from_completion_tokens(
                            response_token_ids,
                            Role.ASSISTANT
                        )

                        messages.extend(new_messages)
                        # logger.info(new_messages)

                        last_message = messages[-1]

                        if last_message.channel == 'final':
                            break
                        
                        # If majority found, do not execute more tools
                        if _should_stop():
                            cancelled_early = True
                            break
                        
                        if last_message.recipient == "python":
                            tool_uses += 1
                            # print(f"🐍 Executing Python code...")

                            # If majority found, don't execute tool
                            if _should_stop():
                                cancelled_early = True
                                break

                            t_tool0 = time.perf_counter()
                            response_messages = python_tool.process_sync_plus(last_message)
                            tool_time_total += (time.perf_counter() - t_tool0)
                            tool_text = "\n".join(_message_text(m) for m in response_messages).strip()
                            if tool_text.startswith("[ERROR]"):
                                tool_timeouts += 1
                            # logger.info(response_messages)
                            messages.extend(response_messages)
                    
                    final_msg = next(
                        (
                            m for m in reversed(messages)
                            if m.author.role == Role.ASSISTANT and m.channel == "final"
                        ),
                        None,
                    )

                    final_text = _message_text(final_msg).strip() if final_msg else ""
            
                    # Render full conversation
                    # return self.encoding.decode_utf8(
                    #     self.encoding.render_conversation_for_training(
                    #         Conversation.from_messages(messages),
                    #         RenderConversationConfig(auto_drop_analysis=True)
                    #     )
                    # )

                    tokens_per_second = (
                        (total_tokens_processed / model_time_total) if model_time_total > 0 else None
                    )

                    stats: Dict[str, Any] = {
                        "cancelled_early": cancelled_early,                                 # (0)
                        "num_tool_uses": tool_uses,                                         # (1)
                        "tool_execution_time_s": tool_time_total,                           # (2)
                        "num_tool_timeouts": tool_timeouts,                                 # (3)
                        "num_model_calls": model_calls,                                     # (4)
                        "model_wait_time_s": model_time_total,                              # (5)
                        "total_prefill_tokens": total_prefill_tokens,                       # (6)
                        "total_generated_tokens": total_generated_tokens,                   # (7)
                        "total_tokens_processed": total_tokens_processed,                   # (8)
                        "tokens_per_second": tokens_per_second,                             # (9)
                        "max_prefill_tokens_single_call": max_prefill_tokens_single_call,   # (10)
                        "max_decode_tokens_single_call": max_decode_tokens_single_call,     # (11)
                        "max_prefill_iter": max_prefill_iter,                               # (12)
                        "max_decode_iter": max_decode_iter,                                 # (13)
                    }

                finally:
                    if sandbox is not None:
                        self._release_sandbox(sandbox)
                    
                
                return final_text, stats


        except Exception as e:
            logger.error(f"Error in generation: {e}")
            stats = {
                    "cancelled_early": cancelled_early,                                 # (0)
                    "num_tool_uses": tool_uses,                                         # (1)
                    "tool_execution_time_s": tool_time_total,                           # (2)
                    "num_tool_timeouts": tool_timeouts,                                 # (3)
                    "num_model_calls": model_calls,                                     # (4)
                    "model_wait_time_s": model_time_total,                              # (5)
                    "total_prefill_tokens": total_prefill_tokens,                       # (6)
                    "total_generated_tokens": total_generated_tokens,                   # (7)
                    "total_tokens_processed": total_tokens_processed,                   # (8)
                    "tokens_per_second": (total_tokens_processed / model_time_total) if model_time_total > 0 else None, # (9)
                    "max_prefill_tokens_single_call": max_prefill_tokens_single_call,   # (10)
                    "max_decode_tokens_single_call": max_decode_tokens_single_call,     # (11)
                    "max_prefill_iter": max_prefill_iter,                               # (12)
                    "max_decode_iter": max_decode_iter,                                 # (13)
                    "error": repr(e)
                }
            
            return "", stats
        

    def gptoss_generate_with_python_tool_batch_text_early_return(
        self,
        prompts: List[str],
        majority_threshold: int,
        reasoning_budget: int = 125000,
        reasoning_time: Optional[float] = None,
        max_workers: Optional[int] = 1,
        python_tool_timeout: float = 5.0,
        max_new_tokens: int = 8192,
        # NEW:
        return_early: bool = True,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Batch generation with majority voting.

        return_early=True  (default):
            - same behavior as before: returns immediately when an answer reaches majority_threshold
            - unfinished responses remain "" and stats remain {}
            - cancels futures that haven't started (best effort) and shuts down executor non-blocking

        return_early=False:
            - when majority reached: set stop_event but DO NOT return
            - continue draining futures fully (as_completed) so workers can exit cooperatively
            and return their own stats (usually cancelled_early=True)
            - no explicit future cancellation is performed (per request)
        """
        if not prompts:
            return [], []

        n = len(prompts)
        responses: List[str] = [""] * n
        stats: List[Dict[str, Any]] = [{} for _ in range(n)]

        stop_event = threading.Event()

        clusters: List[Dict[str, Any]] = []   # each: {"rep": str, "count": int}
        clusters_lock = threading.Lock()

        if max_workers is None or max_workers < 1:
            max_workers = 1

        shutdown_nonblocking_called = False
        majority_reached = False
        majority_rep: Optional[str] = None

        def _safe_is_equiv(a: str, b: str) -> bool:
            try:
                return bool(is_equiv(a, b))
            except Exception:
                return False

        def _extract_pred_rep(text: str) -> Optional[str]:
            """
            Prefer boxed content (supports non-integers).
            Fallback to scan_for_answer (integer) -> string.
            """
            rep = extract_last_boxed_content(text)
            if rep is not None:
                rep = rep.strip()
                if rep:
                    return rep

            # fallback: old integer extractor
            ans_int = scan_for_answer(text)
            if ans_int is not None:
                return str(ans_int)

            return None

        def _update_clusters_and_check_majority(rep: str) -> Tuple[bool, str]:
            """
            Add rep into an existing equivalence class or create a new one.
            Returns (reached_majority, majority_rep_for_class_if_reached_else_rep_or_existing_rep).
            Caller must hold clusters_lock.
            """
            # Try to place into existing cluster
            for c in clusters:
                if _safe_is_equiv(rep, c["rep"]):
                    c["count"] += 1
                    if c["count"] >= majority_threshold:
                        return True, c["rep"]
                    return False, c["rep"]

            # New cluster
            clusters.append({"rep": rep, "count": 1})
            if 1 >= majority_threshold:
                return True, rep
            return False, rep

        def _run_one(i: int, p: str) -> Tuple[int, str, Dict[str, Any]]:
            r, s = self.gptoss_generate_with_python_tool_single_text_early_return(
                prompt=p,
                stop_event=stop_event,
                reasoning_budget=reasoning_budget,
                python_tool_timeout=python_tool_timeout,
                reasoning_time=reasoning_time,
                max_new_tokens=max_new_tokens,
            )
            return i, r, s

        ex = ThreadPoolExecutor(max_workers=max_workers)
        futures: List[Future] = []

        try:
            futures = [ex.submit(_run_one, i, prompts[i]) for i in range(n)]

            for fut in as_completed(futures):
                # If we are in "return early" mode, keep your existing behavior:
                if return_early and stop_event.is_set():
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    ex.shutdown(wait=False, cancel_futures=True)
                    shutdown_nonblocking_called = True
                    return responses, stats

                try:
                    i, r, s = fut.result()
                    responses[i] = r
                    stats[i] = s

                    # Only attempt majority vote if not already reached (small optimization)
                    if not majority_reached:
                        rep = _extract_pred_rep(r)
                        if rep is not None:
                            with clusters_lock:
                                reached, rep0 = _update_clusters_and_check_majority(rep)
                                if reached:
                                    majority_reached = True
                                    majority_rep = rep0
                                    stop_event.set()

                                    if return_early:
                                        # Same behavior as now: cancel what we can, return immediately
                                        for f in futures:
                                            if f is not fut and not f.done():
                                                f.cancel()
                                        ex.shutdown(wait=False, cancel_futures=True)
                                        shutdown_nonblocking_called = True
                                        return responses, stats
                                    # else: do NOT cancel; continue draining futures so workers return stats

                except Exception as e:
                    logger.error(f"Batch worker failed: {e}\n{traceback.format_exc()}")

            # If return_early=False, we intentionally reach here after draining all futures.
            return responses, stats

        finally:
            if not shutdown_nonblocking_called:
                try:
                    ex.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass
        

    def gptoss_generate_with_python_tool_batch_text_early_return_integers_only(
        self,
        prompts: List[str],
        majority_threshold: int,
        reasoning_budget: int = 125000,
        reasoning_time: Optional[float] = None,
        max_workers: Optional[int] = 1,
        python_tool_timeout: float = 5.0,
        max_new_tokens: int = 8192,
        # NEW:
        return_early: bool = True,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Batch generation with majority voting.

        return_early=True  (default):
            - same behavior as before: returns immediately when an answer reaches majority_threshold
            - unfinished responses remain "" and stats remain {}
            - cancels futures that haven't started (best effort) and shuts down executor non-blocking

        return_early=False:
            - when majority reached: set stop_event but DO NOT return
            - continue draining futures fully (as_completed) so workers can exit cooperatively
            and return their own stats (usually cancelled_early=True)
            - no explicit future cancellation is performed (per request)
        """
        if not prompts:
            return [], []

        n = len(prompts)
        responses: List[str] = [""] * n
        stats: List[Dict[str, Any]] = [{} for _ in range(n)]

        stop_event = threading.Event()

        counts = collections.Counter()
        counts_lock = threading.Lock()

        if max_workers is None or max_workers < 1:
            max_workers = 1

        shutdown_nonblocking_called = False
        majority_reached = False
        majority_answer: Optional[int] = None

        def _run_one(i: int, p: str) -> Tuple[int, str, Dict[str, Any]]:
            r, s = self.gptoss_generate_with_python_tool_single_text_early_return(
                prompt=p,
                stop_event=stop_event,
                reasoning_budget=reasoning_budget,
                python_tool_timeout=python_tool_timeout,
                reasoning_time=reasoning_time,
                max_new_tokens=max_new_tokens,
            )
            return i, r, s

        ex = ThreadPoolExecutor(max_workers=max_workers)
        futures: List[Future] = []

        try:
            futures = [ex.submit(_run_one, i, prompts[i]) for i in range(n)]

            for fut in as_completed(futures):
                # If we are in "return early" mode, keep your existing behavior:
                if return_early and stop_event.is_set():
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    ex.shutdown(wait=False, cancel_futures=True)
                    shutdown_nonblocking_called = True
                    return responses, stats

                try:
                    i, r, s = fut.result()
                    responses[i] = r
                    stats[i] = s

                    # Only attempt majority vote if not already reached (small optimization)
                    if not majority_reached:
                        ans = scan_for_answer(r)
                        if ans is not None and isinstance(ans, int) and ans >= 0:
                            with counts_lock:
                                counts[ans] += 1
                                if counts[ans] >= majority_threshold:
                                    majority_reached = True
                                    majority_answer = ans
                                    stop_event.set()

                                    if return_early:
                                        # Same behavior as now: cancel what we can, return immediately
                                        for f in futures:
                                            if f is not fut and not f.done():
                                                f.cancel()
                                        ex.shutdown(wait=False, cancel_futures=True)
                                        shutdown_nonblocking_called = True
                                        return responses, stats
                                    # else: do NOT cancel; continue draining futures so workers return stats

                except Exception as e:
                    logger.error(f"Batch worker failed: {e}\n{traceback.format_exc()}")

            # If return_early=False, we intentionally reach here after draining all futures.
            return responses, stats

        finally:
            if not shutdown_nonblocking_called:
                try:
                    ex.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass
                
    def gptoss_generate_from_prompt(
        self,
        prompt: str,
        *,
        system_identity: Optional[str] = None,
        developer_instructions: Optional[str] = None,
        estimate_reasoning: str = "medium",
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        time_budget_s: float = 360.0,
        return_meta: bool = False,
    ) -> str | Tuple[str, Dict[str, Any]]:
        """
        POST directly to SGLang /generate with Harmony input_ids, and parse output_ids back into Harmony messages.

        This avoids relying on client.completions.create() returning token_ids.
        """

        if temperature is None:
            temperature = float(self.temperature)
        if top_p is None:
            top_p = float(self.top_p)
        if seed is None:
            seed = int(self.random_seed)

        _HARMONY_ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = _HARMONY_ENC.stop_tokens_for_assistant_actions()

        # ----- Build Harmony messages -----
        system_content = (
            SystemContent.new()
            .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
            .with_reasoning_effort(
                reasoning_effort=self.gptoss_get_estimate_reasoning_effort_enum(estimate_reasoning)
            )
        )
        if system_identity:
            system_content = system_content.with_model_identity(system_identity)

        messages: List[Message] = [Message.from_role_and_content(Role.SYSTEM, system_content)]

        if developer_instructions:
            developer_content = DeveloperContent.new().with_instructions(developer_instructions)
            messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_content))

        messages.append(Message.from_role_and_content(Role.USER, prompt))

        prompt_ids = _HARMONY_ENC.render_conversation_for_completion(
            Conversation.from_messages(messages),
            Role.ASSISTANT,
        )

        # ----- Direct /generate call (returns output_ids) -----
        def _sglang_generate_with_ids(
            base_url: str,
            prompt_ids: list[int],
            *,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            stop_token_ids: list[int],
            seed: int,
            timeout_s: float,
        ) -> Tuple[str, List[int], Dict[str, Any]]:
            payload = {
                "input_ids": list(map(int, prompt_ids)),
                "rid": str(uuid.uuid4()),
                "sampling_params": {
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "stop_token_ids": list(map(int, stop_token_ids)),
                    # "seed": int(seed),
                    # Keep Harmony markers in decoding
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                    # Optional: don’t trim stop tokens from decoded text
                    "no_stop_trim": True,
                },
                "stream": False,
            }

            r = requests.post(f"{base_url.rstrip('/')}/generate", json=payload, timeout=timeout_s)
            if r.status_code == 400:
                # SGLang tends to include the real reason in body
                raise RuntimeError(f"SGLang /generate returned 400: {r.text}")
            r.raise_for_status()
            data = r.json()

            text = data.get("text", "")
            output_ids = data.get("output_ids", None)
            meta = data.get("meta_info", {}) or {}

            if not output_ids:
                raise RuntimeError(f"/generate returned no output_ids. keys={list(data.keys())} meta={meta!r}")

            return text, list(map(int, output_ids)), meta

        t0 = time.perf_counter()
        text, output_ids, meta_info = _sglang_generate_with_ids(
            self.native_base_url,
            prompt_ids,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            stop_token_ids=stop_token_ids,
            seed=int(seed),
            timeout_s=float(time_budget_s),
        )
        elapsed = time.perf_counter() - t0

        # ----- Parse Harmony messages back into structured messages -----
        parsed_messages = _HARMONY_ENC.parse_messages_from_completion_tokens(
            output_ids,
            Role.ASSISTANT,
        )

        def _message_text(msg: Message) -> str:
            if not getattr(msg, "content", None):
                return ""
            return "".join(
                c.text
                for c in msg.content
                if isinstance(c, TextContent) and getattr(c, "text", None) is not None
            )

        final_text = ""
        for msg in reversed(parsed_messages):
            if getattr(msg, "channel", None) == "final":
                final_text = _message_text(msg).strip()
                break

        if not final_text and parsed_messages:
            final_text = _message_text(parsed_messages[-1]).strip()

        if not return_meta:
            return final_text

        # meta_info from /generate is often richer than OpenAI usage()
        meta: Dict[str, Any] = {
            "elapsed_s": elapsed,
            "sglang_meta_info": meta_info,
            "prompt_len_tokens": len(prompt_ids),
            "output_len_tokens": len(output_ids),
            # "raw_text": text,  # decoded text as returned by /generate (may include special tokens)
            # "parsed_messages": parsed_messages,
        }
        return final_text, meta
    
    def gptoss_generate_from_prompts_batch(
        self,
        prompts: List[str],
        *,
        max_workers: Optional[int] = None,
        # forward args to single-call method
        system_identity: Optional[str] = None,
        developer_instructions: Optional[str] = None,
        estimate_reasoning: str = "medium",
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        time_budget_s: float = 360.0,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Batch-generate using ThreadPoolExecutor.

        Returns:
        final_texts: list[str]
        metas: list[dict]
        where final_texts[i] and metas[i] correspond to prompts[i].
        """

        n = len(prompts)
        if n == 0:
            return [], []

        if max_workers is None:
            # sensible default: avoid oversubscribing; tune as you like
            max_workers = min(32, n)

        final_texts: List[str] = [""] * n
        metas: List[Dict[str, Any]] = [{} for _ in range(n)]

        def _one(idx: int, prompt: str) -> Tuple[int, str, Dict[str, Any]]:
            try:
                text, meta = self.gptoss_generate_from_prompt(
                    prompt,
                    system_identity=system_identity,
                    developer_instructions=developer_instructions,
                    estimate_reasoning=estimate_reasoning,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    time_budget_s=time_budget_s,
                    return_meta=True,
                )
                # meta should already be a dict, but ensure it
                if not isinstance(meta, dict):
                    meta = {"_warning": "meta_not_dict", "meta_repr": repr(meta)}
                return idx, text, meta
            except Exception as e:
                # keep alignment: return empty text and error meta for this idx
                return idx, "", {
                    "error": True,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_one, i, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                idx, text, meta = fut.result()
                final_texts[idx] = text
                metas[idx] = meta

        return final_texts, metas


    def shutdown(self) -> None:
        if not getattr(self, "_launched_here", False):
            return
        self.stop_server()

def main():
    from math_verify import parse, verify

    import importlib
    mv = importlib.import_module("math_verify")

    v = getattr(mv, "__version__", None)
    if v is None:
        from importlib.metadata import version as pkg_version
        v = pkg_version("math_verify")

    print(v)
    ap = argparse.ArgumentParser()


    # model server settings
    ap.add_argument("--model_path", default='openai/gpt-oss-120b')
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--log_level", default="info")
    ap.add_argument("--served_model_name", default="sglang_model")
    ap.add_argument("--dtype", default=None)               # auto/half/bfloat16/...
    ap.add_argument("--kv_cache_dtype", default=None)      # auto/bf16/fp8_...
    ap.add_argument("--context_length", type=int, default=131072) # max_model_len
    ap.add_argument("--mem_fraction_static", type=float, default=None) # gpu_memory_utilization
    ap.add_argument("--chunked_prefill_size", type=int, default=16384)
    ap.add_argument("--enable_torch_compile", action="store_true")
    ap.add_argument("--allow_auto_truncate", dest="allow_auto_truncate", action="store_true")
    ap.add_argument("--no_allow_auto_truncate", dest="allow_auto_truncate", action="store_false")
    ap.set_defaults(allow_auto_truncate=True)
    ap.add_argument("--tp_size", type=int, default=1) # tensor_parallel usually set to 1
    ap.add_argument("--dp_size", type=int, default=1) # data_parallel, set to 1 for single gpu.
    ap.add_argument("--tool_call_parser", default=None)
    ap.add_argument("--reasoning_parser", default=None)
    # ap.add_argument("--max_running_requests", type=int, default=None)
    # ap.add_argument("--max_queued_requests", type=int, default=None)
    # ap.add_argument("--max_total_tokens", type=int, default=None)
    ap.add_argument("--enable_metrics", action="store_true")
    ap.add_argument("--log_requests", action="store_true")
    ap.add_argument("--log_requests_level", type=int, default=None)

    # generation settings
    ap.add_argument("--reasoning", default="high") # only for gpt-oss
    ap.add_argument("--estimate_reasoning", type=str, default='medium') # only for gpt-oss
    ap.add_argument("--sampling_defaults", type=str, default='openai')
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--random_seed", type=int, default=2026010799)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--reasoning_effort", type=str, default='high')
    # ap.add_argument("--min_p", type=float, default=0.02)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--majority_threshold", type=int, default=3)
    ap.add_argument("--reasoning_budget", type=int, default=65536)
    ap.add_argument("--python_tool_timeout", type=float, default=5.0)

    # client and data settings

    # Inference System settings (RSA, Streaming, Code_tool, etc.)

    args = ap.parse_args()

    inference_engine = SGLangServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        served_model_name=args.served_model_name,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
        enable_torch_compile=args.enable_torch_compile,
        allow_auto_truncate=args.allow_auto_truncate,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        enable_metrics=args.enable_metrics,
        log_requests=args.log_requests,
        log_requests_level=args.log_requests_level,
        temperature=args.temperature,
        random_seed=args.random_seed,
        top_p=args.top_p,
        reasoning_effort=args.reasoning_effort
    )


    question_1 = {
        'id': 'imo-bench-geometry-029',
        'problem': r'Let $XYZ$ be a triangle with $\angle X = 120^\circ$, $J$ be the incenter, and $N$ be the midpoint of $YZ$. The line passing through $N$ and parallel to $XJ$ intersects the circle with diameter $YZ$ at points $U$ and $V$ ($X$ and $U$ lie on the same semiplane with respect to $YZ$). The line passing through $U$ and perpendicular to $VJ$ intersects $XY$ and $XZ$ at points $R$ and $S$ respectively. Find the value of $\angle RJS$ in terms of degree.',
        'answer': '90',
        'problem_type': 'Geometry'
    }

    question_2 = {
        'id': 'imo-bench-geometry-062',
        'problem': r"Let $PQRS$ be a convex quadrilateral with $PQ=2, PS=7,$ and $RS=3$ such that the bisectors of acute angles $\angle{QPS}$ and $\angle{PSR}$ intersect at the midpoint of $\overline{QR}.$ Find the square of the area of $PQRS.$",
        'answer': '180',
        'problem_type': 'Geometry'
    }



    question_3 = {
        'id': 'imo-bench-geometry-072',
        'problem': r"Let $XYZ$ be a triangle inscribed in circle $(O)$ that is tangent to the sides $YZ, ZX, XY$ at points $U, V, W$ respectively. Assume that $M$ is the intersection of $YV$ and $ZW, N$ is the centroid of triangle $UVW, R$ is the symmetric point of $M$ about $N$. If $UR$ meets $VW$ at $S, T$ is on $VW$ such that $WT = VS$, compute $\angle UNV + \angle WNT$ in terms of degree.",
        'answer': '180',
        'problem_type': 'Geometry'
    }

    question_4 = {
        'id': 'imo-bench-geometry-095',
        'problem': r"In quadrilateral $PQRS$, $\angle QPS=\angle PQR=110^{\circ}$, $\angle QRS=35^{\circ}$, $\angle RSP=105^{\circ}$, and $PR$ bisects $\angle QPS$. Find $\angle PQS$ in terms of degree.",
        'answer': '40',
        'problem_type': 'Geometry'
    }

    question_5 = {
        'id': 'imo-bench-geometry-100',
        'problem': r"Triangle $XYZ$ is given with angles $\angle XYZ = 60^o$ and $\angle  YZX = 100^o$. On the sides $XY$ and $XZ$, the points $P$ and $Q$ are chosen, respectively, in such a way that $\angle  QPZ = 2\angle  PZY = 2\angle  ZXY$. Find the angle $\angle  YQP$ in terms of degree.",
        'answer': '10',
        'problem_type': 'Geometry'
    }

    format_prompt = r"Output the final answer within \boxed{}."
    questions = [question_1, question_2, question_3, question_4, question_5]
    # questions = [question_5]

    for qi, q in enumerate(questions, start=1):
        final_prompt = q["problem"] + "\n" + format_prompt
        prompts = [final_prompt] * 4

        logger.info(f"[Q{qi}] id={q['id']} running batch: n_prompts=8, majority_threshold=4")

        responses, batch_stats = inference_engine.gptoss_generate_with_python_tool_batch_text_early_return(
            prompts=prompts,
            majority_threshold=4,
            reasoning_budget=args.reasoning_budget,
            reasoning_time=None,
            max_workers=4,
            python_tool_timeout=args.python_tool_timeout,
            max_new_tokens=args.max_new_tokens,
        )

        for r in responses:
            print(f'####################__{responses.index(r)}__#####################')
            logger.info(r)

        # Log the extracted boxed answers (blanks remain blanks)
        final_answers = [scan_for_answer(r) for r in responses]
        logger.info(f"[Q{qi}] final boxed answers (8 samples, blanks kept): {final_answers}")

        # If you ALSO want to see which ones were blank:
        blank_idxs = [i for i, r in enumerate(responses) if not r]
        if blank_idxs:
            logger.info(f"[Q{qi}] blank response indices (early-returned): {blank_idxs}")




if __name__ == "__main__":
    main()




