from __future__ import annotations
import os
# os.environ.setdefault("PYTHON_EXECUTION_BACKEND", "dangerously_use_uv")
import argparse
import time
# import subprocess
from typing import Union, List, Optional, Callable, Dict, Any, Tuple, Mapping
from rewards.math import last_boxed_only_string, remove_boxed, is_equiv
# import kaggle_evaluation.aimo_3_inference_server
# import pandas as pd
import polars as pl
from loguru import logger
# from concurrent.futures import ThreadPoolExecutor
# from model_serving.vllm_model import VLLMModel
from model_serving.sglang_server import SGLangServer
from model_serving.inference_helpers import (
    extract_boxed_text,
    load_aimo3_csv_polars,
    load_aimo3_dicts_polars,
)
from prompts.prompt_helpers import prompt_dictionary
import os
import json
from pathlib import Path
from collections import Counter

def aggregate_generation_stats_per_question(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-generation stats (one dict per attempt) into per-question statistics.

    Expected keys per element (as you described):
      - total_prefill_tokens
      - total_generated_tokens
      - num_model_calls
      - num_tool_uses
      - tokens_per_second
      - max_prefill_tokens_single_call
      - max_decode_tokens_single_call
    """
    def _to_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    def _to_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _min_avg_max(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"min": None, "avg": None, "max": None}
        return {
            "min": min(values),
            "avg": sum(values) / len(values),
            "max": max(values),
        }

    total_prefill = 0
    total_generated = 0

    model_calls_vals: List[float] = []
    tool_uses_vals: List[float] = []
    tps_vals: List[float] = []
    max_prefill_single_vals: List[float] = []
    max_decode_single_vals: List[float] = []

    for s in stats_list or []:
        total_prefill += _to_int(s.get("total_prefill_tokens", 0))
        total_generated += _to_int(s.get("total_generated_tokens", 0))

        model_calls_vals.append(_to_float(s.get("num_model_calls", 0)))
        tool_uses_vals.append(_to_float(s.get("num_tool_uses", 0)))
        tps_vals.append(_to_float(s.get("tokens_per_second", 0.0)))
        max_prefill_single_vals.append(_to_float(s.get("max_prefill_tokens_single_call", 0)))
        max_decode_single_vals.append(_to_float(s.get("max_decode_tokens_single_call", 0)))

    aggregated: Dict[str, Any] = {
        "num_generations": len(stats_list),

        # 1) / 2)
        "total_prefill_tokens_all_generations": total_prefill,
        "total_generated_tokens_all_generations": total_generated,

        # 3)
        "num_model_calls_min_avg_max": _min_avg_max(model_calls_vals),

        # 4) / 5) (same thing; your spec repeats it)
        "num_tool_uses_min_avg_max": _min_avg_max(tool_uses_vals),

        # 6)
        "tokens_per_second_min_avg_max": _min_avg_max(tps_vals),

        # 7)
        "max_prefill_tokens_single_call_min_avg_max": _min_avg_max(max_prefill_single_vals),

        # 8)
        "max_decode_tokens_single_call_min_avg_max": _min_avg_max(max_decode_single_vals),
    }

    return aggregated

def _normalize_vote_token(x) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", "")
    if s.endswith("."):
        s = s[:-1].strip()
    return s if s else None

def majority_vote(predictions: list[str]) -> str:
    """
    Majority vote over predictions.
    - Ignores empty/None/whitespace predictions.
    - If tie, returns the first occurring tied value (stable tie-break).
    - If all empty, returns "".
    """
    cleaned = []
    for p in predictions:
        t = _normalize_vote_token(p)
        if t is not None:
            cleaned.append(t)

    if not cleaned:
        return ""

    counts = Counter(cleaned)
    top_count = max(counts.values())
    tied = {k for k, v in counts.items() if v == top_count}

    # stable tie-break: first occurrence in cleaned
    for p in cleaned:
        if p in tied:
            return p
    return ""  # should never hit

def _safe_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s.isdigit():
        return int(s)
    return None

def main():
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
    # ap.add_argument("--reasoning", default="high") # only for gpt-oss
    ap.add_argument("--reasoning_effort", type=str, default='high')
    ap.add_argument("--estimate_reasoning", type=str, default='medium') # only for gpt-oss
    ap.add_argument("--sampling_defaults", type=str, default='openai')
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--random_seed", type=int, default=2026010799)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=65536) # max_new tokens per generation
    ap.add_argument("--max_workers", type=int, default=8)
    # ap.add_argument("--min_p", type=float, default=0.02)
    ap.add_argument("--majority_threshold", type=int, default=3)


    # client and data settings
    ap.add_argument("--dataset_name", type=str, default='amalgamated1')
    ap.add_argument("--output_folder", type=str, default="./eval_aimo3/tirsc_7/gpt-oss-120b/")
    # Inference System settings (RSA, Streaming, Code_tool, etc.)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--reasoning_budget", type=int, default=125000)
    ap.add_argument("--python_tool_timeout", type=float, default=10.0)
    ap.add_argument(
        "--launch_server",
        action="store_true",
        help="If set, launch SGLang server locally. If not set, only connect (no model loading).",
    )
    ap.set_defaults(launch_server=False)

    # NEW: connect-only base URL (scheme://host:port). Example: http://10.42.0.15:5000
    ap.add_argument(
        "--base_url_override",
        type=str,
        default=None,
        help="If provided, connect to existing SGLang server at this base URL (e.g. http://HOST:PORT).",
    )

    args = ap.parse_args()

    dataset_csv_path = f"./data/aimo3/aimo3_{args.dataset_name}.csv"
    problems_df, answers_df = load_aimo3_csv_polars(csv_path=dataset_csv_path)

    start = 0
    end = 999
    length = end - start + 1  # 59

    problems_df = problems_df.slice(start, length)
    answers_df  = answers_df.slice(start, length)

    print("n_problems =", problems_df.height, "n_answers =", answers_df.height)
    # assert problems_df.height == 59 and answers_df.height == 59

    print('######################################################')
    print(problems_df)
    print('######################################################')
    print(answers_df)

    # ------------------------------
    # (1) Open output JSONL file once
    # ------------------------------

    def float_tag(x: float, ndp: int = 3) -> str:
        """
        Convert a float to a filename-safe string.
        Example: 0.95 -> '0p95', 1.0 -> '1p0'
        """
        s = f"{x:.{ndp}f}".rstrip("0").rstrip(".")  # trim trailing zeros
        if s == "-0":  # edge case
            s = "0"
        return s.replace(".", "p")
    

    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    temp_tag = float_tag(args.temperature, ndp=3)
    top_p_tag = float_tag(args.top_p, ndp=3)

    out_name = (
        f"{args.dataset_name}_tirsc7"
        f"_re{args.reasoning_effort}"
        f"_t{temp_tag}"
        f"_p{top_p_tag}"
        f"_n{args.max_new_tokens}"
        f"_seed{args.random_seed}.jsonl"
    )
    out_path = Path(args.output_folder) / out_name

    # Build answer lookup: id -> answer
    # (Assumes answers_df has columns: "id", "answer")
    answers_map = {
        row["id"]: row["answer"]
        for row in answers_df.iter_rows(named=True)
    }

    inference_engine = SGLangServer(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        served_model_name=args.served_model_name,       # must match what friend served
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        context_length=args.context_length,             # still needed for max_new computation
        mem_fraction_static=args.mem_fraction_static,
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
        reasoning_effort=args.reasoning_effort,

        launch_server=args.launch_server,
        base_url_override=args.base_url_override,
    )
    format_prompt = r"Output the final answer within \boxed{}."
    # results = []  # empty list to store per-problem dicts


    # Iterate through rows of problems_df
    try:
        with open(out_path, "a", encoding="utf-8") as f_jsonl:
            for row in problems_df.iter_rows(named=True):
                # 1) Look at id and problem
                problem_id = row["id"]  
                problem_text = row["problem"]

                # 2) Create a dictionary with keys id, predictions, pred, answer, statistics
                # item = {
                #     "id": problem_id,
                #     "predictions": None,
                #     "pred": None,
                #     "answer": None,
                #     "is_correct": None,
                #     "generation_statistics": None,
                #     "question_statistics": None,
                # }


                formatted_prompt = f"{problem_text} {format_prompt}"
                prompts = [formatted_prompt] * int(args.population)


                # ---- timing start ----
                t0 = time.perf_counter()

                # (2) call the SGLangServer batch generation method
                responses, stats = inference_engine.gptoss_generate_with_python_tool_batch_text_early_return(
                    prompts=prompts,
                    majority_threshold=args.majority_threshold,
                    reasoning_budget=args.reasoning_budget,
                    reasoning_time=None,
                    max_workers=args.max_workers,
                    python_tool_timeout=args.python_tool_timeout,
                    max_new_tokens=args.max_new_tokens, 
                )
                # ---- timing end ----
                question_time_s = time.perf_counter() - t0

                # (3) predictions from model
                predictions = [extract_boxed_text(r) for r in responses]
                # (4) majority vote
                pred = majority_vote(predictions)
                # (5) answer from answers_df
                answer = answers_map.get(problem_id, None)
                # (6) correctness
                pred_i = _safe_int(pred)
                ans_i  = _safe_int(answer)
                if pred_i is not None and ans_i is not None:
                    is_correct = (pred_i == ans_i)
                else:
                    is_correct = (str(pred).strip() == str(answer).strip())

                # (7) store generation_statistics + question_statistics as-is
                question_statistics = aggregate_generation_stats_per_question(stats_list=stats)
                # optionally: also store it inside question_statistics
                question_statistics["wall_time_s"] = question_time_s
                item = {
                    "id": problem_id,
                    "predictions": predictions,
                    "pred": pred,
                    "answer": answer,
                    "is_correct": is_correct,
                    "generation_statistics": stats,
                    "question_statistics": question_statistics,
                    "wall_time_s": question_time_s,  # <--- add this
                }

                # (8) append to JSONL
                f_jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_jsonl.flush()

                # results.append(item)
                logger.info(item)  # (optional) log per item (less spammy than logger.info(results))
                print("##########################################################")
                logger.info(
                    f"Done solving qid={problem_id} | correct={is_correct} | wall_time_s={question_time_s:.3f}"
                )


        logger.info(f"All data written to {out_path}")

    finally:
        # Always attempt to stop the server
        try:
            inference_engine.shutdown()
            logger.info("SGLangServer shut down successfully.")
        except Exception:
            logger.exception("Failed to shut down SGLangServer cleanly.")


if __name__ == "__main__":
    main()
    