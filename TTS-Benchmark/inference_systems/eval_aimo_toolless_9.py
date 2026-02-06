from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rewards.math import last_boxed_only_string, remove_boxed, is_equiv
from model_serving.inference_helpers import (
    # extract_boxed_text,
    scan_for_answer,
    extract_last_boxed_content,
    load_aimo3_csv_polars,
    load_aimo3_dicts_polars,
)
from model_serving.sglang_server import SGLangServer
from loguru import logger

# --- you already have these elsewhere in your script ---
# from model_serving.sglang_server import SGLangServer
# from your_module import load_aimo3_csv_polars, extract_last_boxed_content, is_equiv


# =========================
# Aggregation (NEW META)
# =========================

def aggregate_generation_stats_per_question(meta_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-generation meta dicts (one per attempt) into per-question statistics.

    Expected keys per element:
      - prompt_len_tokens
      - output_len_tokens

    Returns:
      - num_generations
      - total_prompt_len_tokens_all_generations
      - total_output_len_tokens_all_generations
      - max_prompt_len_tokens
      - max_output_len_tokens
    """

    def _to_int(x: Any, default: int = 0) -> int:
        try:
            if x is None:
                return default
            return int(x)
        except Exception:
            return default

    total_prompt = 0
    total_output = 0
    max_prompt: Optional[int] = None
    max_output: Optional[int] = None

    for m in meta_list or []:
        p = _to_int(m.get("prompt_len_tokens"), 0)
        o = _to_int(m.get("output_len_tokens"), 0)

        total_prompt += p
        total_output += o

        max_prompt = p if max_prompt is None else max(max_prompt, p)
        max_output = o if max_output is None else max(max_output, o)

    return {
        "num_generations": len(meta_list or []),
        "total_prompt_len_tokens_all_generations": total_prompt,
        "total_output_len_tokens_all_generations": total_output,
        "max_prompt_len_tokens": max_prompt,
        "max_output_len_tokens": max_output,
    }


# =========================
# Majority + evaluation helpers (unchanged)
# =========================

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


def evaluate_k_answers_math(
    k_answers: List[Optional[str]],
    gt: str
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    k_answers: list of extracted boxed contents, e.g. ["1/2", "\\frac{1}{2}", "0.5", None, ""]
    gt: ground truth string

    Returns:
      (metrics_dict, majority_pred_rep)
    """
    extracted: List[str] = []
    for a in k_answers:
        extracted.append("" if a is None else str(a).strip())

    correct_bools = [bool(is_equiv(e, gt)) for e in extracted]
    mean_acc = float(sum(correct_bools) / max(1, len(correct_bools)))
    pass_at_k = float(1.0 if any(correct_bools) else 0.0)

    clusters: List[Dict[str, Any]] = []
    for e in extracted:
        if not e:
            continue
        placed = False
        for c in clusters:
            if bool(is_equiv(e, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": e, "count": 1})

    if not clusters:
        majority_pred_rep: Optional[str] = None
        majority_vote = 0.0
    else:
        best = max(clusters, key=lambda c: c["count"])
        majority_pred_rep = best["rep"]
        majority_vote = float(bool(is_equiv(majority_pred_rep, gt)))

    metrics = {
        "pred_accuracies": [float(b) for b in correct_bools],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote,
    }
    return metrics, majority_pred_rep


def majority_vote(predictions: list[str]) -> str:
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

    for p in cleaned:
        if p in tied:
            return p
    return ""


def _safe_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s.isdigit():
        return int(s)
    return None


# =========================
# Main (UPDATED: no python tool)
# =========================

def main():
    ap = argparse.ArgumentParser()

    # model server settings
    ap.add_argument("--model_path", default="openai/gpt-oss-120b")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--log_level", default="info")
    ap.add_argument("--served_model_name", default="sglang_model")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--kv_cache_dtype", default=None)
    ap.add_argument("--context_length", type=int, default=131072)
    ap.add_argument("--mem_fraction_static", type=float, default=None)
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--dp_size", type=int, default=1)
    ap.add_argument("--tool_call_parser", default=None)
    ap.add_argument("--reasoning_parser", default=None)
    ap.add_argument("--enable_metrics", action="store_true")
    ap.add_argument("--log_requests", action="store_true")
    ap.add_argument("--log_requests_level", type=int, default=None)

    # generation settings
    ap.add_argument("--reasoning_effort", type=str, default="high")
    ap.add_argument("--estimate_reasoning", type=str, default="medium")
    ap.add_argument("--sampling_defaults", type=str, default="openai")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--random_seed", type=int, default=2026020499)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=65536)
    ap.add_argument("--max_workers", type=int, default=8)

    # kept for compatibility, but no longer used in this no-tool version:
    ap.add_argument("--majority_threshold", type=int, default=3)
    ap.add_argument("--reasoning_budget", type=int, default=125000)
    ap.add_argument("--python_tool_timeout", type=float, default=10.0)

    # client and data settings
    ap.add_argument("--dataset_name", type=str, default="amalgamated1")
    ap.add_argument("--output_folder", type=str, default="./eval_aimo3/tirsc_7/gpt-oss-120b/")
    ap.add_argument("--population", type=int, default=8)

    ap.add_argument(
        "--launch_server",
        action="store_true",
        help="If set, launch SGLang server locally. If not set, only connect (no model loading).",
    )
    ap.set_defaults(launch_server=False)

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
    length = end - start + 1

    problems_df = problems_df.slice(start, length)
    answers_df = answers_df.slice(start, length)

    print("n_problems =", problems_df.height, "n_answers =", answers_df.height)

    def float_tag(x: float, ndp: int = 3) -> str:
        s = f"{x:.{ndp}f}".rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s.replace(".", "p")

    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    temp_tag = float_tag(args.temperature, ndp=3)
    top_p_tag = float_tag(args.top_p, ndp=3)

    out_name = (
        f"{args.dataset_name}_ARIAAgentToolless"
        f"_re{args.reasoning_effort}"
        f"_t{temp_tag}"
        f"_p{top_p_tag}"
        f"_n{args.max_new_tokens}"
        f"_seed{args.random_seed}.jsonl"
    )
    out_path = Path(args.output_folder) / out_name

    answers_map = {row["id"]: row["answer"] for row in answers_df.iter_rows(named=True)}

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

    try:
        with open(out_path, "a", encoding="utf-8") as f_jsonl:
            for row in problems_df.iter_rows(named=True):
                problem_id = row["id"]
                problem_text = row["problem"]

                formatted_prompt = f"{problem_text} {format_prompt}"
                prompts = [formatted_prompt] * int(args.population)

                # ---- timing start ----
                t0 = time.perf_counter()

                # (NEW) batch generation WITHOUT python tool
                responses, metas = inference_engine.gptoss_generate_from_prompts_batch(
                    prompts=prompts,
                    max_workers=args.max_workers,
                    estimate_reasoning=args.estimate_reasoning,
                    max_tokens=args.max_new_tokens,   # maps to max_new_tokens in /generate wrapper
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.random_seed,
                    time_budget_s=1800.0,              # adjust if you want CLI arg
                )

                question_time_s = time.perf_counter() - t0
                # ---- timing end ----

                boxed_preds: List[Optional[str]] = [extract_last_boxed_content(r) for r in responses]

                answer = answers_map.get(problem_id, None)
                gt = "" if answer is None else str(answer).strip()

                eval_metrics, majority_rep = evaluate_k_answers_math(boxed_preds, gt)
                pred = majority_rep if majority_rep is not None else ""
                is_correct = bool(eval_metrics["majority_vote_correct"])

                pred_int = _safe_int(pred)
                ans_int = _safe_int(gt)

                # (UPDATED) question stats now aggregate prompt/output token lengths from metas
                question_statistics = aggregate_generation_stats_per_question(meta_list=metas)
                question_statistics["wall_time_s"] = question_time_s

                item = {
                    "id": problem_id,
                    "predictions": boxed_preds,
                    "pred": pred,
                    "answer": answer,
                    "eval_metrics": eval_metrics,
                    # keep raw per-generation info
                    "generation_statistics": metas,          # metas now, not old tool+timing stats
                    "question_statistics": question_statistics,
                    "wall_time_s": question_time_s,
                    "pred_int": pred_int,
                    "answer_int": ans_int,
                }

                f_jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_jsonl.flush()

                logger.info(item)
                print("##########################################################")
                logger.info(f"Done solving qid={problem_id} | correct={is_correct} | wall_time_s={question_time_s:.3f}")

        logger.info(f"All data written to {out_path}")

    finally:
        try:
            inference_engine.shutdown()
            logger.info("SGLangServer shut down successfully.")
        except Exception:
            logger.exception("Failed to shut down SGLangServer cleanly.")


if __name__ == "__main__":
    main()