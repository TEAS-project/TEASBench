import argparse
import os
import re
import random
import collections # we are using Counter but we don't want a name clash
from typing import Union, List, Optional, Callable, Dict, Any, Tuple
import json
import time
import numpy as np
# from rewards.math import last_boxed_only_string, remove_boxed, is_equiv
# import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl
from loguru import logger
from functools import partial

# from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Histogram, Metric

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort
)


# =========================
# vllm metrics helpers
# =========================




# =========================
# Data Loading
# =========================


def load_aimo3_csv_polars(csv_path: str):
    """
    Load the AIMO3-format CSV and return:
      1. DataFrame with columns: id, problem
      2. DataFrame with columns: id, answer
    """
    # Read CSV
    df = pl.read_csv(csv_path)

    # First dataframe: id + problem
    problems_df = df.select([
        pl.col("id"),
        pl.col("problem")
    ])

    # Second dataframe: id + answer
    answers_df = df.select([
        pl.col("id"),
        pl.col("answer")
    ])

    return problems_df, answers_df


# =========================
# prompts
# =========================

prompt_dictionary = {
    # generic helpers
    # think step by step was taken from the aime25 dataset problem text. aimo3 dataset does not have this. Need to include it to reproduce RSA
    'think_step_by_step': 'Lets think step by step and output the final answer within \\boxed{}.',

    # format hints
    'rg_format_hint' : '<answer>...</answer>',
    'supergpqa_format_hint' : '\\boxed{}. Only include the correct option letter in \\boxed{}; for example \\boxed{A}',
    'math_format_hint' : '\\boxed{}',

    # problem kind labels
    "rg_problem_kind": "problem",
    "supergpqa_problem_kind": "multiple-choice problem",
    "math_problem_kind": "math problem",

    # aggregation templates: single-candidate
    "agg_single_preamble": (
        "You are given a {problem_kind} and a candidate solution. "
        "The candidate may be incomplete or contain errors. "
        "Refine this trajectory and produce an improved, higher-quality solution. "
        "If it is entirely wrong, attempt a new strategy. "
        "End with the final result in {format_hint}.\n"
    ),
    "agg_single_candidate_header": "Candidate solution (may contain mistakes):\n",
    "agg_single_candidate_block": "---- Candidate ----\n{candidate}\n",
    "agg_single_tail": (
        "Now refine the candidate into an improved solution. "
        "Provide clear reasoning and end with the final answer in {format_hint}."
    ),

    # aggregation templates: multi-candidate
    "agg_multi_preamble": (
        "You are given a {problem_kind} and several candidate solutions. "
        "Some candidates may be incorrect or contain errors. "
        "Aggregate the useful ideas and produce a single, high-quality solution. "
        "Reason carefully; if candidates disagree, choose the correct path. "
        "If all are incorrect, then attempt a different strategy."
        "End with the final result in {format_hint}.\n"
    ),
    "agg_multi_candidates_header": "Candidate solutions (may contain mistakes):\n",
    "agg_multi_candidate_block": "---- Solution {i} ----\n{candidate}\n",
    "agg_multi_tail": (
        "Now write a single improved solution. "
        "Provide clear reasoning and end with the final answer in {format_hint}."
    ),

    # common headers
    "agg_problem_header": "Problem:\n{question}\n",
}

def render_chat_template(tokenizer, prompt: str) -> str:
    chat_message = [
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(chat_message, tokenize=False, add_generation_prompt=True)


def render_chat_template_gpt_non_streaming(
        tokenizer,
        prompt: str,
        reasoning
        ) -> List[int]:
    convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(reasoning)),
        Message.from_role_and_content(Role.USER, prompt),
    ]
    )
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    return prefill_ids


def aggregate_prompt(
        question: str,
        candidate_answers: List[str],
        task: str,
        prompt_dictionary: dict
        ) -> str:
    # pick problem kind + format hint from dict
    if task == "rg":
        problem_kind = prompt_dictionary["rg_problem_kind"]
        format_hint = prompt_dictionary["rg_format_hint"]
    elif task == "supergpqa":
        problem_kind = prompt_dictionary["supergpqa_problem_kind"]
        format_hint = prompt_dictionary["supergpqa_format_hint"]
    else:
        problem_kind = prompt_dictionary["math_problem_kind"]
        format_hint = prompt_dictionary["math_format_hint"]

    parts = []

    # preamble
    if len(candidate_answers) == 1:
        preamble = prompt_dictionary["agg_single_preamble"].format(
            problem_kind=problem_kind,
            format_hint=format_hint,
        )
    else:
        preamble = prompt_dictionary["agg_multi_preamble"].format(
            problem_kind=problem_kind,
            format_hint=format_hint,
        )
    parts.append(preamble)

    # problem header
    parts.append(
        prompt_dictionary["agg_problem_header"].format(
            question=question.strip()
        )
    )

    # candidate(s) section
    if len(candidate_answers) == 1:
        parts.append(prompt_dictionary["agg_single_candidate_header"])
        ans_str = (candidate_answers[0] or "").strip()
        parts.append(
            prompt_dictionary["agg_single_candidate_block"].format(
                candidate=ans_str
            )
        )
        parts.append(
            prompt_dictionary["agg_single_tail"].format(
                format_hint=format_hint
            )
        )
    else:
        parts.append(prompt_dictionary["agg_multi_candidates_header"])
        for i, ans in enumerate(candidate_answers, 1):
            ans_str = (ans or "").strip()
            parts.append(
                prompt_dictionary["agg_multi_candidate_block"].format(
                    i=i,
                    candidate=ans_str,
                )
            )
        parts.append(
            prompt_dictionary["agg_multi_tail"].format(
                format_hint=format_hint
            )
        )

    return "\n".join(parts)



def build_prompt(
    tokenizer: Any,
    question: str,
    candidate_answers: Optional[List[Optional[str]]],
    task: str,
    chat_template_fn: Callable[[Any, str], List],
    prompt_dictionary: dict,
) -> Union[str, List[int]]:
    """
    Build the new prompt to feed into the LLM.

    - If `candidate_answers` is None or contains no *real* (non-empty) strings,
      we just use the base `question`.
    - Otherwise, we call `aggregate_prompt(...)` to build an aggregation prompt.
    """

    # Decide whether we actually have any meaningful candidates
    has_real_candidate = (
        candidate_answers is not None
        and any(
            (c is not None) and isinstance(c, str) and c.strip() != ""
            for c in candidate_answers
        )
    )

    if has_real_candidate:
        # At least one real candidate string -> aggregation / refinement prompt
        prompt = aggregate_prompt(
            question=question,
            candidate_answers=[c for c in candidate_answers if c is not None],
            task=task,
            prompt_dictionary=prompt_dictionary,
        )
    else:
        # No candidates / only Nones / only empty strings -> solve-from-scratch
        prompt = question + ' ' + prompt_dictionary['think_step_by_step']

    return chat_template_fn(tokenizer, prompt)

# =========================
# verifier
# =========================

# leaving out for now. See deepseek-r2 verifier


# =========================
# Evaluation
# =========================

# Not needed for AIMO
# Needed for ARIA project
# Implement later

# =========================
# Model wrapper using vLLM
# =========================
 

class Model:
    """Model wrapper that loads a vLLM LLM, tokenizer, and sampling params."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        tp_size: int = 1,
        gpu_memory_utilization: float = 0.84,
        enforce_eager = False,
        dtype: str = "bfloat16",
        seed: int = 2025999901,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        max_new_tokens: int = None,
        temperature: float = 1.0,
        trust_remote_code=True,
        disable_log_stats=True
    ):
        # Save config
        self.model_name = model_name
        self.tp_size = tp_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.dtype = dtype
        self.seed = seed
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.trust_remote_code = trust_remote_code
        self.diable_log_stats = disable_log_stats

        # These will be fully constructed in __init__
        self.llm: Optional[LLM] = None
        self.tokenizer: Optional[Any] = None
        self.sampling_params: Optional[SamplingParams] = None

        # ---- Build LLM (mirrors your snippet) ----
        if "nemo" in self.model_name:
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tp_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=self.enforce_eager,
                dtype=self.dtype,
                seed=self.seed,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                kv_cache_dtype="auto",
                mamba_ssm_cache_dtype="float32",
                trust_remote_code=self.trust_remote_code,
                disable_log_stats=disable_log_stats,
            )
        else:
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tp_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=self.enforce_eager,
                dtype=self.dtype,
                seed=self.seed,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                kv_cache_dtype="auto",
                trust_remote_code=self.trust_remote_code,
                disable_log_stats=disable_log_stats,
            )

        # ---- Tokenizer ----
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_name,
        #     trust_remote_code=self.trust_remote_code,
        # )

        self.tokenizer = self.llm.get_tokenizer()

        # ---- Sampling params (mirrors your snippet) ----
        if "gpt" in self.model_name:
            
            # initialize these if and only if we are using gpt, otherwise don't waste memory
            self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

            self.sampling_params = SamplingParams(
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                stop_token_ids=self.stop_token_ids
            )
        else:
            self.sampling_params = SamplingParams(
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )


    def generate_single(self, prompt: Union[str, List[int]]) -> str:
        """
        Generate a single response for a single prompt string.

        Returns the text of the first output.
        """
        if self.llm is None or self.sampling_params is None:
            logger.error("Model is not initialized correctly. Either self.llm is None or self.sampling_params is None. Please check __init__ of the Model class.")
            raise RuntimeError("Model is not initialized correctly.")

        # GPT-style models: feed token IDs via TokensPrompt, like your friend's code
        if "gpt" in self.model_name.lower():

            if not isinstance(prompt, list):
                logger.error(f'GPT model expects token IDs (List[int]) from build_prompt, but got {type(prompt)}')
                raise TypeError(
                    f"GPT model expects token IDs (List[int]) from build_prompt, "
                    f"but got {type(prompt)}"
                )
            # Encode the prompt to token IDs
            prompts = [TokensPrompt(prompt_token_ids=prompt)]

            outs = self.llm.generate(
                prompts,
                sampling_params=self.sampling_params,
            )
        else:
            # Non-GPT models: pass raw string, vLLM will tokenize internally
            outs = self.llm.generate(
                [prompt],
                sampling_params=self.sampling_params,
            )

        # Extract the first output text
        if not outs or not outs[0].outputs:
            return ""

        return outs[0].outputs[0].text



    def generate_batch(self, prompts: List[Union[str, List[int]]]) -> List[str]:

        if self.llm is None or self.sampling_params is None:
            logger.error("Model is not initialized correctly. Either self.llm is None or self.sampling_params is None. Please check __init__ of the Model class.")
            raise RuntimeError("Model is not initialized correctly.")

        if not prompts:
            return []

        if "gpt" in self.model_name.lower():
            vllm_prompts = []

            # erro checking
            for p in prompts:
                if not isinstance(p, list):
                    logger.error(f'GPT model expects token IDs (List[int]) in batch, but got {type(p)}')
                    logger.error(p)
                    raise TypeError(
                        f"GPT model expects token IDs (List[int]) in batch, "
                        f"but got {type(p)}"
                    )
                vllm_prompts.append(TokensPrompt(prompt_token_ids=p))

            outs = self.llm.generate(
                vllm_prompts,
                sampling_params=self.sampling_params,
            )
        else:

            # error checking
            for p in prompts:
                if not isinstance(p, str):
                    logger.error(f'Non-GPT model expects string prompts in batch, but got {type(p)}')
                    logger.error(p)
                    raise TypeError(
                        f"Non-GPT model expects string prompts in batch, "
                        f"but got {type(p)}"
                    )
            outs = self.llm.generate(
                prompts,
                sampling_params=self.sampling_params,
            )

        responses: List[str] = []
        for out in outs:
            if out.outputs:
                responses.append(out.outputs[0].text)
            else:
                responses.append("")

        return responses
    

    def generate(self, prompt_or_prompts: Union[str, List[int], List[Union[str, List[int]]]]):
        """
        Convenience method:
        - If given a single prompt (str or List[int]), returns a single string.
        - If given a list of prompts, returns a list of strings.
        """
        # GPT single: List[int] (flat)
        if isinstance(prompt_or_prompts, list) and prompt_or_prompts and isinstance(prompt_or_prompts[0], int):
            return self.generate_single(prompt_or_prompts)  # type: ignore[arg-type]

        # batch: list of prepared prompts
        if isinstance(prompt_or_prompts, list):
            return self.generate_batch(prompt_or_prompts)  # type: ignore[arg-type]

        # single non-GPT: string
        return self.generate_single(prompt_or_prompts)  # type: ignore[arg-type]

# =========================
# Helper functions
# =========================



def generate_candidate_groups(
    previous_candidates: Optional[List[str]],
    parallel: int,
    k: int,
) -> List[Optional[List[str]]]:
    """
    Given the pool of previous candidates, create 'parallel' groups of size 'k'
    to use for aggregation prompts.

    - If previous_candidates is None -> first loop: we return [None] * parallel,
      meaning "solve from scratch" prompts.
    """
    if previous_candidates is None:
        return [None for _ in range(parallel)]

    if len(previous_candidates) <= k:
        # Not enough variety; just reuse the whole pool
        return [previous_candidates for _ in range(parallel)]

    groups = []
    for _ in range(parallel):
        groups.append(random.sample(previous_candidates, k))
    return groups


def extract_boxed_answer(text: str) -> Optional[int]:
    """
    Try to extract an integer answer from \\boxed{...}.
    If not found or not an integer, return None.
    """
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not matches:
        return None
    candidate = matches[-1].strip()  # last boxed
    # Keep only digits (AIMO wants 0..99999)
    if re.fullmatch(r"\d+", candidate):
        val = int(candidate)
        if 0 <= val <= 99999:
            return val
    return None


def extract_fallback_integer(text: str) -> Optional[int]:
    """
    Fallback: extract the last integer in the text, if any.
    """
    ints = re.findall(r"\d+", text)
    if not ints:
        return None
    val = int(ints[-1])
    if 0 <= val <= 99999:
        return val
    return None


def extract_answer_int(text: str) -> Optional[int]:
    """
    Unified answer extractor: try \\boxed{}, then fallback integer.
    """
    ans = extract_boxed_answer(text)
    if ans is not None:
        return ans
    return extract_fallback_integer(text)


def choose_final_answer(candidates: List[str]) -> int:
    """
    Given a list of solution texts, extract integer answers and perform majority vote.
    If no valid answers are found, return 0 as a safe default.
    """
    extracted = [extract_answer_int(c) for c in candidates]
    valid = [a for a in extracted if a is not None]

    if not valid:
        return 0

    counts = collections.Counter(valid)
    # Most common answer; if tie, Counter.most_common handles it deterministically
    final_answer, _ = counts.most_common(1)[0]
    return final_answer



# =========================
# Recursive Self-Aggregation for ONE problem
# =========================


# use applyRSA for aimo3

def applyRSA(
    model: Model,
    question: str,
    initial_candidates: Optional[List[str]],
    task: str,
    prompt_dictionary: dict,
    sequential: int,
    parallel: int,
    k: int,
    reasoning: str = "medium",
) -> List[str]:
    """
    RSA-style iterative refinement for a single problem.

    Args:
        model: An initialised Model() object.
        question: The base problem text.
        initial_candidates: A list of previous candidates, or None for the first step.
        task: 'math', 'rg', 'supergpqa', ...
        prompt_dictionary: dictionary containing all necessary prompt fragments.
        sequential: Number of refinement iterations (outer loop).
        parallel: Number of parallel generations per iteration.
        k: Number of sampled candidates to aggregate per prompt.
        reasoning: ReasoningEffort for GPT models.

    Returns:
        A list of final candidate solutions after `sequential` iterations.
    """

    # ---------- 1. Choose chat_template_fn ----------
    name = model.model_name.lower()

    if "gpt" in name:
        # Convert user string → ReasoningEffort enum
        if reasoning == "low":
            effort = ReasoningEffort.LOW
        elif reasoning == "high":
            effort = ReasoningEffort.HIGH
        else:
            effort = ReasoningEffort.MEDIUM

        chat_template_fn = partial(render_chat_template_gpt_non_streaming, reasoning=effort)
    else:
        chat_template_fn = render_chat_template

    # ---------- 2. Initialise candidates ----------
    current_candidates = initial_candidates

    if sequential <= 0:
        return current_candidates or []

    # ---------- 3. RSA sequential refinement ----------
    for _ in range(sequential):

        # Create parallel groups of size k using your new function
        candidate_groups = generate_candidate_groups(
            previous_candidates=current_candidates,
            parallel=parallel,
            k=k,
        )

        # Prepare prompts (string or token-IDs depending on model)
        prepared_prompts = []
        for group in candidate_groups:
            prepared_prompts.append(
                build_prompt(
                    tokenizer=model.tokenizer,
                    question=question,
                    candidate_answers=group,
                    task=task,
                    chat_template_fn=chat_template_fn,
                    prompt_dictionary=prompt_dictionary,
                )
            )

        # ---------- 4. Model.generate (batch mode) ----------

        responses = model.generate(prepared_prompts)

        # Normalize output into list[str]
        if isinstance(responses, str):
            responses = [responses]

        # These responses become the candidates for the next iteration
        current_candidates = list(responses)

    # ---------- 5. Return final set of candidates ----------
    return current_candidates


# use applyRSA_2 for EIDF runs to track metrics and per loop solutions

def applyRSA_2(
    model: Model,
    question: str,
    initial_candidates: Optional[List[str]],
    task: str,
    prompt_dictionary: dict,
    sequential: int,
    parallel: int,
    k: int,
    reasoning: str = "medium",
    problem_id: Optional[str] = None,
    loop_trace_dfs: Optional[List[pl.DataFrame]] = None,
) -> Tuple[List[str], List[List[str]]]:
    """
    RSA-style iterative refinement for a single problem.

    Args:
        model: An initialised Model() object.
        question: The base problem text.
        initial_candidates: A list of previous candidates, or None for the first step.
        task: 'math', 'rg', 'supergpqa', ...
        prompt_dictionary: dictionary containing all necessary prompt fragments.
        sequential: Number of refinement iterations (outer loop).
        parallel: Number of parallel generations per iteration.
        k: Number of sampled candidates to aggregate per prompt.
        problem_id: Optional[str] = None,
        loop_trace_dfs: Optional[List[pl.DataFrame]] = None
        reasoning: ReasoningEffort for GPT models.

    Returns:
        A list of final candidate solutions after `sequential` iterations.
    """

    # ---------- 1. Choose chat_template_fn ----------
    name = model.model_name.lower()

    if "gpt" in name:
        # Convert user string → ReasoningEffort enum
        if reasoning == "low":
            effort = ReasoningEffort.LOW
        elif reasoning == "high":
            effort = ReasoningEffort.HIGH
        else:
            effort = ReasoningEffort.MEDIUM

        chat_template_fn = partial(render_chat_template_gpt_non_streaming, reasoning=effort)
    else:
        chat_template_fn = render_chat_template

    # ---------- 2. Initialise candidates ----------
    current_candidates = initial_candidates

    if sequential <= 0:
        return current_candidates or [], []

    # NEW: store all candidates per loop
    all_loop_candidates: List[List[str]] = []

    # ---------- 3. RSA sequential refinement ----------
    for loop_idx in range(sequential):

        # Create parallel groups of size k using your new function
        candidate_groups = generate_candidate_groups(
            previous_candidates=current_candidates,
            parallel=parallel,
            k=k,
        )

        # Prepare prompts (string or token-IDs depending on model)
        prepared_prompts = []
        for group in candidate_groups:
            prepared_prompts.append(
                build_prompt(
                    tokenizer=model.tokenizer,
                    question=question,
                    candidate_answers=group,
                    task=task,
                    chat_template_fn=chat_template_fn,
                    prompt_dictionary=prompt_dictionary,
                )
            )

        # ---------- 4. Model.generate (batch mode) ----------

        responses = model.generate(prepared_prompts)

        # Normalize output into list[str]
        if isinstance(responses, str):
            responses = [responses]

        # These responses become the candidates for the next iteration
        current_candidates = list(responses)

        all_loop_candidates.append(current_candidates)


        # ---------- NEW: choose answer and log per-loop ----------
        try:
            chosen_answer = choose_final_answer(current_candidates)
        except Exception as e:
            logger.error(f"Error choosing final answer for id={problem_id}, loop={loop_idx}: {e}")
            chosen_answer = 0

        if (
            problem_id is not None
            and loop_trace_dfs is not None
            and 0 <= loop_idx < len(loop_trace_dfs)
        ):
            # Append a new row <id, answer> to the correct loop DataFrame
            new_row = pl.DataFrame({"id": [problem_id], "answer": [chosen_answer]})
            loop_trace_dfs[loop_idx] = loop_trace_dfs[loop_idx].vstack(new_row)

    # ---------- 5. Return final set of candidates ----------
    return current_candidates, all_loop_candidates




# global instance used by Kaggle's inference server
# model = Model()


# =========================
# AIMO3 predict() entrypoint
# =========================

# use _predict_full for aimo3

def _predict_full(
    id_: pl.Series,
    problem: pl.Series,
    model: Model,
    task: str,
    prompt_dictionary: dict,
    sequential: int,
    parallel: int,
    k: int,
    reasoning: str
):
    id_val = id_.item(0)
    problem_text = problem.item(0)

    final_candidates = applyRSA(
        model=model,
        question=problem_text,
        initial_candidates=None,
        task=task,
        prompt_dictionary=prompt_dictionary,
        sequential=sequential,
        parallel=parallel,
        k=k,
        reasoning=reasoning,
    )

    prediction_int = choose_final_answer(final_candidates)
    prediction_int = max(0, min(99999, int(prediction_int)))

    return pl.DataFrame({"id": id_val, "answer": prediction_int})


# use _predict_full_2 for EIDF runs

def _predict_full_2(
    id_: pl.Series,
    problem: pl.Series,
    model: Model,
    task: str,
    prompt_dictionary: dict,
    sequential: int,
    parallel: int,
    k: int,
    reasoning: str,
    loop_trace_dfs: List[pl.DataFrame],
):
    id_val = id_.item(0)
    problem_text = problem.item(0)

    final_candidates, all_loop_candidates = applyRSA_2(
        model=model,
        question=problem_text,
        initial_candidates=None,
        task=task,
        prompt_dictionary=prompt_dictionary,
        sequential=sequential,
        parallel=parallel,
        k=k,
        reasoning=reasoning,
        problem_id=id_val,
        loop_trace_dfs=loop_trace_dfs,
    )

    prediction_int = choose_final_answer(final_candidates)
    prediction_int = max(0, min(99999, int(prediction_int)))

    pred_df = pl.DataFrame({"id": id_val, "answer": prediction_int})

    # return both the final prediction DF and all loop candidates
    return pred_df, all_loop_candidates


# =========================
# Global model & config (Kaggle-style)
# =========================

_GLOBAL_MODEL: Optional[Model] = None

LLM_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
TP_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.84
ENFORCE_EAGER = False
DTYPE = "bfloat16"
SEED = 2025112899
MAX_MODEL_LEN: Optional[int] = None
MAX_NUM_SEQS: Optional[int] = None
MAX_NEW_TOKENS = 8192
TEMPERATURE = 1.0
DISABLE_LOG_STATS = True
REASONING = "medium"  # for GPT models

# RSA config
RSA_TASK = "math_problem"
LOOPS = 10
POPULATION = 16
K = 4

def _ensure_model_loaded() -> Model:
    """
    Lazy-load the global vLLM Model using the current global config.
    Mirrors the Kaggle AIMO3 notebook behavior.
    """
    global _GLOBAL_MODEL

    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL = Model(
            model_name=LLM_MODEL_PATH,
            tp_size=TP_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=ENFORCE_EAGER,
            dtype=DTYPE,
            seed=SEED,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            trust_remote_code=True,
            disable_log_stats=DISABLE_LOG_STATS,
        )
    return _GLOBAL_MODEL


def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """
    AIMO3-style predict entrypoint (Kaggle-like).

    - Uses global config variables.
    - Lazily loads the vLLM model on first call.
    - Delegates to _predict_full (no logging of per-loop answers).
    """
    # Make sure the global model exists
    model = _ensure_model_loaded()

    return _predict_full(
        id_=id_,
        problem=problem,
        model=model,
        task=RSA_TASK,
        prompt_dictionary=prompt_dictionary,
        sequential=LOOPS,
        parallel=POPULATION,
        k=K,
        reasoning=REASONING,
    )


# =========================
# main
# =========================

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default='Qwen/Qwen2.5-1.5B-Instruct')
    # ap.add_argument("--dataset_csv_path", default="./data/aime25/aimo3_format.csv")
    ap.add_argument("--dataset_name", default='aimo3_aime25')
    ap.add_argument("--output", default="./eval_aimo3")
    # ap.add_argument("--output", default="./eval")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--population", type=int, default=16)
    # ap.add_argument("--summarize-cot", action="store_true")
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.84)
    ap.add_argument('--max_model_len', type=int, default=None)
    ap.add_argument('--max_num_seqs', type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto","float16","bfloat16"])
    ap.add_argument("--seed", type=int, default=2025112899)
    # ap.add_argument("--resume", action='store_true', default=False)
    # ap.add_argument("--remove_checkpoint", action='store_true', default=False)
    # ap.add_argument("--self_verify", action='store_true', default=False)
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--disable_log_stats", default=True)

    args = ap.parse_args()


    # ------------------------------
    # Fill global config from CLI
    # ------------------------------

    global LLM_MODEL_PATH
    global TP_SIZE
    global GPU_MEMORY_UTILIZATION
    global ENFORCE_EAGER
    global DTYPE
    global SEED
    global MAX_MODEL_LEN
    global MAX_NUM_SEQS
    global MAX_NEW_TOKENS
    global TEMPERATURE
    global DISABLE_LOG_STATS
    global REASONING
    global RSA_TASK
    global LOOPS
    global POPULATION
    global K
    global _GLOBAL_MODEL
    

    LLM_MODEL_PATH = args.model
    TP_SIZE = args.tp_size
    GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization
    ENFORCE_EAGER = False  # or add a CLI flag if you want to control this
    DTYPE = args.dtype
    SEED = args.seed
    MAX_MODEL_LEN = args.max_model_len
    MAX_NUM_SEQS = args.max_num_seqs
    MAX_NEW_TOKENS = args.max_new_tokens
    TEMPERATURE = args.temperature
    DISABLE_LOG_STATS = args.disable_log_stats
    REASONING = args.reasoning

    RSA_TASK = "math_problem"
    LOOPS = args.loops
    POPULATION = args.population
    K = args.k

    # Ensure fresh lazy-load each run
    _GLOBAL_MODEL = None


    # ------------------------------
    # Load dataset
    # ------------------------------
    
    dataset_csv_path = f'./data/aimo3/aimo3_{args.dataset_name}.csv'
    problems_df, answers_df = load_aimo3_csv_polars(csv_path=dataset_csv_path)

    gt_map: Dict[str, int] = {
        row["id"]: int(row["answer"])
        for row in answers_df.iter_rows(named=True)
    }

    loop_trace_dfs: List[pl.DataFrame] = [
        pl.DataFrame(schema={"id": pl.Utf8, "answer": pl.Int64})
        for _ in range(args.loops)
    ]

    prediction_dfs: list[pl.DataFrame] = []

    # Per-question perf rows (time, tokens, tps)
    per_question_metrics_rows: List[Dict[str, Any]] = []

    # Per-loop global stats accumulators
    loop_stats: List[Dict[str, Any]] = [
        {
            "loop": loop_idx,
            "total_candidates": 0,
            "correct_candidates": 0,
            "total_questions": 0,
            "questions_pass_at_k": 0,
            "questions_majority_correct": 0,
            "tokens_per_response": [],  # list[int] of token counts
        }
        for loop_idx in range(args.loops)
    ]


    # ------------------------------
    # Main evaluation loop on EIDF
    #   - uses applyRSA_2 / _predict_full_2
    #   - logs per-loop answers into loop_trace_dfs
    #   - uses the same lazy-loaded global model as predict()
    # ------------------------------
    for row in problems_df.iter_rows(named=True):
        id_val = row["id"]
        problem_text = row["problem"]

        id_series = pl.Series("id", [id_val])
        problem_series = pl.Series("problem", [problem_text])

        model = _ensure_model_loaded()

        # Ground-truth answer for this question
        gt = int(gt_map[id_val])

        # Measure time for this question
        q_start = time.time()
        pred_df, all_loop_candidates = _predict_full_2(
            id_=id_series,
            problem=problem_series,
            model=model,
            task=RSA_TASK,
            prompt_dictionary=prompt_dictionary,
            sequential=LOOPS,
            parallel=POPULATION,
            k=K,
            reasoning=REASONING,
            loop_trace_dfs=loop_trace_dfs,
        )
        q_end = time.time()
        q_time = q_end - q_start

        prediction_dfs.append(pred_df)

        # ------------------------------
        # NEW: update per-loop metrics & per-question token counts
        # ------------------------------
        total_tokens_for_question = 0

        for loop_idx, candidates in enumerate(all_loop_candidates):
            if loop_idx >= len(loop_stats):
                # safety check (should not happen if lengths match)
                break

            stats = loop_stats[loop_idx]
            stats["total_questions"] += 1
            stats["total_candidates"] += len(candidates)

            # Candidate-level correctness and tokens
            correct_bools = []
            tokens_this_loop: List[int] = []

            for cand in candidates:
                # token count for this candidate (completion only, approx)
                # we do NOT add special tokens to be closer to vLLM behavior
                token_ids = model.tokenizer.encode(
                    cand,
                    add_special_tokens=False
                )
                tok_len = len(token_ids)
                tokens_this_loop.append(tok_len)
                stats["tokens_per_response"].append(tok_len)
                total_tokens_for_question += tok_len

                # correctness
                ans = extract_answer_int(cand)
                is_correct = (ans is not None and ans == gt)
                correct_bools.append(is_correct)

            # update candidate-level accuracy
            stats["correct_candidates"] += sum(correct_bools)

            # pass@k (any correct in the population)
            if any(correct_bools):
                stats["questions_pass_at_k"] += 1

            # majority-vote accuracy for this loop
            majority_ans = choose_final_answer(candidates)
            if majority_ans == gt:
                stats["questions_majority_correct"] += 1

        # Tokens-per-second for this question
        if q_time > 0:
            tps = total_tokens_for_question / q_time
        else:
            tps = 0.0

        per_question_metrics_rows.append(
            {
                "id": id_val,
                "time_sec": float(q_time),
                "total_tokens": int(total_tokens_for_question),
                "tokens_per_sec": float(tps),
            }
        )

    # Concatenate all 1-row DFs into a single predictions_df
    predictions_df = pl.concat(prediction_dfs)

    logger.info(predictions_df)


    # ---------------------------------------------------------
    # Compute accuracy by joining predictions with ground truth
    # ---------------------------------------------------------

    # Rename answer column in answers_df to avoid collision
    answers_aligned = answers_df.rename({"answer": "gt"})

    # Join on id
    eval_df = predictions_df.join(
        answers_aligned,
        on="id",
        how="left"
    )

    # boolean: 1 if prediction == ground truth, else 0
    eval_df = eval_df.with_columns([
        (pl.col("answer") == pl.col("gt")).alias("correct")
    ])

    # compute accuracy
    num_correct = eval_df["correct"].sum()
    total = eval_df.height
    accuracy = float(num_correct) / float(total) if total > 0 else 0.0

    logger.info(f"Accuracy: {accuracy:.6f}  ({num_correct}/{total})")
    logger.info(eval_df)


    # logger.info('exitting on line 1015 to prevent saving csvs...')
    # exit(1)

    # Save final per-problem predictions

    rsa_settings_folder_name = f"p-{args.population}_k-{args.k}_r-{args.reasoning}_s-{args.seed}"

    output_directory = os.path.join(
        args.output,
        args.model.split('/')[-1],
        args.dataset_name,
        rsa_settings_folder_name
        )
    logger.info(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    
    per_question_metrics_df = pl.DataFrame(per_question_metrics_rows)

    final_answer_csv_path = os.path.join(
        output_directory,
        f"final_answers_p{args.population}_k{args.k}_r{args.reasoning}_predictions.csv"
        )
    per_question_csv_path = os.path.join(
        output_directory,
        f"per_question_perf_p{args.population}_k{args.k}_r{args.reasoning}.csv",
    )


    eval_df.write_csv(final_answer_csv_path)
    logger.info(f"Saved predictions (with gt + correct) to {final_answer_csv_path}")

    per_question_metrics_df.write_csv(per_question_csv_path)
    logger.info(
        f"Saved per-question performance metrics to {per_question_csv_path}"
    )

    # ------------------------------
    # Save per-loop predictions with gt + correct
    # ------------------------------
    for loop_idx, df in enumerate(loop_trace_dfs, start=1):
        # df has columns: id, answer (for that loop)
        loop_eval_df = df.join(
            answers_aligned,  # has id, gt
            on="id",
            how="left",
        ).with_columns([
            (pl.col("answer") == pl.col("gt")).alias("correct")
        ])

        loop_csv_path = os.path.join(
            output_directory,
            f"loop_{loop_idx}_p{args.population}_k{args.k}_r{args.reasoning}_predictions.csv",
        )
        loop_eval_df.write_csv(loop_csv_path)
        logger.info(
            f"Saved loop {loop_idx} predictions (with gt + correct) to {loop_csv_path}"
        )

 # ------------------------------
    # Compute per-loop summary metrics and save to JSONL
    # ------------------------------
    loop_summaries: List[Dict[str, Any]] = []

    for stats in loop_stats:
        loop_idx = stats["loop"]
        total_candidates = stats["total_candidates"]
        total_questions = stats["total_questions"]
        correct_candidates = stats["correct_candidates"]
        questions_pass_at_k = stats["questions_pass_at_k"]
        questions_majority_correct = stats["questions_majority_correct"]
        tokens_list = stats["tokens_per_response"]

        # mean_acc_k: candidate-level accuracy
        mean_acc_k = (
            float(correct_candidates) / float(total_candidates)
            if total_candidates > 0
            else 0.0
        )

        # mean_pass_at_k: fraction of questions with any correct candidate
        mean_pass_at_k = (
            float(questions_pass_at_k) / float(total_questions)
            if total_questions > 0
            else 0.0
        )

        # mean_majority_acc: fraction of questions where majority vote is correct
        mean_majority_acc = (
            float(questions_majority_correct) / float(total_questions)
            if total_questions > 0
            else 0.0
        )

        # Token distribution stats per response
        if tokens_list:
            arr = np.array(tokens_list, dtype=np.float64)
            mean_tokens = float(arr.mean())
            min_tokens = float(arr.min())
            p25_tokens = float(np.percentile(arr, 25))
            median_tokens = float(np.percentile(arr, 50))
            p75_tokens = float(np.percentile(arr, 75))
            max_tokens = float(arr.max())
        else:
            mean_tokens = min_tokens = p25_tokens = median_tokens = p75_tokens = max_tokens = 0.0

        summary = {
            "loop": loop_idx,
            "mean_acc_k": mean_acc_k,
            "mean_pass_at_k": mean_pass_at_k,
            "mean_majority_acc": mean_majority_acc,
            "mean_tokens_per_response": mean_tokens,
            "min_tokens_per_response": min_tokens,
            "p25_tokens_per_response": p25_tokens,
            "median_tokens_per_response": median_tokens,
            "p75_tokens_per_response": p75_tokens,
            "max_tokens_per_response": max_tokens,
            "total_candidates": int(total_candidates),
            "total_questions": int(total_questions),
        }
        loop_summaries.append(summary)

    loop_metrics_jsonl_path = os.path.join(
        output_directory,
        f"loop_metrics_p{args.population}_k{args.k}_r{args.reasoning}.jsonl",
    )

    with open(loop_metrics_jsonl_path, "w", encoding="utf-8") as f:
        for summary in loop_summaries:
            f.write(json.dumps(summary) + "\n")

    logger.info(f"Saved per-loop metrics to {loop_metrics_jsonl_path}")

if __name__ == "__main__":
    # print('{}')
    main()