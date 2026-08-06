#!/usr/bin/env python3
"""Shared node-aggregate prefill-rate resolver.

One function, `resolve_prefill_rate`, decides how a run's prefill token rate is
published, and both sidecar producers (`compute_sparsity_metrics.py` and
`compute_cost.py`) call it — two independent formulas is how the same run once
shipped two prefill rates differing by exactly the batch size.

Every disposition is a pure function of the run's own recorded evidence:

  identity-bs1   The run pins prefill batch 1 (its batch regime, or a measured
                 average batch of 1): the per-request rate IS the node rate, so
                 `prefill_tokens_per_request / ttft` is aggregation-exact.
  trace-exact    A per-run exact-profile sidecar exists (written by
                 `compute_prefill_profile.py` from the run's own trace): the
                 nominal attempted prompt tokens over the exact physical
                 prefill-step elapsed time.
  hybrid-rung1   Concurrent run whose per-request prefill fits a single
                 scheduler pass: `tokens/request x mean prefill batch / mean
                 prefill pass latency`.
  hybrid-rung2   Concurrent long-context run where chunking may split a
                 request across passes, breaking the one-pass identity:
                 `(tokens/request / ttft) x mean prefill batch`.
  null           Anything else, with the missing evidence named. There is no
                 fallback to a dataset constant or to a harness-computed rate:
                 an unlabelled fallback is how three different quantities were
                 once published under one column.

Token bases: `nominal-attempted` is the run's recorded client-side prompt-token
count over the attempted cohort (cache-inclusive, refusals counted — the ruled
published basis). `configured-input-target` is used only where the recorded
count is absent and the run itself witnesses a fixed input target twice over:
its batch-regime name records the target AND its launch record carries exactly
one matching `--target-input-tokens` flag.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional


# A measured mean prefill batch this close to 1 is a single-stream run: the
# aggregation identity (per-request rate == node rate) holds to within 1%.
IDENTITY_BATCH_TOLERANCE = 1.01

# Single-pass-by-shape criterion for the estimate rungs. Engines split a
# request's prefill across several passes only when it exceeds the scheduler's
# per-pass token budget. The smallest budget any recorded engine configuration
# applies is 2048 tokens (vLLM's historical chunked-prefill default floor;
# SGLang's chunked_prefill_size default and every explicitly configured budget
# in the archive are larger), so a per-request prefill within 2048 tokens is
# admitted in one pass under EVERY configuration observed, while a longer one
# may be chunked under at least some. Deliberately the conservative floor, not
# the run's own configured budget: the rung-1 estimator was validated against
# exact traces under this split, and a generous per-run budget would move
# long-context runs onto the unvalidated side of that validation.
SINGLE_PASS_PREFILL_TOKEN_BUDGET = 2048

# Fixed-input batch regime: the directory name is run metadata recording the
# configured target, e.g. `batch-size-default_input1024_output1024`.
FIXED_INPUT_REGIME_RE = re.compile(r"^batch-size-default_input(\d+)_output(\d+)$")

TARGET_INPUT_FLAG_RE = re.compile(r"(?:^|\s)--target-input-tokens\s+(\d+)(?=\s|$)")

# Schema the resolver accepts from the exact-profile sidecar. Anything else is
# ignored (fail-closed), never partially decoded.
PREFILL_PROFILE_SCHEMA = "prefill-profile/1"


def _finite_pos(value) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value > 0
    )


def _null(reason: str) -> dict:
    return {
        "value": None, "basis": None, "method": None,
        "token_basis": None, "reason": reason,
    }


def _resolved(value: float, basis: str, method: str, token_basis: str) -> dict:
    return {
        "value": value, "basis": basis, "method": method,
        "token_basis": token_basis, "reason": None,
    }


def configured_input_target(batch_regime: str, launch_text: Optional[str]) -> Optional[int]:
    """The run's fixed input-token target, or None.

    Two independent run-native witnesses are required and must agree: the
    batch-regime directory name records the target, and the run's own launch
    record carries exactly one `--target-input-tokens` flag with the same
    value. Either alone is insufficient — a regime name without the flag may
    be a mislabelled directory, and a flag without the regime name would make
    the disposition depend on which file happens to survive.
    """
    match = FIXED_INPUT_REGIME_RE.fullmatch(batch_regime or "")
    if not match or not launch_text:
        return None
    target = int(match.group(1))
    flags = TARGET_INPUT_FLAG_RE.findall(launch_text)
    if flags != [str(target)]:
        return None
    return target


def valid_prefill_profile(profile) -> bool:
    """True when `profile` is a usable exact-profile sidecar.

    Fail-closed: an unknown schema, or a non-positive numerator/denominator,
    rejects the whole profile rather than salvaging fields from it.
    """
    return (
        isinstance(profile, dict)
        and profile.get("schema") == PREFILL_PROFILE_SCHEMA
        and _finite_pos(profile.get("prefill_step_elapsed_s"))
        and _finite_pos(profile.get("prefill_nominal_attempted_tokens"))
        and _finite_pos(profile.get("prefill_forwarded_tokens"))
    )


def resolve_prefill_rate(
    metrics: dict,
    profile: Optional[dict] = None,
    *,
    batch_regime: str = "",
    launch_text: Optional[str] = None,
) -> dict:
    """Resolve one run's node-aggregate prefill token rate.

    Returns `{value, basis, method, token_basis, reason}`. `basis` is
    `measured` (aggregation-exact) or `estimated` (Policy A structural
    hybrid); a null `value` carries the missing evidence in `reason` and
    nulls the other fields.
    """
    perf = (metrics or {}).get("performance") or {}
    batch_profile = (metrics or {}).get("batch_token_profile") or {}
    prompt = batch_profile.get("prefill_tokens_per_request")
    batch = batch_profile.get("prefill_avg_batch_size")
    ttft = perf.get("ttft")
    pass_s = perf.get("prefill_pass_latency_s")

    # 1. Identity: prefill batch pinned to 1, by regime or by measurement.
    #    The regime name wins over a recorded average above 1: batch-size-1
    #    pins concurrency server-side, so such a profile is an accounting
    #    artefact (the cost producer already discards it for the same reason).
    regime_bs1 = (batch_regime or "").startswith("batch-size-1")
    measured_bs1 = _finite_pos(batch) and batch <= IDENTITY_BATCH_TOLERANCE
    if regime_bs1 or measured_bs1:
        if not _finite_pos(prompt):
            return _null("no-token-evidence")
        if not _finite_pos(ttft):
            return _null("no-latency-evidence")
        return _resolved(
            float(prompt) / ttft, "measured", "identity-bs1", "nominal-attempted",
        )

    # 2. Exact: the run's own trace supported a complete physical prefill
    #    profile. Published on the nominal-attempted numerator (the ruled
    #    basis); the forwarded-token variant stays in the profile as evidence.
    if valid_prefill_profile(profile):
        return _resolved(
            profile["prefill_nominal_attempted_tokens"]
            / profile["prefill_step_elapsed_s"],
            "measured", "trace-exact", "nominal-attempted",
        )

    # 3. Estimated: Policy A structural hybrid over the summary fields.
    if _finite_pos(batch) and batch > IDENTITY_BATCH_TOLERANCE:
        if _finite_pos(prompt):
            tokens: float = float(prompt)
            token_basis = "nominal-attempted"
        else:
            target = configured_input_target(batch_regime, launch_text)
            if target is None:
                return _null("no-token-evidence")
            tokens = target
            token_basis = "configured-input-target"

        if tokens <= SINGLE_PASS_PREFILL_TOKEN_BUDGET:
            if not _finite_pos(pass_s):
                return _null("no-latency-evidence")
            return _resolved(
                tokens * float(batch) / float(pass_s),
                "estimated", "hybrid-rung1", token_basis,
            )
        if not _finite_pos(ttft):
            return _null("no-latency-evidence")
        return _resolved(
            (tokens / ttft) * float(batch),
            "estimated", "hybrid-rung2", token_basis,
        )

    # 4. Null: no usable prefill batch evidence.
    return _null("no-batch-evidence")


# ---------------------------------------------------------------------------
# Run-tree helpers shared by both sidecar producers.

def _swap_prefix(name: str, old: str, new: str) -> str:
    if name == f"{old}.json":
        return f"{new}.json"
    if name.startswith(f"{old}_"):
        return new + name[len(old):]
    return f"{new}_{name}"


def prefill_profile_path_for(metrics_path: Path) -> Path:
    return metrics_path.with_name(
        _swap_prefix(metrics_path.name, "metrics", "prefill_profile")
    )


def load_prefill_profile(metrics_path: Path) -> Optional[dict]:
    import json

    path = prefill_profile_path_for(metrics_path)
    if not path.is_file():
        return None
    try:
        profile = json.loads(path.read_text())
    except Exception:
        return None
    return profile if valid_prefill_profile(profile) else None


def launch_text_for(metrics_path: Path) -> Optional[str]:
    path = metrics_path.with_name("run.sh")
    if not path.is_file():
        return None
    try:
        return path.read_text()
    except Exception:
        return None


def batch_regime_for(metrics_path: Path) -> str:
    """The run's batch-regime directory name (`.../<regime>/<run>/metrics*.json`)."""
    return metrics_path.parent.parent.name


def resolve_for_metrics_path(metrics_path: Path, metrics: dict) -> dict:
    """Convenience wrapper: gather the run-local evidence and resolve."""
    return resolve_prefill_rate(
        metrics,
        load_prefill_profile(metrics_path),
        batch_regime=batch_regime_for(metrics_path),
        launch_text=launch_text_for(metrics_path),
    )
