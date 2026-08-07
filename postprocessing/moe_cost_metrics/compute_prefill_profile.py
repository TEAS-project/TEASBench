#!/usr/bin/env python3
"""Write per-run exact prefill profiles from the runs' own traces.

For every run that carries a per-request extend trace (`detailed_results*.jsonl`
beside its metrics file), reconstruct the physical prefill work: the sum of
tokens actually forwarded in prefill-carrying steps and the sum of those steps'
elapsed time. The result is a `prefill_profile*.json` sidecar beside the run,
which is the ONLY thing that makes the shared prefill-rate resolver
(`prefill_rate.py`) publish a `trace-exact` rate — presence of the sidecar is
the condition, and the sidecar itself carries the evidence it was built from.

A profile is written only when every gate passes; a run failing any gate gets
no sidecar, never a partial one. The gates:

  - exactly one trace file and a readable client output record beside it;
  - the client record is complete (contiguous 0-based indices);
  - the recording window is provable: at most a bounded, whole-request leading
    preamble is excluded, the retained request ids are complete, disjoint from
    the preamble, and count-match the client success cohort — and a preamble
    may be excluded at all only when every client attempt succeeded, since
    with any client failure a trimmed count-match cannot prove membership;
  - every accepted step carries decodable per-request token evidence
    (integer `extend_len`, a request id, one positive elapsed time) and each
    retained request completes exactly once (`is_last_chunk`);
  - one recorder rank per physical global pass: the run is TP-only
    (`tensor_parallel_size == num_gpus`, no DP/PP) and the accepted step
    indices are strictly increasing, so no rank wrote a pass twice;
  - the run's nominal attempted prompt-token total exists and bounds the
    forwarded sum from above (a cache hit can only shrink forwarded work).

Both numerator bases are recorded: `prefill_forwarded_tokens` (physical) and
`prefill_nominal_attempted_tokens` (the ruled published basis, copied from the
run's own metrics). Elapsed time counts every accepted prefill-carrying step at
its full latency, including steps that also carried decode work — the profile
counts such steps once and is never additive with decode time.

Runs without a trace are skipped and any existing profile is left alone (the
trace archive is not present in every checkout); runs WITH a trace that fails a
gate have any stale profile removed. Output is deterministic (sorted keys, no
timestamp), so regeneration over an unchanged tree is a no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Optional

if __package__:
    from .prefill_rate import prefill_profile_path_for
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prefill_rate import prefill_profile_path_for


PROFILE_SCHEMA = "prefill-profile/1"
RECORDER_SCHEMA = "per-request-extend-trace-v1"
# Historical recorder files predate an explicit clear/start/stop marker, so a
# bounded, whole-request leading preamble (server warm-up probes) may precede
# the client cohort. The bound keeps the search deterministic and refuses a
# window whose start cannot be pinned this tightly.
MAX_LEADING_PREAMBLE_STEPS = 4


class GateFailure(Exception):
    """A run whose trace cannot support an exact profile. Not an error."""


def _fail(reason: str):
    raise GateFailure(reason)


def _finite_pos(value) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value > 0
    )


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_digest(values: list) -> str:
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def profile_steps(prefills: list[tuple[int, dict]]) -> dict:
    """Token/time profile of a set of prefill-carrying step records.

    Same computation the frozen review evidence was built with: forwarded
    tokens are the sum of per-request `extend_len` over the steps, elapsed is
    the sum of each step's full recorded latency, and a request is complete
    when exactly one of its chunks is marked last.
    """
    ids: set[str] = set()
    completion_counts: dict[str, int] = {}
    forwarded = 0
    elapsed = 0.0
    indices: list[int] = []
    for pos, row in prefills:
        per_req = row.get("per_req_info")
        duration = row.get("ttft", row.get("latency"))
        if not (isinstance(per_req, list) and per_req):
            _fail("step-missing-per-request-evidence")
        if not (isinstance(duration, (int, float)) and math.isfinite(duration)
                and duration > 0):
            _fail("step-missing-elapsed")
        indices.append(int(row["index"]))
        elapsed += float(duration)
        for req in per_req:
            req_id = req.get("req_id")
            extend_len = req.get("extend_len")
            if req_id is None:
                _fail("chunk-missing-request-id")
            if not (isinstance(extend_len, int) and not isinstance(extend_len, bool)
                    and extend_len >= 0):
                _fail("chunk-missing-extend-len")
            req_id = str(req_id)
            ids.add(req_id)
            forwarded += extend_len
            if req.get("is_last_chunk") is True:
                completion_counts[req_id] = completion_counts.get(req_id, 0) + 1
    completed = {rid for rid, count in completion_counts.items() if count == 1}
    duplicates = sorted(rid for rid, count in completion_counts.items() if count != 1)
    incomplete = sorted(ids - completed)
    return {
        "ids": ids, "completed": completed, "duplicate_completions": duplicates,
        "incomplete": incomplete, "forwarded": forwarded, "elapsed": elapsed,
        "indices": indices,
    }


def select_client_prefills(
    trace_rows: list[dict], client_successes: int, client_failures: int,
) -> tuple[list[tuple[int, dict]], list[tuple[int, dict]]]:
    """Split the trace's prefill steps into (excluded preamble, client cohort).

    Only a bounded, whole-request leading preamble may be excluded, and the
    retained request set must exactly match the client success count, with no
    excluded request reappearing. Minimal trimming is chosen so a full file
    that already matches the cohort is never cut.

    The client record carries no server request ids, so a count match is the
    only membership evidence. An untrimmed window is provable on the count
    alone: a client-failed request whose prefill reached the server would
    inflate the count past the success cohort. A trimmed window is provable
    only when every client attempt succeeded — with any failure, a window
    that count-matches after trimming cannot be told apart from one that
    dropped a real success and kept the failed request's steps.
    """
    prefills = [
        (pos, row) for pos, row in enumerate(trace_rows)
        if row.get("forward_mode") == "prefill"
    ]
    if not prefills:
        _fail("no-prefill-steps")
    empty = {"ids": set(), "completed": set(), "duplicate_completions": [],
             "incomplete": []}
    max_trim = (
        min(MAX_LEADING_PREAMBLE_STEPS, len(prefills))
        if client_failures == 0 else 0
    )
    for trim in range(0, max_trim + 1):
        excluded, accepted = prefills[:trim], prefills[trim:]
        try:
            acc = profile_steps(accepted)
            exc = profile_steps(excluded) if excluded else dict(empty)
        except GateFailure:
            continue
        if (
            len(acc["ids"]) == client_successes
            and acc["ids"] == acc["completed"]
            and not acc["duplicate_completions"] and not acc["incomplete"]
            and exc["ids"] == exc["completed"]
            and not exc["duplicate_completions"] and not exc["incomplete"]
            and exc["ids"].isdisjoint(acc["ids"])
        ):
            return excluded, accepted
    _fail("no-exact-client-cohort-window")
    raise AssertionError  # unreachable


def count_mixed_steps(accepted: list[tuple[int, dict]], trace_rows: list[dict]) -> int:
    """Physical steps that carried prefill AND decode work.

    The vLLM recorder represents one mixed step as adjacent prefill and decode
    rows with the identical full step time; the prefill row is accepted once
    and its decode twin never re-enters the denominator.
    """
    mixed = 0
    for pos, row in accepted:
        if pos + 1 >= len(trace_rows):
            continue
        twin = trace_rows[pos + 1]
        if (
            twin.get("forward_mode") == "decode"
            and twin.get("index") == row.get("index") + 1
            and twin.get("tpot") == row.get("ttft")
        ):
            mixed += 1
    return mixed


def build_profile(metrics_path: Path) -> Optional[dict]:
    """Build one run's profile dict, or raise GateFailure."""
    run_dir = metrics_path.parent
    traces = sorted(run_dir.glob("detailed_results*.jsonl"))
    if not traces:
        return None  # no trace: not a gate failure, simply nothing to profile
    if len(traces) != 1:
        _fail("ambiguous-trace-files")
    trace_path = traces[0]
    output_path = run_dir / "output_data.jsonl"
    metadata_path = run_dir / "metadata.json"
    if not output_path.is_file():
        _fail("no-client-output-record")
    if not metadata_path.is_file():
        _fail("no-metadata")

    metadata = json.loads(metadata_path.read_text())
    env = metadata.get("system_environment") or {}
    hardware = metadata.get("hardware") or {}
    engine = env.get("inference_engine")
    if not engine:
        _fail("no-recorded-engine")
    # One recorder rank per physical global pass: provable only for TP-only
    # runs (the recorders gate on TP rank 0; under DP/PP several ranks hold
    # distinct passes and this reconstruction would be wrong, not just noisy).
    # Fail closed on ABSENT layout evidence too: a metadata file recording
    # neither the TP degree nor the GPU count proves nothing about the rank
    # layout, and None == None must not read as a proof.
    tp = env.get("tensor_parallel_size")
    num_gpus = hardware.get("num_gpus")
    def _pos_int(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if not (_pos_int(tp) and _pos_int(num_gpus)):
        _fail("rank-rule-unprovable-no-layout-evidence")
    if tp != num_gpus:
        _fail("rank-rule-unprovable-tp")
    if env.get("data_parallel_size") not in (None, 1):
        _fail("rank-rule-unprovable-dp")
    if env.get("pipeline_parallel_size") not in (None, 1):
        _fail("rank-rule-unprovable-pp")

    client_rows = read_jsonl(output_path)
    attempts = len(client_rows)
    if attempts <= 0:
        _fail("empty-client-record")
    successes = sum(row.get("success") is True for row in client_rows)
    client_indices = [row.get("index") for row in client_rows]
    if not all(isinstance(i, int) for i in client_indices) or sorted(
        client_indices
    ) != list(range(attempts)):
        _fail("client-record-incomplete")

    trace_rows = read_jsonl(trace_path)
    excluded, accepted = select_client_prefills(
        trace_rows, successes, attempts - successes)
    acc = profile_steps(accepted)
    exc = profile_steps(excluded) if excluded else {
        "ids": set(), "forwarded": 0, "elapsed": 0.0, "indices": []}
    if sorted(acc["indices"]) != acc["indices"] or len(set(acc["indices"])) != len(
        acc["indices"]
    ):
        _fail("pass-indices-not-single-writer")
    if not (acc["forwarded"] > 0 and acc["elapsed"] > 0):
        _fail("no-physical-prefill-work")

    metrics = json.loads(metrics_path.read_text())
    attempted_tokens = (metrics.get("batch_token_profile") or {}).get("prefill_tokens")
    if not _finite_pos(attempted_tokens):
        _fail("no-nominal-token-total")
    if not acc["forwarded"] <= attempted_tokens:
        _fail("forwarded-exceeds-nominal")

    return {
        "schema": PROFILE_SCHEMA,
        "recorder_schema": RECORDER_SCHEMA,
        "engine": engine,
        "engine_version": env.get("inference_engine_version"),
        "prefill_forwarded_tokens": acc["forwarded"],
        "prefill_step_elapsed_s": acc["elapsed"],
        "prefill_nominal_attempted_tokens": attempted_tokens,
        "prefill_physical_steps": len(accepted),
        "prefill_mixed_steps": count_mixed_steps(accepted, trace_rows),
        "excluded_leading_steps": len(excluded),
        "excluded_leading_forwarded_tokens": exc["forwarded"],
        "excluded_leading_elapsed_s": exc["elapsed"],
        "cohort": {
            "attempts": attempts,
            "successes": successes,
            "failures": attempts - successes,
            "trace_request_ids": len(acc["ids"]),
            "trace_request_ids_sha256": list_digest(sorted(acc["ids"])),
        },
        "rank_rule": {
            "tensor_parallel_size": env.get("tensor_parallel_size"),
            "data_parallel_size": env.get("data_parallel_size", 1) or 1,
            "pipeline_parallel_size": env.get("pipeline_parallel_size", 1) or 1,
        },
        "evidence": {
            "trace_file": trace_path.name,
            "trace_sha256": sha256(trace_path),
            "trace_size_bytes": trace_path.stat().st_size,
            "client_output_file": output_path.name,
            "client_output_sha256": sha256(output_path),
        },
    }


def find_metrics_files(root: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in ("metrics_*.json", "metrics.json"):
        for p in root.rglob(pat):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).parent,
                        help="Tree to scan for metrics files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"error: root {root} not found", file=sys.stderr)
        return 2

    written = unchanged = no_trace = gated = removed = 0
    for metrics_path in find_metrics_files(root):
        out_path = prefill_profile_path_for(metrics_path)
        try:
            profile = build_profile(metrics_path)
        except GateFailure as gate:
            gated += 1
            print(f"  [gate] {metrics_path.parent.relative_to(root)}: {gate}")
            # A trace is present but cannot support a profile: an existing
            # sidecar is stale evidence and must not keep resolving the run.
            if out_path.exists() and not args.dry_run:
                out_path.unlink()
                removed += 1
            continue
        except Exception as error:
            gated += 1
            print(f"  [gate] {metrics_path.parent.relative_to(root)}: "
                  f"unreadable run evidence ({error})")
            continue
        if profile is None:
            no_trace += 1
            continue
        payload = json.dumps(profile, indent=2, sort_keys=True) + "\n"
        if out_path.exists() and out_path.read_text() == payload:
            unchanged += 1
            continue
        if args.dry_run:
            print(f"  [dry-run] would write {out_path.relative_to(root.parent)}")
        else:
            out_path.write_text(payload)
        written += 1

    print(f"\nWrote {written} prefill profiles ({unchanged} unchanged, "
          f"{no_trace} runs without a trace, {gated} gated, {removed} stale removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
