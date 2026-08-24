"""Post-hoc auditor for a SWE-bench-on-k8s run directory: decides what to
retry, prepares the run for a resume pass, and gates whether a run is
complete enough to publish.

Why this exists (see plan doc + CONTRACT.md "The problem in one paragraph"):
`agentic-driver.sh` drives SWE-bench Lite from an EIDF login node because
EIDF grants no pod RBAC, so every per-task swe-rex sandbox is reached over
`kubectl port-forward`. When a tunnel drops mid-task the task dies, a row
still lands in `results.jsonl` with an empty patch, and the run exits 0
reporting success. In the evidence run that motivated this module, 55/100
tasks produced no patch; 8 of those died from a dropped sandbox tunnel
(`swerex/runtime/remote.py:205 run_in_session` -> `_request` ->
`aiohttp ... ServerDisconnectedError: Server disconnected`), and the other
47 were legitimate outcomes (the agent gave up, hit its cost/format/context
limit, or an outer `timeout` SIGKILLed it) that a retry must NOT touch --
giving those tasks a second sample for free biases accuracy upward.

This module is the single place that draws that line. It is kept as a
plain, stdlib-only, importable module (not buried in the driver shell
script) because the classification logic is JSON-heavy and needs to be
unit-testable without a cluster:

    import swebench_run_audit               # pipeline/k8s/lib is on PYTHONPATH
    python -m swebench_run_audit <command>   # same module, run as a script

Three of the four subcommands below are documented in full in CONTRACT.md
("Run-audit module" / "Retry classification rules"); `teas-output` is newer
and lives only here for now:

    retry-list RUN_DIR [--retry-timeouts 0|1]
        stdout: one task_id per line -- the tasks to re-run (nothing else;
        the driver reads this with `mapfile`). stderr: human summary.
        Always exits 0 -- an empty retry list is not an error, and a bug in
        this auditor must never be able to abort the driver's retry loop.

    prune RUN_DIR --tasks-file FILE --attempt N
        Archives results.jsonl, rewrites it to drop the retried tasks
        (deduplicated, last occurrence wins), and clears the per-task state
        that AgentCAP would otherwise double-count or wrongly resurrect on
        a resume pass. Exit 0 on success, non-zero on failure.

    report RUN_DIR --attempts N [--out completeness.json]
        Writes a completeness report and exits 1 if the run is not safe to
        publish yet.

    teas-output RUN_DIR [--model NAME] [--wall-time-s N] [--dataset NAME]
        Works around AgentCAP's write_teas_outputs() never succeeding on the
        --evaluator swebench-k8s path: SWEBenchK8sEvaluator writes report.json
        two levels under the wrong root (RUN_DIR/eval_k8s/<iid>/report.json),
        while AgentCAP's _swebench_reports() globs three levels under
        RUN_DIR/logs/run_evaluation/. Copies (never moves) each report into
        that expected layout, then calls write_teas_outputs(). Always exits
        0 -- including when the run is genuinely incomplete and
        write_teas_outputs() correctly refuses to export (that is not this
        command's problem to solve; `report`, above, is the publish gate).

All classification logic lives in plain functions that take a run_dir path
(load_results, classify_task, build_retry_report, ...) rather than inside
argparse callbacks, specifically so tests can call them directly against
fixture directories without going through the CLI.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

# --- Rule 3: tunnel-drop signatures grepped out of the sweagent logs -------
# These are the tracebacks/exceptions observed when a kubectl port-forward
# tunnel dies out from under a running SWE-agent step. Both log files are
# written with `write_text` -- truncated fresh per attempt
# (agent_cap/agents/strategies_sweagent.py:211-212) -- so they always
# describe only the most recent attempt, never a stale earlier one.
_LOG_SIGNATURES = (
    "ServerDisconnectedError",
    "Server disconnected",
    "Cannot connect to host 127.0.0.1:",
    "SessionExistsError",
)

# Rule 2: AgentCAP wraps any exception from provider.acquire() in exactly
# this prefix (agent_cap/agents/strategies_sweagent.py:130-137), covering
# "sandbox job create failed", "sandbox pod not Running after ...", and
# "swerex not alive after ...".
_SIDECAR_ERROR_PREFIX = "k8s sidecar failed:"

# Outer `timeout` SIGKILL exit code (rule 4).
_TIMEOUT_RC = 124

REASON_LABELS = {
    "complete": "has a patch",
    "pf_drop_running": "port-forward dropped mid-task (drop journal)",
    "sidecar_error": "k8s sidecar failed (provider.acquire raised)",
    "log_signature": "tunnel-drop signature in sweagent stdout/stderr log",
    "timeout_first_retry": "outer timeout (sweagent_rc=124), first retry",
    "timeout_retry_exhausted": "outer timeout (sweagent_rc=124), retry already spent",
    "timeout_not_retried": "outer timeout (sweagent_rc=124), --retry-timeouts 0",
    "no_evidence": "no infra evidence (agent gave up / exit_cost / exit_format / "
                    "exit_context / rc=0 empty patch)",
}


# ---------------------------------------------------------------------------
# Low-level JSONL / archive readers
# ---------------------------------------------------------------------------

def read_jsonl(path):
    """Yield parsed dicts from a JSONL file, tolerating a missing file and
    skipping blank/unparseable lines.

    AgentCAP's own resume loader (`_load_resume`, agent_cap/agents/cli.py
    :735-749) treats an unparseable line as skippable rather than fatal --
    a run directory whose last write was killed mid-line can have a torn
    trailing row, and refusing to read the file at all would be worse than
    dropping the one broken row. Every reader in this module goes through
    here so that guarantee holds everywhere, not just in `prune`.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def load_results(run_dir):
    """{task_id: row} for the *current* results.jsonl, last occurrence wins.

    Mirrors the write-back in cli.py:636-661, which patches only the last
    row per task_id when it re-writes results.jsonl after finalize() --
    if results.jsonl somehow already has duplicate rows (the exact bug
    `prune` exists to prevent), this keeps the same notion of "current"
    AgentCAP itself uses, rather than e.g. the first occurrence.
    """
    out = {}
    for row in read_jsonl(Path(run_dir) / "results.jsonl"):
        tid = row.get("task_id")
        if isinstance(tid, str) and tid:
            out[tid] = row
    return out


def load_attempt_archives(run_dir):
    """Sorted [(attempt_no, {task_id: row}), ...] for results.attempt-N.jsonl.

    These archives are `prune`'s own doing (see below) -- a fresh run has
    none, which is fine, they're only consulted for the "already retried
    this timeout once" check (rule 4) and the regression check in `report`.
    """
    run_dir = Path(run_dir)
    archives = []
    for p in sorted(run_dir.glob("results.attempt-*.jsonl")):
        try:
            n = int(p.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        rows = {}
        for row in read_jsonl(p):
            tid = row.get("task_id")
            if isinstance(tid, str) and tid:
                rows[tid] = row
        archives.append((n, rows))
    archives.sort(key=lambda t: t[0])
    return archives


def load_journal(run_dir):
    """Drop-journal events, or [] if the journal is absent.

    `$TEASBENCH_PF_EVENTS` (see CONTRACT.md) is new -- runs from before it
    existed, including the evidence run this module was built and verified
    against, have no `portforward-events.jsonl` at all. That must degrade
    to "no evidence from the journal", not an error: rule 3 (the log-
    signature grep) has to stand on its own for those runs.
    """
    return list(read_jsonl(Path(run_dir) / "portforward-events.jsonl"))


ATTEMPT_MARKER_EVENT = "attempt_start"


def _numeric(value):
    """value as a float if it is a real number, else None (bool is not)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def journal_attempt_window(events):
    """`ts` of the last attempt_start marker, or None if there is none.

    agentic-driver.sh stamps one marker into the journal at the top of every
    attempt, so this module can scope drop evidence to the attempt whose
    verdict is actually being decided. Runs recorded before the marker
    existed have none; those must keep the original whole-run behaviour
    rather than be silently re-judged, because their verdicts are already
    published.
    """
    start = None
    for ev in events:
        if ev.get("event") != ATTEMPT_MARKER_EVENT:
            continue
        ts = _numeric(ev.get("ts"))
        if ts is not None:
            start = ts
    return start


def journal_running_drop_labels(run_dir):
    """task_ids (journal `label`) with >=1 pf_drop while phase == "running",
    counting only drops in the current attempt.

    Phase is load-bearing (CONTRACT.md "Drop journal format"):
    `_PortForwardSandbox.start()` already restarts the tunnel while the
    sandbox pod is still pip-installing swe-rex, and that is normal
    startup churn, not evidence of anything -- it must be phase
    "startup" and must NOT make a task eligible for retry. Only a drop the
    babysitter itself observed, after `/is_alive` first succeeded (phase
    "running"), is real infrastructure evidence.

    The attempt window is load-bearing for the same reason. The journal
    spans the whole run, so without one a single recovered drop in attempt 1
    keeps its task in this set for every later attempt: rule 1 in
    classify_task() fires ahead of the `no_evidence` fallback, so the task
    stays infrastructure-incomplete however cleanly it re-runs. That both
    masks the genuine-failure verdict such a task has earned and stalls the
    driver's retry loop on tasks that can never clear -- the loop stops on
    "no progress" with the run still gated as incomplete.

    A drop event carrying no usable `ts` cannot be placed in any window, so
    it is ignored once a window exists. Ignoring rather than including it is
    deliberate: counting unplaceable evidence would reinstate exactly the
    stale-evidence bug this window exists to close. Every event the
    port-forward provider writes carries a ts.
    """
    events = load_journal(run_dir)
    since = journal_attempt_window(events)
    labels = set()
    for ev in events:
        if ev.get("event") != "pf_drop" or ev.get("phase") != "running":
            continue
        if since is not None:
            ts = _numeric(ev.get("ts"))
            if ts is None or ts < since:
                continue
        label = ev.get("label")
        if isinstance(label, str):
            labels.add(label)
    return labels


def load_task_ids_file(path):
    """One task_id per line from a --tasks-file, order-preserving, deduped."""
    ids = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        ids.append(line)
    return ids


# ---------------------------------------------------------------------------
# Classification (CONTRACT.md "Retry classification rules")
# ---------------------------------------------------------------------------

def is_incomplete(row):
    """A task is incomplete iff it produced no patch text -- exactly the
    condition under which AgentCAP's own evaluator refuses to buffer it for
    grading at all (`evaluate()` in evaluators_swebench.py: `if not
    output_text.strip(): return EvalResult(passed=False, ...)` without
    touching `self._buffer`). That, not `errors`, is the ground truth for
    "did this task contribute a prediction": a row can carry `errors` and
    still have a patch (a warning after a successful submit), and a row
    with `errors == []` and `sweagent_rc == 0` can still be an empty-patch
    "agent finished, submitted nothing" outcome.
    """
    return not (row.get("output_text") or "").strip()


def has_sidecar_error(row):
    """Rule 2: provider.acquire() raised and AgentCAP recorded it."""
    errors = row.get("errors") or []
    return (bool(errors) and isinstance(errors[0], str)
            and errors[0].startswith(_SIDECAR_ERROR_PREFIX))


def has_log_signature(run_dir, task_id):
    """Rule 3: grep task_<id>/sweagent_std{out,err}.log for a tunnel-drop
    traceback. Missing/unreadable log files are not evidence either way."""
    task_dir = Path(run_dir) / f"task_{task_id}"
    for name in ("sweagent_stdout.log", "sweagent_stderr.log"):
        p = task_dir / name
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if any(sig in text for sig in _LOG_SIGNATURES):
            return True
    return False


def prior_timeout_attempts(run_dir, task_id):
    """How many results.attempt-*.jsonl archives already show sweagent_rc
    == 124 for this task -- i.e. how many times its one rule-4 retry has
    already been spent.

    Rule 4 is retry-once-only: under a fixed per-instance call/step budget
    a task that timed out will time out again deterministically, so a
    second retry just burns cluster time for nothing. Once `prune` has
    rewritten results.jsonl to drop the row, the attempt archives are the
    only surviving record that "this task already got its retry".
    """
    count = 0
    for _, rows in load_attempt_archives(run_dir):
        row = rows.get(task_id)
        if row is not None and (row.get("extras") or {}).get("sweagent_rc") == _TIMEOUT_RC:
            count += 1
    return count


def classify_task(run_dir, task_id, row, retry_timeouts, running_drop_labels):
    """Classify one task's current row. Returns (should_retry, reason).

    `reason` is a short, stable tag (see REASON_LABELS) shared by the
    retry-list summary and the report's per-task classification -- keep
    the two in sync if a rule is ever added here.
    """
    if not is_incomplete(row):
        return False, "complete"

    # Rule 1: journal evidence of a drop seen after the tunnel came up.
    if task_id in running_drop_labels:
        return True, "pf_drop_running"

    # Rule 2: provider.acquire() itself raised.
    if has_sidecar_error(row):
        return True, "sidecar_error"

    # Rule 3: a tunnel-drop signature in the agent's own log output.
    if has_log_signature(run_dir, task_id):
        return True, "log_signature"

    # Rule 4: outer `timeout` SIGKILL, gated by --retry-timeouts and capped
    # at one retry per task (see prior_timeout_attempts).
    sweagent_rc = (row.get("extras") or {}).get("sweagent_rc")
    if sweagent_rc == _TIMEOUT_RC:
        if not retry_timeouts:
            return False, "timeout_not_retried"
        if prior_timeout_attempts(run_dir, task_id) >= 1:
            return False, "timeout_retry_exhausted"
        return True, "timeout_first_retry"

    # No positive evidence at all: SWE-agent's own exit_cost / exit_format /
    # exit_context exits, or sweagent_rc == 0 with an empty patch (the agent
    # finished and submitted nothing). CONTRACT.md is explicit these must
    # never be retried -- doing so gives some tasks a second sample and
    # biases accuracy upward relative to tasks that only ran once.
    return False, "no_evidence"


def build_retry_report(run_dir, retry_timeouts):
    """{task_id: (should_retry, reason)} for every task in results.jsonl,
    plus whether the drop journal exists at all (informational)."""
    run_dir = Path(run_dir)
    rows = load_results(run_dir)
    running_drop_labels = journal_running_drop_labels(run_dir)
    classifications = {
        tid: classify_task(run_dir, tid, row, retry_timeouts, running_drop_labels)
        for tid, row in rows.items()
    }
    journal_present = (run_dir / "portforward-events.jsonl").exists()
    return classifications, journal_present


# ---------------------------------------------------------------------------
# retry-list
# ---------------------------------------------------------------------------

def cmd_retry_list(args):
    run_dir = Path(args.run_dir)
    retry_timeouts = bool(args.retry_timeouts)
    try:
        classifications, journal_present = build_retry_report(run_dir, retry_timeouts)

        retry_ids = sorted(tid for tid, (should, _reason) in classifications.items() if should)
        for tid in retry_ids:
            print(tid)

        counts = Counter(reason for _should, reason in classifications.values())
        total = len(classifications)
        complete = counts.get("complete", 0)
        incomplete = total - complete
        not_retried = incomplete - len(retry_ids)

        print(f"swebench_run_audit retry-list: {run_dir}", file=sys.stderr)
        print(f"  drop journal: {'present' if journal_present else 'absent (older run, or journalling off)'}",
              file=sys.stderr)
        print(f"  --retry-timeouts {int(retry_timeouts)}", file=sys.stderr)
        print(f"  {total} task(s): {complete} complete, {incomplete} incomplete", file=sys.stderr)
        print(f"  retrying {len(retry_ids)}:", file=sys.stderr)
        for reason in ("pf_drop_running", "sidecar_error", "log_signature", "timeout_first_retry"):
            n = counts.get(reason, 0)
            if n:
                print(f"    {n:3d}  {REASON_LABELS[reason]}", file=sys.stderr)
        if not_retried:
            print(f"  NOT retrying {not_retried} (deliberate):", file=sys.stderr)
            for reason in ("no_evidence", "timeout_retry_exhausted", "timeout_not_retried"):
                n = counts.get(reason, 0)
                if n:
                    print(f"    {n:3d}  {REASON_LABELS[reason]}", file=sys.stderr)
        return 0
    except Exception as exc:
        # CONTRACT.md: retry-list always exits 0 -- a bug in this auditor
        # must never be able to abort the driver's retry loop. Fail safe to
        # "nothing to retry" (stdout already empty at this point) rather
        # than crash the caller.
        print(f"swebench_run_audit retry-list: ERROR: {exc}", file=sys.stderr)
        return 0


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

def prune_run(run_dir, tasks_file, attempt):
    """Archive results.jsonl, rewrite it without the retried tasks
    (deduplicated, last occurrence wins), and clear the per-task state a
    retry must not inherit. Returns a summary dict. Raises on failure.

    See CONTRACT.md "Facts about AgentCAP" for exactly why each step below
    is required:
      - --resume skips any task_id already present in results.jsonl,
        errors or not (cli.py:503-506, _load_resume at :735-749) -- so the
        retried tasks MUST be removed from results.jsonl or they will
        never be re-run at all.
      - cli.py:636-661 patches only the LAST occurrence of a task_id when
        it writes results.jsonl back after finalize() -- a duplicate row
        silently makes AgentCAP's own denominator 101-for-100 and skews
        metrics.json. Dedup here is not optional.
      - task_<id>/stream_stats.jsonl is opened in append mode
        (agent_cap/sweagent_streaming/__init__.py:320) and summed
        whole-file (strategies_sweagent.py:234-252) -- left in place, a
        retried task would report attempt-1 + attempt-2 tokens as one
        number.
      - task_<id>/sweagent_traj/ is globbed by mtime for the first *.traj
        with a non-empty info.submission (strategies_sweagent.py:216-226)
        -- left in place, a retry that itself fails to produce a patch can
        silently inherit the earlier attempt's patch instead of reporting
        empty.
    """
    run_dir = Path(run_dir)
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"no results.jsonl in {run_dir}")

    task_ids = load_task_ids_file(tasks_file)
    archive_path = run_dir / f"results.attempt-{attempt}.jsonl"

    archived = False
    if archive_path.exists():
        # Idempotency: if a previous prune for this exact --attempt crashed
        # after archiving but before finishing the rewrite/cleanup below, a
        # re-run must be safe. "Never overwrite an existing archive" (the
        # contract) and "be idempotent" only both hold if we skip the copy
        # here (the existing attempt-N.jsonl is the only surviving record
        # of the PRE-retry state; overwriting it with an already-pruned
        # file would corrupt `report`'s resolved-regression check) and then
        # let the rewrite/cleanup below run anyway -- both are naturally
        # idempotent (pruning an already-pruned file removes nothing more;
        # deleting an already-deleted stream_stats.jsonl is a no-op).
        pass
    else:
        shutil.copy2(results_path, archive_path)
        archived = True

    rows_before = 0
    unparseable = 0
    keep = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows_before += 1
            try:
                row = json.loads(line)
            except Exception:
                # Torn trailing line from a killed process. AgentCAP's own
                # _load_resume (cli.py:735-749) skips these rather than
                # aborting -- prune runs on exactly the kind of run that
                # just had a process killed mid-write, so it must be at
                # least as forgiving.
                unparseable += 1
                continue
            tid = row.get("task_id") if isinstance(row, dict) else None
            if not isinstance(tid, str) or not tid:
                unparseable += 1
                continue
            keep[tid] = row  # last occurrence wins (matches cli.py:636-661)

    duplicates_dropped = rows_before - unparseable - len(keep)

    rows_pruned = 0
    for tid in task_ids:
        if tid in keep:
            del keep[tid]
            rows_pruned += 1

    # Write via temp file + atomic rename so a process killed mid-write
    # (the same failure mode this whole module exists to recover from)
    # cannot leave results.jsonl half-written or truncated.
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(run_dir), prefix=".results.jsonl.tmp-")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for row in keep.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, results_path)  # atomic: same filesystem as run_dir
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    stream_stats_removed = []
    traj_removed = []
    missing_task_dirs = []
    for tid in task_ids:
        task_dir = run_dir / f"task_{tid}"
        if not task_dir.is_dir():
            # Defensive, not exceptional: a task can be in the retry set
            # because of journal/sidecar evidence recorded before its
            # task_<id> dir was ever created (e.g. the sandbox never came
            # up at all), so "missing" is an expected case, not a bug.
            missing_task_dirs.append(tid)
            continue
        stream_stats = task_dir / "stream_stats.jsonl"
        if stream_stats.exists():
            stream_stats.unlink()
            stream_stats_removed.append(tid)
        traj_dir = task_dir / "sweagent_traj"
        if traj_dir.exists():
            shutil.rmtree(traj_dir)
            traj_removed.append(tid)

    return {
        "archive": str(archive_path),
        "archived": archived,
        "rows_before": rows_before,
        "unparseable_dropped": unparseable,
        "duplicates_dropped": duplicates_dropped,
        "rows_pruned": rows_pruned,
        "rows_after": len(keep),
        "stream_stats_removed": stream_stats_removed,
        "traj_removed": traj_removed,
        "missing_task_dirs": missing_task_dirs,
    }


def cmd_prune(args):
    try:
        summary = prune_run(args.run_dir, args.tasks_file, args.attempt)
    except Exception as exc:
        print(f"swebench_run_audit prune: ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"swebench_run_audit prune: {args.run_dir} (attempt {args.attempt})")
    print(f"  archive: {summary['archive']}"
          + ("" if summary["archived"] else "  (already existed, not overwritten)"))
    print(f"  results.jsonl: {summary['rows_before']} row(s) read"
          + (f", {summary['unparseable_dropped']} unparseable dropped" if summary["unparseable_dropped"] else "")
          + (f", {summary['duplicates_dropped']} duplicate(s) collapsed" if summary["duplicates_dropped"] else "")
          + f", {summary['rows_pruned']} pruned -> {summary['rows_after']} row(s) kept")
    print(f"  stream_stats.jsonl removed for {len(summary['stream_stats_removed'])} task(s)")
    print(f"  sweagent_traj/ removed for {len(summary['traj_removed'])} task(s)")
    if summary["missing_task_dirs"]:
        print(f"  task dir already absent for {len(summary['missing_task_dirs'])} task(s) (ok): "
              + ", ".join(summary["missing_task_dirs"]))
    return 0


# ---------------------------------------------------------------------------
# stack-health
# ---------------------------------------------------------------------------
#
# What the engine did to the agent's ability to act, as opposed to what the
# agent did with it. A serving stack that mangles tool calls produces the same
# empty patch as a model that could not solve the task, and the results alone
# cannot tell those apart -- which matters, because TEASBench deliberately does
# NOT patch engine defects (see the revert of the sglang gpt-oss parser fix):
# an engine that degrades a model's realised capability is part of what the
# benchmark measures. Measuring it is the whole point; hiding it is not.
#
# Sampled per attempt, before `prune`, because prune deletes
# task_<id>/sweagent_traj/ and task_<id>/stream_stats.jsonl for every task it
# retries. A single pass at the end of a run would only ever see each task's
# final attempt.

STACK_HEALTH_FILE = "stack-health.json"
STACK_HEALTH_SCHEMA = 1

# SWE-agent logs one of these per response it could not parse a tool call from
# (`Requerying model after FunctionCallingFormatError`). Engine-agnostic: it
# says the harness got no usable tool call, not why.
_FORMAT_ERROR_MARKER = "FunctionCallingFormatError"

# Tool-call payload sitting in the reasoning channel instead of tool_calls --
# the harmony-format failure the sglang gpt-oss parser produced. Matched
# against the reasoning body only.
_STRANDED_HINTS = ('{"command"', '{"path"', "functions.", "<|constrain|>",
                   "<|channel|>", '{"file_text"', '{"old_str"')

_MESSAGE_RE = re.compile(r"Message\((?P<body>.{0,4000}?)\)\)\]", re.S)
_REASONING_RE = re.compile(
    r"reasoning_content=(?P<q>['\"])(?P<body>.{0,4000}?)(?<!\\)(?P=q)", re.S)


def unwrap_log(text):
    """Undo the fixed-column hard wrap SWE-agent's console logger applies.

    The logger pads every line to a fixed width and continues on the next line
    with a deep indent, breaking Python reprs mid-token
    (`accepted_pr` / `ediction_tokens=None`). Rejoining therefore strips the
    trailing pad and concatenates with no separator.

    This is lossy at wraps that fall on a real space -- "failing test" comes
    back as "failingtest" -- which is why callers only ever test the result for
    the presence of punctuation-dense markers, never for prose.
    """
    out = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if out and line.startswith("            ") and not stripped.lstrip().startswith(("\U0001f920", "\U0001f916")):
            out[-1] += stripped.lstrip()
        else:
            out.append(stripped)
    return "\n".join(out)


def scrape_stranded_tool_calls(log_text):
    """(stranded, examined) messages that carried a tool call nowhere useful.

    Best-effort by construction, and the caller must treat it as such: the
    rejected responses exist nowhere structured. SWE-agent drops them from
    both `history` and `trajectory`, so the only surviving copy is the
    litellm response repr in the debug log. That repr is SWE-agent's internal
    formatting with no stability guarantee, hence (None, 0) rather than a
    confident zero when nothing parses.
    """
    if not log_text or _FORMAT_ERROR_MARKER not in log_text:
        return 0, 0
    text = unwrap_log(log_text)
    stranded = 0
    examined = 0
    for msg in _MESSAGE_RE.finditer(text):
        body = msg.group("body")
        if "tool_calls=None" not in body and "tool_calls=[]" not in body:
            continue
        # Examined counts every tool-call-less message we could read, not just
        # the ones carrying a reasoning channel. An engine that populates no
        # reasoning_content at all (vLLM does not) strands nothing by
        # definition -- scoring that "undetermined" would report a blind spot
        # where there is actually a clean negative.
        examined += 1
        found = _REASONING_RE.search(body)
        if found and any(hint in found.group("body") for hint in _STRANDED_HINTS):
            stranded += 1
    if examined == 0:
        # The log says responses were rejected, but nothing in it parsed as a
        # message. Report "could not tell", never "none happened".
        return None, 0
    return stranded, examined


def _traj_files(task_dir):
    return sorted(Path(task_dir).glob("sweagent_traj/*/*.traj"))


def sample_task_health(task_dir):
    """Per-task counters for one attempt. Never raises: a task directory in
    any state at all must not be able to abort the driver's loop."""
    task_dir = Path(task_dir)
    out = {"responses": 0, "no_tool_call": 0, "empty_action_steps": 0,
           "hit_format_limit": False, "stranded": 0, "stranded_examined": 0,
           "stranded_undetermined": False}

    stats = task_dir / "stream_stats.jsonl"
    if stats.exists():
        out["responses"] = sum(1 for _ in read_jsonl(stats))

    log = task_dir / "sweagent_stdout.log"
    if log.exists():
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        out["no_tool_call"] = text.count(_FORMAT_ERROR_MARKER)
        stranded, examined = scrape_stranded_tool_calls(text)
        if stranded is None:
            out["stranded_undetermined"] = True
        else:
            out["stranded"] = stranded
            out["stranded_examined"] = examined

    for traj in _traj_files(task_dir):
        try:
            data = json.loads(traj.read_text(encoding="utf-8"))
        except Exception:
            continue
        exit_status = (data.get("info") or {}).get("exit_status") or ""
        # Both the bare limit and "submitted (exit_format)" -- the agent can
        # hit the format limit and still autosubmit a patch, so this is not
        # the same question as "did the task produce a patch".
        if "exit_format" in exit_status:
            out["hit_format_limit"] = True
        for step in (data.get("trajectory") or []):
            if not (step.get("action") or "").strip():
                out["empty_action_steps"] += 1
    return out


def sample_stack_health(run_dir, attempt):
    """Sample this attempt and merge it into RUN_DIR/stack-health.json."""
    run_dir = Path(run_dir)
    totals = {"tasks": 0, "responses": 0, "no_tool_call": 0,
              "empty_action_steps": 0, "tasks_hit_format_limit": 0,
              "stranded": 0, "stranded_examined": 0,
              "tasks_stranded_undetermined": 0}
    for task_dir in sorted(run_dir.glob("task_*")):
        if not task_dir.is_dir():
            continue
        one = sample_task_health(task_dir)
        totals["tasks"] += 1
        totals["responses"] += one["responses"]
        totals["no_tool_call"] += one["no_tool_call"]
        totals["empty_action_steps"] += one["empty_action_steps"]
        totals["tasks_hit_format_limit"] += 1 if one["hit_format_limit"] else 0
        totals["stranded"] += one["stranded"]
        totals["stranded_examined"] += one["stranded_examined"]
        totals["tasks_stranded_undetermined"] += 1 if one["stranded_undetermined"] else 0

    path = run_dir / STACK_HEALTH_FILE
    doc = _read_json(path) or {}
    if doc.get("schema") != STACK_HEALTH_SCHEMA:
        doc = {"schema": STACK_HEALTH_SCHEMA, "attempts": {}}
    doc.setdefault("attempts", {})
    # Re-sampling the same attempt overwrites it rather than double-counting;
    # the driver calls this once per attempt but must be safe to re-run.
    doc["attempts"][str(attempt)] = totals
    doc["totals"] = _sum_attempts(doc["attempts"])
    doc["stranded_detection"] = "best-effort-log-scrape"
    doc["notes"] = (
        "no_tool_call counts responses the harness could not parse a tool call "
        "from, for any engine. stranded is the subset whose payload was found "
        "in the reasoning channel instead; it is scraped from SWE-agent's "
        "debug log because rejected responses are kept nowhere structured, so "
        "treat it as indicative. tasks_stranded_undetermined counts tasks "
        "whose log had rejections that could not be parsed at all.")
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def _sum_attempts(attempts):
    total = {}
    for one in attempts.values():
        for key, value in one.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
    return total


def cmd_stack_health(args):
    try:
        doc = sample_stack_health(args.run_dir, args.attempt)
    except Exception as exc:
        # Advisory telemetry must never be what stops a run.
        print(f"swebench_run_audit stack-health: WARNING: {exc}", file=sys.stderr)
        return 0
    one = doc["attempts"][str(args.attempt)]
    responses = one["responses"]
    pct = (100.0 * one["no_tool_call"] / responses) if responses else 0.0
    print(f"swebench_run_audit stack-health: attempt {args.attempt}")
    print(f"  {one['responses']} response(s) over {one['tasks']} task(s)")
    print(f"  {one['no_tool_call']} with no parseable tool call ({pct:.1f}%)"
          + (f", {one['stranded']} stranded in the reasoning channel"
             if one["stranded_examined"] else ""))
    print(f"  {one['tasks_hit_format_limit']} task(s) hit the format limit")
    if one["tasks_stranded_undetermined"]:
        print(f"  {one['tasks_stranded_undetermined']} task(s): rejections present "
              "but the log did not parse (stranded count is a floor)")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _resolved_value(row):
    """True/False/None grade for a row, preferring eval_details.resolved
    (what the batch evaluator itself wrote) over eval_passed (a copy of
    the same value cli.py derives from it) -- fall back to eval_passed
    only when eval_details is missing/short."""
    details = row.get("eval_details") or {}
    if isinstance(details.get("resolved"), bool):
        return details["resolved"]
    if isinstance(row.get("eval_passed"), bool):
        return row["eval_passed"]
    return None


def find_resolved_regressions(run_dir):
    """Tasks whose grade went resolved:true -> false across successive
    results.attempt-*.jsonl snapshots, ending with the current
    results.jsonl.

    `finalize()` re-grades every buffered instance from scratch each
    attempt and writes predictions.json/eval_k8s_results.json wholesale,
    no merge (agent_cap/agents/evaluators_swebench.py:169, 236-251) -- so a
    retry pass that hits a transient exec-pod failure on some OTHER,
    already-resolved task can silently downgrade it even though nothing
    about that task changed. A report that only looked at the current file
    would never catch this; it must be surfaced explicitly.
    """
    run_dir = Path(run_dir)
    snapshots = [(f"attempt-{n}", rows) for n, rows in load_attempt_archives(run_dir)]
    snapshots.append(("current", load_results(run_dir)))

    last_seen = {}  # task_id -> (snapshot_label, resolved)
    regressions = []
    for label, rows in snapshots:
        for tid, row in rows.items():
            resolved = _resolved_value(row)
            if resolved is None:
                continue
            prev = last_seen.get(tid)
            if prev is not None and prev[1] is True and resolved is False:
                regressions.append({"task_id": tid, "from_attempt": prev[0], "to_attempt": label})
            last_seen[tid] = (label, resolved)
    return regressions


def tally_journal(run_dir):
    events = load_journal(run_dir)
    by_label, by_event, by_phase, by_reason = Counter(), Counter(), Counter(), Counter()
    for ev in events:
        by_label[ev.get("label", "?")] += 1
        by_event[ev.get("event", "?")] += 1
        by_phase[ev.get("phase", "?")] += 1
        if "reason" in ev:
            by_reason[ev.get("reason", "?")] += 1
    return {
        "total_events": len(events),
        "by_label": dict(by_label),
        "by_event": dict(by_event),
        "by_phase": dict(by_phase),
        "by_reason": dict(by_reason),
    }


def classify_for_report(run_dir, rows):
    """{task_id: {"status": ..., "reason": ...}} using retry_timeouts=True
    unconditionally -- `report` has no --retry-timeouts flag of its own
    (that is a driver-loop policy knob, CONTRACT.md's RETRY_TIMEOUTS), so
    it evaluates rule 4 the same way retry-list would with the flag on,
    then relies on prior_timeout_attempts to tell an unexhausted timeout
    (still fixable -> infra-incomplete) from an exhausted one (not fixable
    by retrying again -> genuinely-failed)."""
    run_dir = Path(run_dir)
    running_drop_labels = journal_running_drop_labels(run_dir)
    out = {}
    for tid, row in rows.items():
        should_retry, reason = classify_task(run_dir, tid, row, True, running_drop_labels)
        if not is_incomplete(row):
            status = "complete"
        elif should_retry:
            status = "infra-incomplete"
        else:
            status = "genuinely-failed"
        out[tid] = {"status": status, "reason": reason}
    return out


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_completeness_report(run_dir, attempts):
    """Build the full completeness report as a plain dict. No side effects
    -- callers decide whether/where to write it."""
    run_dir = Path(run_dir)
    rows = load_results(run_dir)
    classification = classify_for_report(run_dir, rows)
    archives = load_attempt_archives(run_dir)
    regressions = find_resolved_regressions(run_dir)
    journal_tally = tally_journal(run_dir)
    journal_present = (run_dir / "portforward-events.jsonl").exists()

    task_dirs = sorted(p.name[len("task_"):] for p in run_dir.glob("task_*") if p.is_dir())
    total_lines = sum(1 for _ in read_jsonl(run_dir / "results.jsonl"))
    patched = sum(1 for r in rows.values() if not is_incomplete(r))

    # predictions.json is a JSON *array*, one entry per task whose patch was
    # non-empty at evaluate() time (evaluators_swebench.py:66-81 only
    # buffers non-empty output_text) -- it is NEVER expected to reach
    # results.jsonl's total row count, only its *patched* row count.
    # eval_k8s_results.json is a dict keyed the same way. Comparing either
    # against the total row count would make this gate fail on every
    # ordinary SWE-bench run (most tasks don't produce a patch), which
    # would make it useless as a publish gate -- comparing against patched
    # rows is what actually catches "the batch evaluator silently dropped
    # some patched tasks", which is the failure this check exists for.
    predictions = _read_json(run_dir / "predictions.json")
    predictions_count = len(predictions) if isinstance(predictions, (list, dict)) else 0
    eval_k8s = _read_json(run_dir / "eval_k8s_results.json")
    eval_k8s_count = len(eval_k8s) if isinstance(eval_k8s, dict) else 0

    status_counts = Counter(v["status"] for v in classification.values())
    infra_incomplete = status_counts.get("infra-incomplete", 0)

    exit_reasons = []
    if infra_incomplete:
        exit_reasons.append(f"{infra_incomplete} task(s) still infrastructure-incomplete")
    if predictions_count < patched:
        exit_reasons.append(
            f"predictions.json has {predictions_count} entries, short of {patched} patched task(s)")
    if eval_k8s_count < patched:
        exit_reasons.append(
            f"eval_k8s_results.json has {eval_k8s_count} entries, short of {patched} patched task(s)")

    return {
        "run_dir": str(run_dir),
        "attempts_requested": attempts,
        "attempts_found": len(archives),
        "journal_present": journal_present,
        "counts": {
            "task_dirs": len(task_dirs),
            "results_jsonl_lines": total_lines,
            "results_jsonl_tasks": len(rows),
            "results_jsonl_patched": patched,
            "predictions_json_entries": predictions_count,
            "eval_k8s_results_entries": eval_k8s_count,
        },
        "classification": classification,
        "classification_summary": dict(status_counts),
        "journal_tally": journal_tally,
        "resolved_regressions": regressions,
        "complete": not exit_reasons,
        "exit_reasons": exit_reasons,
    }


def cmd_report(args):
    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "completeness.json"

    report = build_completeness_report(run_dir, args.attempts)

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    c = report["counts"]
    print(f"swebench_run_audit report: {run_dir}")
    print(f"  task dirs: {c['task_dirs']}   "
          f"results.jsonl: {c['results_jsonl_lines']} line(s), {c['results_jsonl_tasks']} distinct task(s)")
    if c["results_jsonl_lines"] != c["results_jsonl_tasks"]:
        print(f"  WARNING: {c['results_jsonl_lines'] - c['results_jsonl_tasks']} duplicate task_id "
              f"row(s) in results.jsonl -- run prune before publishing")
    print(f"  patched: {c['results_jsonl_patched']}   "
          f"predictions.json: {c['predictions_json_entries']}   "
          f"eval_k8s_results.json: {c['eval_k8s_results_entries']}")
    print(f"  classification: " + ", ".join(f"{k}={v}" for k, v in sorted(report["classification_summary"].items())))
    print(f"  attempts requested: {report['attempts_requested']}   "
          f"attempt archive(s) found: {report['attempts_found']}")
    if report["journal_present"]:
        print(f"  drop journal: {report['journal_tally']['total_events']} event(s)")
    else:
        print("  drop journal: absent (older run, or journalling off)")
    if report["resolved_regressions"]:
        print(f"  WARNING: {len(report['resolved_regressions'])} task(s) regressed resolved:true -> false:")
        for r in report["resolved_regressions"]:
            print(f"    {r['task_id']}: {r['from_attempt']} -> {r['to_attempt']}")
    print(f"  wrote {out_path}")

    if report["exit_reasons"]:
        print("NOT COMPLETE -- do not publish:")
        for reason in report["exit_reasons"]:
            print(f"  - {reason}")
        return 1
    print("COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# teas-output
# ---------------------------------------------------------------------------

# Fixed, not derived from run_dir or anything else: the relocated tree always
# lives under run_dir's OWN logs/ subtree, so there is no cross-run collision
# to guard against by making this fancier, and keeping it a plain constant
# means relocation needs no AgentCAP import at all (see relocate_swebench_
# reports below). "agentcap-unified" matches the model_name default that
# SWEBenchEvaluator (the docker path, in the same evaluators_swebench.py) and
# AgentCAP/scripts/run_swebench_modal_100.sh both already use for this exact
# path component -- swebench-k8s runs are launched with no `--judge
# model_name=...` override (pipeline/configs/config.yaml: "No judge: block --
# swebench-k8s grading uses the official SWE-bench test..."), so this is the
# value that keeps a relocated leaf looking like an ordinary swebench-harness
# leaf instead of inventing a new naming scheme. Deliberately distinct from
# the new subcommand's own --model flag, which only feeds write_teas_outputs'
# TEAS-metadata model_name -- a different, human-facing field.
_SWEBENCH_TEAS_RUN_ID = "swebench_k8s"
_SWEBENCH_TEAS_MODEL_DIR = "agentcap-unified"


def relocate_swebench_reports(run_dir):
    """Copy each RUN_DIR/eval_k8s/<iid>/report.json into the three-level
    RUN_DIR/logs/run_evaluation/<run_id>/<model>/<iid>/report.json layout
    that AgentCAP's _swebench_reports() globs (agent_cap/agents/teas_output.py
    -- `reports_root.glob("*/*/*/report.json")`).

    SWEBenchK8sEvaluator.finalize() (agent_cap/agents/evaluators_swebench.py)
    dumps get_eval_report()'s output verbatim to eval_k8s/<iid>/report.json --
    already the exact single-key {instance_id: {..., "resolved": bool, ...}}
    shape _swebench_reports() wants, just two levels deep under the wrong
    root. This is a pure relocation (copy2, never move/rename) -- eval_k8s/
    stays SWEBenchK8sEvaluator's own record, untouched.

    Idempotent: copy2 onto an unchanged source overwrites with identical
    bytes, so a second run neither duplicates nor corrupts anything -- no
    extra bookkeeping needed to detect "already relocated".

    Never raises: a missing or malformed report.json is recorded in
    "skipped" with a reason and left alone, not fatal to the run.

    Returns {"relocated": [iid, ...], "skipped": [(iid, reason), ...],
    "dest_root": str}.
    """
    run_dir = Path(run_dir)
    eval_k8s_dir = run_dir / "eval_k8s"
    dest_root = (run_dir / "logs" / "run_evaluation"
                 / _SWEBENCH_TEAS_RUN_ID / _SWEBENCH_TEAS_MODEL_DIR)
    relocated = []
    skipped = []
    if not eval_k8s_dir.is_dir():
        return {"relocated": relocated, "skipped": skipped, "dest_root": str(dest_root)}

    for task_dir in sorted(p for p in eval_k8s_dir.iterdir() if p.is_dir()):
        iid = task_dir.name
        report_path = task_dir / "report.json"
        if not report_path.is_file():
            # Expected, not a bug: a task whose patch failed to apply, or
            # whose acquire_exec() itself raised, never reaches
            # get_eval_report() and so never gets a report.json at all.
            skipped.append((iid, "no report.json under eval_k8s/ (task never reached grading)"))
            continue
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append((iid, f"unreadable/invalid report.json: {exc}"))
            continue
        # Same shape check _swebench_reports() itself does, so a bad file is
        # caught here with a clear per-task reason rather than surfacing as
        # a confusing TeasOutputError deep inside AgentCAP later.
        if not isinstance(payload, dict) or len(payload) != 1:
            skipped.append((iid, "report.json is not a single-key {instance_id: {...}} dict"))
            continue
        _, body = next(iter(payload.items()))
        if not isinstance(body, dict) or not isinstance(body.get("resolved"), bool):
            skipped.append((iid, "report.json missing boolean 'resolved' field"))
            continue

        dest_dir = dest_root / iid
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, dest_dir / "report.json")
        relocated.append(iid)

    return {"relocated": relocated, "skipped": skipped, "dest_root": str(dest_root)}


def run_teas_output(run_dir, model=None, wall_time_s=0.0, dataset="swe-bench-lite"):
    """Relocate eval_k8s/ reports, then call AgentCAP's write_teas_outputs.

    Never raises -- every failure mode here (AgentCAP not importable, no
    results.jsonl rows, write_teas_outputs itself raising -- including the
    EXPECTED "SWE-bench evaluation is incomplete" TeasOutputError on a run
    that genuinely lost tasks to infrastructure drops) is recorded in the
    returned summary's "warnings" list instead of propagated. This command
    has no business failing a driver over TEAS-output bookkeeping; the
    completeness gate (`report`, above) is the only thing scoped to gate
    publishing.

    Returns {"relocated": [...], "skipped_reports": [...], "dest_root": ...,
    "teas_path": str|None, "warnings": [...]}.
    """
    run_dir = Path(run_dir)
    summary = {
        "relocated": [],
        "skipped_reports": [],
        "dest_root": None,
        "teas_path": None,
        "warnings": [],
    }

    try:
        reloc = relocate_swebench_reports(run_dir)
    except Exception as exc:
        summary["warnings"].append(f"report relocation failed: {exc}")
        reloc = {"relocated": [], "skipped": [], "dest_root": None}
    summary["relocated"] = reloc["relocated"]
    summary["skipped_reports"] = reloc["skipped"]
    summary["dest_root"] = reloc["dest_root"]

    try:
        from agent_cap.agents.teas_output import write_teas_outputs
    except Exception as exc:
        # Not just ImportError: a missing transitive dependency of AgentCAP
        # can surface as some other exception type at import time, and all
        # of them mean the same thing here -- "can't write TEAS outputs".
        summary["warnings"].append(
            f"AgentCAP not importable, skipping TEAS output writing: {exc}")
        return summary

    rows = list(read_jsonl(run_dir / "results.jsonl"))
    if not rows:
        summary["warnings"].append("no rows in results.jsonl; nothing to write")
        return summary

    try:
        teas_path = write_teas_outputs(
            run_dir,
            rows,
            dataset=dataset,
            wall_time_s=float(wall_time_s or 0.0),
            model_name=model,
        )
    except Exception as exc:
        summary["warnings"].append(f"TEAS output writing failed: {exc}")
        return summary

    if teas_path is None:
        summary["warnings"].append("TEAS writer produced no output")
    else:
        summary["teas_path"] = str(teas_path)
    return summary


def cmd_teas_output(args):
    try:
        summary = run_teas_output(
            args.run_dir, model=args.model, wall_time_s=args.wall_time_s,
            dataset=args.dataset,
        )
    except Exception as exc:
        # This command always exits 0 -- a bug here must never be able to
        # fail the driver over TEAS-output bookkeeping (see run_teas_output's
        # docstring). run_teas_output already catches everything it knows
        # about; this is defense in depth for anything it doesn't.
        print(f"swebench_run_audit teas-output: ERROR: {exc}", file=sys.stderr)
        return 0

    print(f"swebench_run_audit teas-output: {args.run_dir}")
    print(f"  eval_k8s reports relocated: {len(summary['relocated'])} -> {summary['dest_root']}")
    if summary["skipped_reports"]:
        print(f"  skipped {len(summary['skipped_reports'])} report(s):")
        for iid, reason in summary["skipped_reports"]:
            print(f"    {iid}: {reason}")
    if summary["teas_path"]:
        teas_path = Path(summary["teas_path"])
        print(f"  wrote TEAS outputs: {teas_path.parent} ({teas_path.name} et al.)")
    for warning in summary["warnings"]:
        print(f"  WARNING: {warning}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="swebench_run_audit",
        description="Retry-set classifier / pruner / completeness gate for a "
                     "SWE-bench-on-k8s run directory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("retry-list", help="print task_ids eligible for retry")
    p_list.add_argument("run_dir")
    p_list.add_argument("--retry-timeouts", type=int, choices=(0, 1), default=1,
                         help="retry sweagent_rc=124 (outer timeout) tasks once; "
                              "default matches the driver's RETRY_TIMEOUTS default")
    p_list.set_defaults(func=cmd_retry_list)

    p_prune = sub.add_parser("prune", help="archive+rewrite results.jsonl for a resume pass")
    p_prune.add_argument("run_dir")
    p_prune.add_argument("--tasks-file", required=True, help="file of task_ids to retry, one per line")
    p_prune.add_argument("--attempt", type=int, required=True, help="attempt number, for the archive filename")
    p_prune.set_defaults(func=cmd_prune)

    p_report = sub.add_parser("report", help="write + print a completeness report")
    p_report.add_argument("run_dir")
    p_report.add_argument("--attempts", type=int, required=True, help="how many attempts this run made")
    p_report.add_argument("--out", default=None, help="default: RUN_DIR/completeness.json")
    p_report.set_defaults(func=cmd_report)

    p_health = sub.add_parser(
        "stack-health",
        help="sample engine-vs-agent health for one attempt into stack-health.json")
    p_health.add_argument("run_dir")
    p_health.add_argument("--attempt", type=int, required=True,
                          help="attempt number being sampled; re-sampling the "
                               "same attempt overwrites rather than double-counts")
    p_health.set_defaults(func=cmd_stack_health)

    p_teas = sub.add_parser(
        "teas-output",
        help="relocate eval_k8s/ reports into AgentCAP's expected layout and "
             "write TEAS-format outputs (metadata/metrics/detailed-results/output-data)")
    p_teas.add_argument("run_dir")
    p_teas.add_argument("--model", default=None,
                         help="model name for TEAS metadata, passed through to "
                              "write_teas_outputs -- unrelated to the fixed "
                              "report-path model component (agentcap-unified)")
    p_teas.add_argument("--wall-time-s", type=float, default=0.0,
                         help="generation wall-clock time in seconds, for TEAS "
                              "performance metrics (default: 0.0)")
    p_teas.add_argument("--dataset", default="swe-bench-lite",
                         help="TEAS dataset name (default: swe-bench-lite)")
    p_teas.set_defaults(func=cmd_teas_output)

    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
