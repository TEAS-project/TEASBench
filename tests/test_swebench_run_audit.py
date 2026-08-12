"""Tests for swebench_run_audit - the retry classifier / pruner /
completeness gate for a SWE-bench-on-k8s run directory (CONTRACT.md
"Run-audit module" / "Retry classification rules"). No cluster, no
network: every test builds a small synthetic run directory under
tempfile.TemporaryDirectory() and calls the module's plain functions
directly (they're written to take a run_dir path precisely so tests
don't have to go through the argparse CLI - see the module docstring)."""

import argparse
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline" / "k8s" / "lib"))

import swebench_run_audit as audit

# teas-output calls into AgentCAP (agent_cap.agents.teas_output.write_teas_
# outputs). AgentCAP is a sibling repo, read-only reference, not a dependency
# of this repo's own package -- add it to sys.path only for tests that want a
# real write_teas_outputs() call, and skip those specific tests (rather than
# fail) if it isn't importable in this environment. This must NOT be
# required for correctness: swebench_run_audit.cmd_teas_output() has to work
# (by warning, not crashing) even when AgentCAP genuinely can't be imported,
# and that path is exercised separately by force via sys.modules patching,
# not by this being absent.
AGENTCAP_REPO = REPO.parent / "AgentCAP"
if AGENTCAP_REPO.is_dir():
    sys.path.insert(0, str(AGENTCAP_REPO))
try:
    import agent_cap.agents.teas_output  # noqa: F401
    AGENTCAP_AVAILABLE = True
except Exception:
    AGENTCAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixture helpers - deliberately hand-written and minimal (CONTRACT.md /
# the task brief both call out NOT copying the real evidence run wholesale).
# ---------------------------------------------------------------------------

def write_jsonl(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_row(task_id, output_text="", errors=None, extras=None,
             eval_details=None, eval_passed=None):
    row = {"task_id": task_id, "output_text": output_text}
    if errors is not None:
        row["errors"] = errors
    if extras is not None:
        row["extras"] = extras
    if eval_details is not None:
        row["eval_details"] = eval_details
    if eval_passed is not None:
        row["eval_passed"] = eval_passed
    return row


def write_task_log(run_dir, task_id, stdout=None, stderr=None):
    task_dir = Path(run_dir) / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    if stdout is not None:
        (task_dir / "sweagent_stdout.log").write_text(stdout, encoding="utf-8")
    if stderr is not None:
        (task_dir / "sweagent_stderr.log").write_text(stderr, encoding="utf-8")
    return task_dir


def write_journal(run_dir, events):
    write_jsonl(Path(run_dir) / "portforward-events.jsonl", events)


def write_eval_k8s_report(run_dir, instance_id, resolved, extra_body=None):
    """RUN_DIR/eval_k8s/<iid>/report.json, shaped exactly like
    SWEBenchK8sEvaluator.finalize()'s get_eval_report() dump (a single-key
    {instance_id: {..., "resolved": bool, ...}} dict) -- see
    agent_cap/agents/evaluators_swebench.py."""
    task_dir = Path(run_dir) / "eval_k8s" / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    body = {"resolved": resolved}
    if extra_body:
        body.update(extra_body)
    (task_dir / "report.json").write_text(
        json.dumps({instance_id: body}), encoding="utf-8")
    return task_dir


def make_teas_row(task_id, output_text):
    """A results.jsonl row with the fields write_teas_outputs() actually
    reads (total_usage/tool_calls/e2e_latency_s) -- distinct from make_row()
    above, which is shaped for the retry-classification tests instead."""
    return {
        "task_id": task_id,
        "output_text": output_text,
        "e2e_latency_s": 1.5,
        "tool_calls": 1,
        "total_usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "completion_tokens": 5,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "requests": 1,
        },
    }


# ---------------------------------------------------------------------------
# Rule-by-rule retry classification
# ---------------------------------------------------------------------------

class RetrySignalTests(unittest.TestCase):
    """Each of the four positive retry signals fires on its own, and the
    negative cases (CONTRACT.md "Never retry") stay negative even when
    they superficially resemble a positive case."""

    def test_rule1_running_phase_pf_drop_triggers_retry(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_journal(run_dir, [
                {"ts": 1.0, "label": "t1", "event": "pf_drop", "phase": "running",
                 "reason": "probe_failed"},
            ])
            row = make_row("t1", output_text="")
            labels = audit.journal_running_drop_labels(run_dir)
            should_retry, reason = audit.classify_task(run_dir, "t1", row, False, labels)
            self.assertTrue(should_retry)
            self.assertEqual(reason, "pf_drop_running")

    def test_rule1_startup_phase_pf_drop_does_not_trigger_retry(self):
        """The single most important negative case: phase="startup" fires
        on every normal sandbox start while the pod pip-installs swe-rex,
        and must NOT make the task retryable - only a drop the babysitter
        itself observed after /is_alive first succeeded (phase="running")
        is real infrastructure evidence."""
        with tempfile.TemporaryDirectory() as run_dir:
            write_journal(run_dir, [
                {"ts": 1.0, "label": "t1", "event": "pf_drop", "phase": "startup",
                 "reason": "process_exited"},
            ])
            row = make_row("t1", output_text="")
            labels = audit.journal_running_drop_labels(run_dir)
            self.assertEqual(labels, set())  # startup rows never enter the set at all
            should_retry, reason = audit.classify_task(run_dir, "t1", row, False, labels)
            self.assertFalse(should_retry)
            self.assertEqual(reason, "no_evidence")

    def test_rule2_sidecar_error_triggers_retry(self):
        with tempfile.TemporaryDirectory() as run_dir:
            row = make_row("t2", output_text="",
                            errors=["k8s sidecar failed: sandbox pod not Running after 1200s (t2)"])
            should_retry, reason = audit.classify_task(run_dir, "t2", row, False, set())
            self.assertTrue(should_retry)
            self.assertEqual(reason, "sidecar_error")

    def test_rule2_unrelated_error_does_not_trigger_retry(self):
        with tempfile.TemporaryDirectory() as run_dir:
            row = make_row("t2b", output_text="", errors=["sweagent rc=1"])
            should_retry, reason = audit.classify_task(run_dir, "t2b", row, False, set())
            self.assertFalse(should_retry)
            self.assertEqual(reason, "no_evidence")

    def test_rule3_log_signatures_trigger_retry(self):
        signatures = [
            ("ServerDisconnectedError", "stdout"),
            ("Server disconnected", "stderr"),
            ("Cannot connect to host 127.0.0.1:41235", "stdout"),
            ("SessionExistsError", "stderr"),
        ]
        for signature, which in signatures:
            with self.subTest(signature=signature, which=which):
                with tempfile.TemporaryDirectory() as run_dir:
                    tid = "t3"
                    kwargs = {"stdout": f"garbage\n{signature}\nmore garbage\n"} if which == "stdout" \
                        else {"stderr": f"garbage\n{signature}\nmore garbage\n"}
                    write_task_log(run_dir, tid, **kwargs)
                    row = make_row(tid, output_text="")
                    should_retry, reason = audit.classify_task(run_dir, tid, row, False, set())
                    self.assertTrue(should_retry, f"{signature} in {which} should be retryable")
                    self.assertEqual(reason, "log_signature")

    def test_rule3_clean_log_does_not_trigger_retry(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_task_log(run_dir, "t3b", stdout="agent gave up politely\n",
                            stderr="no traceback here\n")
            row = make_row("t3b", output_text="")
            should_retry, reason = audit.classify_task(run_dir, "t3b", row, False, set())
            self.assertFalse(should_retry)
            self.assertEqual(reason, "no_evidence")

    def test_rule4_timeout_retried_only_with_flag(self):
        with tempfile.TemporaryDirectory() as run_dir:
            row = make_row("t4", output_text="", errors=["sweagent rc=124"],
                            extras={"sweagent_rc": 124})
            should_retry_off, reason_off = audit.classify_task(run_dir, "t4", row, False, set())
            self.assertFalse(should_retry_off)
            self.assertEqual(reason_off, "timeout_not_retried")

            should_retry_on, reason_on = audit.classify_task(run_dir, "t4", row, True, set())
            self.assertTrue(should_retry_on)
            self.assertEqual(reason_on, "timeout_first_retry")

    def test_rule4_timeout_not_retried_twice(self):
        """A task already present in a results.attempt-*.jsonl archive
        with sweagent_rc==124 has already spent its one rule-4 retry."""
        with tempfile.TemporaryDirectory() as run_dir:
            write_jsonl(Path(run_dir) / "results.attempt-1.jsonl", [
                make_row("t4b", output_text="", extras={"sweagent_rc": 124}),
            ])
            row = make_row("t4b", output_text="", extras={"sweagent_rc": 124})
            should_retry, reason = audit.classify_task(run_dir, "t4b", row, True, set())
            self.assertFalse(should_retry)
            self.assertEqual(reason, "timeout_retry_exhausted")

    def test_task_with_patch_never_retried_regardless_of_other_evidence(self):
        """A task with a patch is never retried, whatever else is true of
        it - even if it also happens to carry sidecar-error and
        running-phase-drop evidence (e.g. a warning logged after a
        successful late submit)."""
        with tempfile.TemporaryDirectory() as run_dir:
            row = make_row(
                "t5", output_text="diff --git a/x.py b/x.py\n+pass\n",
                errors=["k8s sidecar failed: something"],
                extras={"sweagent_rc": 124},
            )
            labels = {"t5"}  # even with a running-phase drop on file
            should_retry, reason = audit.classify_task(run_dir, "t5", row, True, labels)
            self.assertFalse(should_retry)
            self.assertEqual(reason, "complete")

    def test_rc_zero_empty_patch_never_retried(self):
        with tempfile.TemporaryDirectory() as run_dir:
            row = make_row("t6", output_text="", extras={"sweagent_rc": 0})
            should_retry, reason = audit.classify_task(run_dir, "t6", row, True, set())
            self.assertFalse(should_retry)
            self.assertEqual(reason, "no_evidence")

    def test_sweagent_own_exit_reasons_never_retried(self):
        for tag in ("exit_cost", "exit_format", "exit_context"):
            with self.subTest(tag=tag):
                with tempfile.TemporaryDirectory() as run_dir:
                    row = make_row(f"t7-{tag}", output_text="", errors=[tag],
                                    extras={"sweagent_rc": 1})
                    should_retry, reason = audit.classify_task(
                        run_dir, f"t7-{tag}", row, True, set())
                    self.assertFalse(should_retry)
                    self.assertEqual(reason, "no_evidence")

    def test_build_retry_report_end_to_end_only_returns_running_phase(self):
        """Integration check that the wiring from a real run_dir through
        build_retry_report matches the per-rule unit tests above: a run
        with one startup churn event and one genuine running-phase drop
        retries only the second task."""
        with tempfile.TemporaryDirectory() as run_dir:
            write_journal(run_dir, [
                {"ts": 1.0, "label": "ok-task", "event": "pf_drop", "phase": "startup",
                 "reason": "process_exited"},
                {"ts": 2.0, "label": "dropped-task", "event": "pf_drop", "phase": "running",
                 "reason": "probe_failed"},
            ])
            write_jsonl(Path(run_dir) / "results.jsonl", [
                make_row("ok-task", output_text=""),
                make_row("dropped-task", output_text=""),
                make_row("complete-task", output_text="diff --git a/x b/x\n"),
            ])
            classifications, journal_present = audit.build_retry_report(run_dir, False)
            self.assertTrue(journal_present)
            self.assertEqual(classifications["ok-task"], (False, "no_evidence"))
            self.assertEqual(classifications["dropped-task"], (True, "pf_drop_running"))
            self.assertEqual(classifications["complete-task"], (False, "complete"))


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

class PruneTests(unittest.TestCase):

    def _make_run(self, run_dir):
        """3 tasks: t1 retried (has stream_stats + traj), t2 retried but
        has no per-task dir at all (defensive case), t3 NOT retried (must
        be left completely untouched)."""
        write_jsonl(Path(run_dir) / "results.jsonl", [
            make_row("t1", output_text=""),
            make_row("t2", output_text=""),
            make_row("t3", output_text="diff --git a/x b/x\n"),
        ])
        t1_dir = Path(run_dir) / "task_t1"
        t1_dir.mkdir()
        (t1_dir / "stream_stats.jsonl").write_text('{"tokens": 1}\n', encoding="utf-8")
        traj_dir = t1_dir / "sweagent_traj"
        traj_dir.mkdir()
        (traj_dir / "run.traj").write_text("{}", encoding="utf-8")

        t3_dir = Path(run_dir) / "task_t3"
        t3_dir.mkdir()
        (t3_dir / "stream_stats.jsonl").write_text('{"tokens": 5}\n', encoding="utf-8")
        (t3_dir / "sweagent_traj").mkdir()
        (t3_dir / "sweagent_traj" / "run.traj").write_text("{}", encoding="utf-8")

        tasks_file = Path(run_dir) / "retry_tasks.txt"
        tasks_file.write_text("t1\nt2\n", encoding="utf-8")
        return tasks_file

    def test_prune_archives_dedups_and_cleans_only_retried_tasks(self):
        with tempfile.TemporaryDirectory() as run_dir:
            tasks_file = self._make_run(run_dir)

            summary = audit.prune_run(run_dir, tasks_file, 1)

            archive_path = Path(run_dir) / "results.attempt-1.jsonl"
            self.assertTrue(archive_path.exists())
            archived_ids = {json.loads(l)["task_id"] for l in archive_path.read_text().splitlines()}
            self.assertEqual(archived_ids, {"t1", "t2", "t3"})

            kept = {json.loads(l)["task_id"]
                    for l in (Path(run_dir) / "results.jsonl").read_text().splitlines()}
            self.assertEqual(kept, {"t3"})

            self.assertFalse((Path(run_dir) / "task_t1" / "stream_stats.jsonl").exists())
            self.assertFalse((Path(run_dir) / "task_t1" / "sweagent_traj").exists())
            # t3 was never in the retry set - must be completely untouched.
            self.assertTrue((Path(run_dir) / "task_t3" / "stream_stats.jsonl").exists())
            self.assertTrue((Path(run_dir) / "task_t3" / "sweagent_traj").exists())

            self.assertEqual(summary["rows_pruned"], 2)
            self.assertEqual(summary["rows_after"], 1)
            self.assertIn("t2", summary["missing_task_dirs"])
            self.assertEqual(sorted(summary["stream_stats_removed"]), ["t1"])
            self.assertEqual(sorted(summary["traj_removed"]), ["t1"])

    def test_prune_deduplicates_keeping_last_occurrence(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_jsonl(Path(run_dir) / "results.jsonl", [
                make_row("dup", output_text="", extras={"attempt": "first"}),
                make_row("other", output_text="diff\n"),
                make_row("dup", output_text="", extras={"attempt": "second"}),
            ])
            tasks_file = Path(run_dir) / "tasks.txt"
            tasks_file.write_text("", encoding="utf-8")  # retry nothing, just dedup

            summary = audit.prune_run(run_dir, tasks_file, 1)

            rows = [json.loads(l) for l in (Path(run_dir) / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            dup_row = next(r for r in rows if r["task_id"] == "dup")
            self.assertEqual(dup_row["extras"]["attempt"], "second")
            self.assertEqual(summary["duplicates_dropped"], 1)

    def test_prune_tolerates_torn_trailing_line(self):
        with tempfile.TemporaryDirectory() as run_dir:
            results_path = Path(run_dir) / "results.jsonl"
            with results_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(make_row("good", output_text="diff\n")) + "\n")
                f.write('{"task_id": "torn", "output_text": "diff --git a/x')  # killed mid-write
            tasks_file = Path(run_dir) / "tasks.txt"
            tasks_file.write_text("", encoding="utf-8")

            summary = audit.prune_run(run_dir, tasks_file, 1)

            self.assertEqual(summary["unparseable_dropped"], 1)
            rows = [json.loads(l) for l in results_path.read_text().splitlines()]
            self.assertEqual([r["task_id"] for r in rows], ["good"])

    def test_prune_is_idempotent(self):
        with tempfile.TemporaryDirectory() as run_dir:
            tasks_file = self._make_run(run_dir)

            summary1 = audit.prune_run(run_dir, tasks_file, 1)
            archive_bytes_1 = (Path(run_dir) / "results.attempt-1.jsonl").read_bytes()
            results_bytes_1 = (Path(run_dir) / "results.jsonl").read_bytes()

            # Re-running prune for the SAME attempt number must not fail,
            # must not overwrite the existing archive, and must leave the
            # already-pruned results.jsonl/cleaned task dirs unchanged.
            summary2 = audit.prune_run(run_dir, tasks_file, 1)

            self.assertFalse(summary2["archived"])  # archive already existed
            self.assertEqual((Path(run_dir) / "results.attempt-1.jsonl").read_bytes(), archive_bytes_1)
            self.assertEqual((Path(run_dir) / "results.jsonl").read_bytes(), results_bytes_1)
            self.assertEqual(summary2["rows_pruned"], 0)  # t1/t2 already gone from results.jsonl
            self.assertEqual(summary1["archive"], summary2["archive"])


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

class ReportTests(unittest.TestCase):

    def _report_args(self, run_dir, attempts=1, out=None):
        return argparse.Namespace(run_dir=str(run_dir), attempts=attempts, out=out)

    def test_report_exits_nonzero_when_infra_incomplete_task_remains(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_journal(run_dir, [
                {"ts": 1.0, "label": "stuck", "event": "pf_drop", "phase": "running",
                 "reason": "probe_failed"},
            ])
            write_jsonl(Path(run_dir) / "results.jsonl", [
                make_row("stuck", output_text=""),
            ])
            rc = audit.cmd_report(self._report_args(run_dir))
            self.assertEqual(rc, 1)
            report = json.loads((Path(run_dir) / "completeness.json").read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["classification"]["stuck"]["status"], "infra-incomplete")

    def test_report_exits_zero_when_run_is_clean(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_jsonl(Path(run_dir) / "results.jsonl", [
                make_row("solved", output_text="diff --git a/x b/x\n+pass\n"),
                make_row("gave-up", output_text=""),  # no infra evidence -> genuinely-failed
            ])
            (Path(run_dir) / "predictions.json").write_text(
                json.dumps([{"instance_id": "solved", "model_patch": "diff"}]), encoding="utf-8")
            (Path(run_dir) / "eval_k8s_results.json").write_text(
                json.dumps({"solved": {"resolved": True}}), encoding="utf-8")

            rc = audit.cmd_report(self._report_args(run_dir))
            self.assertEqual(rc, 0)
            report = json.loads((Path(run_dir) / "completeness.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["exit_reasons"], [])
            self.assertEqual(report["classification"]["gave-up"]["status"], "genuinely-failed")

    def test_report_surfaces_resolved_regression_across_attempts(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_jsonl(Path(run_dir) / "results.attempt-1.jsonl", [
                make_row("flaky", output_text="diff\n", eval_details={"resolved": True}),
            ])
            write_jsonl(Path(run_dir) / "results.jsonl", [
                make_row("flaky", output_text="diff\n", eval_details={"resolved": False}),
            ])
            report = audit.build_completeness_report(run_dir, attempts=2)
            self.assertEqual(len(report["resolved_regressions"]), 1)
            regression = report["resolved_regressions"][0]
            self.assertEqual(regression["task_id"], "flaky")
            self.assertEqual(regression["from_attempt"], "attempt-1")
            self.assertEqual(regression["to_attempt"], "current")


# ---------------------------------------------------------------------------
# teas-output
# ---------------------------------------------------------------------------

class TeasOutputTests(unittest.TestCase):
    """swebench_run_audit teas-output: relocate eval_k8s/ reports into the
    layout AgentCAP's _swebench_reports() expects, then call
    write_teas_outputs(). See the module docstring's "teas-output" entry and
    relocate_swebench_reports()/run_teas_output()'s own docstrings for why."""

    def _teas_args(self, run_dir, model=None, wall_time_s=0.0, dataset="swe-bench-lite"):
        return argparse.Namespace(
            run_dir=str(run_dir), model=model, wall_time_s=wall_time_s, dataset=dataset)

    # -- relocation: layout, idempotency, missing/malformed reports --------

    def test_relocation_produces_three_level_layout(self):
        """The exact RUN_DIR/logs/run_evaluation/<run_id>/<model>/<iid>/
        report.json shape _swebench_reports()'s `glob("*/*/*/report.json")`
        matches -- checked here with the identical glob pattern, without
        importing AgentCAP, so this test always runs."""
        with tempfile.TemporaryDirectory() as run_dir:
            write_eval_k8s_report(run_dir, "astropy__astropy-1", True)

            summary = audit.relocate_swebench_reports(run_dir)

            self.assertEqual(summary["relocated"], ["astropy__astropy-1"])
            self.assertEqual(summary["skipped"], [])

            dest = (Path(run_dir) / "logs" / "run_evaluation"
                    / audit._SWEBENCH_TEAS_RUN_ID / audit._SWEBENCH_TEAS_MODEL_DIR
                    / "astropy__astropy-1" / "report.json")
            self.assertTrue(dest.is_file())
            self.assertEqual(
                json.loads(dest.read_text()),
                {"astropy__astropy-1": {"resolved": True}},
            )

            reports_root = Path(run_dir) / "logs" / "run_evaluation"
            matches = list(reports_root.glob("*/*/*/report.json"))
            self.assertEqual(matches, [dest])

            # eval_k8s/ itself is left completely untouched -- copy, not move.
            self.assertTrue(
                (Path(run_dir) / "eval_k8s" / "astropy__astropy-1" / "report.json").is_file())

    def test_relocation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_eval_k8s_report(run_dir, "t1", True)
            write_eval_k8s_report(run_dir, "t2", False)

            summary1 = audit.relocate_swebench_reports(run_dir)
            dest_root = Path(summary1["dest_root"])
            tree1 = sorted(str(p.relative_to(dest_root)) for p in dest_root.rglob("*") if p.is_file())
            contents1 = {p: (dest_root / p).read_bytes() for p in tree1}

            summary2 = audit.relocate_swebench_reports(run_dir)
            tree2 = sorted(str(p.relative_to(dest_root)) for p in dest_root.rglob("*") if p.is_file())
            contents2 = {p: (dest_root / p).read_bytes() for p in tree2}

            self.assertEqual(summary1["relocated"], summary2["relocated"])
            self.assertEqual(summary1["skipped"], summary2["skipped"])
            self.assertEqual(tree1, tree2)  # no duplicated files
            self.assertEqual(contents1, contents2)  # no corruption

    def test_missing_and_malformed_reports_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as run_dir:
            write_eval_k8s_report(run_dir, "good", True)

            bad_json_dir = Path(run_dir) / "eval_k8s" / "bad-json"
            bad_json_dir.mkdir(parents=True)
            (bad_json_dir / "report.json").write_text("{not valid json", encoding="utf-8")

            # task dir exists (e.g. patch-apply failed before grading) but no
            # report.json was ever written -- the expected case, not a bug.
            (Path(run_dir) / "eval_k8s" / "no-report").mkdir(parents=True)

            bad_shape_dir = Path(run_dir) / "eval_k8s" / "bad-shape"
            bad_shape_dir.mkdir(parents=True)
            (bad_shape_dir / "report.json").write_text(
                json.dumps({"a": 1, "b": 2}), encoding="utf-8")  # not single-key

            summary = audit.relocate_swebench_reports(run_dir)

            self.assertEqual(summary["relocated"], ["good"])
            skipped_ids = {iid for iid, _reason in summary["skipped"]}
            self.assertEqual(skipped_ids, {"bad-json", "no-report", "bad-shape"})
            for _iid, reason in summary["skipped"]:
                self.assertTrue(reason)  # every skip carries a human-readable reason

    # -- always exits 0 ------------------------------------------------------

    def test_cmd_teas_output_exits_zero_when_agentcap_unimportable(self):
        """Forced via sys.modules patching (not by AGENTCAP_REPO simply being
        absent from sys.path) so this holds regardless of whether AgentCAP
        happens to be reachable in a given environment: setting the
        submodule to None in sys.modules makes `from
        agent_cap.agents.teas_output import write_teas_outputs` raise
        ImportError unconditionally."""
        with tempfile.TemporaryDirectory() as run_dir:
            write_jsonl(Path(run_dir) / "results.jsonl", [
                make_teas_row("t1", "diff --git a/x b/x\n+pass\n"),
            ])
            with unittest.mock.patch.dict(sys.modules, {"agent_cap.agents.teas_output": None}):
                rc = audit.cmd_teas_output(self._teas_args(run_dir))
                self.assertEqual(rc, 0)

                summary = audit.run_teas_output(run_dir)
            self.assertIsNone(summary["teas_path"])
            self.assertTrue(any("not importable" in w for w in summary["warnings"]))

    @unittest.skipUnless(AGENTCAP_AVAILABLE, "AgentCAP not importable in this environment")
    def test_cmd_teas_output_exits_zero_when_no_reports_exist(self):
        """Distinct from the unimportable case above: AgentCAP genuinely
        loads here, so this exercises write_teas_outputs()/resolve_quality()
        actually refusing (no report, no explicit-failure provenance) --
        still must not fail the command."""
        with tempfile.TemporaryDirectory() as run_dir:
            write_jsonl(Path(run_dir) / "results.jsonl", [
                make_teas_row("t1", "diff --git a/x b/x\n+pass\n"),
            ])
            # No eval_k8s/ directory at all.
            rc = audit.cmd_teas_output(self._teas_args(run_dir))
            self.assertEqual(rc, 0)

            summary = audit.run_teas_output(run_dir)
            self.assertEqual(summary["relocated"], [])
            self.assertIsNone(summary["teas_path"])
            self.assertTrue(summary["warnings"])

    # -- end to end: synthetic all-complete fixture --------------------------

    @unittest.skipUnless(AGENTCAP_AVAILABLE, "AgentCAP not importable in this environment")
    def test_all_complete_fixture_writes_all_four_teas_files(self):
        with tempfile.TemporaryDirectory() as run_dir:
            resolved_map = {
                "task-a": True, "task-b": True, "task-c": True, "task-d": False,
            }
            rows = [
                make_teas_row(tid, f"diff --git a/{tid}.py b/{tid}.py\n+pass\n")
                for tid in resolved_map
            ]
            write_jsonl(Path(run_dir) / "results.jsonl", rows)
            for tid, resolved in resolved_map.items():
                write_eval_k8s_report(run_dir, tid, resolved)

            summary = audit.run_teas_output(
                run_dir, model="test-model", wall_time_s=120.0, dataset="swe-bench-lite")

            self.assertEqual(summary["warnings"], [])
            self.assertEqual(len(summary["relocated"]), len(resolved_map))
            self.assertIsNotNone(summary["teas_path"])
            self.assertTrue(Path(summary["teas_path"]).is_file())

            for pattern in (
                "metadata_swe-bench-lite_*.json",
                "metrics_swe-bench-lite_*.json",
                "detailed-results_swe-bench-lite_*.jsonl",
                "output-data_swe-bench-lite_*.jsonl",
            ):
                matches = list(Path(run_dir).glob(pattern))
                self.assertEqual(len(matches), 1, f"expected 1 match for {pattern}, got {matches}")

            metrics_path = next(Path(run_dir).glob("metrics_swe-bench-lite_*.json"))
            metrics = json.loads(metrics_path.read_text())
            passed = sum(1 for v in resolved_map.values() if v)
            total = len(resolved_map)
            self.assertEqual(metrics["quality"], {
                "acc": round(passed / total, 4),
                "total_examples": total,
                "passed": passed,
            })


if __name__ == "__main__":
    unittest.main()
