"""Attempt-scoped drop evidence, and the stack-health sampler.

Both exist because of the same run: two tasks that failed for a reason the
benchmark is supposed to measure (sglang's gpt-oss parser mangling tool calls)
were reported as infrastructure-incomplete and retried until the loop gave up.
The scoping fix stops stale drop evidence masking a genuine failure; the
sampler makes the engine-side degradation visible instead of invisible.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "pipeline" / "k8s" / "lib"
DRIVER_TEMPLATE = REPO / "pipeline" / "templates" / "agentic-driver.sh"
sys.path.insert(0, str(LIB))

import swebench_run_audit as audit  # noqa: E402


def write_journal(run_dir, events):
    path = Path(run_dir) / "portforward-events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


DROP1 = {"ts": 100.0, "label": "task-a", "event": "pf_drop", "phase": "running"}
DROP2 = {"ts": 300.0, "label": "task-b", "event": "pf_drop", "phase": "running"}
STARTUP = {"ts": 105.0, "label": "task-c", "event": "pf_drop", "phase": "startup"}
MARK2 = {"ts": 200.0, "event": "attempt_start", "attempt": 2}


class AttemptScopingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_drop_before_the_current_attempt_is_not_evidence(self):
        """The actual defect. task-a lost its tunnel in attempt 1 and recovered;
        attempt 2 saw no drop at all. Counting attempt 1's drop again is what
        pinned such tasks as infrastructure-incomplete for the whole run."""
        write_journal(self.run_dir, [DROP1, MARK2, DROP2])
        labels = audit.journal_running_drop_labels(self.run_dir)
        self.assertEqual(labels, {"task-b"})

    def test_a_journal_with_no_marker_keeps_whole_run_behaviour(self):
        """Runs recorded before the marker existed are already published; they
        must not be silently re-judged by a newer auditor."""
        write_journal(self.run_dir, [DROP1, DROP2])
        self.assertEqual(audit.journal_running_drop_labels(self.run_dir),
                         {"task-a", "task-b"})

    def test_startup_phase_is_still_ignored_inside_the_window(self):
        write_journal(self.run_dir, [MARK2, STARTUP, DROP2])
        self.assertEqual(audit.journal_running_drop_labels(self.run_dir), {"task-b"})

    def test_only_the_last_marker_defines_the_window(self):
        write_journal(self.run_dir, [
            DROP1, MARK2, DROP2,
            {"ts": 400.0, "event": "attempt_start", "attempt": 3},
            {"ts": 500.0, "label": "task-d", "event": "pf_drop", "phase": "running"},
        ])
        self.assertEqual(audit.journal_running_drop_labels(self.run_dir), {"task-d"})

    def test_a_drop_exactly_on_the_boundary_counts(self):
        write_journal(self.run_dir, [
            MARK2, {"ts": 200.0, "label": "task-e", "event": "pf_drop", "phase": "running"}])
        self.assertEqual(audit.journal_running_drop_labels(self.run_dir), {"task-e"})

    def test_unplaceable_drop_is_ignored_once_a_window_exists(self):
        """No ts means it cannot be placed. Counting it would reinstate the
        stale-evidence bug the window closes."""
        write_journal(self.run_dir, [MARK2, {"label": "task-f", "event": "pf_drop",
                                             "phase": "running"}])
        self.assertEqual(audit.journal_running_drop_labels(self.run_dir), set())

    def test_a_bool_ts_is_not_a_timestamp(self):
        write_journal(self.run_dir, [{"ts": True, "event": "attempt_start", "attempt": 2},
                                     DROP1])
        # No usable marker -> whole-run behaviour, not a window at ts=1.
        self.assertEqual(audit.journal_running_drop_labels(self.run_dir), {"task-a"})

    def test_scoping_flips_the_classifier_from_retry_to_genuine_failure(self):
        """End-to-end: the empty-patch, rc=0 row that used to be retried
        forever must now reach the no_evidence verdict."""
        write_journal(self.run_dir, [DROP1, MARK2])
        row = {"task_id": "task-a", "output_text": "",
               "extras": {"sweagent_rc": 0, "has_patch": False}, "errors": []}
        labels = audit.journal_running_drop_labels(self.run_dir)
        should_retry, reason = audit.classify_task(self.run_dir, "task-a", row, True, labels)
        self.assertFalse(should_retry)
        self.assertEqual(reason, "no_evidence")

        # And the same row IS still retried when the drop is inside the window.
        write_journal(self.run_dir, [MARK2, {**DROP1, "ts": 250.0}])
        labels = audit.journal_running_drop_labels(self.run_dir)
        should_retry, reason = audit.classify_task(self.run_dir, "task-a", row, True, labels)
        self.assertTrue(should_retry)
        self.assertEqual(reason, "pf_drop_running")


STRANDED_LOG = """\
🤖 DEBUG    response
            choices=[Choices(finish_reason='stop', index=0,
            message=Message(content=None, role='assistant', tool_calls=None,
            function_call=None, reasoning_content='We need to locate failing
            test.{"command":"view","path":"/testbed/tests"}'
            , provider_specific_fields=None))],
🤠 WARN     Requerying model after FunctionCallingFormatError (1th requery)
"""

CLEAN_LOG = """\
🤖 DEBUG    response
            choices=[Choices(finish_reason='stop', index=0,
            message=Message(content='I am done thinking', role='assistant',
            tool_calls=None, function_call=None,
            provider_specific_fields=None))],
🤠 WARN     Requerying model after FunctionCallingFormatError (1th requery)
"""


class StackHealthTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def make_task(self, name, log="", responses=0, exit_status=None, actions=()):
        d = self.run_dir / f"task_{name}"
        d.mkdir(parents=True)
        if log:
            (d / "sweagent_stdout.log").write_text(log, encoding="utf-8")
        if responses:
            (d / "stream_stats.jsonl").write_text(
                "".join(json.dumps({"completion_tokens": 1}) + "\n" for _ in range(responses)),
                encoding="utf-8")
        if exit_status is not None or actions:
            td = d / "sweagent_traj" / "abc123"
            td.mkdir(parents=True)
            (td / "abc123.traj").write_text(json.dumps({
                "info": {"exit_status": exit_status or "submitted"},
                "trajectory": [{"action": a} for a in actions],
            }), encoding="utf-8")
        return d

    def test_stranded_tool_call_is_detected(self):
        stranded, examined = audit.scrape_stranded_tool_calls(STRANDED_LOG)
        self.assertEqual(stranded, 1)
        self.assertEqual(examined, 1)

    def test_a_rejection_with_no_reasoning_channel_is_a_clean_negative(self):
        """An engine that populates no reasoning_content strands nothing by
        definition. Reporting that as 'undetermined' would claim a blind spot
        where there is a real negative -- which is how vLLM reads."""
        stranded, examined = audit.scrape_stranded_tool_calls(CLEAN_LOG)
        self.assertEqual(stranded, 0)
        self.assertEqual(examined, 1)

    def test_unparseable_log_reports_undetermined_not_zero(self):
        """The whole point of marking this best-effort: a SWE-agent version
        that changes its log format must not read as 'the problem went away'."""
        stranded, examined = audit.scrape_stranded_tool_calls(
            "WARN Requerying model after FunctionCallingFormatError\n")
        self.assertIsNone(stranded)
        self.assertEqual(examined, 0)

    def test_a_log_with_no_rejections_is_zero_not_undetermined(self):
        self.assertEqual(audit.scrape_stranded_tool_calls("nothing here"), (0, 0))

    def test_format_limit_counts_the_autosubmitted_variant(self):
        """`submitted (exit_format)` is the same engine-side failure; the agent
        just autosubmitted anyway. Counting only the bare status undercounts
        it by roughly half on a real run."""
        self.make_task("a", exit_status="submitted (exit_format)")
        self.make_task("b", exit_status="exit_format")
        self.make_task("c", exit_status="submitted")
        doc = audit.sample_stack_health(self.run_dir, 1)
        self.assertEqual(doc["attempts"]["1"]["tasks_hit_format_limit"], 2)

    def test_totals_accumulate_across_attempts(self):
        self.make_task("a", log=STRANDED_LOG, responses=3)
        audit.sample_stack_health(self.run_dir, 1)
        doc = audit.sample_stack_health(self.run_dir, 2)
        self.assertEqual(doc["totals"]["responses"], 6)
        self.assertEqual(sorted(doc["attempts"]), ["1", "2"])

    def test_resampling_one_attempt_overwrites_rather_than_double_counts(self):
        self.make_task("a", log=STRANDED_LOG, responses=3)
        audit.sample_stack_health(self.run_dir, 1)
        doc = audit.sample_stack_health(self.run_dir, 1)
        self.assertEqual(doc["totals"]["responses"], 3)

    def test_report_is_marked_best_effort(self):
        self.make_task("a", log=STRANDED_LOG, responses=1)
        doc = audit.sample_stack_health(self.run_dir, 1)
        self.assertEqual(doc["stranded_detection"], "best-effort-log-scrape")

    def test_sampling_never_raises_on_a_broken_task_dir(self):
        d = self.make_task("a", responses=1)
        td = d / "sweagent_traj" / "zz"
        td.mkdir(parents=True)
        (td / "zz.traj").write_text("{not json", encoding="utf-8")
        doc = audit.sample_stack_health(self.run_dir, 1)
        self.assertEqual(doc["attempts"]["1"]["tasks"], 1)

    def test_empty_action_steps_are_counted(self):
        self.make_task("a", actions=["ls", "", "  ", "grep x"])
        doc = audit.sample_stack_health(self.run_dir, 1)
        self.assertEqual(doc["attempts"]["1"]["empty_action_steps"], 2)


class DriverWiringTests(unittest.TestCase):
    """The template, not a generated copy: these must hold for every row."""

    @classmethod
    def setUpClass(cls):
        cls.body = DRIVER_TEMPLATE.read_text()

    def test_attempt_marker_is_stamped_inside_the_retry_loop(self):
        self.assertIn('"event": "attempt_start"', self.body)
        at = self.body.index('"event": "attempt_start"')
        self.assertGreater(at, self.body.index("while :; do"),
                           "the marker must be stamped per attempt, not once per run")
        self.assertLess(at, self.body.index("agent_cap exit:"),
                        "the marker must precede the work it delimits")

    def test_stack_health_runs_before_prune_can_delete_the_evidence(self):
        self.assertIn("stack-health", self.body)
        at = self.body.index('swebench_run_audit stack-health')
        self.assertLess(at, self.body.index("swebench_run_audit prune"),
                        "prune deletes sweagent_traj/ and stream_stats.jsonl")

    def test_stack_health_failure_cannot_end_the_run(self):
        at = self.body.index('swebench_run_audit stack-health')
        block = self.body[at:at + 300]
        self.assertIn("||", block, "advisory telemetry must not abort the loop")

    def test_stack_health_is_not_published(self):
        at = self.body.index("NO_PUBLISH=(")
        self.assertIn("stack-health.json", self.body[at:at + 300])


class CliTests(unittest.TestCase):
    def test_stack_health_subcommand_exits_zero_on_an_empty_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-m", "swebench_run_audit", "stack-health", tmp,
                 "--attempt", "1"],
                cwd=str(LIB), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue((Path(tmp) / "stack-health.json").exists())


if __name__ == "__main__":
    unittest.main()
