import json
import re
import tempfile
import unittest
from pathlib import Path

from postprocessing.moe_cost_metrics.prefill_rate import (
    PREFILL_PROFILE_SCHEMA,
    SINGLE_PASS_PREFILL_TOKEN_BUDGET,
    configured_input_target,
    load_prefill_profile,
    prefill_profile_path_for,
    resolve_for_metrics_path,
    resolve_prefill_rate,
)


def metrics(prompt=None, batch=None, ttft=None, pass_s=None, nominal=None):
    performance = {}
    if ttft is not None:
        performance["ttft"] = ttft
    if pass_s is not None:
        performance["prefill_pass_latency_s"] = pass_s
    profile = {}
    if prompt is not None:
        profile["prefill_tokens_per_request"] = prompt
    if batch is not None:
        profile["prefill_avg_batch_size"] = batch
    if nominal is not None:
        profile["prefill_tokens"] = nominal
    return {"performance": performance, "batch_token_profile": profile}


def exact_profile(**overrides):
    profile = {
        "schema": PREFILL_PROFILE_SCHEMA,
        "prefill_forwarded_tokens": 90_000,
        "prefill_step_elapsed_s": 25.0,
        "prefill_nominal_attempted_tokens": 100_000,
    }
    profile.update(overrides)
    return profile


LAUNCH_FIXED = "runner --datasets arena-hard --target-input-tokens 1024 --ignore-eos"


class IdentityArmTests(unittest.TestCase):

    def test_measured_batch_one_is_aggregation_exact(self):
        res = resolve_prefill_rate(metrics(prompt=120.0, batch=1.0, ttft=0.5))
        self.assertEqual(res, {
            "value": 240.0, "basis": "measured", "method": "identity-bs1",
            "token_basis": "nominal-attempted", "reason": None,
        })

    def test_batch_regime_name_pins_identity_over_recorded_batch(self):
        # batch-size-1 pins concurrency server-side; a profile average above 1
        # on such a run is an accounting artefact, not concurrency.
        res = resolve_prefill_rate(
            metrics(prompt=120.0, batch=7.0, ttft=0.5), batch_regime="batch-size-1"
        )
        self.assertEqual(res["method"], "identity-bs1")
        self.assertEqual(res["value"], 240.0)

    def test_identity_beats_an_exact_profile(self):
        # A batch-1 run needs no aggregation correction; its published value
        # keeps the (cache-inclusive) nominal identity it always had.
        res = resolve_prefill_rate(
            metrics(prompt=120.0, batch=1.0, ttft=0.5), exact_profile()
        )
        self.assertEqual(res["method"], "identity-bs1")

    def test_identity_without_tokens_is_null(self):
        res = resolve_prefill_rate(metrics(batch=1.0, ttft=0.5))
        self.assertEqual(res["reason"], "no-token-evidence")
        self.assertIsNone(res["value"])

    def test_identity_without_ttft_is_null(self):
        res = resolve_prefill_rate(metrics(prompt=120.0, batch=1.0))
        self.assertEqual(res["reason"], "no-latency-evidence")


class ExactArmTests(unittest.TestCase):

    def test_exact_profile_publishes_nominal_over_physical_elapsed(self):
        res = resolve_prefill_rate(
            metrics(prompt=390.0, batch=64.0, ttft=2.0, pass_s=0.5, nominal=100_000),
            exact_profile(),
        )
        self.assertEqual(res, {
            "value": 100_000 / 25.0, "basis": "measured", "method": "trace-exact",
            "token_basis": "nominal-attempted", "reason": None,
        })

    def test_repaired_metrics_numerator_invalidates_the_profile(self):
        # The profile's numerator is a build-time copy of the run's nominal
        # token total; a run whose metrics were repaired afterwards must not
        # keep publishing the frozen copy as measured.
        res = resolve_prefill_rate(
            metrics(prompt=390.0, batch=64.0, ttft=2.0, pass_s=0.5, nominal=90_000),
            exact_profile(),
        )
        self.assertEqual(res["method"], "hybrid-rung1")
        self.assertEqual(res["basis"], "estimated")

    def test_profile_without_a_live_metrics_numerator_is_stale_evidence(self):
        res = resolve_prefill_rate(
            metrics(prompt=390.0, batch=64.0, ttft=2.0, pass_s=0.5), exact_profile()
        )
        self.assertEqual(res["method"], "hybrid-rung1")

    def test_stale_profile_with_no_estimate_evidence_is_null(self):
        res = resolve_prefill_rate(
            metrics(batch=64.0, ttft=2.0, nominal=90_000), exact_profile()
        )
        self.assertEqual(res["value"], None)
        self.assertEqual(res["reason"], "no-token-evidence")

    def test_unknown_profile_schema_is_ignored_not_partially_decoded(self):
        res = resolve_prefill_rate(
            metrics(prompt=390.0, batch=64.0, ttft=2.0, pass_s=0.5),
            exact_profile(schema="prefill-profile/999"),
        )
        self.assertEqual(res["method"], "hybrid-rung1")

    def test_degenerate_profile_fields_reject_the_profile(self):
        for bad in (
            exact_profile(prefill_step_elapsed_s=0.0),
            exact_profile(prefill_nominal_attempted_tokens=None),
            exact_profile(prefill_forwarded_tokens=-1),
        ):
            res = resolve_prefill_rate(
                metrics(prompt=390.0, batch=64.0, ttft=2.0, pass_s=0.5), bad
            )
            self.assertEqual(res["method"], "hybrid-rung1")


class EstimateArmTests(unittest.TestCase):

    def test_short_prompt_uses_rung1_pass_latency_form(self):
        res = resolve_prefill_rate(metrics(prompt=120.0, batch=32.0, ttft=1.0, pass_s=0.5))
        self.assertEqual(res, {
            "value": 120.0 * 32.0 / 0.5, "basis": "estimated",
            "method": "hybrid-rung1", "token_basis": "nominal-attempted",
            "reason": None,
        })

    def test_long_prompt_uses_rung2_ttft_form(self):
        res = resolve_prefill_rate(
            metrics(prompt=11_000.0, batch=8.0, ttft=4.0, pass_s=0.5)
        )
        self.assertEqual(res, {
            "value": (11_000.0 / 4.0) * 8.0, "basis": "estimated",
            "method": "hybrid-rung2", "token_basis": "nominal-attempted",
            "reason": None,
        })

    def test_rung_boundary_sits_at_the_single_pass_budget(self):
        at = resolve_prefill_rate(metrics(
            prompt=float(SINGLE_PASS_PREFILL_TOKEN_BUDGET), batch=4.0, ttft=1.0, pass_s=0.5))
        above = resolve_prefill_rate(metrics(
            prompt=float(SINGLE_PASS_PREFILL_TOKEN_BUDGET + 1), batch=4.0, ttft=1.0, pass_s=0.5))
        self.assertEqual(at["method"], "hybrid-rung1")
        self.assertEqual(above["method"], "hybrid-rung2")

    def test_rung1_does_not_need_ttft(self):
        # The short-prompt estimate runs on pass latency alone, so a run whose
        # first-token latency was withheld keeps its estimate.
        res = resolve_prefill_rate(metrics(prompt=130.0, batch=85.0, pass_s=0.7))
        self.assertEqual(res["method"], "hybrid-rung1")

    def test_rung1_without_pass_latency_is_null_not_a_ttft_fallback(self):
        res = resolve_prefill_rate(metrics(prompt=130.0, batch=85.0, ttft=0.1))
        self.assertEqual(res["reason"], "no-latency-evidence")
        self.assertIsNone(res["value"])

    def test_rung2_without_ttft_is_null(self):
        res = resolve_prefill_rate(metrics(prompt=11_000.0, batch=8.0, pass_s=0.5))
        self.assertEqual(res["reason"], "no-latency-evidence")

    def test_configured_target_fills_a_missing_token_count(self):
        res = resolve_prefill_rate(
            metrics(batch=16.0, ttft=1.0, pass_s=2.0),
            batch_regime="batch-size-default_input1024_output1024",
            launch_text=LAUNCH_FIXED,
        )
        self.assertEqual(res, {
            "value": 1024 * 16.0 / 2.0, "basis": "estimated",
            "method": "hybrid-rung1", "token_basis": "configured-input-target",
            "reason": None,
        })

    def test_recorded_tokens_win_over_a_configured_target(self):
        res = resolve_prefill_rate(
            metrics(prompt=1109.0, batch=16.0, ttft=1.0, pass_s=2.0),
            batch_regime="batch-size-default_input1024_output1024",
            launch_text=LAUNCH_FIXED,
        )
        self.assertEqual(res["token_basis"], "nominal-attempted")
        self.assertEqual(res["value"], 1109.0 * 16.0 / 2.0)

    def test_missing_tokens_and_no_witnessed_target_is_null(self):
        res = resolve_prefill_rate(metrics(batch=16.0, ttft=1.0, pass_s=2.0))
        self.assertEqual(res["reason"], "no-token-evidence")

    def test_no_batch_evidence_is_null(self):
        res = resolve_prefill_rate(metrics(prompt=120.0, ttft=0.5, pass_s=0.5))
        self.assertEqual(res["reason"], "no-batch-evidence")
        self.assertEqual(
            res, {"value": None, "basis": None, "method": None,
                  "token_basis": None, "reason": "no-batch-evidence"})


class ConfiguredTargetWitnessTests(unittest.TestCase):
    """Both run-native witnesses are required and must agree."""

    def test_regime_and_sole_flag_agreeing_yields_the_target(self):
        self.assertEqual(
            configured_input_target(
                "batch-size-default_input1024_output1024", LAUNCH_FIXED),
            1024,
        )

    def test_regime_without_launch_record_is_refused(self):
        self.assertIsNone(configured_input_target(
            "batch-size-default_input1024_output1024", None))

    def test_launch_flag_without_fixed_regime_is_refused(self):
        self.assertIsNone(configured_input_target("batch-size-default", LAUNCH_FIXED))

    def test_disagreeing_flag_value_is_refused(self):
        self.assertIsNone(configured_input_target(
            "batch-size-default_input2048_output1024", LAUNCH_FIXED))

    def test_repeated_flag_is_refused(self):
        self.assertIsNone(configured_input_target(
            "batch-size-default_input1024_output1024",
            LAUNCH_FIXED + " --target-input-tokens 1024"))


class RunTreeHelperTests(unittest.TestCase):

    def test_resolve_for_metrics_path_reads_profile_regime_and_launch(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "batch-size-default_input1024_output1024" / "20990101-0000"
            run.mkdir(parents=True)
            metrics_path = run / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics(batch=16.0, pass_s=2.0, nominal=100_000)))
            (run / "run.sh").write_text(LAUNCH_FIXED + "\n")
            res = resolve_for_metrics_path(
                metrics_path, json.loads(metrics_path.read_text()))
            self.assertEqual(res["token_basis"], "configured-input-target")
            self.assertEqual(res["value"], 1024 * 16.0 / 2.0)

            profile_path = prefill_profile_path_for(metrics_path)
            self.assertEqual(profile_path.name, "prefill_profile.json")
            profile_path.write_text(json.dumps(exact_profile()))
            res = resolve_for_metrics_path(
                metrics_path, json.loads(metrics_path.read_text()))
            self.assertEqual(res["method"], "trace-exact")

    def test_malformed_profile_file_loads_as_none(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            metrics_path = run / "metrics.json"
            metrics_path.write_text("{}")
            (run / "prefill_profile.json").write_text("not json")
            self.assertIsNone(load_prefill_profile(metrics_path))


class DesignRuleTests(unittest.TestCase):

    def test_resolver_module_carries_no_run_coordinate_constants(self):
        """D148: dispositions are conditions on run-native evidence.

        The shipped module must contain no model, dataset-instance, GPU,
        provider or run-date constant — nothing that names WHICH runs get a
        disposition rather than WHAT evidence earns it. The only numeric
        policy constants allowed are the physically-justified identity
        tolerance and single-pass budget.
        """
        source = (
            Path(__file__).resolve().parents[1]
            / "postprocessing" / "moe_cost_metrics" / "prefill_rate.py"
        ).read_text()
        # Strip docstrings and comments: prose may justify a physical constant
        # by naming the engines it was derived from; executable code may not
        # name anything that picks runs.
        body = re.sub(r'"""[\s\S]*?"""', "", source)
        body = "\n".join(line.split("#", 1)[0] for line in body.splitlines())
        banned = [
            # model instances
            "deepseek", "qwen", "kimi", "gpt-oss", "oss-120b", "oss-20b",
            # dataset instances
            "gsm8k", "arena", "longbench",
            # accelerators / providers
            "a100", "h100", "h200", "b200", "b300", "mi355", "gb10",
            "blackhole", "vastai", "eidf", "dgx", "tenstorrent",
            # engines
            "sglang", "vllm",
        ]
        lowered = body.lower()
        for token in banned:
            self.assertNotIn(token, lowered, f"coordinate constant {token!r}")
        # No run-timestamp/date literals (e.g. 20260803) anywhere in code.
        self.assertIsNone(re.search(r"\b20\d{6}\b", body))
        self.assertIsNone(re.search(r"\b20\d{2}-\d{2}-\d{2}\b", body))


if __name__ == "__main__":
    unittest.main()
