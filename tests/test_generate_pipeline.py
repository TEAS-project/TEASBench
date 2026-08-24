import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from postprocessing.aggregate_results import parse_run_path

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "pipeline"
EXPERIMENTS = REPO / "experiments"
STUDY_LAUNCHER = PIPELINE / "k8s" / "helpers" / "run_study_block.sh"
STUDY_PREFLIGHT_RUNNER = PIPELINE / "k8s" / "helpers" / "run_study_preflight.sh"
STUDY_PREFLIGHT_COLLECTOR = (
    PIPELINE / "k8s" / "helpers" / "study_preflight_collector.yaml")
STUDY_GUARD = PIPELINE / "k8s" / "helpers" / "study_guard.py"
STUDY_TERMINAL_VALIDATOR = (
    PIPELINE / "k8s" / "helpers" / "study_terminal_validate.sh")

# Benchmarks that route through the agentic family (see utils.AGENTIC_BENCHMARKS),
# with the experiments CSV that exercises each one and the container count its
# generated YAML's pod spec is expected to have (1, except mcp-atlas's fixed
# tool-server sidecar making 2 -- see pipeline/configs/config.yaml's
# sidecar_containers rule and section 1 of docs/agentic-pipeline-design.md).
AGENTIC_CSVS_AND_EXPECTED_CONTAINERS = {
    "imo-answerbench": (EXPERIMENTS / "imo-answerbench-eidf.csv", 1),
    "mcp-atlas": (EXPERIMENTS / "mcp-atlas-eidf.csv", 2),
    "swe-bench-lite": (EXPERIMENTS / "swe-bench-lite-eidf.csv", 1),
}

# Literal secret values that must never appear in a generated YAML -- the
# flawed prior art (TEASBench-jpr-pipeline-eidf-mcp-atlas) inlined these
# directly in its mcp-atlas sidecar instead of referencing k8s secrets by name.
FORBIDDEN_LITERAL_SECRETS = ("YOUR_GITHUB_TOKEN_HERE", "YOUR_BRAVE_API_KEY_HERE")


def load_module_from_path(module_name, path):
    """Import a module by file path under an unambiguous name, so loading
    pipeline/utils.py or pipeline/template.py for direct unit testing can't
    collide with an unrelated same-named module elsewhere on sys.path."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NoDuplicateKeysSafeLoader(yaml.SafeLoader):
    """A yaml.safe_load that raises on a mapping with a repeated key, instead
    of silently keeping only the last occurrence (PyYAML's default). This is
    exactly the defect class the flawed prior art had: a duplicate top-level
    `containers:` key in the pod spec, which silently discarded the sidecar
    container rather than erroring -- see
    TEASBench-jpr-pipeline-eidf-mcp-atlas/pipeline/templates/agentic-sidecar.yaml."""


def _no_duplicates_constructor(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key in YAML mapping: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDuplicateKeysSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates_constructor)


def load_yaml_no_duplicates(text):
    return yaml.load(text, Loader=_NoDuplicateKeysSafeLoader)


def run_generate(pipeline_dir, csv_file, target_dir, results_repo=None):
    """Invoke generate.py exactly as it is meant to be run: from within a
    pipeline/ directory (relative reads of configs/config.yaml and
    templates/*.yaml), via the interpreter that has pandas/pyyaml
    (sys.executable, so the test works under ~/pyvenvs/teasbench/bin/python)."""
    cmd = [sys.executable, "generate.py", "--csv_file", str(csv_file),
           "--target_dir", str(target_dir)]
    if results_repo is not None:
        cmd += ["--results_repo", results_repo]
    return subprocess.run(cmd, cwd=str(pipeline_dir), text=True,
                          capture_output=True, check=True)


# Last commit before the agentic-pipeline generalisation ("Fix agentic cost
# computation"). The MoE regression tests below compare against this, not HEAD.
BASELINE_REF = os.environ.get("TEASBENCH_BASELINE_REF", "f22e96e")


def make_baseline_pipeline_copy(tmp_root):
    """Materialise the pre-change pipeline/ into a fresh directory nested
    inside this repo (so `git rev-parse HEAD`, run with this directory as cwd,
    walks up to the real .git -- no git stash involved).

    The baseline is pinned to BASELINE_REF, NOT to HEAD. It was HEAD while this
    work was uncommitted, but once the work lands HEAD becomes the *post*-change
    code and every comparison below silently turns into new-vs-new -- i.e. the
    MoE byte-identity guarantee would keep passing while testing nothing at all.
    Override with TEASBENCH_BASELINE_REF if the history is ever rewritten.

    Returns the path to the baseline pipeline/ directory.
    """
    baseline = Path(tmp_root) / "pipeline"
    (baseline / "configs").mkdir(parents=True)
    (baseline / "templates").mkdir(parents=True)

    for rel in ("utils.py", "generate.py", "template.py",
                "configs/config.yaml", "templates/template.yaml"):
        blob = subprocess.check_output(
            ["git", "show", f"{BASELINE_REF}:pipeline/{rel}"], cwd=str(REPO), text=True)
        (baseline / rel).write_text(blob)

    return baseline


def add_blank_length_columns(src_csv, dst_csv):
    """Copy src_csv to dst_csv with two blank trailing columns,
    'input_length' and 'output_length', appended to every row.

    Needed only to work around a pre-existing, pipeline-unrelated issue
    documented in MoeRegressionTests.test_pre_change_code_cannot_run_against_the_checked_in_csv:
    the pre-change generate.py unconditionally evaluates
    `row.input_length`/`row.output_length` for every row (see git show
    $BASELINE_REF:pipeline/generate.py), but experiments/moe-experiments-eidf.csv has
    never had those columns (git log -p confirms it was committed without
    them). Pandas' Series.__getattr__ raises AttributeError for a genuinely
    absent column (as opposed to one that is present but NaN), so the
    pre-change code cannot execute against the checked-in fixture at all in
    this environment -- not something introduced by this refactor. Adding
    blank columns makes every row equivalent to 'column present but empty',
    which both the old and new code already treat identically to 'column
    absent' (NaN / not-a-key both mean "no fixed length"), so this changes
    nothing about the experiment parameters actually being swept.
    """
    with open(src_csv, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0] + ["input_length", "output_length"]
    out_rows = [header] + [row + ["", ""] for row in rows[1:] if row]
    with open(dst_csv, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)


class MoeRegressionTests(unittest.TestCase):
    """C8.1 (historical): during the agentcap refactor itself, the MoE path
    had to stay byte-identical to BASELINE_REF -- that was the acceptance
    criterion for the refactor workstream, and is still checked by
    test_moe_yaml_generation_treats_blank_length_columns_as_a_no_op below,
    minus the final historical-byte-identity assertion. That assertion was
    retired when main's independent post-fork MoE changes (PVC archive
    layout, apt-get robustification, expert-distribution-copy-command, the
    vLLM --use-chat-api rule) were merged into this branch: those are real,
    intended changes to generated MoE YAML, so BASELINE_REF output is now
    expected to differ, not a regression to guard against."""

    def test_pre_change_code_cannot_run_against_the_checked_in_csv(self):
        """Documents a pre-existing, refactor-unrelated fixture/code mismatch
        discovered while writing the regression test below: the pre-change
        generate.py hard-codes `row.input_length`/`row.output_length` (no
        .get(), no try/except), but experiments/moe-experiments-eidf.csv has
        no such columns (confirmed with `git log -p -- experiments/moe-experiments-eidf.csv`,
        which shows the file was committed without them). Running the
        unmodified pre-change code against the unmodified checked-in CSV therefore
        fails outright, in this repo, on this pandas version, independent of
        anything in this workstream. This test pins that fact down so it
        isn't mistaken for a regression introduced here; the byte-identical
        comparison test below works around it with add_blank_length_columns.
        """
        moe_csv = EXPERIMENTS / "moe-experiments-eidf.csv"
        with tempfile.TemporaryDirectory(dir=str(REPO)) as tmp_root:
            baseline_pipeline = make_baseline_pipeline_copy(tmp_root)
            out_dir = Path(tmp_root) / "out"
            out_dir.mkdir()
            with self.assertRaises(subprocess.CalledProcessError) as ctx:
                run_generate(baseline_pipeline, moe_csv, out_dir)
            self.assertIn("input_length", ctx.exception.stderr)

    def test_moe_yaml_generation_treats_blank_length_columns_as_a_no_op(self):
        """Renamed from test_moe_yaml_generation_byte_identical_to_pre_change_code:
        this no longer compares against BASELINE_REF's generated bytes (see the
        class docstring for why that comparison was retired), only that blank
        input_length/output_length columns are indistinguishable from the
        columns being absent altogether, and that the real fixture CSV still
        generates one file per row under current code."""
        moe_csv = EXPERIMENTS / "moe-experiments-eidf.csv"
        self.assertTrue(moe_csv.exists(), f"missing fixture CSV: {moe_csv}")

        with tempfile.TemporaryDirectory(dir=str(REPO)) as tmp_root:
            augmented_csv = Path(tmp_root) / "moe-experiments-eidf.augmented.csv"
            add_blank_length_columns(moe_csv, augmented_csv)

            new_out_real_csv = Path(tmp_root) / "new_out_real_csv"
            new_out_augmented_csv = Path(tmp_root) / "new_out_augmented_csv"
            new_out_real_csv.mkdir()
            new_out_augmented_csv.mkdir()

            run_generate(PIPELINE, moe_csv, new_out_real_csv)
            run_generate(PIPELINE, augmented_csv, new_out_augmented_csv)

            new_files_real = sorted(p.name for p in new_out_real_csv.iterdir())
            new_files_augmented = sorted(p.name for p in new_out_augmented_csv.iterdir())
            self.assertTrue(new_files_real, "no YAML files were generated at all")
            self.assertEqual(new_files_real, new_files_augmented,
                              "generated file names differ (real CSV vs. augmented CSV)")

            for name in new_files_real:
                new_bytes_real = (new_out_real_csv / name).read_bytes()
                new_bytes_augmented = (new_out_augmented_csv / name).read_bytes()
                self.assertEqual(
                    new_bytes_real, new_bytes_augmented,
                    f"{name}: new code produced different output for the real "
                    f"CSV vs. the same rows with blank input_length/output_length "
                    f"columns appended -- the two should be indistinguishable")


class AgenticYamlGenerationTests(unittest.TestCase):
    """C8.2-C8.5: correctness checks on the generated agentic YAMLs, run
    against every row of every checked-in agentic experiments CSV (all three
    benchmarks, both engines, every GPU row) plus the agentic smoke tests."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(dir=str(REPO))
        cls.out_dir = Path(cls._tmp.name)
        cls.csvs = [csv_path for csv_path, _ in AGENTIC_CSVS_AND_EXPECTED_CONTAINERS.values()]
        cls.csvs.append(EXPERIMENTS / "agentic-smoke-tests-eidf.csv")
        for csv_path in cls.csvs:
            run_generate(PIPELINE, csv_path, cls.out_dir)
        cls.generated = sorted(cls.out_dir.glob("*.yaml"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def expected_container_count(self, filename):
        for benchmark, (_, expected) in AGENTIC_CSVS_AND_EXPECTED_CONTAINERS.items():
            if benchmark in filename:
                return expected
        self.fail(f"could not determine which benchmark generated {filename}")

    def test_at_least_one_yaml_was_generated_per_benchmark(self):
        self.assertTrue(self.generated, "no agentic YAML files were generated at all")
        for benchmark in AGENTIC_CSVS_AND_EXPECTED_CONTAINERS:
            matching = [p for p in self.generated if benchmark in p.name]
            self.assertTrue(matching, f"no generated YAML filename contains {benchmark!r}")

    def test_generated_yaml_parses_with_exactly_one_containers_key_and_expected_count(self):
        for path in self.generated:
            text = path.read_text()
            with self.subTest(file=path.name):
                doc = load_yaml_no_duplicates(text)  # raises on a duplicate key, e.g. two 'containers:'
                self.assertIsInstance(doc, dict)
                pod_spec = doc["spec"]["template"]["spec"]
                self.assertIn("containers", pod_spec)
                containers = pod_spec["containers"]
                self.assertIsInstance(containers, list)
                expected = self.expected_container_count(path.name)
                self.assertEqual(
                    len(containers), expected,
                    f"{path.name}: expected {expected} container(s), got "
                    f"{len(containers)} ({[c.get('name') for c in containers]})")

    def test_no_unresolved_placeholder_in_any_generated_yaml(self):
        placeholder_re = re.compile(r"@[a-zA-Z_]+@")
        for path in self.generated:
            with self.subTest(file=path.name):
                leftover = placeholder_re.findall(path.read_text())
                self.assertEqual(leftover, [], f"{path.name}: unresolved placeholder(s) {leftover}")

    def test_no_literal_secret_values_in_any_generated_yaml(self):
        for path in self.generated:
            text = path.read_text()
            with self.subTest(file=path.name):
                for secret in FORBIDDEN_LITERAL_SECRETS:
                    self.assertNotIn(secret, text, f"{path.name}: contains literal secret {secret!r}")

    @unittest.skipUnless(shutil.which("bash"), "bash not available on this system")
    def test_embedded_bash_scripts_are_syntactically_valid(self):
        """Extracts every container's args[0] block scalar (the actual bash
        script k8s will run) from every generated file and syntax-checks it
        with `bash -n` (parses only, never executes)."""
        for path in self.generated:
            doc = yaml.safe_load(path.read_text())
            containers = doc["spec"]["template"]["spec"]["containers"]
            for container in containers:
                script = container["args"][0]
                with self.subTest(file=path.name, container=container["name"]):
                    result = subprocess.run(["bash", "-n"], input=script, text=True,
                                            capture_output=True)
                    self.assertEqual(
                        result.returncode, 0,
                        f"{path.name} container {container['name']!r}: "
                        f"bash -n reported a syntax error:\n{result.stderr}")

    def test_mcp_atlas_secrets_are_referenced_by_name_not_inlined(self):
        """More targeted than the literal-string check above: the mcp-atlas
        sidecar's GITHUB_TOKEN/BRAVE_API_KEY must come from secretKeyRef, not
        a literal value: field."""
        mcp_atlas_files = [p for p in self.generated if "mcp-atlas" in p.name]
        self.assertTrue(mcp_atlas_files)
        for path in mcp_atlas_files:
            doc = yaml.safe_load(path.read_text())
            containers = doc["spec"]["template"]["spec"]["containers"]
            sidecar = next(c for c in containers if c["name"] == "mcp-atlas-sidecar")
            env_by_name = {e["name"]: e for e in sidecar["env"]}
            for name in ("GITHUB_TOKEN", "BRAVE_API_KEY"):
                with self.subTest(file=path.name, env=name):
                    self.assertIn("valueFrom", env_by_name[name])
                    self.assertIn("secretKeyRef", env_by_name[name]["valueFrom"])
                    self.assertNotIn("value", env_by_name[name])


class ResultsRepoDirRoundTripTests(unittest.TestCase):
    """C8.3: results_repo_dir output for each benchmark matches the spec's
    path convention and round-trips through
    postprocessing/aggregate_results.py:parse_run_path's 6 fixed levels."""

    @classmethod
    def setUpClass(cls):
        cls.utils = load_module_from_path("_pipeline_utils_under_test", PIPELINE / "utils.py")

    def assert_round_trips(self, params, expected_dataset_dir_value):
        """expected_dataset_dir_value is what parse_run_path's 'dataset' field
        should come back as (for agentic rows this is the bare benchmark
        string, since parse_dataset_dir's '<dataset>_<n>samples' regex does
        not match a benchmark name with no trailing '_Nsamples')."""
        full_dir = self.utils.results_repo_dir(params)
        family = "agentic" if self.utils.benchmark_family(params) == "agentic" else "moe"
        self.assertTrue(full_dir.startswith(f"{family}/"),
                         f"{full_dir!r} does not start with {family}/")

        # aggregate_results.py is invoked with --results_dir pointing at the
        # platform-level root (.../agentic or .../moe, NOT the repo root --
        # see the parse_run_path docstring and its 6-fixed-level comment), so
        # rel_parts for the round trip start after that leading segment.
        rel_dir = full_dir.split("/", 1)[1]
        rel_parts = tuple(rel_dir.split("/")) + ("20260727-1200", "metrics_x.json")

        parsed = parse_run_path(rel_parts)
        self.assertIsNotNone(parsed, f"parse_run_path returned None for {rel_parts!r}")
        self.assertEqual(parsed["platform"], params.get("platform", "eidf"))
        self.assertEqual(parsed["inference_engine"], params["inference_engine"])
        self.assertEqual(parsed["model"], params["model"].lower())
        self.assertEqual(parsed["dataset"], expected_dataset_dir_value)
        self.assertEqual(parsed["hw_type x num_hw"],
                          f"{params['gpu'].lower()}x{params['num_gpu']}")
        self.assertEqual(parsed["batch_size"], params["batch_size"])
        self.assertEqual(parsed["run_timestamp"], "20260727-1200")
        return full_dir

    def test_imo_answerbench_path_convention(self):
        params = {"family": "agentic", "benchmark": "imo-answerbench", "inference_engine": "sglang",
                  "model": "gpt-oss-120b", "gpu": "A100", "num_gpu": 2,
                  "batch_size": "default", "num_tasks": 100}
        full_dir = self.assert_round_trips(params, "imo-answerbench")
        self.assertEqual(
            full_dir,
            "agentic/eidf/sglang/gpt-oss-120b/imo-answerbench/a100x2/batch-size-default")

    def test_mcp_atlas_path_convention(self):
        params = {"family": "agentic", "benchmark": "mcp-atlas", "inference_engine": "vllm",
                  "model": "gpt-oss-120b", "gpu": "H100", "num_gpu": 2,
                  "batch_size": "default", "num_tasks": 60}
        full_dir = self.assert_round_trips(params, "mcp-atlas")
        self.assertEqual(
            full_dir,
            "agentic/eidf/vllm/gpt-oss-120b/mcp-atlas/h100x2/batch-size-default")

    def test_swe_bench_lite_path_convention(self):
        params = {"family": "agentic", "benchmark": "swe-bench-lite", "inference_engine": "sglang",
                  "model": "gpt-oss-120b", "gpu": "H200", "num_gpu": 1,
                  "batch_size": "default", "num_tasks": 100}
        full_dir = self.assert_round_trips(params, "swe-bench-lite")
        self.assertEqual(
            full_dir,
            "agentic/eidf/sglang/gpt-oss-120b/swe-bench-lite/h200x1/batch-size-default")

    def test_agentic_path_uses_explicit_platform_when_given(self):
        params = {"family": "agentic", "benchmark": "mcp-atlas", "inference_engine": "sglang",
                  "model": "gpt-oss-120b", "gpu": "H200", "num_gpu": 8,
                  "batch_size": "default", "num_tasks": 60, "platform": "vastai"}
        full_dir = self.assert_round_trips(params, "mcp-atlas")
        self.assertEqual(
            full_dir,
            "agentic/vastai/sglang/gpt-oss-120b/mcp-atlas/h200x8/batch-size-default")

    def test_moe_path_convention_unaffected(self):
        """The MoE branch must still produce the pre-existing convention
        (moe/<platform>/... with the dataset_<n>samples level), now taking
        platform from the parameters instead of a hardcoded 'eidf'."""
        params = {"family": "moe", "inference_engine": "sglang", "model": "gpt-oss-120b",
                  "dataset": "gsm8k", "num_samples": 256, "gpu": "A100",
                  "num_gpu": 1, "batch_size": "default",
                  "input_length": None, "output_length": None}
        full_dir = self.utils.results_repo_dir(params)
        self.assertEqual(
            full_dir,
            "moe/eidf/sglang/gpt-oss-120b/gsm8k_256samples/a100x1/batch-size-default")

        rel_parts = tuple(full_dir.split("/", 1)[1].split("/")) + ("20260101-0000", "metrics.json")
        parsed = parse_run_path(rel_parts)
        self.assertEqual(parsed["dataset"], "gsm8k")
        self.assertEqual(parsed["num_samples"], "256")
        self.assertEqual(parsed["batch_size"], "default")


class LoginNodeDriverGenerationTests(unittest.TestCase):
    """The generated login-node driver (swe-bench-lite on k8s) must be runnable,
    not merely syntactically valid.

    generate.py substitutes @tokens@ everywhere in a template, comments
    included. A token whose value spans multiple lines therefore turns a
    one-line comment into one commented line plus live continuation lines --
    valid shell, so `bash -n` passes, but the script dies at run time on the
    first variable those lines reference. That shipped once and aborted a run
    with `line 67: RUN_DIR: unbound variable`.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(dir=str(REPO))
        cls.out_dir = Path(cls._tmp.name)
        run_generate(PIPELINE, EXPERIMENTS / "swe-bench-lite-eidf.csv", cls.out_dir)
        cls.drivers = sorted(cls.out_dir.glob("*.sh"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_drivers_were_generated(self):
        self.assertTrue(self.drivers, "no driver .sh generated for swe-bench-lite on k8s")

    def test_real_image_preflight_gates_the_run_before_the_engine_starts(self):
        """The sandbox pod command runs nowhere but inside a pod, so a mistake
        in it is invisible until every task has burned its 600s sandbox
        timeout. Only --real-image exercises it: the default preflight swaps in
        a busybox httpd and never runs the swe-rex install at all. The gate
        must also sit ahead of section [2], or a failure costs GPU time."""
        for driver in self.drivers:
            with self.subTest(driver=driver.name):
                body = driver.read_text()
                self.assertIn("preflight_portforward.py", body)
                self.assertIn("--real-image", body)
                self.assertIn("SKIP_PREFLIGHT", body)
                self.assertLess(body.index("preflight_portforward.py"),
                                body.index("[2] Starting the engine"),
                                "preflight must gate the run before the GPU job")

    def test_run_inputs_are_snapshotted_before_anything_can_fail(self):
        """This script and the engine manifest are what a failed run has to be
        reconstructed from, so they are copied into $RUN_DIR as soon as it
        exists. Doing it at publish time instead put them out of reach of
        exactly the runs that need them: section [5] is gated on the
        completeness gate passing, on --push, and on the results-repo clone
        succeeding, and a run that trips any of those keeps nothing."""
        for driver in self.drivers:
            with self.subTest(driver=driver.name):
                body = driver.read_text()
                for cp in ('cp "${BASH_SOURCE[0]}" "$RUN_DIR/"',
                           'cp "$ENGINE_MANIFEST" "$RUN_DIR/"'):
                    self.assertEqual(body.count(cp), 1,
                                     f"expected exactly one {cp}")
                    at = body.index(cp)
                    self.assertGreater(at, body.index('mkdir -p "$RUN_DIR"'),
                                       "cannot copy before the directory exists")
                    for section in ("[2] Starting the engine",
                                    "[3] Completeness gate",
                                    "[5] Results"):
                        self.assertLess(at, body.index(section),
                                        f"snapshot must precede {section}")

    def test_benign_teas_warning_is_annotated_without_hiding_the_exit_status(self):
        """AgentCAP prints a TEAS-writer warning on every attempt of this path
        that cannot succeed and does not matter -- the driver writes the leaf
        itself after the gate. Six bare warnings make a good run read as a
        failed one, so the driver annotates them. The filter must not swallow
        the client's exit status, which drives the whole retry loop."""
        for driver in self.drivers:
            with self.subTest(driver=driver.name):
                body = driver.read_text()
                start = body.index("| awk '")
                block = body[start:body.index("| tee -a", start)]
                self.assertIn("NOTE: expected on the swebench-k8s path", block)
                # RC must still be read from the client, not from awk or tee.
                self.assertIn("RC=${PIPESTATUS[0]}", body)

                # Behavioural: the rendered filter annotates the warning and
                # leaves the upstream exit status intact.
                harness = (
                    'f() { printf "A\\n'
                    'WARNING: TEAS output writing failed: SWE-bench quality is not available x\\n'
                    'B\\n"; return 7; }\n'
                    'f 2>&1 \\\n  ' + block + '| cat\n'
                    'echo "RC=${PIPESTATUS[0]}"\n')
                proc = subprocess.run(["bash", "-c", harness],
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn("NOTE: expected on the swebench-k8s path", proc.stdout)
                self.assertIn("RC=7", proc.stdout,
                              "the annotation filter must not mask the client's exit code")
                self.assertIn("B", proc.stdout, "non-matching lines must pass through")

    def test_retry_loop_keeps_going_and_stops_on_no_progress(self):
        """With MAX_ATTEMPTS at 2 the loop gets exactly one retry, which
        clears only part of the infrastructure backlog and leaves the rest to
        be caught by the completeness gate. The loop has to keep going while it
        is still shrinking the retry list -- and stop when it is not, so a
        cluster dropping tunnels faster than tasks finish cannot burn every
        attempt for nothing."""
        for driver in self.drivers:
            with self.subTest(driver=driver.name):
                body = driver.read_text()
                self.assertIn('MAX_ATTEMPTS="${MAX_ATTEMPTS:-50}"', body)
                self.assertIn("PREV_RETRY_COUNT", body)
                self.assertIn(
                    'if [ -n "$PREV_RETRY_COUNT" ] && '
                    '[ "$RETRY_COUNT" -ge "$PREV_RETRY_COUNT" ]; then',
                    body)

    def test_driver_runs_to_arg_parsing(self):
        """`--help` exits 0, which means everything above the arg loop ran
        under `set -u`. That is the region placeholder expansion corrupts, and
        it needs no cluster, no venv and no network to exercise."""
        env = dict(os.environ,
                   TEASBENCH_ENV_PREFIX="/nonexistent",
                   TEASBENCH_ROOT="/nonexistent",
                   AGENTCAP_DIR="/nonexistent",
                   SWEAGENT_DIR="/nonexistent")
        for driver in self.drivers:
            with self.subTest(driver=driver.name):
                r = subprocess.run(["bash", str(driver), "--help"],
                                   text=True, capture_output=True, env=env)
                self.assertEqual(
                    r.returncode, 0,
                    f"{driver.name} --help exited {r.returncode}\n"
                    f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}")

    def test_no_multiline_placeholder_left_inside_a_comment(self):
        """Direct check on the cause: a generated comment line must not be
        followed by continuation lines that the template had inside it."""
        for driver in self.drivers:
            with self.subTest(driver=driver.name):
                lines = driver.read_text().splitlines()
                for i, line in enumerate(lines[:-1]):
                    if line.lstrip().startswith("#") and line.rstrip().endswith("\\"):
                        nxt = lines[i + 1].strip()
                        self.assertTrue(
                            nxt.startswith("#") or not nxt,
                            f"{driver.name}:{i + 1} a comment ends in a line "
                            f"continuation and the next line is live code "
                            f"({nxt!r}) -- a multi-line @token@ was expanded "
                            f"inside a comment")


class ReplicationStudyTests(unittest.TestCase):
    """The controlled repeatability / engine-build study: the 72-leaf CSV
    is balanced as the study design freezes it, study rows pin their engine build
    into the image tag, and the study-<block> ingestion marker level survives
    the parse_run_path round trip without breaking timestamp parsing."""

    STUDY_CSV = EXPERIMENTS / "replication-study-eidf.csv"

    @classmethod
    def setUpClass(cls):
        cls.utils = load_module_from_path("_pipeline_utils_study", PIPELINE / "utils.py")
        with open(cls.STUDY_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        cls.rows = rows

        # Generate one block from each hardware stratum end-to-end via the launcher's own CSV filter
        # (awk column 10 + sed ref-fill), so a CSV column change that breaks
        # run_study_block.sh breaks this test too.
        cls.tmp = tempfile.mkdtemp()
        cls.generated_by_block = {}
        for block in ("E1", "E4"):
            block_csv = Path(cls.tmp) / f"{block.lower()}.csv"
            subprocess.run(
                f"awk -F, 'NR==1 || $10==\"{block}\"' " + str(cls.STUDY_CSV)
                + " | sed 's/,$/,abc1234/' > " + str(block_csv),
                shell=True, check=True)
            outdir = Path(cls.tmp) / f"out-{block.lower()}"
            run_generate(PIPELINE, block_csv, outdir)
            cls.generated_by_block[block] = sorted(outdir.glob("*.yaml"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def leaves(self, block):
        return sorted((r for r in self.rows if r["study_block"] == block),
                      key=lambda r: int(r["study_order"]))

    def launcher_environment(self, root, dirty=False, kubectl_script=None):
        root = Path(root)
        bin_dir = root / "bin"
        bin_dir.mkdir(exist_ok=True)
        git_stub = bin_dir / "git"
        dirty_output = 'echo " M pipeline/utils.py"' if dirty else ":"
        git_stub.write_text(
            "#!/bin/sh\n"
            "case \"$*\" in\n"
            f"  *\"status --porcelain\"*) {dirty_output} ;;\n"
            "  *\"rev-parse --short HEAD\"*) echo aaaaaaa ;;\n"
            "  *\"rev-parse HEAD\"*) echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;\n"
            "  *\"ls-remote\"*) printf 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\\trefs/heads/main\\n' ;;\n"
            "  *) echo \"unexpected git call: $*\" >&2; exit 91 ;;\n"
            "esac\n")
        git_stub.chmod(0o755)
        curl_stub = bin_dir / "curl"
        curl_stub.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' "
            "'{\"digest\":\"sha256:1111111111111111111111111111111111111111111111111111111111111111\"}'\n")
        curl_stub.chmod(0o755)
        if kubectl_script is not None:
            kubectl_stub = bin_dir / "kubectl"
            kubectl_stub.write_text(kubectl_script)
            kubectl_stub.chmod(0o755)
            sleep_stub = bin_dir / "sleep"
            sleep_stub.write_text("#!/bin/sh\nexit 0\n")
            sleep_stub.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
        env["STUDY_STATE_DIR"] = str(root / "state")
        env["TMPDIR"] = str(root)
        env["TMUX"] = "test"
        return env

    def run_study_launcher(self, root, *args, dirty=False, kubectl_script=None):
        return subprocess.run(
            [str(STUDY_LAUNCHER), *args], cwd=str(REPO),
            env=self.launcher_environment(
                root, dirty=dirty, kubectl_script=kubectl_script),
            text=True, capture_output=True)

    @staticmethod
    def run_guard(*args):
        return subprocess.run(
            [sys.executable, str(STUDY_GUARD), *map(str, args)],
            text=True, capture_output=True)

    @staticmethod
    def write_complete_study_manifest(state, block):
        state = Path(state)
        receipts = state / "receipts"
        receipts.mkdir(parents=True, exist_ok=True)
        submitted_yamls = state / "submitted-yamls"
        submitted_yamls.mkdir(parents=True, exist_ok=True)
        with open(EXPERIMENTS / "replication-study-eidf.csv", newline="") as handle:
            coordinates = {
                int(row["study_order"]): row for row in csv.DictReader(handle)
                if row["study_block"] == block
            }
        records = []
        for order in range(1, 13):
            coordinate = coordinates[order]
            job = f"study-{block.lower()}-{order}"
            job_uid = f"00000000-0000-4000-8000-{order:012d}"
            image_ref = "example.invalid/engine@sha256:" + "f" * 64
            yaml_path = submitted_yamls / f"{job}.yaml"
            yaml_path.write_text(yaml.safe_dump({
                "apiVersion": "batch/v1", "kind": "Job",
                "metadata": {"name": job},
                "spec": {"template": {"spec": {"containers": [
                    {"name": "server", "image": image_ref}]}}},
            }, sort_keys=False))
            yaml_hash = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
            publish_path = f"moe/eidf/study-{block.lower()}/run-{order}"
            output_path = f"/mnt/develop/archive/{publish_path}"
            artifact_keys = [
                "metadata", "metrics", "launch_yaml", "detailed_results",
                "output_data", "timings", "pip_freeze"]
            if coordinate["inference_engine"] == "sglang":
                artifact_keys.append("expert_distribution_bundle")
            hashes = {key: str(order % 10) * 64 for key in artifact_keys}
            hashes["launch_yaml"] = yaml_hash
            receipt = {
                "receipt_version": 1, "status": "validated",
                "study_id": "controlled-variation-2026-x2",
                "block": block.lower(), "planned_order": order,
                "job": job, "job_uid": job_uid,
                "node": "gpu-a", "output_path": output_path,
                "publish_path": publish_path, "publication": "development",
                "inference_engine": coordinate["inference_engine"],
                "engine_version": coordinate["engine_version"],
                "dataset": coordinate["dataset"],
                "teasbench_commit": "a" * 7, "moe_cap_commit": "e" * 7,
                "image_ref": image_ref,
                "metadata_sha256": hashes["metadata"],
                "metrics_sha256": hashes["metrics"],
                "job_yaml_sha256": hashes["launch_yaml"],
                "artifact_sha256": hashes,
                "quality": {"total": 256, "attempted": 256,
                            "served": 256, "completed": 256},
            }
            receipt_path = receipts / f"{job}.json"
            receipt_path.write_text(json.dumps(receipt))
            records.append({
                "study_id": "controlled-variation-2026-x2", "block": block,
                "planned_order": order, "outcome": "complete", "node": "gpu-a",
                "image_id": "docker-pullable://example.invalid/engine@sha256:" + "f" * 64,
                "job": job, "job_uid": receipt["job_uid"],
                "yaml": yaml_path.name, "yaml_path": str(yaml_path),
                "yaml_sha256": yaml_hash,
                "teasbench_commit": "a" * 40, "moe_cap_ref": "e" * 40,
                "output_path": output_path, "receipt_path": str(receipt_path),
                "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            })
        (state / f"manifest-{block}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records))

    def test_matrix_is_the_frozen_72_leaves(self):
        """6 blocks x 4 arms x 3 datasets, split 3/3 across A100x2/H100x2."""
        self.assertEqual(len(self.rows), 72)
        arms = {("vllm", "0.16.0"), ("vllm", "0.21.0"),
                ("sglang", "0.5.9"), ("sglang", "0.5.12.post1")}
        datasets = {"gsm8k", "arena-hard", "longbench_v1"}
        for block in ("E1", "E2", "E3", "E4", "E5", "E6"):
            leaves = self.leaves(block)
            self.assertEqual(len(leaves), 12, block)
            self.assertEqual([int(r["study_order"]) for r in leaves], list(range(1, 13)))
            combos = {(r["inference_engine"], r["engine_version"], r["dataset"])
                      for r in leaves}
            self.assertEqual(combos, {(e, v, d) for (e, v) in arms for d in datasets})
            expected_gpu = "A100" if block in ("E1", "E2", "E3") else "H100"
            for r in leaves:
                self.assertEqual(
                    (r["family"], r["model"], r["num_samples"], r["gpu"],
                     r["num_gpu"], r["batch_size"]),
                    ("moe", "gpt-oss-120b", "256", expected_gpu, "2", "default"))

    def test_study_publish_does_not_push_or_succeed_after_commit_failure(self):
        document = yaml.safe_load(self.generated_by_block["E1"][0].read_text())
        script = document["spec"]["template"]["spec"]["containers"][0]["args"][0]
        start = script.index('if git -C "$CLONE" diff --cached --quiet; then')
        end = script.index('rm -rf "$CLONE"', start)
        publish_block = script[start:end]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            trace = root / "git-trace"
            publish_state = root / "publish-state"
            git_stub = bin_dir / "git"
            git_stub.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *\" diff --cached --quiet\"*) exit 1 ;;\n"
                f"  *\" commit \"*) echo commit >> '{trace}'; exit 42 ;;\n"
                f"  *\" push \"*) echo push >> '{trace}'; exit 0 ;;\n"
                "  *) exit 0 ;;\n"
                "esac\n")
            git_stub.chmod(0o755)
            harness = (
                "STUDY_PUBLISH_OK=0\n"
                f"CLONE='{root / 'clone'}'\n"
                "PUBLISH_SUBDIR='moe/eidf/study-e1/run'\n"
                f"PVC_RUN_OUTPUT_DIR='{root / 'pvc-run'}'\n"
                + publish_block
                + f"\nprintf '%s' \"$STUDY_PUBLISH_OK\" > '{publish_state}'\n")
            env = os.environ.copy()
            env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
            proc = subprocess.run(
                ["bash"], input=harness, env=env, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(trace.read_text().splitlines(), ["commit"])
            self.assertEqual(publish_state.read_text(), "0")
            self.assertIn("git commit failed", proc.stdout)
            self.assertNotIn("RESULTS PUBLISHED", proc.stdout)

    def test_order_balance_and_pair_adjacency(self):
        """Control/alternate legs of the same engine x dataset run
        back-to-back (so a truncated block still yields complete pairs), and
        engine order / endpoint order are each 3/3 across the six blocks."""
        control = {"vllm": "0.16.0", "sglang": "0.5.9"}
        first_engines, first_endpoints = [], []
        for block in ("E1", "E2", "E3", "E4", "E5", "E6"):
            leaves = self.leaves(block)
            for i in range(0, 12, 2):
                a, b = leaves[i], leaves[i + 1]
                self.assertEqual(a["inference_engine"], b["inference_engine"], block)
                self.assertEqual(a["dataset"], b["dataset"], block)
                self.assertNotEqual(a["engine_version"], b["engine_version"], block)
            first_engines.append(leaves[0]["inference_engine"])
            first_endpoints.append(
                "control" if leaves[0]["engine_version"] ==
                control[leaves[0]["inference_engine"]] else "alternate")
        self.assertEqual(sorted(first_engines), ["sglang"] * 3 + ["vllm"] * 3)
        self.assertEqual(sorted(first_endpoints), ["alternate"] * 3 + ["control"] * 3)

    def test_generated_yaml_pins_build_hardware_marker_and_producer(self):
        """Each generated leaf carries its row's exact image tag, the
        study path, TP2 command and allocation, MoE-CAP checkout pin, and the
        fresh x2 study identity -- and its job name stays under the k8s label
        limit. E1 and E4 exercise both hardware strata."""
        base = {"vllm": "vllm/vllm-openai", "sglang": "lmsysorg/sglang"}
        products = {"A100": "NVIDIA-A100-SXM4-80GB",
                    "H100": "NVIDIA-H100-80GB-HBM3"}
        tp_flags = {"vllm": "--tensor-parallel-size 2", "sglang": "--tp-size 2"}
        for block in ("E1", "E4"):
            generated = self.generated_by_block[block]
            self.assertEqual(len(generated), 12)
            by_name = {p.name: p.read_text() for p in generated}
            for row in self.leaves(block):
                stem = None
                for name, text in by_name.items():
                    if (f'engine_version: "{row["engine_version"]}"' in text
                            and f'--datasets {row["dataset"]}' in text
                            and f'planned_order: {row["study_order"]},' in text):
                        stem, yaml_text = name, text
                        break
                self.assertIsNotNone(stem, f"no YAML for row {row}")
                self.assertIn(f'image: {base[row["inference_engine"]]}:'
                              f'v{row["engine_version"]}', yaml_text)
                self.assertNotIn(f'v{row["engine_version"]}-cu130', yaml_text)
                self.assertIn(tp_flags[row["inference_engine"]], yaml_text)
                self.assertIn(f'nvidia.com/gpu.product: {products[row["gpu"]]}',
                              yaml_text)
                self.assertEqual(yaml_text.count("nvidia.com/gpu: 2"), 1)
                self.assertIn(f"/batch-size-default/study-{block.lower()}/$timestamp",
                              yaml_text)
                # Pin must fail the run if the checkout fails, not fall back to main.
                self.assertIn("git checkout --quiet --detach abc1234 || ", yaml_text)
                self.assertIn('study_id: "controlled-variation-2026-x2"', yaml_text)
                self.assertIn(f'block_id: "{block.lower()}"', yaml_text)
                self.assertIn("pip freeze > $PVC_RUN_OUTPUT_DIR/pip_freeze.txt", yaml_text)
                self.assertIn("arena_baseline_sha256", yaml_text)
                parsed_job = load_yaml_no_duplicates(yaml_text)
                env_entries = parsed_job["spec"]["template"]["spec"][
                    "containers"][0]["env"]
                self.assertEqual(
                    [entry["name"] for entry in env_entries].count("k8s_job_uid"), 1)
                generate_name = re.search(r"generateName: (\S+)", yaml_text).group(1)
                self.assertLessEqual(len(generate_name) + 5, 63, generate_name)

    def test_launcher_dry_run_executes_successfully_with_diagnostic_options(self):
        """Exercise the launcher through its /bin/bash shebang. In particular,
        this proves YAML collection does not depend on Bash-4 mapfile."""
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.run_study_launcher(
                tmp, "E1", "--dry-run", "--no-pin",
                "--results-repo", "Scratch_Results")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("Planned order for E1:", proc.stdout)
            self.assertIn("Dry run: nothing submitted", proc.stdout)
            self.assertEqual(proc.stdout.count(".yaml"), 12)

    def test_preflight_runner_dry_run_generates_exact_excluded_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            collector = load_yaml_no_duplicates(
                STUDY_PREFLIGHT_COLLECTOR.read_text())
            collector_command = collector["spec"]["containers"][0]["command"]
            self.assertIn("while true", collector_command[-1])
            self.assertNotIn("86400", collector_command[-1])

            proc = subprocess.run(
                [str(STUDY_PREFLIGHT_RUNNER), "--dry-run"], cwd=str(REPO),
                env=self.launcher_environment(tmp), text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("Dry run: four YAMLs generated", proc.stdout)
            yamls = list(Path(tmp).glob("teas-preflight-*/*.yaml"))
            self.assertEqual(len(yamls), 4)
            combinations = set()
            for yaml_path in yamls:
                text = yaml_path.read_text()
                self.assertIn("/compatibility-preflight/$timestamp", text)
                self.assertNotIn("/study-e1/$timestamp", text)
                self.assertIn("STUDY_PREFLIGHT=1", text)
                self.assertIn(".compatibility_preflight =", text)
                self.assertRegex(text, r"image: \S+@sha256:[0-9a-f]{64}")
                engine = re.search(r'STUDY_ENGINE="([^"]+)"', text).group(1)
                version = re.search(r'STUDY_ENGINE_VERSION="([^"]+)"', text).group(1)
                combinations.add((engine, version))
            self.assertEqual(combinations, {
                ("vllm", "0.16.0"), ("vllm", "0.21.0"),
                ("sglang", "0.5.9"), ("sglang", "0.5.12.post1")})
            self.assertFalse(
                (Path(tmp) / "state" / "a100x2-compatibility-preflight.jsonl").exists())

    def test_preflight_runner_one_command_collects_through_live_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kubectl_impl = root / "mock_kubectl.py"
            kubectl_impl.write_text(r'''#!/usr/bin/env python3
import hashlib, json, os, re, shutil, sys
from pathlib import Path
import yaml

root = Path(os.environ["MOCK_KUBE_ROOT"])
state_path = root / "kube-state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"count": 0}
args = sys.argv[1:]
joined = " ".join(args)

def save():
    state_path.write_text(json.dumps(state))

if " cp " in f" {joined} ":
    (root / "forbidden-kubectl-cp").write_text("called")
    raise SystemExit(97)
if "create" in args:
    yaml_path = Path(args[args.index("-f") + 1])
    if yaml_path.name == "study_preflight_collector.yaml":
        print("pod/mock-preflight-collector")
        raise SystemExit(0)
    document = yaml.safe_load(yaml_path.read_text())
    state["count"] += 1
    count = state["count"]
    generate_name = document["metadata"]["generateName"]
    job = f"{generate_name}{count:05d}"
    uid = f"11111111-1111-4111-8111-{count:012d}"
    pod = f"preflight-pod-{count}"
    container = document["spec"]["template"]["spec"]["containers"][0]
    script = container["args"][0]
    engine = re.search(r'STUDY_ENGINE="([^"]+)"', script).group(1)
    version = re.search(r'STUDY_ENGINE_VERSION="([^"]+)"', script).group(1)
    teas = re.search(r'TEASBENCH_COMMIT="([0-9a-f]{40})"', script).group(1)
    image_ref = container["image"]
    image_tag = re.search(r'STUDY_IMAGE_TAG="([^"]+)"', script).group(1)
    pod_dir = f"/mnt/develop/outputs/mock/compatibility-preflight/run-{count}"
    local_dir = root / "pvc" / pod_dir.removeprefix("/mnt/develop/")
    local_dir.mkdir(parents=True)
    gpu_uuids = [f"GPU-{count}-a", f"GPU-{count}-b"]
    metadata = {
        "model_config": {"model_name": "unsloth/gpt-oss-120b"},
        "hardware": {"num_gpus": 2, "gpu_type": "NVIDIA-A100-SXM4-80GB"},
        "system_environment": {
            "inference_engine": engine, "inference_engine_version": version,
            "teasbench_commit": teas, "moe_cap_commit": "e" * 7},
        "compatibility_preflight": {
            "dataset": "longbench_v1", "num_samples": 256,
            "batch_size": "default", "gpu": "A100", "num_gpu": 2,
            "node": "gpu-a", "job_name": job, "job_uid": uid,
            "gpu_uuids": gpu_uuids, "image_ref": image_ref}}
    metrics = {"quality": {"total": 256, "attempted": 256,
                           "served": 256, "completed": 256}}
    metadata_path = local_dir / "metadata.json"
    metrics_path = local_dir / "metrics.json"
    metadata_path.write_text(json.dumps(metadata))
    metrics_path.write_text(json.dumps(metrics))
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt = {
        "study_id": "controlled-variation-2026-x2",
        "kind": "excluded-compatibility-preflight",
        "inference_engine": engine, "engine_version": version,
        "gpu": "A100", "num_gpu": 2, "dataset": "longbench_v1",
        "num_samples": 256, "batch_size": "default", "outcome": "complete",
        "teasbench_commit": teas, "moe_cap_ref": "e" * 40,
        "image_tag": image_tag, "image_ref": image_ref,
        "job_uid": uid, "job_name": job, "node": "gpu-a",
        "gpu_uuids": gpu_uuids, "completed_at": "2026-08-17T12:00:00Z",
        "artifact_dir": pod_dir, "metadata_sha256": sha(metadata_path),
        "metrics_sha256": sha(metrics_path), "job_yaml_sha256": sha(yaml_path)}
    state.update({"job": job, "uid": uid, "pod": pod, "receipt": receipt})
    save()
    print(f"{job}\t{uid}")
    raise SystemExit(0)
if "wait" in args or "delete" in args:
    raise SystemExit(0)
if "exec" in args:
    source = Path(args[-1])
    local = root / "pvc" / str(source).removeprefix("/mnt/develop/")
    sys.stdout.buffer.write(local.read_bytes())
    raise SystemExit(0)
if "get" in args:
    if "{.metadata.uid}" in joined:
        print(state["uid"])
    elif "{.items[0].metadata.name}" in joined:
        print(state["pod"])
    elif "status.phase" in joined:
        print("Succeeded")
    elif "Complete" in joined:
        print("True")
    elif "Failed" in joined:
        pass
    elif "terminated.message" in joined:
        print(json.dumps(state["receipt"], separators=(",", ":")))
    raise SystemExit(0)
raise SystemExit(f"unexpected kubectl call: {joined}")
''')
            kubectl_script = f'''#!/bin/sh
exec python3 '{kubectl_impl}' "$@"
'''
            shared = root / "job-configs"
            shared.mkdir()
            env = self.launcher_environment(tmp, kubectl_script=kubectl_script)
            env["STUDY_JOB_CONFIGS_DIR"] = str(shared)
            env["MOCK_KUBE_ROOT"] = str(root)
            proc = subprocess.run(
                [str(STUDY_PREFLIGHT_RUNNER)], cwd=str(REPO), env=env,
                text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            manifest = root / "state" / "a100x2-compatibility-preflight.jsonl"
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertEqual(
                {(r["inference_engine"], r["engine_version"]) for r in records},
                {("vllm", "0.16.0"), ("vllm", "0.21.0"),
                 ("sglang", "0.5.9"), ("sglang", "0.5.12.post1")})
            self.assertTrue(
                (root / "state" /
                 "a100x2-compatibility-preflight.validated.json").is_file())
            self.assertFalse((root / "forbidden-kubectl-cp").exists())
            for record in records:
                job_yaml = next(Path(record["artifact_dir"]).glob("*.yaml"))
                document = load_yaml_no_duplicates(job_yaml.read_text())
                self.assertIn("generateName", document["metadata"])
                self.assertNotIn("name", document["metadata"])
                self.assertTrue(record["job_name"].startswith(
                    document["metadata"]["generateName"]))

    def test_launcher_rejects_unsafe_actual_launch_options(self):
        cases = (
            (("E1", "--no-pin"), "--no-pin is allowed only with --dry-run"),
            (("E1", "--skip-image-check"),
             "--skip-image-check is allowed only with --dry-run"),
            (("E1", "--results-repo", "TEAS_Results_Private"),
             "study runs must write to TEAS_Development_Results_Private"),
        )
        for index, (arguments, message) in enumerate(cases):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as tmp:
                proc = self.run_study_launcher(tmp, *arguments)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(message, proc.stderr)

    def test_launcher_reconcile_rejects_leaf_without_ambiguous_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_resume_state(tmp)
            kubectl_marker = Path(tmp) / "unexpected-kubectl"
            kubectl_script = f'''#!/bin/sh
touch '{kubectl_marker}'
exit 0
'''
            proc = self.run_study_launcher(
                tmp, "E4", "--only", "2", "--reconcile-identity",
                kubectl_script=kubectl_script)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("has no ambiguous identity to reconcile", proc.stderr)
            self.assertFalse(kubectl_marker.exists())

            submitted = self.run_study_launcher(
                tmp, "E4", "--only", "1", "--reconcile-identity",
                kubectl_script=kubectl_script)
            self.assertNotEqual(submitted.returncode, 0)
            self.assertIn(
                "has no ambiguous identity to reconcile", submitted.stderr)
            self.assertFalse(kubectl_marker.exists())

    def test_launcher_rejects_dirty_actual_launch_but_not_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.run_study_launcher(tmp, "E1", dirty=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("working tree is dirty", proc.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.run_study_launcher(
                tmp, "E1", "--dry-run", "--skip-image-check", dirty=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("dry-run output is verification-only", proc.stdout)

    def test_launcher_enforces_every_first_launch_predecessor(self):
        schedule = ("E1", "E4", "E2", "E5", "E3", "E6")
        for index, block in enumerate(schedule[1:], 1):
            predecessor = schedule[index - 1]
            with self.subTest(block=block), tempfile.TemporaryDirectory() as tmp:
                state = Path(tmp) / "state"
                state.mkdir()
                for earlier in schedule[:index - 1]:
                    self.write_complete_study_manifest(state, earlier)
                proc = self.run_study_launcher(tmp, block)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(
                    f"{block} is blocked until {predecessor} has 12 successful leaves",
                    proc.stderr)

    def test_launcher_rejects_out_of_order_later_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            state.mkdir()
            self.write_complete_study_manifest(state, "E4")
            proc = self.run_study_launcher(tmp, "E1")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(
                "state is out of order: E4 has records before first launch of E1",
                proc.stderr)

    def test_launcher_preserves_state_on_commit_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            state.mkdir()
            (state / "teasbench_commit").write_text("old1234\n")
            (state / "moe_cap_ref").write_text("e" * 40 + "\n")
            proc = self.run_study_launcher(tmp, "E1")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Preserve", proc.stderr)
            self.assertIn("move the state directory aside", proc.stderr)
            self.assertNotIn("delete", proc.stderr.lower())
            self.assertTrue((state / "teasbench_commit").exists())

    def test_launcher_rejects_ambiguous_repeat_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state"
            state.mkdir()
            records = [
                {"study_id": "controlled-variation-2026-x2", "block": "E1",
                 "planned_order": 1, "outcome": "failed", "node": "gpu-a"},
                {"study_id": "controlled-variation-2026-x2", "block": "E1",
                 "planned_order": 1, "outcome": "complete", "node": "gpu-b"},
            ]
            (state / "manifest-E1.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            proc = self.run_study_launcher(tmp, "E1", "--only", "1")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("records multiple nodes", proc.stderr)
            self.assertIn("cannot recover a unique node", proc.stderr)

    def test_launcher_starts_e1_without_compatibility_preflight_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_marker = Path(tmp) / "e1-create-reached"
            kubectl_script = f'''#!/bin/sh
case "$*" in
  *" create "*) touch '{create_marker}'; exit 72 ;;
  *) exit 91 ;;
esac
'''
            proc = self.run_study_launcher(
                tmp, "E1", kubectl_script=kubectl_script)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("create failed before Kubernetes returned", proc.stderr)
            self.assertTrue(create_marker.exists())
            self.assertNotIn("compatibility-preflight", proc.stderr)
            state = Path(tmp) / "state"
            pins = (state / "image-digests.tsv").read_text().splitlines()
            self.assertEqual(len(pins), 4)
            self.assertTrue(all("\tsha256:" in line for line in pins))
            self.assertFalse(
                (state / "a100x2-compatibility-preflight.validated.json").exists())

    def test_launcher_rejects_retired_preflight_evidence_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.run_study_launcher(
                tmp, "E1", "--preflight-evidence", "obsolete.jsonl")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("unknown option --preflight-evidence", proc.stdout)
            self.assertFalse((Path(tmp) / "state").exists())

    def test_launcher_validates_timeout_before_creating_state(self):
        invalid_values = ("0", "169", "999", "+1", "1h", "1+2",
                          "1;touch-should-never-run")
        for value in invalid_values:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                proc = self.run_study_launcher(
                    tmp, "E1", "--dry-run", "--leaf-timeout-hours", value)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("decimal integer from 1 to 168", proc.stderr)
                self.assertFalse((Path(tmp) / "state").exists())

    def test_launcher_exclusive_lock_rejects_a_second_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "state" / ".launcher.lock"
            lock.mkdir(parents=True)
            (lock / "owner").write_text("pid=123\n")
            proc = self.run_study_launcher(
                tmp, "E1", "--dry-run", "--skip-image-check")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("launcher lock is held", proc.stderr)
            self.assertTrue(lock.is_dir())

    def make_resume_state(self, root, *, served=256):
        root = Path(root)
        state = root / "state"
        state.mkdir()
        (state / "teasbench_commit").write_text("a" * 40 + "\n")
        (state / "moe_cap_ref").write_text("e" * 40 + "\n")
        digest = "sha256:" + "1" * 64
        (state / "image-digests.tsv").write_text(
            "".join(f"{tag}\t{digest}\n" for tag in (
                "lmsysorg/sglang:v0.5.12.post1", "lmsysorg/sglang:v0.5.9",
                "vllm/vllm-openai:v0.16.0", "vllm/vllm-openai:v0.21.0")))
        job = "study-e4-sglang-059-gptoss120b-gsm8k-h100x2-abcde"
        uid = "11111111-1111-4111-8111-111111111111"
        yaml_name = "study-e4-sglang-059-gptoss120b-gsm8k-h100x2.yaml"
        submitted_dir = state / "submitted-yamls"
        submitted_dir.mkdir()
        submitted_yaml = submitted_dir / f"{job}.yaml"
        submitted_yaml.write_text(yaml.safe_dump({
            "apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": job},
            "spec": {"template": {"spec": {"containers": [{
                "name": "server", "image": "lmsysorg/sglang@" + digest}]}}},
        }, sort_keys=False))
        submitted_yaml_hash = hashlib.sha256(submitted_yaml.read_bytes()).hexdigest()
        submitted = {
            "study_id": "controlled-variation-2026-x2", "block": "E4",
            "planned_order": 1, "job": job, "job_uid": uid,
            "yaml": yaml_name, "yaml_path": str(submitted_yaml),
            "yaml_sha256": submitted_yaml_hash,
            "node": "gpu-h", "image_id": "",
            "moe_cap_ref": "e" * 40, "teasbench_commit": "a" * 40,
            "submitted_at": "2026-08-17T10:00:00Z", "finished_at": "",
            "outcome": "submitted", "receipt_path": "",
            "receipt_sha256": "", "output_path": "",
        }
        (state / "manifest-E4.jsonl").write_text(json.dumps(submitted) + "\n")
        publish_path = "moe/eidf/sglang/gpt-oss-120b/gsm8k_256samples/" \
                       "h100x2/batch-size-default/study-e4/run"
        hashes = {key: "2" * 64 for key in (
            "metadata", "metrics", "launch_yaml", "detailed_results",
            "output_data", "timings", "pip_freeze",
            "expert_distribution_bundle")}
        hashes["launch_yaml"] = submitted_yaml_hash
        receipt = {
            "receipt_version": 1, "status": "validated",
            "study_id": "controlled-variation-2026-x2", "block": "e4",
            "planned_order": 1, "job": job, "job_uid": uid, "node": "gpu-h",
            "output_path": "/mnt/develop/archive/" + publish_path,
            "publish_path": publish_path, "publication": "development",
            "inference_engine": "sglang", "engine_version": "0.5.9",
            "dataset": "gsm8k",
            "teasbench_commit": "a" * 7, "moe_cap_commit": "e" * 7,
            "image_ref": "lmsysorg/sglang@" + digest,
            "metadata_sha256": hashes["metadata"],
            "metrics_sha256": hashes["metrics"],
            "job_yaml_sha256": hashes["launch_yaml"],
            "artifact_sha256": hashes,
            "quality": {"total": 256, "attempted": 256,
                        "served": served, "completed": served},
        }
        receipt_source = root / "mock-receipt.json"
        receipt_source.write_text(json.dumps(receipt))
        return state, receipt_source

    def make_ambiguous_resume_state(self, root):
        state, receipt = self.make_resume_state(root)
        manifest = state / "manifest-E4.jsonl"
        record = json.loads(manifest.read_text())
        record["job_uid"] = ""
        record["outcome"] = "identity-unknown"
        manifest.write_text(json.dumps(record) + "\n")
        return state, receipt, record

    @staticmethod
    def polling_kubectl(root, receipt_source=None, failed=False):
        root = Path(root)
        job = "study-e4-sglang-059-gptoss120b-gsm8k-h100x2-abcde"
        uid = "11111111-1111-4111-8111-111111111111"
        deleted = root / "mock-job-deleted"
        message_case = ":"
        if receipt_source is not None:
            generated_root = Path(receipt_source).parent
            message_case = (
                f"yaml_path=$(find '{generated_root}' -type f "
                f"-name '{job}.yaml' "
                "| head -1); "
                "yaml_sha=$(sha256sum \"$yaml_path\" | cut -d' ' -f1); "
                f"jq --arg sha \"$yaml_sha\" '.job_yaml_sha256=$sha | "
                f".artifact_sha256.launch_yaml=$sha' '{receipt_source}'")
        complete = "" if failed else "True"
        failed_value = "True" if failed else ""
        return (
            "#!/bin/sh\n"
            "case \"$*\" in\n"
            f"  *\" create \"*) printf '{job}\\t{uid}\\n' ;;\n"
            f"  *\" delete job \"*) touch '{deleted}'; exit 0 ;;\n"
            f"  *\"ownerReferences\"*) [ -f '{deleted}' ] || "
            "printf 'mock-pod\\t11111111-1111-4111-8111-111111111111\\n' ;;\n"
            f"  *\"{{.metadata.uid}}\"*) [ -f '{deleted}' ] || "
            "echo 11111111-1111-4111-8111-111111111111 ;;\n"
            "  *\"status.phase\"*) echo Succeeded ;;\n"
            "  *\"spec.nodeName\"*) echo gpu-h ;;\n"
            "  *\"imageID\"*) echo 'docker-pullable://engine@sha256:" + "1" * 64 + "' ;;\n"
            f"  *\"Complete\"*) echo '{complete}' ;;\n"
            f"  *\"Failed\"*) echo '{failed_value}' ;;\n"
            f"  *\"terminated.message\"*) {message_case} ;;\n"
            "  *) : ;;\n"
            "esac\n")

    def recovering_kubectl(self, root, receipt_source, *, already_terminal=False):
        script = self.polling_kubectl(root, receipt_source)
        if already_terminal:
            return script
        complete_counter = Path(root) / "complete-polls"
        return script.replace(
            '  *"Complete"*) echo \'True\' ;;',
            f'''  *"Complete"*)
    n=$(cat '{complete_counter}' 2>/dev/null || echo 0)
    n=$((n + 1)); echo "$n" > '{complete_counter}'
    [ "$n" -gt 1 ] && echo True ;;
'''.rstrip())

    def test_launcher_reconciliation_restores_frozen_config_before_monitoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, receipt, record = self.make_ambiguous_resume_state(tmp)
            shared = Path(tmp) / "job-configs"
            shared.mkdir()
            env = self.launcher_environment(
                tmp, kubectl_script=self.recovering_kubectl(tmp, receipt))
            env["STUDY_JOB_CONFIGS_DIR"] = str(shared)
            proc = subprocess.run(
                [str(STUDY_LAUNCHER), "E4", "--only", "1",
                 "--reconcile-identity"],
                cwd=str(REPO), env=env, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            restored = shared / f'{record["job"]}.yaml'
            frozen = Path(record["yaml_path"])
            self.assertEqual(restored.read_bytes(), frozen.read_bytes())
            self.assertEqual(
                hashlib.sha256(restored.read_bytes()).hexdigest(),
                record["yaml_sha256"])
            records = [json.loads(line) for line in
                       (state / "manifest-E4.jsonl").read_text().splitlines()]
            self.assertEqual([r["outcome"] for r in records[-2:]],
                             ["submitted", "complete"])

    def test_launcher_reconciliation_rejects_unrestorable_or_wrong_config(self):
        for case in ("copy-failed", "hash-mismatch"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                state, receipt, record = self.make_ambiguous_resume_state(tmp)
                shared = Path(tmp) / "job-configs"
                if case == "hash-mismatch":
                    shared.mkdir()
                env = self.launcher_environment(
                    tmp, kubectl_script=self.recovering_kubectl(tmp, receipt))
                env["STUDY_JOB_CONFIGS_DIR"] = str(shared)
                if case == "hash-mismatch":
                    cp_stub = Path(tmp) / "bin" / "cp"
                    cp_stub.write_text(
                        "#!/bin/sh\nprintf 'wrong config\\n' > \"$2\"\n")
                    cp_stub.chmod(0o755)
                proc = subprocess.run(
                    [str(STUDY_LAUNCHER), "E4", "--only", "1",
                     "--reconcile-identity"],
                    cwd=str(REPO), env=env, text=True, capture_output=True)
                self.assertNotEqual(proc.returncode, 0)
                expected = ("could not restore frozen shared config" if
                            case == "copy-failed" else
                            "restored shared config hash mismatch")
                self.assertIn(expected, proc.stderr)
                self.assertFalse((shared / f'{record["job"]}.yaml').exists())
                records = [json.loads(line) for line in
                           (state / "manifest-E4.jsonl").read_text().splitlines()]
                self.assertEqual(records[-1]["outcome"], "identity-unknown")

    def test_launcher_reconciliation_does_not_leave_terminal_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, receipt, record = self.make_ambiguous_resume_state(tmp)
            shared = Path(tmp) / "job-configs"
            shared.mkdir()
            destination = shared / f'{record["job"]}.yaml'
            destination.write_text("stale config\n")
            env = self.launcher_environment(
                tmp, kubectl_script=self.recovering_kubectl(
                    tmp, receipt, already_terminal=True))
            env["STUDY_JOB_CONFIGS_DIR"] = str(shared)
            proc = subprocess.run(
                [str(STUDY_LAUNCHER), "E4", "--only", "1",
                 "--reconcile-identity"],
                cwd=str(REPO), env=env, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("already terminal; no shared config restored", proc.stdout)
            self.assertFalse(destination.exists())
            records = [json.loads(line) for line in
                       (state / "manifest-E4.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["outcome"], "complete")

    def test_launcher_resume_requires_a_valid_terminal_receipt(self):
        cases = (("success", 256, True, False),
                 ("quality-255", 255, False, False),
                 ("missing-receipt", 256, False, False),
                 ("publish-failed-job", 256, False, True))
        for label, served, expected_success, job_failed in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                state, receipt = self.make_resume_state(tmp, served=served)
                receipt_for_stub = None if label == "missing-receipt" else receipt
                proc = self.run_study_launcher(
                    tmp, "E4", "--only", "1",
                    kubectl_script=self.polling_kubectl(
                        tmp, receipt_for_stub, failed=job_failed))
                self.assertEqual(proc.returncode == 0, expected_success,
                                 proc.stdout + proc.stderr)
                records = [json.loads(line) for line in
                           (state / "manifest-E4.jsonl").read_text().splitlines()]
                expected_outcome = (
                    "complete" if expected_success else
                    "cleanup-confirmed" if job_failed else "validation-failed")
                self.assertEqual(records[-1]["outcome"], expected_outcome)
                if job_failed:
                    self.assertEqual(records[-2]["outcome"], "failed")
                if expected_success:
                    self.assertTrue(Path(records[-1]["receipt_path"]).is_file())
                    self.assertEqual(records[-1]["output_path"],
                                     json.loads(receipt.read_text())["output_path"])

    def test_launcher_cleanup_is_uid_bound_and_waits_for_job_and_pods(self):
        uid = "11111111-1111-4111-8111-111111111111"
        replacement_uid = "99999999-9999-4999-8999-999999999999"

        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self.make_resume_state(tmp)
            script = self.polling_kubectl(tmp, failed=True).replace(
                f"touch '{Path(tmp) / 'mock-job-deleted'}'; exit 0",
                "exit 17")
            proc = self.run_study_launcher(
                tmp, "E4", "--only", "1", kubectl_script=script)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("deletion request failed for exact job", proc.stderr)
            self.assertNotIn("cleanup confirmed", proc.stdout)
            records = [json.loads(line) for line in
                       (state / "manifest-E4.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["outcome"], "failed")

            create_marker = Path(tmp) / "unsafe-retry-create"
            retry_script = f'''#!/bin/sh
case "$*" in
  *" create "*) touch '{create_marker}'; exit 0 ;;
  *) : ;;
esac
'''
            retry = self.run_study_launcher(
                tmp, "E4", "--only", "1", kubectl_script=retry_script)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("automatic resubmission is prohibited", retry.stderr)
            self.assertFalse(create_marker.exists())

            orphan_create = Path(tmp) / "orphan-replacement-create"
            orphan_script = f'''#!/bin/sh
case "$*" in
  *" create "*) touch '{orphan_create}'; exit 0 ;;
  *"ownerReferences"*)
    printf 'orphan-pod\t{uid}\n' ;;
  *"{{.metadata.uid}}"*) exit 0 ;;
  *) : ;;
esac
'''
            orphan_retry = self.run_study_launcher(
                tmp, "E4", "--only", "1", "--reconcile-identity",
                kubectl_script=orphan_script)
            self.assertNotEqual(orphan_retry.returncode, 0)
            self.assertIn("orphan-pod", orphan_retry.stderr)
            self.assertIn("still exists", orphan_retry.stderr)
            self.assertFalse(orphan_create.exists())
            records = [json.loads(line) for line in
                       (state / "manifest-E4.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["outcome"], "failed")

            safe_create = Path(tmp) / "safe-replacement-create"
            safe_script = f'''#!/bin/sh
case "$*" in
  *" create "*) touch '{safe_create}'; exit 9 ;;
  *"ownerReferences"*) exit 0 ;;
  *"{{.metadata.uid}}"*) exit 0 ;;
  *) : ;;
esac
'''
            safe_retry = self.run_study_launcher(
                tmp, "E4", "--only", "1", "--reconcile-identity",
                kubectl_script=safe_script)
            self.assertNotEqual(safe_retry.returncode, 0)
            self.assertTrue(safe_create.exists())
            records = [json.loads(line) for line in
                       (state / "manifest-E4.jsonl").read_text().splitlines()]
            self.assertEqual(records[-2]["outcome"], "cleanup-confirmed")
            self.assertEqual(records[-2]["job_uid"], uid)
            self.assertEqual(records[-1]["outcome"], "create-failed")

        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self.make_resume_state(tmp)
            root = Path(tmp)
            requested = root / "delete-requested"
            job_count = root / "job-polls"
            pod_count = root / "pod-polls"
            script = f'''#!/bin/sh
case "$*" in
  *" delete job "*) touch '{requested}'; exit 0 ;;
  *"ownerReferences"*)
    n=$(cat '{pod_count}' 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > '{pod_count}'
    [ "$n" -gt 3 ] || printf 'mock-pod\t{uid}\n' ;;
  *"{{.metadata.uid}}"*)
    if [ ! -f '{requested}' ]; then echo {uid}; exit 0; fi
    n=$(cat '{job_count}' 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > '{job_count}'
    [ "$n" -gt 2 ] || echo {uid} ;;
  *"status.phase"*) echo Failed ;;
  *"spec.nodeName"*) echo gpu-h ;;
  *"imageID"*) echo 'docker-pullable://engine@sha256:{'1' * 64}' ;;
  *"Complete"*) : ;;
  *"Failed"*) echo True ;;
  *) : ;;
esac
'''
            proc = self.run_study_launcher(
                tmp, "E4", "--only", "1", kubectl_script=script)
            self.assertNotEqual(proc.returncode, 0)  # scientific outcome remains failed
            self.assertIn("cleanup confirmed", proc.stdout)
            self.assertGreaterEqual(int(job_count.read_text()), 3)
            self.assertGreaterEqual(int(pod_count.read_text()), 4)
            records = [json.loads(line) for line in
                       (state / "manifest-E4.jsonl").read_text().splitlines()]
            self.assertEqual(records[-2]["outcome"], "failed")
            self.assertEqual(records[-1]["outcome"], "cleanup-confirmed")

        with tempfile.TemporaryDirectory() as tmp:
            self.make_resume_state(tmp)
            root = Path(tmp)
            uid_queries = root / "uid-queries"
            delete_marker = root / "unsafe-delete"
            script = f'''#!/bin/sh
case "$*" in
  *" delete job "*) touch '{delete_marker}'; exit 0 ;;
  *"{{.metadata.uid}}"*)
    n=$(cat '{uid_queries}' 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > '{uid_queries}'
    if [ "$n" -eq 1 ]; then echo {uid}; else echo {replacement_uid}; fi ;;
  *"status.phase"*) echo Failed ;;
  *"spec.nodeName"*) echo gpu-h ;;
  *"imageID"*) echo 'docker-pullable://engine@sha256:{'1' * 64}' ;;
  *"Complete"*) : ;;
  *"Failed"*) echo True ;;
  *) : ;;
esac
'''
            proc = self.run_study_launcher(
                tmp, "E4", "--only", "1", kubectl_script=script)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("refusing to delete replacement job", proc.stderr)
            self.assertFalse(delete_marker.exists())

    def test_launcher_captures_generated_identity_and_blocks_ambiguous_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self.make_resume_state(tmp)
            prior = json.loads((state / "manifest-E4.jsonl").read_text())
            prior["outcome"] = "cleanup-confirmed"
            (state / "manifest-E4.jsonl").write_text(json.dumps(prior) + "\n")
            # The create response returns the API-server generated name and UID.
            # The local config copy then fails, exercising exact UID cleanup.
            proc = self.run_study_launcher(
                tmp, "E4", "--only", "1",
                kubectl_script=self.polling_kubectl(tmp))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("cleanup confirmed", proc.stdout)
            records = [json.loads(line) for line in
                       (state / "manifest-E4.jsonl").read_text().splitlines()]
            self.assertEqual(records[-3]["outcome"], "submitted")
            self.assertEqual(records[-2]["outcome"], "config-copy-failed")
            self.assertEqual(records[-1]["outcome"], "cleanup-confirmed")
            self.assertEqual(records[-1]["job_uid"],
                             "11111111-1111-4111-8111-111111111111")
            self.assertEqual(
                records[-1]["job"],
                "study-e4-sglang-059-gptoss120b-gsm8k-h100x2-abcde")
            submitted = load_yaml_no_duplicates(
                Path(records[-1]["yaml_path"]).read_text())
            self.assertIn("generateName", submitted["metadata"])
            self.assertNotIn("name", submitted["metadata"])
            self.assertEqual(
                hashlib.sha256(
                    Path(records[-1]["yaml_path"]).read_bytes()).hexdigest(),
                records[-1]["yaml_sha256"])

        for label, create_case, outcome in (
                ("malformed", '*" create "*) exit 0 ;;', "identity-unknown"),
                ("failed", '*" create "*) exit 9 ;;', "create-failed")):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                state, _ = self.make_resume_state(tmp)
                prior = json.loads((state / "manifest-E4.jsonl").read_text())
                prior["outcome"] = "cleanup-confirmed"
                (state / "manifest-E4.jsonl").write_text(json.dumps(prior) + "\n")
                script = f'''#!/bin/sh
case "$*" in
  {create_case}
  *) : ;;
esac
'''
                proc = self.run_study_launcher(
                    tmp, "E4", "--only", "1", kubectl_script=script)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("inspect the namespace before any retry", proc.stderr)
                records = [json.loads(line) for line in
                           (state / "manifest-E4.jsonl").read_text().splitlines()]
                self.assertEqual(records[-1]["outcome"], outcome)
                self.assertEqual(records[-1]["job"], "")
                self.assertEqual(records[-1]["job_uid"], "")
                self.assertTrue(Path(records[-1]["yaml_path"]).is_file())
                blocked = self.run_guard(
                    "repeat-action", "--manifest", state / "manifest-E4.jsonl",
                    "--block", "E4", "--order", "1")
                self.assertNotEqual(blocked.returncode, 0)
                self.assertIn("no captured server-generated Job name", blocked.stderr)

    def test_study_marker_level_round_trips_and_keeps_a_pure_timestamp(self):
        """parse_run_path keeps the standard 6-level prefix and reports
        'study-e1/<ts>' as the run id; the timestamp stays its own PURE
        component, which is what the dashboard assembler's run_date() needs."""
        params = {"family": "moe", "inference_engine": "vllm", "model": "gpt-oss-120b",
                  "dataset": "gsm8k", "num_samples": 256, "gpu": "A100",
                  "num_gpu": 2, "batch_size": "default",
                  "input_length": None, "output_length": None,
                  "study_block": "E1", "study_order": 2,
                  "engine_version": "0.21.0"}
        full_dir = self.utils.results_repo_dir(params)
        self.assertEqual(
            full_dir,
            "moe/eidf/vllm/gpt-oss-120b/gsm8k_256samples/a100x2/"
            "batch-size-default/study-e1")
        rel_parts = tuple(full_dir.split("/", 1)[1].split("/")) + ("20260901-1010", "metrics.json")
        parsed = parse_run_path(rel_parts)
        self.assertEqual(parsed["dataset"], "gsm8k")
        self.assertEqual(parsed["batch_size"], "default")
        self.assertEqual(parsed["run_timestamp"], "study-e1/20260901-1010")

    def test_study_row_without_engine_version_fails_generation(self):
        """A study row must pin its build; silently inheriting the config
        default would turn an alternate-build leaf into a control run."""
        params = {"family": "moe", "inference_engine": "vllm", "model": "gpt-oss-120b",
                  "dataset": "gsm8k", "num_samples": 256, "gpu": "H100",
                  "num_gpu": 1, "batch_size": "default", "study_block": "E1"}
        with self.assertRaises(ValueError):
            self.utils.get_run_name(params)
        # study_block on an agentic row is an error, not silently ignored.
        with self.assertRaises(ValueError):
            self.utils.study_fields({"family": "agentic", "benchmark": "mcp-atlas",
                                     "study_block": "E1", "engine_version": "0.5.9"})

    def test_direct_generation_rejects_every_mutated_study_coordinate_field(self):
        valid = dict(self.leaves("E1")[0])
        self.assertEqual(self.utils.study_fields(valid), ("e1", "0.16.0"))
        mutations = {
            "family": "agentic", "model": "another-model", "dataset": "gsm8k-extra",
            "num_samples": "255", "gpu": "H100", "num_gpu": "1",
            "batch_size": "8", "inference_engine": "sglang",
            "engine_version": "9.9.9", "study_block": "E4", "study_order": "12",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                row = {**valid, field: value}
                with self.assertRaises(ValueError):
                    self.utils.study_fields(row)
        for field, value in (("family", " moe "), ("study_block", "e1"),
                             ("study_order", "01"), ("num_samples", "0256"),
                             ("num_gpu", "02")):
            with self.subTest(field=field, noncanonical=value):
                with self.assertRaises(ValueError):
                    self.utils.study_fields({**valid, field: value})
        with self.assertRaises(ValueError):
            self.utils.study_fields({
                "family": "moe", "model": "gpt-oss-120b", "dataset": "gsm8k",
                "num_samples": "256", "gpu": "A100", "num_gpu": "2",
                "batch_size": "default", "inference_engine": "vllm",
                "engine_version": "0.16.0", "study_order": "1"})

    def test_non_study_rows_unaffected(self):
        """A row with no study_block keeps its historical name and path."""
        params = {"family": "moe", "inference_engine": "sglang", "model": "gpt-oss-120b",
                  "dataset": "gsm8k", "num_samples": 256, "gpu": "H100",
                  "num_gpu": 1, "batch_size": "default",
                  "input_length": None, "output_length": None}
        self.assertEqual(self.utils.get_run_name(params),
                         "sglang_gptoss120b_gsm8k_ns256_H100x1_bsd")
        self.assertEqual(
            self.utils.results_repo_dir(params),
            "moe/eidf/sglang/gpt-oss-120b/gsm8k_256samples/h100x1/batch-size-default")


class StudyGuardTests(unittest.TestCase):
    STUDY_ID = "controlled-variation-2026-x2"
    MOE_CAP_REF = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    TEASBENCH_COMMIT = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    COMBINATIONS = (
        ("vllm", "0.16.0", "1"),
        ("vllm", "0.21.0", "2"),
        ("sglang", "0.5.9", "3"),
        ("sglang", "0.5.12.post1", "4"),
    )

    def run_guard(self, *args):
        return subprocess.run(
            [sys.executable, str(STUDY_GUARD), *map(str, args)],
            text=True, capture_output=True)

    @staticmethod
    def file_sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def test_prepare_job_preserves_generate_name_in_unique_durable_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "study-job.yaml"
            source.write_text(
                "apiVersion: batch/v1\n"
                "kind: Job\n"
                "metadata:\n"
                "  generateName: study-e1-vllm-0160-gptoss120b-gsm8k-a100x2-\n"
                "  labels:\n"
                "    kueue.x-k8s.io/queue-name: eidf230ns-user-queue\n"
                "spec:\n"
                "  template:\n"
                "    spec:\n"
                "      containers: []\n")
            outputs = []
            for _ in range(2):
                proc = self.run_guard(
                    "prepare-job", "--yaml", source,
                    "--state-dir", Path(tmp) / "submitted")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                fields = proc.stdout.strip().split("\t")
                self.assertEqual(len(fields), 2)
                durable, digest = Path(fields[0]), fields[1]
                self.assertEqual(durable.read_bytes(), source.read_bytes())
                self.assertEqual(self.file_sha256(durable), digest)
                document = load_yaml_no_duplicates(durable.read_text())
                self.assertIn("generateName", document["metadata"])
                self.assertNotIn("name", document["metadata"])
                outputs.append(durable)
            self.assertNotEqual(outputs[0], outputs[1])

    def make_preflight_evidence(self, root):
        root = Path(root)
        pins_path = root / "image-digests.tsv"
        manifest_path = root / "a100x2-compatibility-preflight.jsonl"
        bases = {"vllm": "vllm/vllm-openai", "sglang": "lmsysorg/sglang"}
        pins = {}
        records = []
        for engine, version, digit in self.COMBINATIONS:
            tag = f"{bases[engine]}:v{version}"
            digest = "sha256:" + digit * 64
            pins[tag] = digest
            artifact_dir = (root / "compatibility-preflight" /
                            f"{engine}-{version.replace('.', '-')}")
            artifact_dir.mkdir(parents=True)
            metadata = {
                "system_environment": {
                    "inference_engine": engine,
                    "inference_engine_version": version,
                    "teasbench_commit": self.TEASBENCH_COMMIT[:7],
                    "moe_cap_commit": self.MOE_CAP_REF[:7],
                },
                "model_config": {"model_name": "unsloth/gpt-oss-120b"},
                "hardware": {"num_gpus": 2,
                             "gpu_type": "NVIDIA-A100-SXM4-80GB"},
                "compatibility_preflight": {
                    "dataset": "longbench_v1", "num_samples": 256,
                    "batch_size": "default", "gpu": "A100", "num_gpu": 2,
                    "gpu_uuids": [f"GPU-{digit}a", f"GPU-{digit}b"],
                    "job_uid": f"job-{engine}-{version}",
                    "job_name": f"preflight-{engine}-{digit}",
                    "node": "gpu-node-a",
                    "image_ref": f"{bases[engine]}@{digest}",
                },
            }
            metrics = {
                "quality": {"total": 256, "attempted": 256,
                            "served": 256, "completed": 256},
            }
            metadata_path = artifact_dir / "metadata.json"
            metrics_path = artifact_dir / "metrics.json"
            job_yaml_path = artifact_dir / "preflight-job.yaml"
            metadata_path.write_text(json.dumps(metadata))
            metrics_path.write_text(json.dumps(metrics))
            image_ref = f"{bases[engine]}@{digest}"
            model_flag = "--model" if engine == "vllm" else "--model-path"
            tp_flag = "--tensor-parallel-size" if engine == "vllm" else "--tp-size"
            if engine == "vllm":
                server_tail = (
                    f"{model_flag} unsloth/gpt-oss-120b --port 30000 "
                    f"--host 0.0.0.0 {tp_flag} 2 --reasoning-parser openai_gptoss")
            else:
                server_tail = (
                    f"{model_flag} unsloth/gpt-oss-120b --port 30000 "
                    "--expert-distribution-recorder-mode stat "
                    f"{tp_flag} 2 --reasoning-parser gpt-oss")
            job_script = (
                f"git checkout --quiet --detach {self.MOE_CAP_REF}\n"
                f"echo '{{\"teasbench_commit\": \"{self.TEASBENCH_COMMIT[:7]}\"}}'\n"
                f"python3 -m moe_cap.systems.{engine} {server_tail} &> server.log &\n"
                "python3 -m moe_cap.runner.openai_api_profile "
                "--model_name unsloth/gpt-oss-120b --datasets longbench_v1 "
                "--num-samples 256 --api-url http://localhost:30000/v1/completions "
                f"--backend {engine} --output_dir /mnt/develop/batch-size-default/"
                "compatibility-preflight/run "
                "--use-chat-api &> client.log\n")
            job_yaml_path.write_text(yaml.safe_dump({
                "apiVersion": "batch/v1", "kind": "Job",
                "metadata": {"name": f"preflight-{engine}-{digit}"},
                "spec": {"template": {"spec": {
                    "containers": [{"name": "server", "image": image_ref,
                                    "resources": {"limits": {"nvidia.com/gpu": 2}},
                                    "args": [job_script]}],
                    "nodeSelector": {
                        "nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"},
                }}},
            }, sort_keys=False))
            records.append({
                "study_id": self.STUDY_ID,
                "kind": "excluded-compatibility-preflight",
                "gpu": "A100",
                "num_gpu": 2,
                "dataset": "longbench_v1",
                "num_samples": 256,
                "batch_size": "default",
                "outcome": "complete",
                "teasbench_commit": self.TEASBENCH_COMMIT,
                "moe_cap_ref": self.MOE_CAP_REF,
                "inference_engine": engine,
                "engine_version": version,
                "image_tag": tag,
                "image_ref": image_ref,
                "job_uid": f"job-{engine}-{version}",
                "job_name": f"preflight-{engine}-{digit}",
                "node": "gpu-node-a",
                "gpu_uuids": [f"GPU-{digit}a", f"GPU-{digit}b"],
                "completed_at": "2020-08-17T12:00:00Z",
                "artifact_dir": str(artifact_dir.resolve()),
                "metadata_sha256": self.file_sha256(metadata_path),
                "metrics_sha256": self.file_sha256(metrics_path),
                "job_yaml_sha256": self.file_sha256(job_yaml_path),
            })
        pins_path.write_text(
            "".join(f"{tag}\t{pins[tag]}\n" for tag in sorted(pins)))
        manifest_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records))
        return pins_path, manifest_path, records

    def test_preflight_validator_hash_binds_complete_four_arm_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            pins, manifest, records = self.make_preflight_evidence(tmp)
            validation_record = Path(tmp) / "validated.json"
            proc = self.run_guard(
                "validate-preflight", "--manifest", manifest,
                "--image-pins", pins,
                "--teasbench-commit", self.TEASBENCH_COMMIT,
                "--moe-cap-ref", self.MOE_CAP_REF,
                "--record", validation_record)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            validation = json.loads(validation_record.read_text())
            self.assertEqual(validation["kind"], "validated-compatibility-preflight")
            self.assertEqual(len(validation["combinations"]), 4)
            self.assertEqual(validation["manifest_sha256"], self.file_sha256(manifest))

            # A hash-consistent artifact with incomplete LongBench service is
            # still invalid; the gate derives completion from metrics.json.
            artifact_dir = Path(records[0]["artifact_dir"])
            metrics_path = artifact_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["quality"]["served"] = 255
            metrics_path.write_text(json.dumps(metrics))
            records[0]["metrics_sha256"] = self.file_sha256(metrics_path)
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            proc = self.run_guard(
                "validate-preflight", "--manifest", manifest,
                "--image-pins", pins,
                "--teasbench-commit", self.TEASBENCH_COMMIT,
                "--moe-cap-ref", self.MOE_CAP_REF)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("served must be 256", proc.stderr)

    def test_preflight_rejects_short_commits_future_duplicate_uid_and_alias_escape(self):
        # Commit pins used by the gate must be full immutable object IDs.
        with tempfile.TemporaryDirectory() as tmp:
            pins, manifest, _ = self.make_preflight_evidence(tmp)
            proc = self.run_guard(
                "validate-preflight", "--manifest", manifest,
                "--image-pins", pins, "--teasbench-commit", "a",
                "--moe-cap-ref", self.MOE_CAP_REF)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("40 lowercase hex digits", proc.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            pins, manifest, records = self.make_preflight_evidence(tmp)
            records[0]["completed_at"] = "2999-01-01T00:00:00Z"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
            proc = self.run_guard(
                "validate-preflight", "--manifest", manifest,
                "--image-pins", pins, "--teasbench-commit", self.TEASBENCH_COMMIT,
                "--moe-cap-ref", self.MOE_CAP_REF)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("in the future", proc.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            pins, manifest, records = self.make_preflight_evidence(tmp)
            records[1]["job_uid"] = records[0]["job_uid"]
            manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
            proc = self.run_guard(
                "validate-preflight", "--manifest", manifest,
                "--image-pins", pins, "--teasbench-commit", self.TEASBENCH_COMMIT,
                "--moe-cap-ref", self.MOE_CAP_REF)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("duplicate job_uid", proc.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            pins, manifest, records = self.make_preflight_evidence(tmp)
            original = Path(records[0]["artifact_dir"])
            escaped = Path(tmp) / "study-e1" / "aliased-run"
            escaped.parent.mkdir()
            shutil.move(str(original), str(escaped))
            original.symlink_to(escaped, target_is_directory=True)
            records[0]["artifact_dir"] = str(original)
            manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
            proc = self.run_guard(
                "validate-preflight", "--manifest", manifest,
                "--image-pins", pins, "--teasbench-commit", self.TEASBENCH_COMMIT,
                "--moe-cap-ref", self.MOE_CAP_REF)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("inside a study block path", proc.stderr)

    def test_preflight_parses_job_and_metadata_instead_of_trusting_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            pins, manifest, records = self.make_preflight_evidence(tmp)
            job_path = next(Path(records[0]["artifact_dir"]).glob("*.yaml"))
            document = yaml.safe_load(job_path.read_text())
            document["spec"]["template"]["spec"]["containers"][0][
                "resources"]["limits"]["nvidia.com/gpu"] = 1
            # A truthful-looking comment must not rescue an actual x1 allocation.
            document["review_note"] = "nvidia.com/gpu: 2 longbench_v1 256 TP2"
            job_path.write_text(yaml.safe_dump(document, sort_keys=False))
            records[0]["job_yaml_sha256"] = self.file_sha256(job_path)
            manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
            proc = self.run_guard(
                "validate-preflight", "--manifest", manifest,
                "--image-pins", pins, "--teasbench-commit", self.TEASBENCH_COMMIT,
                "--moe-cap-ref", self.MOE_CAP_REF)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("nvidia.com/gpu must be 2", proc.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            pins, manifest, records = self.make_preflight_evidence(tmp)
            metadata_path = Path(records[0]["artifact_dir"]) / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["compatibility_preflight"]["job_uid"] = "some-other-job"
            metadata_path.write_text(json.dumps(metadata))
            records[0]["metadata_sha256"] = self.file_sha256(metadata_path)
            manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
            proc = self.run_guard(
                "validate-preflight", "--manifest", manifest,
                "--image-pins", pins, "--teasbench-commit", self.TEASBENCH_COMMIT,
                "--moe-cap-ref", self.MOE_CAP_REF)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("job_uid", proc.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            pins, manifest, records = self.make_preflight_evidence(tmp)
            artifact_dir = Path(records[0]["artifact_dir"])
            metrics_path = artifact_dir / "metrics.json"
            outside = Path(tmp) / "outside-metrics.json"
            outside.write_bytes(metrics_path.read_bytes())
            metrics_path.unlink()
            metrics_path.symlink_to(outside)
            proc = self.run_guard(
                "validate-preflight", "--manifest", manifest,
                "--image-pins", pins, "--teasbench-commit", self.TEASBENCH_COMMIT,
                "--moe-cap-ref", self.MOE_CAP_REF)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("resolves outside artifact_dir", proc.stderr)

        for engine, forbidden_flag in (
                ("vllm", "--gpu-memory-utilization 0.01"),
                ("sglang", "--mem-fraction-static 0.01")):
            with self.subTest(recipe_mutation=engine), tempfile.TemporaryDirectory() as tmp:
                pins, manifest, records = self.make_preflight_evidence(tmp)
                record = next(r for r in records if r["inference_engine"] == engine)
                job_path = next(Path(record["artifact_dir"]).glob("*.yaml"))
                document = yaml.safe_load(job_path.read_text())
                script = document["spec"]["template"]["spec"]["containers"][0]["args"][0]
                document["spec"]["template"]["spec"]["containers"][0]["args"][0] = (
                    script.replace("--port 30000", f"--port 30000 {forbidden_flag}", 1))
                job_path.write_text(yaml.safe_dump(document, sort_keys=False))
                record["job_yaml_sha256"] = self.file_sha256(job_path)
                manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
                proc = self.run_guard(
                    "validate-preflight", "--manifest", manifest,
                    "--image-pins", pins, "--teasbench-commit", self.TEASBENCH_COMMIT,
                    "--moe-cap-ref", self.MOE_CAP_REF)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("server command differs from the frozen", proc.stderr)

    def make_terminal_validation_fixture(self, root, engine="sglang"):
        root = Path(root)
        if engine == "sglang":
            block, order, version, gpu = "e4", 1, "0.5.9", "H100"
            gpu_product = "NVIDIA-H100-80GB-HBM3"
        else:
            block, order, version, gpu = "e1", 1, "0.16.0", "A100"
            gpu_product = "NVIDIA-A100-SXM4-80GB"
        publish_path = (
            f"moe/eidf/{engine}/gpt-oss-120b/gsm8k_256samples/"
            f"{gpu.lower()}x2/batch-size-default/study-{block}/run")
        output = root / "archive" / publish_path
        output.mkdir(parents=True)
        metadata = {
            "model_config": {"model_name": "unsloth/gpt-oss-120b"},
            "hardware": {"num_gpus": 2},
            "system_environment": {
                "inference_engine": engine,
                "inference_engine_version": version,
                "teasbench_commit": "a" * 7, "moe_cap_commit": "e" * 7,
            },
            "study": {
                "study_id": self.STUDY_ID, "block_id": block, "planned_order": order,
                "node": "gpu-h", "job_name": f"study-{block}-job",
                "job_uid": f"uid-{block}-job", "dataset": "gsm8k",
                "num_samples": 256, "gpu": gpu, "num_gpu": 2,
                "batch_size": "default", "gpu_uuids": "GPU-one,GPU-two",
                "gpu_product": gpu_product,
                "arena_baseline_sha256": "",
            },
        }
        metrics = {"quality": {"total": 256, "attempted": 256,
                               "served": 256, "completed": 256}}
        (output / "metadata.json").write_text(json.dumps(metadata))
        (output / "metrics.json").write_text(json.dumps(metrics))
        artifact_names = ["detailed_results.jsonl", "output_data.jsonl", "timings.json",
                          "pip_freeze.txt"]
        if engine == "sglang":
            artifact_names.append("expert_distribution_record.jsonl")
        for name in artifact_names:
            (output / name).write_text("evidence\n")
        (output / "job.yaml").write_text(
            "containers:\n  - name: server\n"
            f"    image: example.invalid/{engine}@sha256:" + "1" * 64 + "\n")
        termination_log = root / "termination-message.json"
        env = os.environ.copy()
        env.update({
            "STUDY_SERVER_OK": "1", "STUDY_CLIENT_OK": "1",
            "STUDY_ENRICH_OK": "1", "STUDY_PUBLISH_OK": "1",
            "PVC_RUN_OUTPUT_DIR": str(output), "STUDY_ID": self.STUDY_ID,
            "STUDY_BLOCK": block, "STUDY_ORDER": str(order), "STUDY_DATASET": "gsm8k",
            "STUDY_ENGINE": engine, "STUDY_ENGINE_VERSION": version,
            "STUDY_GPU": gpu, "STUDY_GPU_PRODUCT": gpu_product,
            "TEASBENCH_COMMIT": "a" * 7, "MOE_CAP_COMMIT": "e" * 7,
            "k8s_node_name": "gpu-h", "k8s_job_name": f"study-{block}-job",
            "k8s_job_uid": f"uid-{block}-job", "PUBLISH_SUBDIR": publish_path,
            "STUDY_TERMINATION_LOG": str(termination_log),
        })
        return output, termination_log, env

    def test_terminal_validator_requires_scientific_and_publication_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, termination_log, env = self.make_terminal_validation_fixture(tmp)
            proc = subprocess.run(
                ["bash", str(STUDY_TERMINAL_VALIDATOR)], env=env,
                text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            receipt = json.loads(termination_log.read_text())
            self.assertEqual(receipt["quality"]["completed"], 256)
            self.assertEqual(set(receipt["artifact_sha256"]), {
                "metadata", "metrics", "launch_yaml", "detailed_results",
                "output_data", "timings", "pip_freeze",
                "expert_distribution_bundle"})

        with tempfile.TemporaryDirectory() as tmp:
            _, termination_log, env = self.make_terminal_validation_fixture(
                tmp, engine="vllm")
            proc = subprocess.run(
                ["bash", str(STUDY_TERMINAL_VALIDATOR)], env=env,
                text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            receipt = json.loads(termination_log.read_text())
            self.assertEqual(receipt["inference_engine"], "vllm")
            self.assertNotIn("expert_distribution_bundle", receipt["artifact_sha256"])

        with tempfile.TemporaryDirectory() as tmp:
            output, _, env = self.make_terminal_validation_fixture(tmp, engine="sglang")
            (output / "expert_distribution_record.jsonl").unlink()
            proc = subprocess.run(
                ["bash", str(STUDY_TERMINAL_VALIDATOR)], env=env,
                text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("missing SGLang expert-distribution artifact", proc.stderr)

        failures = (
            ("server", "STUDY_SERVER_OK", "server did not become ready"),
            ("client", "STUDY_CLIENT_OK", "client did not finish"),
            ("enrichment", "STUDY_ENRICH_OK", "metadata enrichment failed"),
            ("publish", "STUDY_PUBLISH_OK", "publication failed"),
        )
        for label, variable, expected in failures:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                _, _, env = self.make_terminal_validation_fixture(tmp)
                env[variable] = "0"
                proc = subprocess.run(
                    ["bash", str(STUDY_TERMINAL_VALIDATOR)], env=env,
                    text=True, capture_output=True)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(expected, proc.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            output, _, env = self.make_terminal_validation_fixture(tmp)
            metrics = json.loads((output / "metrics.json").read_text())
            metrics["quality"]["served"] = 255
            metrics["quality"]["completed"] = 255
            (output / "metrics.json").write_text(json.dumps(metrics))
            proc = subprocess.run(
                ["bash", str(STUDY_TERMINAL_VALIDATOR)], env=env,
                text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("not all 256", proc.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            output, _, env = self.make_terminal_validation_fixture(tmp)
            (output / "output_data.jsonl").unlink()
            proc = subprocess.run(
                ["bash", str(STUDY_TERMINAL_VALIDATOR)], env=env,
                text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("missing artifact output_data.jsonl", proc.stderr)

    def test_terminal_validator_emits_compact_preflight_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, termination_log, env = self.make_terminal_validation_fixture(
                tmp, engine="vllm")
            metadata_path = output / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["hardware"]["gpu_type"] = "NVIDIA-A100-SXM4-80GB"
            metadata["system_environment"]["teasbench_commit"] = "a" * 40
            metadata.pop("study")
            image_ref = "example.invalid/vllm@sha256:" + "1" * 64
            metadata["compatibility_preflight"] = {
                "dataset": "longbench_v1", "num_samples": 256,
                "batch_size": "default", "gpu": "A100", "num_gpu": 2,
                "node": "gpu-h", "job_name": "study-e1-job",
                "job_uid": "uid-e1-job", "gpu_uuids": ["GPU-one", "GPU-two"],
                "image_ref": image_ref,
            }
            metadata_path.write_text(json.dumps(metadata))
            env.update({
                "STUDY_PREFLIGHT": "1", "STUDY_DATASET": "longbench_v1",
                "STUDY_IMAGE_TAG": "vllm/vllm-openai:v0.16.0",
                "PREFLIGHT_MOE_CAP_REF": "e" * 40,
                "TEASBENCH_COMMIT": "a" * 40,
                "GPU_UUIDS": "GPU-one,GPU-two",
            })
            proc = subprocess.run(
                ["bash", str(STUDY_TERMINAL_VALIDATOR)], env=env,
                text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            evidence = json.loads(termination_log.read_text())
            self.assertEqual(evidence["kind"], "excluded-compatibility-preflight")
            self.assertEqual(evidence["gpu_uuids"], ["GPU-one", "GPU-two"])
            self.assertEqual(evidence["job_yaml_sha256"],
                             self.file_sha256(output / "job.yaml"))
            self.assertLess(len(termination_log.read_bytes()), 4096)

    def test_manifest_guard_checks_completion_and_unique_last_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest-E1.jsonl"
            ReplicationStudyTests.write_complete_study_manifest(tmp, "E1")
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            complete = self.run_guard(
                "block-complete", "--manifest", manifest, "--block", "E1")
            self.assertEqual(complete.returncode, 0, complete.stderr)
            node = self.run_guard(
                "manifest-node", "--manifest", manifest, "--block", "E1")
            self.assertEqual(node.returncode, 0, node.stderr)
            self.assertEqual(node.stdout.strip(), "gpu-a")

            records[-1]["outcome"] = "failed"
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            incomplete = self.run_guard(
                "block-complete", "--manifest", manifest, "--block", "E1")
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn("missing successful leaves 12", incomplete.stderr)

            records[-1]["node"] = "gpu-node-b"
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            ambiguous = self.run_guard(
                "manifest-node", "--manifest", manifest, "--block", "E1")
            self.assertNotEqual(ambiguous.returncode, 0)
            self.assertIn("records multiple nodes", ambiguous.stderr)

    def test_image_guard_persists_digest_patches_yaml_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pin_file = root / "pins.tsv"
            resolved = root / "resolved.tsv"
            yaml_path = root / "job.yaml"
            tag = "vllm/vllm-openai:v0.21.0"
            digest = "sha256:" + "a" * 64
            resolved.write_text(f"{tag}\t{digest}\n")
            yaml_path.write_text(
                f"containers:\n  - name: server\n    image: {tag}\n")
            proc = self.run_guard(
                "pin-images", "--pin-file", pin_file,
                "--resolved-file", resolved, "--persist", yaml_path)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(pin_file.read_text(), f"{tag}\t{digest}\n")
            self.assertIn(f"image: vllm/vllm-openai@{digest}", yaml_path.read_text())

            drift = "sha256:" + "b" * 64
            resolved.write_text(f"{tag}\t{drift}\n")
            second_yaml = root / "second.yaml"
            second_yaml.write_text(
                f"containers:\n  - name: server\n    image: {tag}\n")
            proc = self.run_guard(
                "pin-images", "--pin-file", pin_file,
                "--resolved-file", resolved, "--persist", second_yaml)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("image tag drift", proc.stderr)
            self.assertIn(f"image: {tag}", second_yaml.read_text())


if __name__ == "__main__":
    unittest.main()
