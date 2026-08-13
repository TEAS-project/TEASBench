import csv
import importlib.util
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

        # Generate one block end-to-end via the launcher's own CSV filter
        # (awk column 10 + sed ref-fill), so a CSV column change that breaks
        # eidf/scripts/run_study_block.sh breaks this test too.
        cls.tmp = tempfile.mkdtemp()
        block_csv = Path(cls.tmp) / "e1.csv"
        subprocess.run(
            "awk -F, 'NR==1 || $10==\"E1\"' " + str(cls.STUDY_CSV)
            + " | sed 's/,$/,abc1234/' > " + str(block_csv),
            shell=True, check=True)
        cls.outdir = Path(cls.tmp) / "out"
        run_generate(PIPELINE, block_csv, cls.outdir)
        cls.generated = sorted(cls.outdir.glob("*.yaml"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def leaves(self, block):
        return sorted((r for r in self.rows if r["study_block"] == block),
                      key=lambda r: int(r["study_order"]))

    def test_matrix_is_the_frozen_72_leaves(self):
        """6 blocks x 4 arms x 3 datasets, all gpt-oss-120b / H100x1 / bdef."""
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
            for r in leaves:
                self.assertEqual(
                    (r["family"], r["model"], r["num_samples"], r["gpu"],
                     r["num_gpu"], r["batch_size"]),
                    ("moe", "gpt-oss-120b", "256", "H100", "1", "default"))

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

    def test_generated_yaml_pins_build_marker_and_producer(self):
        """Each generated leaf carries its row's exact image tag, the
        study-e1 path level, the MoE-CAP checkout pin, and the study identity
        jq -- and its job name stays under the k8s label limit."""
        self.assertEqual(len(self.generated), 12)
        by_name = {p.name: p.read_text() for p in self.generated}
        for row in self.leaves("E1"):
            stem = None
            for name, text in by_name.items():
                if (f'engine_version: "{row["engine_version"]}"' in text
                        and f'--datasets {row["dataset"]}' in text
                        and f'planned_order: {row["study_order"]},' in text):
                    stem, yaml_text = name, text
                    break
            self.assertIsNotNone(stem, f"no YAML for row {row}")
            base = {"vllm": "vllm/vllm-openai", "sglang": "lmsysorg/sglang"}
            self.assertIn(f'image: {base[row["inference_engine"]]}:'
                          f'v{row["engine_version"]}', yaml_text)
            self.assertIn("/batch-size-default/study-e1/$timestamp", yaml_text)
            # Pin must fail the run if the checkout fails, not fall back to main.
            self.assertIn("git checkout --quiet --detach abc1234 || ", yaml_text)
            self.assertIn('block_id: "e1"', yaml_text)
            self.assertIn("pip freeze > $PVC_RUN_OUTPUT_DIR/pip_freeze.txt", yaml_text)
            self.assertIn("arena_baseline_sha256", yaml_text)
            generate_name = re.search(r"generateName: (\S+)", yaml_text).group(1)
            self.assertLessEqual(len(generate_name) + 5, 63, generate_name)

    def test_study_marker_level_round_trips_and_keeps_a_pure_timestamp(self):
        """parse_run_path keeps the standard 6-level prefix and reports
        'study-e1/<ts>' as the run id; the timestamp stays its own PURE
        component, which is what the dashboard assembler's run_date() needs."""
        params = {"family": "moe", "inference_engine": "vllm", "model": "gpt-oss-120b",
                  "dataset": "gsm8k", "num_samples": 256, "gpu": "H100",
                  "num_gpu": 1, "batch_size": "default",
                  "input_length": None, "output_length": None,
                  "study_block": "E1", "engine_version": "0.21.0"}
        full_dir = self.utils.results_repo_dir(params)
        self.assertEqual(
            full_dir,
            "moe/eidf/vllm/gpt-oss-120b/gsm8k_256samples/h100x1/"
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


if __name__ == "__main__":
    unittest.main()
