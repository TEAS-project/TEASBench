"""The engine_pre_launch hook, and the sglang gpt-oss parser patch it carries.

The patch itself edits sglang's installed source in place. These tests cover
what can be checked without an sglang install: that the script finds and
rewrites every anchor it expects, refuses to guess when one is missing, is
idempotent, and that generate.py embeds it in the sglang engine Job and in
nothing else.

What they deliberately do not cover: whether the anchors still exist in
whatever sglang image the pipeline points at. That is a property of the
image, not of this repo, and it is re-checked when the image is bumped --
see docs/DEVELOPER_GUIDE.md. The script's own non-zero exit is the guard at
run time.
"""
import ast
import base64
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "pipeline"
EXPERIMENTS = REPO / "experiments"
PATCH_SCRIPT = PIPELINE / "k8s" / "setup" / "patch_sglang_gptoss.py"

# Excerpts of sglang v0.5.12.post1, verbatim apart from the elisions needed to
# make each one a self-contained module: srt/function_call/gpt_oss_detector.py
# and srt/parser/harmony_parser.py. Only the lines the patch anchors on matter,
# and those are byte-for-byte upstream -- including their indentation, which is
# part of every anchor.
STOCK_DETECTOR = r'''import json
import logging
import re

logger = logging.getLogger(__name__)


class GptOssDetector:
    def __init__(self):
        self.bot_token = "<|start|>assistant<|channel|>commentary"

        # Pattern to extract function name and JSON from tool_call event content
        self.tool_extract_pattern = re.compile(
            r"to=([a-zA-Z_][a-zA-Z0-9_.-]*)\s*<\|constrain\|>json<\|message\|>(.*?)(?:<\|call\|>|$)",
            re.DOTALL,
        )

    def has_tool_call(self, text) -> bool:
        """Check if text contains TypeScript-style function call markers."""
        return self.bot_token in text

    def parse_streaming_increment(self, new_text, tools):
        # Quick check if we might have tool calls
        if (
            "<|channel|>commentary to=" not in self._buffer
            and not self.current_tool_name_sent
        ):
            return None

        if new_text:
            if tools:
                if True:
                    # Store tool call info
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": tool_call_info.name,
                        "arguments": json.loads(tool_call_info.parameters),
                    }

                    calls.append(tool_call_info)

    def _extract_tool_call_from_event(self, json_content, function_name, tool_index):
        # Parse JSON arguments
        try:
            arguments = json.loads(json_content) if json_content.strip() else {}
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON arguments: {e}")
            return None

        return ToolCallItem(
            tool_index=tool_index,
            name=function_name,
            parameters=json.dumps(arguments, ensure_ascii=False),
        )
'''

STOCK_HARMONY = r'''import re


class CanonicalStrategy:
    def _parse_partial_analysis(self, text, tokens, message_pos, channel_start):
        channel_end = tokens[message_pos].start
        channel_header = text[channel_start:channel_end]

        channel_type = self._extract_channel_type(channel_header)
        if channel_type != "analysis":
            return None  # Only stream analysis content - tool calls wait for completion

        content_start = tokens[message_pos].end
        content = text[content_start:]
        return content
'''

# Harmony strings in the shapes gpt-oss actually emits. The first is the only
# one the stock pattern was written for; the other two are why tool calls go
# missing.
CANONICAL = '<|channel|>commentary to=functions.str_replace_editor <|constrain|>json<|message|>{"command": "view"}<|call|>'
NO_CONSTRAIN = '<|channel|>commentary to=functions.bash<|message|>{"command": "ls"}<|call|>'
ANALYSIS_CHANNEL = '<|channel|>analysis to=functions.bash code<|message|>{"command": "ls"}<|call|>'


def make_tree(root):
    """A minimal sglang package tree at the layout the patch expects."""
    det = root / "srt" / "function_call" / "gpt_oss_detector.py"
    har = root / "srt" / "parser" / "harmony_parser.py"
    det.parent.mkdir(parents=True, exist_ok=True)
    har.parent.mkdir(parents=True, exist_ok=True)
    det.write_text(STOCK_DETECTOR, encoding="utf-8")
    har.write_text(STOCK_HARMONY, encoding="utf-8")
    return det, har


def run_patch(root):
    return subprocess.run([sys.executable, str(PATCH_SCRIPT), "--root", str(root)],
                          text=True, capture_output=True)


def extract_tool_pattern(text):
    """The compiled-pattern literal out of a detector source file."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('r"to='):
            return ast.literal_eval(stripped.rstrip(","))
    raise AssertionError("no tool_extract_pattern literal found")


class PatchScriptTests(unittest.TestCase):

    def test_applies_every_anchor_and_leaves_valid_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sglang"
            det, har = make_tree(root)
            proc = run_patch(root)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            for path in (det, har):
                text = path.read_text(encoding="utf-8")
                self.assertIn("TEASBENCH_GPTOSS_PATCH", text)
                ast.parse(text)  # raises if a replacement broke the syntax

    def test_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sglang"
            det, har = make_tree(root)
            self.assertEqual(run_patch(root).returncode, 0)
            after_first = (det.read_text(encoding="utf-8"), har.read_text(encoding="utf-8"))
            second = run_patch(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already patched", second.stdout)
            self.assertEqual(after_first,
                             (det.read_text(encoding="utf-8"), har.read_text(encoding="utf-8")))

    def test_missing_anchor_fails_loudly(self):
        """An sglang whose parser code has moved must stop the engine, not be
        served with a parser the patch silently failed to fix."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sglang"
            det, _ = make_tree(root)
            moved = STOCK_DETECTOR.replace(
                "        return self.bot_token in text\n",
                '        return self.bot_token in (text or "")\n')
            self.assertNotIn("        return self.bot_token in text\n", moved)
            det.write_text(moved, encoding="utf-8")
            proc = run_patch(root)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("pattern not found", proc.stderr)

    def test_partial_application_still_fails(self):
        """The harmony file failing after the detector already succeeded must
        still be a non-zero exit."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sglang"
            _, har = make_tree(root)
            har.write_text("import re\n", encoding="utf-8")
            self.assertEqual(run_patch(root).returncode, 2)


class ToolPatternTests(unittest.TestCase):
    """The rewritten pattern must accept the shapes gpt-oss emits, and the
    stock one must be shown to reject them -- otherwise the replacement is
    cargo cult."""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sglang"
            det, _ = make_tree(root)
            self.stock = re.compile(extract_tool_pattern(det.read_text(encoding="utf-8")), re.DOTALL)
            self.assertEqual(run_patch(root).returncode, 0)
            self.patched = re.compile(extract_tool_pattern(det.read_text(encoding="utf-8")), re.DOTALL)

    def test_stock_pattern_only_matches_the_canonical_shape(self):
        self.assertTrue(self.stock.search(CANONICAL))
        self.assertIsNone(self.stock.search(NO_CONSTRAIN))
        self.assertIsNone(self.stock.search(ANALYSIS_CHANNEL))

    def test_patched_pattern_matches_all_three(self):
        for sample in (CANONICAL, NO_CONSTRAIN, ANALYSIS_CHANNEL):
            with self.subTest(sample=sample[:40]):
                match = self.patched.search(sample)
                self.assertIsNotNone(match)
                self.assertTrue(match.group(1).startswith("functions."))
                self.assertIn('"command"', match.group(2))


class GenerationTests(unittest.TestCase):
    """The hook reaches exactly the sglang engine Job and nothing else."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        target = Path(cls._tmp.name)
        subprocess.run([sys.executable, "generate.py",
                        "--csv_file", str(EXPERIMENTS / "swe-bench-lite-eidf.csv"),
                        "--target_dir", str(target)],
                       cwd=str(PIPELINE), text=True, capture_output=True, check=True)
        cls.generated = {p.name: p.read_text(encoding="utf-8") for p in target.iterdir()}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def engine_manifests(self, engine):
        return {n: t for n, t in self.generated.items()
                if n.startswith(engine) and n.endswith(".engine.yaml")}

    def test_sglang_engine_embeds_the_patch_verbatim(self):
        manifests = self.engine_manifests("sglang")
        self.assertTrue(manifests)
        source = PATCH_SCRIPT.read_bytes()
        for name, text in manifests.items():
            with self.subTest(manifest=name):
                args = yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]
                match = re.search(r"echo ([A-Za-z0-9+/=]+) \| base64 -d > /tmp/patch_sglang_gptoss\.py",
                                  args)
                self.assertIsNotNone(match, "no base64 payload in the engine args")
                self.assertEqual(base64.b64decode(match.group(1)), source)
                self.assertLess(args.index(match.group(0)),
                                args.index("sglang.launch_server"),
                                "the patch must run before the server starts")

    def test_payload_is_one_line(self):
        """get() re-indents multi-line replacements to the placeholder's YAML
        indentation. A payload split across lines would have that indentation
        spliced into it."""
        for name, text in self.engine_manifests("sglang").items():
            with self.subTest(manifest=name):
                args = yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]
                payload_lines = [l for l in args.splitlines() if "base64 -d" in l]
                self.assertEqual(len(payload_lines), 1)

    def test_vllm_engine_has_no_pre_launch_step(self):
        manifests = self.engine_manifests("vllm")
        self.assertTrue(manifests)
        for name, text in manifests.items():
            with self.subTest(manifest=name):
                args = yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]["args"][0]
                self.assertNotIn("base64 -d", args)
                self.assertNotIn("patch_sglang_gptoss", args)

    def test_driver_scripts_are_untouched(self):
        for name, text in self.generated.items():
            if name.endswith(".sh"):
                with self.subTest(script=name):
                    self.assertNotIn("patch_sglang_gptoss", text)

    def test_every_generated_manifest_still_parses(self):
        for name, text in self.generated.items():
            if name.endswith(".yaml"):
                with self.subTest(manifest=name):
                    yaml.safe_load(text)


if __name__ == "__main__":
    unittest.main()
