"""Tests for the Vast.ai launch-script generator (pipeline/vast_generate.py).

Covers the agentic family added on top of the MoE-only generator, and pins the
guarantee that the MoE path is unchanged by that work.

No network, no Vast.ai account, no Docker: generation is pure string building.
"""
import base64
import csv
import io
import pathlib
import subprocess
import sys
import tempfile
import unittest

PIPELINE_DIR = pathlib.Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from vast_generate import generate_vast_scripts  # noqa: E402


AGENTIC_HEADER = ("family,benchmark,inference_engine,model,gpu,num_gpu,"
                  "num_tasks,concurrency,batch_size")
MOE_HEADER = "family,inference_engine,model,dataset,num_samples,gpu,num_gpu,batch_size"


def _write_csv(tmpdir, name, lines):
    path = pathlib.Path(tmpdir) / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _generate(tmpdir, csv_path):
    out = pathlib.Path(tmpdir) / "out"
    generate_vast_scripts(csv_path, str(out))
    return {p.name: p.read_text() for p in out.glob("*.sh")}


def _decoded_csv(script_text):
    """Pull BENCHMARK_CSV out of a generated script and decode it."""
    for line in script_text.splitlines():
        if line.startswith("BENCHMARK_CSV="):
            b64 = line.split("=", 1)[1].strip().strip('"')
            return base64.b64decode(b64).decode()
    raise AssertionError("no BENCHMARK_CSV line in generated script")


class AgenticVastGenerationTests(unittest.TestCase):
    def test_agentic_rows_produce_benchmark_named_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "a.csv", [
                AGENTIC_HEADER,
                "agentic,swe-bench-lite,sglang,gpt-oss-120b,H200,1,100,4,default",
            ])
            scripts = _generate(tmp, path)
        self.assertEqual(
            sorted(scripts), ["vast_agentic_swe-bench-lite_sglang_H200x1.sh"])

    def test_per_benchmark_csvs_do_not_overwrite_each_other(self):
        """Three single-benchmark CSVs on the same engine/GPU must yield three
        distinct scripts. Naming only on family+engine+gpu silently collapsed
        them to one."""
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "out"
            for bench, n in (("imo-answerbench", 100), ("mcp-atlas", 60),
                             ("swe-bench-lite", 100)):
                path = _write_csv(tmp, f"{bench}.csv", [
                    AGENTIC_HEADER,
                    f"agentic,{bench},sglang,gpt-oss-120b,H200,1,{n},4,default",
                ])
                generate_vast_scripts(path, str(out))
            self.assertEqual(len(list(out.glob("*.sh"))), 3)

    def test_multi_benchmark_group_keeps_unqualified_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "a.csv", [
                AGENTIC_HEADER,
                "agentic,imo-answerbench,sglang,gpt-oss-120b,H200,1,100,4,default",
                "agentic,mcp-atlas,sglang,gpt-oss-120b,H200,1,60,4,default",
            ])
            scripts = _generate(tmp, path)
        self.assertEqual(sorted(scripts), ["vast_agentic_sglang_H200x1.sh"])

    def test_agentic_rows_use_the_separate_agentic_image(self):
        """The families ship as separate images so agentic deps (SWE-agent,
        swebench, modal, on a brittle engine base image) can never destabilise
        a MoE sweep."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "a.csv", [
                AGENTIC_HEADER,
                "agentic,mcp-atlas,vllm,gpt-oss-120b,H100,2,60,4,default",
            ])
            scripts = _generate(tmp, path)
        text = next(iter(scripts.values()))
        self.assertIn("ghcr.io/teas-project/vllm-agentic:latest", text)
        self.assertNotIn("vllm-bench", text)

    def test_swebench_declares_modal_secrets_and_others_do_not(self):
        """Modal is the SWE-bench sandbox substrate on Vast.ai; the other two
        benchmarks have no sandbox at all and must not demand its tokens."""
        with tempfile.TemporaryDirectory() as tmp:
            swe = _generate(tmp, _write_csv(tmp, "s.csv", [
                AGENTIC_HEADER,
                "agentic,swe-bench-lite,sglang,gpt-oss-120b,H200,1,100,4,default"]))
            imo = _generate(tmp, _write_csv(tmp, "i.csv", [
                AGENTIC_HEADER,
                "agentic,imo-answerbench,sglang,gpt-oss-120b,H200,1,100,4,default"]))
        swe_text = next(iter(swe.values()))
        imo_text = next(iter(imo.values()))
        self.assertIn("MODAL_TOKEN_ID", swe_text)
        self.assertIn("MODAL_TOKEN_SECRET", swe_text)
        self.assertNotIn("MODAL_TOKEN", imo_text)
        # Both judges use Gemini, so both need its key.
        self.assertIn("GEMINI_API_KEY", swe_text)
        self.assertIn("GEMINI_API_KEY", imo_text)

    def test_internal_family_column_is_not_shipped_to_the_container(self):
        """_family is a grouping helper, not part of the experiment definition;
        run_agentic_benchmarks.sh validates the header, so a stray column
        would be a real problem."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "a.csv", [
                AGENTIC_HEADER,
                "agentic,mcp-atlas,sglang,gpt-oss-120b,H200,1,60,4,default",
            ])
            scripts = _generate(tmp, path)
        decoded = _decoded_csv(next(iter(scripts.values())))
        header = next(csv.reader(io.StringIO(decoded)))
        self.assertNotIn("_family", header)
        self.assertEqual(header, AGENTIC_HEADER.split(","))

    def test_no_literal_secret_values_in_generated_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "a.csv", [
                AGENTIC_HEADER,
                "agentic,swe-bench-lite,sglang,gpt-oss-120b,H200,1,100,4,default",
            ])
            scripts = _generate(tmp, path)
        text = next(iter(scripts.values()))
        for leak in ("sk-", "ghp_", "YOUR_", "hf_"):
            self.assertNotIn(leak, text)


class MoERegressionTests(unittest.TestCase):
    """The MoE Vast.ai path must be untouched by the agentic work."""

    def test_moe_rows_keep_their_original_script_name_and_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "m.csv", [
                MOE_HEADER,
                "moe,sglang,gpt-oss-120b,gsm8k,256,H200,8,default",
            ])
            scripts = _generate(tmp, path)
        self.assertEqual(sorted(scripts), ["vast_sglang_H200x8.sh"])
        text = scripts["vast_sglang_H200x8.sh"]
        # Exact pre-agentic header wording.
        self.assertIn("# Vast.ai launch script — sglang on H200x8", text)
        self.assertIn("GIT_TOKEN, HF_TOKEN, OPENAI_API_KEY", text)
        # The MoE image is untouched by the agentic work.
        self.assertIn("ghcr.io/teas-project/sglang-bench:latest", text)
        self.assertNotIn("sglang-agentic", text)

    def test_mixed_csv_splits_into_one_script_per_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "mixed.csv", [
                "family,benchmark,inference_engine,model,dataset,num_samples,gpu,"
                "num_gpu,num_tasks,concurrency,batch_size",
                "moe,,sglang,gpt-oss-120b,gsm8k,256,H200,8,,,default",
                "agentic,mcp-atlas,sglang,gpt-oss-120b,,,H200,8,60,4,default",
            ])
            scripts = _generate(tmp, path)
        self.assertEqual(
            sorted(scripts),
            ["vast_agentic_mcp-atlas_sglang_H200x8.sh", "vast_sglang_H200x8.sh"])


class FamilyColumnTests(unittest.TestCase):
    """`family` is required and authoritative -- never inferred."""

    def test_missing_family_column_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "a.csv", [
                "inference_engine,model,dataset,num_samples,gpu,num_gpu,batch_size",
                "sglang,gpt-oss-120b,gsm8k,256,H200,8,default",
            ])
            with self.assertRaises(ValueError) as cm:
                _generate(tmp, path)
        self.assertIn("family", str(cm.exception))

    def test_unrecognised_family_value_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "a.csv", [
                MOE_HEADER,
                "basic,sglang,gpt-oss-120b,gsm8k,256,H200,8,default",
            ])
            with self.assertRaises(ValueError) as cm:
                _generate(tmp, path)
        self.assertIn("basic", str(cm.exception))

    def test_agentic_family_with_bad_benchmark_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "a.csv", [
                AGENTIC_HEADER,
                "agentic,gsm8k,sglang,gpt-oss-120b,H200,1,100,4,default",
            ])
            with self.assertRaises(ValueError) as cm:
                _generate(tmp, path)
        self.assertIn("gsm8k", str(cm.exception))


class GeneratedScriptSyntaxTests(unittest.TestCase):
    def test_generated_scripts_are_valid_bash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(tmp, "a.csv", [
                AGENTIC_HEADER,
                "agentic,swe-bench-lite,sglang,gpt-oss-120b,H200,1,100,4,default",
                "agentic,mcp-atlas,vllm,gpt-oss-120b,H100,2,60,4,default",
            ])
            out = pathlib.Path(tmp) / "out"
            generate_vast_scripts(path, str(out))
            for script in out.glob("*.sh"):
                r = subprocess.run(["bash", "-n", str(script)],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0,
                                 f"{script.name} is not valid bash: {r.stderr}")


if __name__ == "__main__":
    unittest.main()
