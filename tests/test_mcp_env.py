"""Tests for the MCP Atlas tool-server wiring shared by EIDF and Vast.ai.

The mcp-atlas *server set* is the benchmark definition -- AgentCAP's docs state
that dropping any server changes results -- so the two platforms enabling
different sets would silently make their numbers incomparable. That invariant is
the main thing pinned here, alongside the .env generator's behaviour.

No network, no Docker, no MCP server: the generator is pure text.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))

import yaml  # noqa: E402

from utils import MCP_ENABLED_SERVERS  # noqa: E402

WRITE_MCP_ENV = REPO / "pipeline" / "scripts" / "write_mcp_env.sh"
# The submodule lives in the sibling AgentCAP checkout.
ATLAS_TEMPLATE = (REPO.parent / "AgentCAP-teasbench" / "third_party"
                  / "mcp-atlas" / "env.template")


def _config():
    with open(REPO / "pipeline" / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


class EnabledServersParityTests(unittest.TestCase):
    def test_eidf_sidecar_enables_exactly_the_shared_server_set(self):
        """EIDF passes the list to the sidecar container as an env var."""
        blocks = [
            rule["sidecar_containers"]["value"]
            for rule in _config().get("rules", [])
            if isinstance(rule.get("sidecar_containers"), dict)
            and "value" in rule["sidecar_containers"]
        ]
        self.assertTrue(blocks, "no sidecar_containers rule found")
        found = [b for b in blocks if MCP_ENABLED_SERVERS in b]
        self.assertTrue(
            found,
            "EIDF sidecar_containers does not enable utils.MCP_ENABLED_SERVERS; "
            "the two platforms have drifted apart.")

    def test_vastai_tool_server_setup_enables_the_same_set(self):
        """Vast.ai passes the list to write_mcp_env.sh, which puts it in .env."""
        blocks = [
            rule["tool_server_setup"]["value"]
            for rule in _config().get("rules", [])
            if isinstance(rule.get("tool_server_setup"), dict)
            and "value" in rule["tool_server_setup"]
        ]
        self.assertTrue(blocks, "no tool_server_setup rule found")
        found = [b for b in blocks if MCP_ENABLED_SERVERS in b]
        self.assertTrue(
            found,
            "Vast.ai tool_server_setup does not enable utils.MCP_ENABLED_SERVERS; "
            "the two platforms have drifted apart.")

    def test_server_set_matches_the_archived_reference_runs(self):
        """22 servers, as used by the runs already in the results repo."""
        self.assertEqual(len(MCP_ENABLED_SERVERS.split(",")), 22)


@unittest.skipUnless(ATLAS_TEMPLATE.exists(),
                     "mcp-atlas submodule not checked out")
class WriteMcpEnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.atlas = pathlib.Path(self.tmp.name)
        shutil.copy(ATLAS_TEMPLATE, self.atlas / "env.template")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, env=None, servers=MCP_ENABLED_SERVERS):
        r = subprocess.run(
            ["bash", str(WRITE_MCP_ENV), str(self.atlas), servers],
            capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **(env or {})})
        self.assertEqual(r.returncode, 0, r.stderr)
        return r, (self.atlas / ".env").read_text()

    def _keys(self, text):
        return {line.split("=", 1)[0] for line in text.splitlines() if "=" in line}

    def test_every_template_key_is_present_in_the_output(self):
        """Keys come from the template, so a new upstream key can't be dropped."""
        _, env_text = self._run()
        template_keys = {
            line.split("=", 1)[0]
            for line in (self.atlas / "env.template").read_text().splitlines()
            if line and line[0].isupper() and "=" in line
        }
        self.assertEqual(self._keys(env_text), template_keys)

    def test_supplied_credentials_are_written_and_others_left_empty(self):
        _, env_text = self._run(env={"GITHUB_TOKEN": "tok", "BRAVE_API_KEY": "br"})
        self.assertIn("GITHUB_TOKEN=tok", env_text)
        self.assertIn("BRAVE_API_KEY=br", env_text)
        self.assertIn("ALCHEMY_API_KEY=\n", env_text)

    def test_enabled_servers_is_pinned_not_taken_from_the_environment(self):
        """A stray ENABLED_SERVERS in the container must not redefine the
        benchmark."""
        _, env_text = self._run(env={"ENABLED_SERVERS": "calculator"})
        self.assertIn(f"ENABLED_SERVERS={MCP_ENABLED_SERVERS}", env_text)
        self.assertNotIn("ENABLED_SERVERS=calculator", env_text)

    def test_provenance_lists_key_names_but_never_values(self):
        r, _ = self._run(env={"GITHUB_TOKEN": "supersecret"})
        self.assertIn("GITHUB_TOKEN", r.stdout)
        self.assertNotIn("supersecret", r.stdout)
        self.assertNotIn("supersecret", r.stderr)

    def test_missing_template_fails_loudly(self):
        (self.atlas / "env.template").unlink()
        r = subprocess.run(
            ["bash", str(WRITE_MCP_ENV), str(self.atlas)],
            capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("recurse-submodules", r.stderr)

    def test_env_file_is_not_world_readable(self):
        self._run(env={"GITHUB_TOKEN": "tok"})
        mode = (self.atlas / ".env").stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)


if __name__ == "__main__":
    unittest.main()
