"""Static invariants of pipeline/k8s/setup/setup_swebench_env.sh.

The script builds the login-node environment and cannot be run in CI (it
creates a venv, clones three repos and installs from PyPI). What can be
checked here is the part that is order-dependent and therefore easy to break
by tidying: pip reaches the same final versions whichever order these specs
are installed in, so nothing fails loudly if the order is changed -- it just
leaves the environment littered and every setup ending in a wall of
dependency-conflict errors.
"""
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "pipeline" / "k8s" / "setup" / "setup_swebench_env.sh"
sys.path.insert(0, str(REPO / "pipeline" / "k8s" / "lib"))

import version_check  # noqa: E402


class SpecInstallOrderTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.body = SETUP.read_text(encoding="utf-8")

    def test_ghapi_is_pinned_below_2(self):
        """swebench requires ghapi unpinned and SWE-agent requires ghapi<2.
        Without this pin the first install takes ghapi 2.x -- and its fastspec
        and fasttransport dependencies -- and the second downgrades it, so the
        two are left behind unsatisfiable."""
        match = re.search(r'^GHAPI_SPEC="\$\{GHAPI_SPEC:-([^}]+)\}"', self.body, re.MULTILINE)
        self.assertIsNotNone(match, "GHAPI_SPEC is not defined")
        self.assertTrue(version_check._satisfies_naive(
            "1.1.1", version_check.constraint_of(match.group(1))))
        self.assertFalse(version_check._satisfies_naive(
            "2.1.2", version_check.constraint_of(match.group(1))))

    def test_ghapi_is_installed_before_swebench(self):
        """pip's default --upgrade-strategy is only-if-needed, so an already
        satisfying ghapi is left alone when swebench is installed. That only
        helps if ghapi went in first."""
        match = re.search(r"^for spec in (.+); do$", self.body, re.MULTILINE)
        self.assertIsNotNone(match, "the spec install loop has moved")
        specs = match.group(1)
        self.assertIn('"$GHAPI_SPEC"', specs)
        self.assertLess(specs.index('"$GHAPI_SPEC"'), specs.index('"$SWEBENCH_SPEC"'),
                        "GHAPI_SPEC must be installed before SWEBENCH_SPEC")

    def test_every_spec_in_the_loop_yields_an_importable_module_name(self):
        """The loop derives the module to import by stripping the version
        operators off the spec. A spec whose distribution and module names
        differ needs the explicit mapping the loop already has for swe-rex."""
        match = re.search(r"^for spec in (.+); do$", self.body, re.MULTILINE)
        names = re.findall(r'"\$(\w+)"', match.group(1))
        defined = {n: re.search(rf'^{n}="\$\{{{n}:-([^}}]+)\}}"', self.body, re.MULTILINE)
                   for n in names}
        for name, found in defined.items():
            with self.subTest(spec=name):
                self.assertIsNotNone(found, f"{name} used in the loop but never defined")
                dist = re.split(r"[><=]", found.group(1))[0].strip()
                self.assertTrue(dist, f"{name} has no distribution name")
                self.assertNotIn(" ", dist)


if __name__ == "__main__":
    unittest.main()
