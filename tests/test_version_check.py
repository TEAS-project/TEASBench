"""Unit tests for pipeline/k8s/lib/version_check.py.

This gates whether a run is allowed to start, and it is the check that would
have caught swebench 5 being installed under a `>=2.0` pin -- so its comparison
table is worth pinning down rather than trusting by inspection.
"""

import builtins
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "pipeline" / "k8s" / "lib"
sys.path.insert(0, str(LIB))

import version_check  # noqa: E402


class ConstraintParsingTests(unittest.TestCase):
    def test_strips_distribution_name(self):
        self.assertEqual(version_check.constraint_of("swebench>=2.0,<5"), ">=2.0,<5")
        self.assertEqual(version_check.constraint_of("swe-rex>=1.4.0"), ">=1.4.0")

    def test_accepts_bare_constraint_and_extras(self):
        self.assertEqual(version_check.constraint_of(">=1.4.0"), ">=1.4.0")
        self.assertEqual(version_check.constraint_of("swebench[dev]>=2.0"), ">=2.0")

    def test_unconstrained_spec_yields_empty(self):
        self.assertEqual(version_check.constraint_of("swebench"), "")


class NaiveComparisonTests(unittest.TestCase):
    """The bare-venv fallback, used when `packaging` is unavailable."""

    CASES = [
        # (installed, constraint, expected)
        ("4.1.0", ">=2.0,<5", True),
        ("4.0", ">=2.0,<5", True),
        ("2.0", ">=2.0,<5", True),
        ("5.0.0", ">=2.0,<5", False),   # the swebench 5 regression
        ("5.0.1", ">=2.0,<5", False),
        ("1.9", ">=2.0,<5", False),
        ("1.4.0", ">=1.4.0", True),
        ("1.3.0", ">=1.4.0", False),
        ("2.0", "==2.0", True),
        ("2.0.1", "!=2.0", True),
    ]

    def test_table(self):
        for installed, constraint, expected in self.CASES:
            with self.subTest(installed=installed, constraint=constraint):
                self.assertEqual(
                    version_check._satisfies_naive(installed, constraint), expected)

    def test_shorter_version_padded_not_truncated(self):
        # "5" must compare equal to "5.0.0", so `<5` rejects 5.0.0 rather than
        # letting it through on a length mismatch.
        self.assertFalse(version_check._satisfies_naive("5.0.0", "<5"))
        self.assertTrue(version_check._satisfies_naive("4.9.9", "<5"))


class SatisfiesTests(unittest.TestCase):
    def test_absent_distribution_reports_not_installed(self):
        ok, msg = version_check.satisfies("definitely-not-installed-xyz",
                                          "definitely-not-installed-xyz>=1")
        self.assertFalse(ok)
        self.assertIn("not installed", msg)

    def test_installed_package_satisfied_and_unsatisfied(self):
        ok, msg = version_check.satisfies("pytest", "pytest>=1.0")
        self.assertTrue(ok, msg)
        ok, msg = version_check.satisfies("pytest", "pytest>=999")
        self.assertFalse(ok)
        self.assertIn("does not satisfy", msg)

    def test_unconstrained_spec_passes_when_installed(self):
        ok, msg = version_check.satisfies("pytest", "pytest")
        self.assertTrue(ok, msg)

    def test_falls_back_when_packaging_is_unavailable(self):
        """The venv on a login node may predate `packaging`; the naive path
        must produce the same verdict for the pins this pipeline uses."""
        real_import = builtins.__import__

        def no_packaging(name, *args, **kwargs):
            if name.startswith("packaging"):
                raise ImportError("hidden for test")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = no_packaging
        try:
            ok, _ = version_check.satisfies("pytest", "pytest>=1.0")
            self.assertTrue(ok)
            ok, _ = version_check.satisfies("pytest", "pytest>=999")
            self.assertFalse(ok)
        finally:
            builtins.__import__ = real_import


class CliTests(unittest.TestCase):
    """The driver and the setup script both shell out to this, so the exit
    codes and stream routing are the actual interface."""

    def _run(self, *args):
        return subprocess.run([sys.executable, str(LIB / "version_check.py"), *args],
                              text=True, capture_output=True)

    def test_exit_zero_and_reason_on_stdout_when_satisfied(self):
        r = self._run("pytest", "pytest>=1.0")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("satisfies", r.stdout)

    def test_exit_one_and_reason_on_stderr_when_not(self):
        r = self._run("pytest", "pytest>=999")
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not satisfy", r.stderr)

    def test_usage_error(self):
        self.assertEqual(self._run("only-one-arg").returncode, 2)


if __name__ == "__main__":
    unittest.main()
