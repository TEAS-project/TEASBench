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
import subprocess
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


class ResolveSettingTests(unittest.TestCase):
    """Step 7's five paths, run for real.

    env.sh exports the namespace and the secret name and is normally sourced
    in the working shell, so "already set" must not mean "already decided" --
    that made a re-run of this script unable to change either while reporting
    success. Each path is exercised through the real function text rather than
    a restatement of it.
    """

    @classmethod
    def setUpClass(cls):
        body = SETUP.read_text(encoding="utf-8")
        fn = re.search(r"^resolve_setting\(\) \{.*?^\}$", body, re.MULTILINE | re.DOTALL)
        assert fn, "resolve_setting() not found -- has step 7 been restructured?"
        cls.fn = fn.group(0)

    def run_resolve(self, current, from_flag, tty, stdin=""):
        """Call the real resolve_setting with ok/did/interactive stubbed."""
        harness = "\n".join([
            'ok()  { echo "  ok      $1"; }',
            'did() { echo "  done    $1"; }',
            f'interactive() {{ return {0 if tty else 1}; }}',
            self.fn,
            f'resolve_setting "{current}" {from_flag} "thedefault" '
            '"Prompt text" "label" "--the-flag"',
            'echo "RESOLVED=[$RESOLVED]"',
        ])
        proc = subprocess.run(["bash", "-c", harness], input=stdin,
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # bash only emits `read -p`'s prompt when stdin is a terminal, and
        # these run against a pipe -- so the prompt text is never in the
        # output and cannot be asserted on here. What the prompt offers as its
        # default is covered statically by
        # test_prompts_offer_the_inherited_value_as_default; what the operator
        # gets is covered behaviourally below.
        return proc.stdout + proc.stderr

    def test_flag_wins_and_is_never_re_asked(self):
        out = self.run_resolve("fromflag", 1, tty=True, stdin="typed\n")
        self.assertIn("RESOLVED=[fromflag]", out)
        self.assertIn("given with --the-flag", out)
        self.assertNotIn("typed", out, "the flag path must not read stdin")

    def test_inherited_value_is_offered_as_the_default_and_kept_on_enter(self):
        out = self.run_resolve("frompreviousrun", 0, tty=True, stdin="\n")
        self.assertIn("RESOLVED=[frompreviousrun]", out)
        self.assertIn("inherited from the environment", out)

    def test_inherited_value_can_be_overridden_at_the_prompt(self):
        """The regression this exists for: a re-run must be able to change it."""
        out = self.run_resolve("frompreviousrun", 0, tty=True, stdin="somethingelse\n")
        self.assertIn("RESOLVED=[somethingelse]", out)

    def test_inherited_value_stands_non_interactively_and_says_so(self):
        out = self.run_resolve("frompreviousrun", 0, tty=False)
        self.assertIn("RESOLVED=[frompreviousrun]", out)
        self.assertIn("inherited from the environment", out)
        self.assertIn("--the-flag", out, "must say how to change it")

    def test_unset_prompts_with_the_builtin_default(self):
        out = self.run_resolve("", 0, tty=True, stdin="\n")
        self.assertIn("RESOLVED=[thedefault]", out)

    def test_unset_and_non_interactive_falls_back_with_a_note(self):
        out = self.run_resolve("", 0, tty=False)
        self.assertIn("RESOLVED=[thedefault]", out)
        self.assertIn("non-interactive", out)

    def test_prompts_offer_the_inherited_value_as_default(self):
        """Static, because bash hides the prompt when stdin is not a tty. An
        inherited value must be shown as the default the operator is accepting
        -- offering the built-in default there would invite them to press
        Enter and silently change the namespace."""
        inherited, unset = self.fn.split("elif interactive; then")
        self.assertIn('read -r -p "  $prompt [$current]: " reply', inherited)
        self.assertIn('read -r -p "  $prompt [$fallback]: " reply', unset)

    def test_help_lists_every_accepted_flag(self):
        """The status messages tell the operator to pass a flag, so --help has
        to name it. Also guards the extraction: --help used to print a fixed
        line range, which truncates as soon as the header grows."""
        proc = subprocess.run(["bash", str(SETUP), "--help"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = SETUP.read_text(encoding="utf-8")
        flags = set(re.findall(r"^\s+(--[a-z-]+)\)", body, re.MULTILINE))
        self.assertIn("--namespace", flags)
        for flag in flags:
            with self.subTest(flag=flag):
                self.assertIn(flag, proc.stdout, f"{flag} is accepted but undocumented")
        self.assertNotIn("set -uo pipefail", proc.stdout,
                         "--help must stop at the end of the header")

    def test_both_settings_go_through_it(self):
        body = SETUP.read_text(encoding="utf-8")
        for var, flag in (("TEASBENCH_K8S_NAMESPACE", "--namespace"),
                          ("GIT_TOKEN_K8S_SECRET", "--git-token-secret")):
            with self.subTest(setting=var):
                call = re.search(rf'resolve_setting "\${var}".*?\n{var}="\$RESOLVED"',
                                 body, re.DOTALL)
                self.assertIsNotNone(call, f"{var} does not go through resolve_setting")
                self.assertIn(flag, call.group(0))


if __name__ == "__main__":
    unittest.main()
