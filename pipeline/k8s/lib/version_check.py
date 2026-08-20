"""Does the *installed* distribution satisfy a pip requirement spec?

Importability is not the same question. An older -- or newer -- package imports
perfectly well and then fails somewhere far from the cause:

    swebench 5 moved swebench.harness.test_spec, so `import swebench` succeeds
    and AgentCAP dies mid-run with ModuleNotFoundError before its first task.

    swe-rex satisfying ">=1.4.0" still lacked the `timeout` kwarg SWE-agent's
    recovery path calls with, costing ~46% of a 100-task run its partial
    patches.

Both the setup script and the generated driver ask this question, so the
comparison lives here rather than being written twice: `pipeline/k8s/lib` is
already on the driver's PYTHONPATH (see env.sh), and the setup script invokes
this file by path.

Usage:
    python -m version_check <dist-name> <spec>       # e.g. swebench 'swebench>=2.0,<5'
    -> exit 0 if satisfied; exit 1 with a one-line reason on stderr otherwise

    from version_check import satisfies
    satisfies("swebench", "swebench>=2.0,<5")        # -> (bool, message)
"""

import re
import sys
from importlib import metadata

_OPS = {
    ">=": lambda c: c >= 0,
    "<=": lambda c: c <= 0,
    "==": lambda c: c == 0,
    "!=": lambda c: c != 0,
    ">": lambda c: c > 0,
    "<": lambda c: c < 0,
}


def constraint_of(spec):
    """'swebench>=2.0,<5' -> '>=2.0,<5'. Also accepts a bare constraint."""
    return re.sub(r"^[A-Za-z0-9_.\-\[\]]+", "", spec.strip()).strip()


def _parse_ver(v):
    out = []
    for part in v.split("."):
        m = re.match(r"\d+", part)
        out.append(int(m.group()) if m else 0)
    return out


def _cmp_ver(a, b):
    la, lb = _parse_ver(a), _parse_ver(b)
    n = max(len(la), len(lb))
    la += [0] * (n - len(la))
    lb += [0] * (n - len(lb))
    return (la > lb) - (la < lb)


def _satisfies_naive(installed, constraint):
    """Numeric-component comparison, stdlib only. Deliberately simple: it
    ignores pre-release/epoch/local-version semantics, which is why `packaging`
    is preferred when present. Adequate for the pins this pipeline uses."""
    for part in constraint.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(>=|<=|==|!=|>|<)\s*(.+)$", part)
        if not m:
            continue
        op, ver = m.group(1), m.group(2).strip()
        if not _OPS[op](_cmp_ver(installed, ver)):
            return False
    return True


def satisfies(dist_name, spec):
    """Returns (ok, message). `message` explains a failure, or names the
    installed version on success."""
    try:
        installed = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return False, f"{dist_name} is not installed (required '{spec}')"

    constraint = constraint_of(spec)
    if not constraint:
        return True, f"{dist_name} {installed} (no version constraint)"

    # `packaging` handles pre-releases and epochs properly and is present in any
    # venv pip has populated; the naive comparison is the bare-venv fallback.
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import InvalidVersion, Version

        try:
            ok = Version(installed) in SpecifierSet(constraint, prereleases=True)
        except InvalidVersion:
            ok = _satisfies_naive(installed, constraint)
    except Exception:
        ok = _satisfies_naive(installed, constraint)

    if ok:
        return True, f"{dist_name} {installed} satisfies '{spec}'"
    return False, f"{dist_name} {installed} does not satisfy '{spec}'"


def main(argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} <dist-name> <spec>", file=sys.stderr)
        return 2
    ok, message = satisfies(argv[1], argv[2])
    print(message, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
