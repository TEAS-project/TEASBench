#!/usr/bin/env bash
#
# One-time environment setup for running SWE-bench Lite on EIDF.
#
#     bash pipeline/k8s/setup/setup_swebench_env.sh
#     source ~/teasbench-env/env.sh          # then, in any shell that runs a driver
#
# SWE-bench on EIDF is driven from a login node (EIDF grants pods no RBAC, so
# the driver must create per-task sandbox Jobs with your own credentials -- see
# docs/DEVELOPER_GUIDE.md 5). That means a real Python environment has to exist
# *here*, unlike every other TEASBench run where the Job builds its own.
#
# This script prepares that environment and writes two files:
#
#     $PREFIX/env.sh        exports the generated driver reads. The driver has
#                           no install paths of its own; everything comes from
#                           here, so moving a checkout means re-running this
#                           script, not editing generated scripts.
#     $PREFIX/versions.json resolved commits and versions of every dependency,
#                           which the driver folds into each run's metadata so a
#                           result can be traced to the exact code that produced
#                           it.
#
# Idempotent: every step checks first and does nothing if already satisfied.
# Re-run it after changing a branch, or with --force to rebuild from scratch.
set -uo pipefail

PREFIX="${TEASBENCH_ENV_PREFIX:-$HOME/teasbench-env}"
AGENTCAP_REPO="${AGENTCAP_REPO:-https://github.com/Auto-CAP/AgentCAP.git}"
AGENTCAP_REF="${AGENTCAP_REF:-main}"
SWEAGENT_REPO="${SWEAGENT_REPO:-https://github.com/SWE-agent/SWE-agent.git}"
# Pinned, not "main". SWE-agent upstream moves continuously and this pipeline
# depends on two things that drift with it: the exact litellm.completion block
# AgentCAP's streaming patch string-matches, and the CLI/trajectory surface
# agent_cap/agents/strategies_sweagent.py drives.
#
# Override for a bisect or an upgrade trial; accepts a tag, branch or SHA:
#   SWEAGENT_REF=v1.1.0 bash pipeline/k8s/setup/setup_swebench_env.sh
SWEAGENT_REF="${SWEAGENT_REF:-3ea751c087f32b16e039a2233dd6eefecef325d5}"
SWEREX_SPEC="${SWEREX_SPEC:-swe-rex>=1.4.0}"
SWEBENCH_SPEC="${SWEBENCH_SPEC:-swebench>=2.0}"
PYTHON="${PYTHON:-python3}"
FORCE=0
# Namespace the driver operates in, and the k8s secret it reads GIT_TOKEN from
# at push time (kubectl -n <namespace> get secret <name>, key 'token' -- same
# secret already used via secretKeyRef by the in-cluster agentic Jobs, see
# pipeline/README.md "k8s secrets required"). Left empty here so the prompt
# step below only fires when neither an env var nor a flag supplied one.
TEASBENCH_K8S_NAMESPACE="${TEASBENCH_K8S_NAMESPACE:-}"
GIT_TOKEN_K8S_SECRET="${GIT_TOKEN_K8S_SECRET:-}"

TEASBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)             PREFIX="$2"; shift 2 ;;
        --agentcap-ref)       AGENTCAP_REF="$2"; shift 2 ;;
        --agentcap-repo)      AGENTCAP_REPO="$2"; shift 2 ;;
        --python)             PYTHON="$2"; shift 2 ;;
        --namespace)          TEASBENCH_K8S_NAMESPACE="$2"; shift 2 ;;
        --git-token-secret)   GIT_TOKEN_K8S_SECRET="$2"; shift 2 ;;
        --force)              FORCE=1; shift ;;
        -h|--help)            sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

VENV="$PREFIX/venv"
AGENTCAP_DIR="$PREFIX/AgentCAP"
SWEAGENT_DIR="$PREFIX/swe_agent"
PY="$VENV/bin/python"

step() { printf '\n\033[1m[%s]\033[0m %s\n' "$1" "$2"; }
ok()   { echo "  ok      $1"; }
did()  { echo "  done    $1"; }
die()  { echo "  ERROR   $1" >&2; exit 1; }

echo "=============================================================="
echo "TEASBench SWE-bench environment setup"
echo "  prefix     : $PREFIX"
echo "  AgentCAP   : $AGENTCAP_REPO @ $AGENTCAP_REF"
echo "  TEASBench  : $TEASBENCH_ROOT"
echo "=============================================================="

if [ $FORCE -eq 1 ] && [ -d "$PREFIX" ]; then
    step 0 "--force: removing $PREFIX"
    rm -rf "$PREFIX"
fi
mkdir -p "$PREFIX"

# ---------------------------------------------------------------- venv -----
step 1 "Python environment"
if [ -x "$PY" ]; then
    ok "venv exists ($($PY --version 2>&1))"
else
    command -v "$PYTHON" > /dev/null || die "$PYTHON not found; pass --python."
    "$PYTHON" -m venv "$VENV" || die "could not create a venv at $VENV"
    "$PY" -m pip install --quiet --upgrade pip
    did "created venv ($($PY --version 2>&1))"
fi

# ------------------------------------------------------------- AgentCAP ----
# Cloned rather than pip-installed from a URL: the driver reads task-index JSONs
# and patch_sweagent_streaming.py straight out of the working tree, and a
# checkout is what lets us record the exact commit in versions.json.
step 2 "AgentCAP"
if [ -d "$AGENTCAP_DIR/.git" ]; then
    ok "checkout exists"
    current_ref=$(git -C "$AGENTCAP_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)
    if [ "$current_ref" != "$AGENTCAP_REF" ]; then
        echo "  note    on '$current_ref', switching to '$AGENTCAP_REF'"
        git -C "$AGENTCAP_DIR" fetch --quiet origin "$AGENTCAP_REF" \
            && git -C "$AGENTCAP_DIR" checkout --quiet "$AGENTCAP_REF" \
            || die "could not switch to $AGENTCAP_REF"
        did "switched to $AGENTCAP_REF"
    fi
else
    # --recurse-submodules: mcp-atlas is a submodule. Not needed by SWE-bench,
    # but cloning it now means the same checkout serves mcp-atlas runs later.
    git clone --quiet --recurse-submodules --branch "$AGENTCAP_REF" \
        "$AGENTCAP_REPO" "$AGENTCAP_DIR" || die "clone failed: $AGENTCAP_REPO@$AGENTCAP_REF"
    did "cloned $AGENTCAP_REF"
fi

if "$PY" -c 'import agent_cap' 2>/dev/null; then
    ok "agent_cap importable"
else
    "$PY" -m pip install --quiet -e "$AGENTCAP_DIR" || die "pip install -e AgentCAP failed"
    did "installed agent_cap (editable)"
fi

[ -f "$AGENTCAP_DIR/benchmarks/swe_bench_lite_curated_100.json" ] \
    || die "curated task list missing from $AGENTCAP_DIR -- wrong branch?"
ok "curated-100 task list present"

# ------------------------------------------------- swe-rex and swebench ----
step 3 "swe-rex and swebench"
# Importability alone doesn't prove $SWEREX_SPEC / $SWEBENCH_SPEC is actually
# satisfied -- an older swe-rex already sitting in the venv imports fine, so
# the version pin was silently unenforced. version_satisfies checks the
# *installed* distribution against the spec's constraint with
# importlib.metadata (stdlib only, no `packaging` dep -- this runs against a
# bare venv). If it doesn't satisfy, upgrade and re-check once before die-ing.
version_satisfies() {
    # $1 = distribution name (importlib.metadata key), $2 = full spec, e.g. "swe-rex>=1.4.0"
    "$PY" - "$1" "$2" <<'PYEOF'
import re, sys
from importlib import metadata

dist_name, spec = sys.argv[1], sys.argv[2]
constraint = re.sub(r'^[A-Za-z0-9_.-]+', '', spec).strip()

try:
    installed = metadata.version(dist_name)
except metadata.PackageNotFoundError:
    sys.exit(1)

def parse_ver(v):
    out = []
    for p in v.split('.'):
        m = re.match(r'\d+', p)
        out.append(int(m.group()) if m else 0)
    return out

def cmp_ver(a, b):
    la, lb = parse_ver(a), parse_ver(b)
    n = max(len(la), len(lb))
    la += [0] * (n - len(la))
    lb += [0] * (n - len(lb))
    return (la > lb) - (la < lb)

ops = {'>=': lambda c: c >= 0, '<=': lambda c: c <= 0, '==': lambda c: c == 0,
       '!=': lambda c: c != 0, '>': lambda c: c > 0, '<': lambda c: c < 0}

for part in constraint.split(','):
    part = part.strip()
    if not part:
        continue
    m = re.match(r'^(>=|<=|==|!=|>|<)\s*(.+)$', part)
    if not m:
        continue
    op, ver = m.group(1), m.group(2).strip()
    if not ops[op](cmp_ver(installed, ver)):
        sys.exit(1)
sys.exit(0)
PYEOF
}

for spec in "$SWEREX_SPEC" "$SWEBENCH_SPEC"; do
    mod=$(echo "$spec" | sed 's/[><=].*//' | tr -d ' ')
    imp=$(echo "$mod" | tr '-' '_'); [ "$imp" = "swe_rex" ] && imp=swerex
    if "$PY" -c "import $imp" 2>/dev/null && version_satisfies "$mod" "$spec"; then
        ok "$mod importable and satisfies '$spec'"
    else
        "$PY" -m pip install --quiet --upgrade "$spec" || die "pip install '$spec' failed"
        "$PY" -c "import $imp" 2>/dev/null && version_satisfies "$mod" "$spec" \
            || die "installed '$spec' but the venv still does not satisfy it -- check for a pinned/conflicting requirement elsewhere in the venv"
        did "installed $spec"
    fi
done

# ------------------------------------------------------------ SWE-agent ----
step 4 "SWE-agent (streaming-patched)"
SWEAGENT_MOVED=0
if [ -d "$SWEAGENT_DIR/.git" ]; then
    ok "checkout exists"
else
    # No --branch: it takes tags and branches but not commit SHAs, and
    # SWEAGENT_REF is pinned to one.
    git clone --quiet "$SWEAGENT_REPO" "$SWEAGENT_DIR" \
        || die "clone failed: $SWEAGENT_REPO"
    did "cloned SWE-agent"
fi

# Verify the checkout is actually at $SWEAGENT_REF instead of assuming any
# pre-existing directory is the right one. SWE-agent used to be cloned from an
# unpinned "main", so every checkout already on a login node sits at whatever
# main happened to be that day -- exactly the drift the pin exists to stop.
WANT_SHA="$(git -C "$SWEAGENT_DIR" rev-parse --verify --quiet "${SWEAGENT_REF}^{commit}" || true)"
if [ -z "$WANT_SHA" ]; then
    git -C "$SWEAGENT_DIR" fetch --quiet --tags origin || die "git fetch failed in $SWEAGENT_DIR"
    WANT_SHA="$(git -C "$SWEAGENT_DIR" rev-parse --verify --quiet "${SWEAGENT_REF}^{commit}" || true)"
    [ -n "$WANT_SHA" ] || die "SWEAGENT_REF '$SWEAGENT_REF' not found in $SWEAGENT_REPO"
fi
HAVE_SHA="$(git -C "$SWEAGENT_DIR" rev-parse HEAD)"
if [ "$HAVE_SHA" = "$WANT_SHA" ]; then
    ok "at pinned ref ${SWEAGENT_REF} (${WANT_SHA:0:7})"
else
    # Moving to the pinned ref reverts models.py and so drops the streaming
    # patch. That is fine and self-healing: the patch step below re-applies it
    # whenever the marker is absent. Discarding anything *else* someone edited
    # by hand is not fine, so refuse rather than force-checkout over it.
    # `diff --name-only HEAD` lists tracked modifications only -- untracked
    # build droppings (*.egg-info, __pycache__) are not at risk and must not
    # block the move.
    DIRTY="$(git -C "$SWEAGENT_DIR" diff --name-only HEAD | grep -v '^sweagent/agent/models\.py$' || true)"
    if [ -n "$DIRTY" ]; then
        echo "  local changes beyond the streaming patch:" >&2
        echo "$DIRTY" | sed 's/^/    /' >&2
        die "$SWEAGENT_DIR is at ${HAVE_SHA:0:7}, not the pinned ${WANT_SHA:0:7}, and has local edits (listed above). Move or discard them, or delete $SWEAGENT_DIR and re-run."
    fi
    git -C "$SWEAGENT_DIR" checkout --quiet --force "$WANT_SHA" \
        || die "could not check out '$SWEAGENT_REF' in $SWEAGENT_DIR"
    SWEAGENT_MOVED=1
    did "moved SWE-agent ${HAVE_SHA:0:7} -> pinned ${WANT_SHA:0:7} (streaming patch re-applied below)"
fi

# A ref change can bring different dependencies with it, and `import sweagent`
# would still succeed against the stale editable install -- so reinstall
# unconditionally when we just moved, rather than trusting importability.
if [ "$SWEAGENT_MOVED" -eq 0 ] && "$PY" -c 'import sweagent' 2>/dev/null; then
    ok "sweagent importable"
else
    "$PY" -m pip install --quiet -e "$SWEAGENT_DIR" || die "pip install -e SWE-agent failed"
    did "installed sweagent (editable)"
fi

# The patch routes SWE-agent's single litellm.completion call through
# agent_cap.sweagent_streaming, which is what captures per-call TTFT/TPOT and the
# visible/reasoning/cached token split. Without it a run completes normally and
# every agentic metric is empty -- so this is verified, not assumed.
MARKER=AGENTCAP_STREAMING_PATCH_APPLIED
MODELS_PY="$SWEAGENT_DIR/sweagent/agent/models.py"
[ -f "$MODELS_PY" ] || die "$MODELS_PY not found -- unexpected SWE-agent layout"
if grep -q "$MARKER" "$MODELS_PY"; then
    ok "streaming patch already applied"
else
    "$PY" "$AGENTCAP_DIR/scripts/patch_sweagent_streaming.py" "$SWEAGENT_DIR" \
        || die "patch_sweagent_streaming.py failed"
    grep -q "$MARKER" "$MODELS_PY" \
        || die "patch reported success but the marker is absent -- do not run; metrics would be empty"
    did "applied streaming patch"
fi

# ------------------------------------------- SWE-agent / swe-rex compat ----
# The production failure this guards against (evidence run 20260807-0001):
# SWE-agent's *recovery* path -- attempt_autosubmission_after_error, the code
# that salvages a partial patch when a task hits an error mid-run -- calls
# `self._env.deployment.is_alive(timeout=10)` (sweagent/agent/agents.py:831).
# 46 of that run's 55 failed tasks crashed identically:
#   TypeError: RemoteDeployment.is_alive() got an unexpected keyword argument 'timeout'
# because the installed swe-rex predates the `timeout` kwarg this SWE-agent
# calls with, turning every recoverable error into a total loss of the task's
# work. Step 3's version check alone would not have caught this: a build can
# satisfy ">=1.4.0" and still lack the kwarg, so this checks the actual API
# SWE-agent depends on. Placed here (after step 4 installs SWE-agent) rather
# than folded into step 3, because it is a compatibility assertion between the
# two packages, not a property of swe-rex alone -- it needs both installed to
# mean anything.
step 5 "SWE-agent / swe-rex compatibility"
compat_check() {
    "$PY" - <<'PYEOF'
import inspect
try:
    from swerex.deployment.remote import RemoteDeployment
except Exception as e:
    print(f"FAIL import swerex.deployment.remote.RemoteDeployment: {e}")
    raise SystemExit(0)

try:
    params = inspect.signature(RemoteDeployment.is_alive).parameters
except (TypeError, ValueError) as e:
    print(f"FAIL inspect RemoteDeployment.is_alive: {e}")
    raise SystemExit(0)

if "timeout" in params:
    print("OK")
else:
    print("FAIL RemoteDeployment.is_alive has no 'timeout' parameter")
PYEOF
}

COMPAT_RESULT=$(compat_check)
if [ "$COMPAT_RESULT" = "OK" ]; then
    ok "RemoteDeployment.is_alive accepts timeout= (SWE-agent's recovery path is compatible)"
else
    # Upgrade first, in case upstream has since fixed it -- then the patch
    # below detects the working signature and no-ops.
    echo "  note    $COMPAT_RESULT -- attempting upgrade of '$SWEREX_SPEC'"
    "$PY" -m pip install --quiet --upgrade "$SWEREX_SPEC" || die "pip install --upgrade '$SWEREX_SPEC' failed"
    COMPAT_RESULT=$(compat_check)
    if [ "$COMPAT_RESULT" = "OK" ]; then
        did "upgraded swe-rex; RemoteDeployment.is_alive now accepts timeout="
    else
        # No released swe-rex fixes this -- 1.4.0 is the latest published
        # version and upstream main still overrides is_alive without the
        # kwarg its own AbstractDeployment declares. So patch it, the same
        # way step 4 patches SWE-agent for streaming.
        echo "  note    no swe-rex release fixes this; patching RemoteDeployment.is_alive"
        "$PY" "$TEASBENCH_ROOT/pipeline/k8s/setup/patch_swerex_is_alive.py" \
            || die "patch_swerex_is_alive.py failed -- see the error above"
        COMPAT_RESULT=$(compat_check)
        if [ "$COMPAT_RESULT" = "OK" ]; then
            did "patched swe-rex; RemoteDeployment.is_alive now accepts timeout="
        else
            die "swe-rex/SWE-agent still incompatible after patching: $COMPAT_RESULT -- attempt_autosubmission_after_error (agents.py:831) calls is_alive(timeout=...), so every recoverable task error would crash instead of salvaging a patch. Do not run: ~46% of tasks would lose their partial patch."
        fi
    fi
fi

# ------------------------------------------------------------- kubectl -----
step 6 "kubectl"
if command -v kubectl > /dev/null; then
    ok "kubectl on PATH ($(kubectl version --client -o json 2>/dev/null | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["clientVersion"]["gitVersion"])' 2>/dev/null || echo 'version unknown'))"
else
    echo "  WARN    kubectl not on PATH. It is not installed here because on EIDF"
    echo "          it is normally provided by the login node; load your module or"
    echo "          install it before running a driver."
fi

# ------------------------------------------------- namespace & git token ---
# The driver has no RBAC-granted way to read a secret for you implicitly; it
# needs to know which namespace and which secret by name (see
# templates/agentic-driver.sh step [5]). Asked once here rather than baked
# into the generated driver script, so relocating/re-pointing at a different
# namespace or secret is a re-run of this script, not an edit of generated
# output.
step 7 "Kubernetes namespace and results-push secret"
DEFAULT_NAMESPACE="eidf230ns"
DEFAULT_GIT_TOKEN_SECRET="teas-develop-results-private-ap"

if [ -n "$TEASBENCH_K8S_NAMESPACE" ]; then
    ok "namespace: $TEASBENCH_K8S_NAMESPACE"
elif [ -t 0 ]; then
    read -r -p "  EIDF k8s namespace [$DEFAULT_NAMESPACE]: " TEASBENCH_K8S_NAMESPACE
    TEASBENCH_K8S_NAMESPACE="${TEASBENCH_K8S_NAMESPACE:-$DEFAULT_NAMESPACE}"
    did "namespace: $TEASBENCH_K8S_NAMESPACE"
else
    TEASBENCH_K8S_NAMESPACE="$DEFAULT_NAMESPACE"
    echo "  note    non-interactive; defaulting namespace to $TEASBENCH_K8S_NAMESPACE"
fi

if [ -n "$GIT_TOKEN_K8S_SECRET" ]; then
    ok "GIT_TOKEN secret: $GIT_TOKEN_K8S_SECRET"
elif [ -t 0 ]; then
    read -r -p "  k8s secret holding the results-repo GIT_TOKEN (key 'token') [$DEFAULT_GIT_TOKEN_SECRET]: " GIT_TOKEN_K8S_SECRET
    GIT_TOKEN_K8S_SECRET="${GIT_TOKEN_K8S_SECRET:-$DEFAULT_GIT_TOKEN_SECRET}"
    did "GIT_TOKEN secret: $GIT_TOKEN_K8S_SECRET"
else
    GIT_TOKEN_K8S_SECRET="$DEFAULT_GIT_TOKEN_SECRET"
    echo "  note    non-interactive; defaulting GIT_TOKEN secret to $GIT_TOKEN_K8S_SECRET"
fi

# ------------------------------------------------------------- versions ----
# Recorded now, at the moment the environment is built, so a run's metadata
# describes the code that actually produced it.
step 8 "Recording versions"
"$PY" - "$PREFIX" "$AGENTCAP_DIR" "$SWEAGENT_DIR" "$TEASBENCH_ROOT" "$COMPAT_RESULT" <<'PYEOF'
import json, subprocess, sys
from importlib import metadata
from pathlib import Path

prefix, agentcap, sweagent, teasbench = map(Path, sys.argv[1:5])
compat_result = sys.argv[5]

def commit(repo):
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def ref(repo):
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse",
                                        "--abbrev-ref", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def dist(name):
    try:
        return metadata.version(name)
    except Exception:
        return None

versions = {
    "agentcap": {"commit": commit(agentcap), "ref": ref(agentcap), "path": str(agentcap)},
    "sweagent": {"commit": commit(sweagent), "ref": ref(sweagent), "path": str(sweagent),
                 "streaming_patch": "AGENTCAP_STREAMING_PATCH_APPLIED"},
    "teasbench": {"commit": commit(teasbench), "ref": ref(teasbench), "path": str(teasbench)},
    "swe_rex": dist("swe-rex"),
    "swebench": dist("swebench"),
    "sweagent_dist": dist("sweagent"),
    # attempt_autosubmission_after_error (SWE-agent's recovery path,
    # agents.py:831) calls RemoteDeployment.is_alive(timeout=...); step 5
    # already verified this and would have die'd otherwise, so
    # is_alive_accepts_timeout is always True by the time this file is
    # written. Recorded anyway (not just asserted) so a run's metadata in the
    # results repo is traceable to a known-good environment without having to
    # re-derive it from setup logs -- the same reasoning as the streaming
    # patch marker above.
    "swe_rex_compat": {"is_alive_accepts_timeout": compat_result == "OK",
                        "detail": compat_result},
    "python": sys.version.split()[0],
}
out = prefix / "versions.json"
out.write_text(json.dumps(versions, indent=2) + "\n")
print(f"  done    {out}")
for k in ("agentcap", "sweagent", "teasbench"):
    v = versions[k]
    print(f"          {k:10s} {(v['commit'] or '?')[:8]}  ({v['ref']})")
for k in ("swe_rex", "swebench"):
    print(f"          {k:10s} {versions[k]}")
print(f"          swe_rex_compat  is_alive_accepts_timeout={versions['swe_rex_compat']['is_alive_accepts_timeout']}")
PYEOF
[ -f "$PREFIX/versions.json" ] || die "could not write versions.json"

# ------------------------------------------------------------- env file ----
step 9 "Writing $PREFIX/env.sh"
cat > "$PREFIX/env.sh" <<EOF
# Generated by pipeline/k8s/setup/setup_swebench_env.sh -- do not edit; re-run that.
#
#     source $PREFIX/env.sh
#
# The pipeline-generated driver scripts read these and hold no install paths of
# their own, so relocating a checkout means re-running the setup script rather
# than editing anything generated.
export TEASBENCH_ENV_PREFIX="$PREFIX"
export TEASBENCH_ROOT="$TEASBENCH_ROOT"
export AGENTCAP_DIR="$AGENTCAP_DIR"
export SWEAGENT_DIR="$SWEAGENT_DIR"
export TEASBENCH_VERSIONS_FILE="$PREFIX/versions.json"

# Namespace to operate in, and the k8s secret (key 'token') the driver reads
# GIT_TOKEN from at push time -- see templates/agentic-driver.sh step [5].
export TEASBENCH_K8S_NAMESPACE="$TEASBENCH_K8S_NAMESPACE"
export GIT_TOKEN_K8S_SECRET="$GIT_TOKEN_K8S_SECRET"

# Put the venv first so \`python\` in a generated command is this interpreter.
export PATH="$VENV/bin:\$PATH"
export VIRTUAL_ENV="$VENV"

# The pod providers are imported by dotted path (k8s_pod_providers:...), not
# installed. Only pipeline/k8s/lib goes on the path -- it contains nothing but
# that package, so no other directory in this repo can shadow a real module.
export PYTHONPATH="$TEASBENCH_ROOT/pipeline/k8s/lib\${PYTHONPATH:+:\$PYTHONPATH}"
EOF
did "written"

echo
echo "=============================================================="
echo "Setup complete."
echo
echo "  source $PREFIX/env.sh"
echo
echo "Then generate and run as usual:"
echo "  cd pipeline && python generate.py --csv_file=../experiments/agentic-smoke-tests-eidf.csv --target_dir=./out"
echo "  bash out/<run>.sh"
echo "=============================================================="
