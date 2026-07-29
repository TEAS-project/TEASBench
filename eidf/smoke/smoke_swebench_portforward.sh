#!/usr/bin/env bash
#
# SWE-bench Lite smoke test on EIDF, via PortForwardK8sProvider.
#
# WHY THIS IS A SCRIPT AND NOT A GENERATED JOB
# --------------------------------------------
# Every other TEASBench run is one self-contained k8s Job: submit it, walk away.
# SWE-bench on EIDF cannot be, because EIDF does not grant pods RBAC. The driver
# has to create ~100 per-task sandbox Jobs through the k8s API, and the only
# credentials that can do that here are YOURS -- which live on the login node,
# not in a pod. So the driver runs here and reaches sandboxes over
# `kubectl port-forward`.
#
# The other two agentic benchmarks are unaffected: IMO AnswerBench and MCP Atlas
# never touch the k8s API (MCP Atlas talks to a pod sidecar over localhost), so
# they still run as ordinary generated Jobs from experiments/*.csv.
#
# WHAT IT COVERS
#   engine endpoint -> SWE-agent -> per-task sandbox (port-forward) -> patch ->
#   official grading in an exec container -> TEAS-format outputs
#
# PREREQUISITES
#   - run from an EIDF login node, in the TEASBench repo
#   - kubectl working for the namespace (verify: eidf/preflight/preflight_portforward.py)
#   - a python env with agent_cap installed, plus swe-rex + swebench
#   - a SWE-agent checkout patched by AgentCAP's scripts/patch_sweagent_streaming.py
#   - an OpenAI-compatible LLM endpoint reachable from THIS node
#
# USAGE
#   bash eidf/smoke/smoke_swebench_portforward.sh --llm-url http://127.0.0.1:8000/v1
#   bash eidf/smoke/smoke_swebench_portforward.sh --llm-url ... --num-tasks 1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

NAMESPACE="${TEASBENCH_K8S_NAMESPACE:-eidf230ns}"
LLM_URL="${LLM_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-openai/unsloth/gpt-oss-120b}"
NUM_TASKS=2
CONCURRENCY=1
SWEAGENT_DIR="${SWEAGENT_DIR:-$HOME/swe_agent}"
AGENTCAP_DIR="${AGENTCAP_DIR:-$REPO_ROOT/../AgentCAP-teasbench}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/teasbench-smoke/swebench_$(date +%y%m%d-%H%M)}"
PROVIDER="teasbench.sandbox.k8s:PortForwardK8sProvider"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm-url)      LLM_URL="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --num-tasks)    NUM_TASKS="$2"; shift 2 ;;
        --concurrency)  CONCURRENCY="$2"; shift 2 ;;
        --namespace)    NAMESPACE="$2"; shift 2 ;;
        --sweagent-dir) SWEAGENT_DIR="$2"; shift 2 ;;
        --agentcap-dir) AGENTCAP_DIR="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)      sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

FAILED=0
pass() { echo "  PASS  $1"; }
fail() { echo "  FAIL  $1" >&2; FAILED=1; [ -n "${2:-}" ] && echo "        $2" >&2; }

echo "=============================================================="
echo "SWE-bench Lite smoke -- EIDF, PortForwardK8sProvider"
echo "  namespace   : $NAMESPACE"
echo "  tasks       : $NUM_TASKS (concurrency $CONCURRENCY)"
echo "  output      : $OUTPUT_DIR"
echo "=============================================================="

echo
echo "[1] Prerequisites"

command -v kubectl > /dev/null \
    && pass "kubectl present" \
    || fail "kubectl not on PATH"

kubectl -n "$NAMESPACE" get pods > /dev/null 2>&1 \
    && pass "kubectl can list pods in $NAMESPACE" \
    || fail "kubectl cannot reach $NAMESPACE" \
            "Run eidf/preflight/preflight_portforward.py first."

# pods/portforward is the permission unique to this path.
[ "$(kubectl -n "$NAMESPACE" auth can-i create pods/portforward 2>/dev/null)" = "yes" ] \
    && pass "can create pods/portforward" \
    || fail "cannot create pods/portforward" \
            "The tunnel to each sandbox cannot be established without it."

python3 -c "import agent_cap" 2>/dev/null \
    && pass "agent_cap importable" \
    || fail "agent_cap not importable" "pip install -e $AGENTCAP_DIR"

for mod in swerex swebench; do
    python3 -c "import $mod" 2>/dev/null \
        && pass "$mod importable" \
        || fail "$mod not importable" "pip install 'swe-rex>=1.4.0' swebench"
done

# TEASBench supplies the provider; it is imported by dotted path, so it only
# needs to be on PYTHONPATH -- there is no package to install.
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
python3 -c "from teasbench.sandbox.k8s import PortForwardK8sProvider" 2>/dev/null \
    && pass "PortForwardK8sProvider importable (PYTHONPATH=$REPO_ROOT)" \
    || fail "cannot import the provider from $REPO_ROOT"

# Without the streaming patch the run completes but every agentic metric is
# empty -- a silent, expensive failure. Check it up front.
if [ -f "$SWEAGENT_DIR/sweagent/agent/models.py" ]; then
    if grep -q "AGENTCAP_STREAMING_PATCH_APPLIED" "$SWEAGENT_DIR/sweagent/agent/models.py"; then
        pass "SWE-agent at $SWEAGENT_DIR is streaming-patched"
    else
        fail "SWE-agent at $SWEAGENT_DIR is NOT streaming-patched" \
             "python $AGENTCAP_DIR/scripts/patch_sweagent_streaming.py $SWEAGENT_DIR"
    fi
else
    fail "no SWE-agent checkout at $SWEAGENT_DIR" "Set --sweagent-dir."
fi

INDICES="$AGENTCAP_DIR/benchmarks/swe_bench_lite_curated_100.json"
[ -f "$INDICES" ] \
    && pass "curated task list found" \
    || fail "no task list at $INDICES" "Set --agentcap-dir."

echo
echo "[2] LLM endpoint"
if curl -sS -m 10 -o /dev/null -w "      %{http_code}  $LLM_URL/models\n" "$LLM_URL/models" 2>/dev/null; then
    pass "LLM endpoint reachable from this node"
else
    fail "LLM not reachable at $LLM_URL" \
         "Start the engine (a normal k8s Job is fine) and port-forward it here."
fi

if [ $FAILED -ne 0 ]; then
    echo
    echo "Prerequisites failed -- not starting the run."
    exit 1
fi

echo
echo "[3] Running $NUM_TASKS task(s)"
mkdir -p "$OUTPUT_DIR"

# TEAS_* are read by agent_cap.agents.teas_output when it writes the run's
# metadata/metrics. Set deliberately rather than inferred from a directory name.
export TEAS_BACKEND="swebench-k8s-portforward"
export TEASBENCH_K8S_NAMESPACE="$NAMESPACE"

set -x
python3 -m agent_cap.agents \
    --strategy sweagent \
    --dataset swe-bench-lite \
    --model "$MODEL" \
    --base-url "$LLM_URL" \
    --api-key dummy \
    --task-indices "$INDICES" \
    --num-tasks "$NUM_TASKS" \
    --concurrency "$CONCURRENCY" \
    --sweagent-deployment k8s \
    --sweagent-dir "$SWEAGENT_DIR" \
    --sandbox-provider "$PROVIDER" \
    --exec-provider "$PROVIDER" \
    --evaluator swebench-k8s \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/smoke.log"
RC=${PIPESTATUS[0]}
set +x

echo
echo "[4] Results"
for f in results.jsonl predictions.json; do
    [ -s "$OUTPUT_DIR/$f" ] && pass "$f written" || fail "$f missing or empty"
done
if ls "$OUTPUT_DIR"/metrics_*.json > /dev/null 2>&1; then
    pass "TEAS metrics written: $(basename "$(ls "$OUTPUT_DIR"/metrics_*.json | head -1)")"
    python3 - "$OUTPUT_DIR" <<'PY' || true
import glob, json, sys
m = sorted(glob.glob(f"{sys.argv[1]}/metrics_*.json"))
if m:
    d = json.load(open(m[-1]))
    q, p = d.get("quality", {}), d.get("performance", {})
    print(f"      resolved {q.get('passed')}/{q.get('total_examples')}  "
          f"acc={q.get('acc')}  ttft={p.get('ttft')}  tpot={p.get('tpot')}")
    # Empty streaming metrics = the SWE-agent patch did not take effect.
    if not p.get("ttft"):
        print("      NOTE: ttft is empty -- check the SWE-agent streaming patch.")
PY
else
    fail "no metrics_*.json" "The run did not reach the TEAS output stage."
fi

# Sandboxes are per-task and short-lived; leftovers mean a crash skipped cleanup.
LEFT=$(kubectl -n "$NAMESPACE" get jobs -l app=teasbench-sandbox --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "${LEFT:-0}" -eq 0 ]; then
    pass "no sandbox jobs left behind"
else
    fail "$LEFT sandbox job(s) still present" \
         "kubectl -n $NAMESPACE delete jobs -l app=teasbench-sandbox"
fi

echo
echo "=============================================================="
if [ $FAILED -eq 0 ] && [ "$RC" -eq 0 ]; then
    echo "SMOKE PASSED -- outputs in $OUTPUT_DIR"
    echo "A low or zero accuracy on 2 tasks is normal and not a failure signal;"
    echo "what matters is that every stage ran and wrote its outputs."
else
    echo "SMOKE FAILED (agent_cap rc=$RC) -- see $OUTPUT_DIR/smoke.log"
fi
echo "=============================================================="
[ $FAILED -eq 0 ] && [ "$RC" -eq 0 ]
