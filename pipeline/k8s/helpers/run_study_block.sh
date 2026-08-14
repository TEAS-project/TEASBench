#!/bin/bash
# Launch ONE block of the controlled repeatability / engine-build study on
# EIDF, from a login node. Six identical blocks (E1..E6), each 12 runs:
# {vllm 0.16.0, vllm 0.21.0, sglang 0.5.9, sglang 0.5.12.post1} x
# {gsm8k, arena-hard, longbench_v1}, gpt-oss-120b, batch-default; E1 on A100x1 (pilot); E2-E6 stratum is decided after the pilot (CSV rows still carry the original H100 placeholder - do not launch them until retargeted)
# (experiments/replication-study-eidf.csv).
#
#   ./run_study_block.sh E1              # run block E1 (12 leaves, sequential)
#   ./run_study_block.sh E1 --dry-run    # generate + preflight only
#   ./run_study_block.sh E1 --only 5,6   # repeat specific leaves of a block
#
# Options: --no-pin --skip-image-check --results-repo NAME --leaf-timeout-hours N
#
# Leaves run one at a time in a fixed, balanced order; leaves 2-12 are pinned
# to the node leaf 1 landed on. MoE-CAP and the TEASBench checkout are pinned
# study-wide on first use (state in ~/.teas-replication-study). Run inside
# tmux/screen: a block takes many hours. Complete blocks in order E1 -> E6,
# spread over >= 3 dates. A failed leaf is logged for repeat via --only and
# its results are kept.

set -uo pipefail

# EIDF site facts, mirroring pipeline/configs/sites/eidf.yaml (study is EIDF-only).
NAMESPACE=eidf230ns
JOB_CONFIGS_DIR=/eidfs/eidf230/shared/gpu-service/job-configs
MOE_CAP_REPO=https://github.com/Auto-CAP/MoE-CAP.git
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CSV="$REPO_ROOT/experiments/replication-study-eidf.csv"
STATE_DIR="${STUDY_STATE_DIR:-$HOME/.teas-replication-study}"

RESULTS_REPO=TEAS_Development_Results_Private
DRY_RUN=0
NO_PIN=0
SKIP_IMAGE_CHECK=0
ONLY=""
LEAF_TIMEOUT_HOURS=12

# Usage text = the header comment block, so edits to it can't desync.
usage() { awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "$0"; exit 1; }
die() { echo "ERROR: $*" >&2; exit 1; }
need_val() { [ $# -ge 2 ] || die "$1 needs a value"; }

BLOCK="${1:-}"; [ -n "$BLOCK" ] || usage; shift
BLOCK="$(echo "$BLOCK" | tr '[:lower:]' '[:upper:]')"
[[ "$BLOCK" =~ ^E[1-6]$ ]] || { echo "ERROR: block must be E1..E6"; usage; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --no-pin) NO_PIN=1 ;;
        --skip-image-check) SKIP_IMAGE_CHECK=1 ;;
        --only) need_val "$@"; ONLY="$2"; shift ;;
        --results-repo) need_val "$@"; RESULTS_REPO="$2"; shift ;;
        --leaf-timeout-hours) need_val "$@"; LEAF_TIMEOUT_HOURS="$2"; shift ;;
        *) echo "ERROR: unknown option $1"; usage ;;
    esac
    shift
done
if [ -n "$ONLY" ]; then
    [[ "$ONLY" =~ ^(1[0-2]|[1-9])(,(1[0-2]|[1-9]))*$ ]] \
        || die "--only expects comma-separated leaf numbers 1-12, e.g. --only 5,6"
fi

mkdir -p "$STATE_DIR"
MANIFEST="$STATE_DIR/manifest-$BLOCK.jsonl"
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/teas-study-$BLOCK-XXXXXX")"

echo "== replication study block $BLOCK =="
echo "   repo:       $REPO_ROOT"
echo "   state dir:  $STATE_DIR"
echo "   manifest:   $MANIFEST"
echo "   results to: $RESULTS_REPO"

if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ "$DRY_RUN" -eq 0 ]; then
    echo "WARNING: not inside tmux/screen; a block takes many hours and dies with your ssh session."
fi
[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ] \
    || echo "WARNING: TEASBench working tree is dirty; the recorded teasbench_commit will not describe what runs."

# --- study-wide pins: MoE-CAP commit + TEASBench checkout ---------------------
REF_FILE="$STATE_DIR/moe_cap_ref"
if [ ! -s "$REF_FILE" ]; then
    echo "Resolving MoE-CAP main once for the whole study..."
    git ls-remote "$MOE_CAP_REPO" refs/heads/main | cut -f1 > "$REF_FILE"
    [ -s "$REF_FILE" ] || die "could not resolve MoE-CAP main"
fi
MOE_CAP_REF="$(cat "$REF_FILE")"
echo "   MoE-CAP pinned at $MOE_CAP_REF"

TEASBENCH_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
COMMIT_FILE="$STATE_DIR/teasbench_commit"
if [ -s "$COMMIT_FILE" ] && [ "$(cat "$COMMIT_FILE")" != "$TEASBENCH_COMMIT" ]; then
    die "TEASBench is at $TEASBENCH_COMMIT but the study started at $(cat "$COMMIT_FILE").
       Launch arguments must stay frozen: git checkout $(cat "$COMMIT_FILE") to continue,
       or delete $STATE_DIR to restart the study from scratch."
fi
[ "$DRY_RUN" -eq 1 ] || echo "$TEASBENCH_COMMIT" > "$COMMIT_FILE"

# --- generate this block's 12 job YAMLs ---------------------------------------
# Column 10 = study_block; rows are already in planned order. Fill the trailing
# moe_cap_ref column with the pin.
TMP_CSV="$WORKDIR/block.csv"
awk -F, -v blk="$BLOCK" 'NR==1 || $10==blk' "$CSV" \
    | sed "s/,\$/,$MOE_CAP_REF/" > "$TMP_CSV"
N_LEAVES=$(( $(wc -l < "$TMP_CSV") - 1 ))
[ "$N_LEAVES" -eq 12 ] || die "expected 12 leaves for $BLOCK, got $N_LEAVES"

PY=python3
if ! $PY -c "import pandas, yaml" 2>/dev/null; then
    VENV="$STATE_DIR/venv"
    [ -x "$VENV/bin/python3" ] || { echo "Creating venv (pandas, pyyaml)...";
        python3 -m venv "$VENV" && "$VENV/bin/pip" -q install pandas pyyaml; }
    PY="$VENV/bin/python3"
fi

echo "Generating job YAMLs..."
GEN_LOG="$WORKDIR/generate.log"
(cd "$REPO_ROOT/pipeline" && \
    "$PY" generate.py --csv_file="$TMP_CSV" --target_dir="$WORKDIR" \
                      --results_repo="$RESULTS_REPO") > "$GEN_LOG" 2>&1 \
    || { cat "$GEN_LOG"; die "generate.py failed"; }
# "  wrote <file>" lines come out in CSV row order == planned block order.
mapfile -t LEAF_FILES < <(grep -o 'wrote .*\.yaml' "$GEN_LOG" | awk '{print $2}')
[ "${#LEAF_FILES[@]}" -eq 12 ] || { cat "$GEN_LOG"; die "expected 12 YAMLs, got ${#LEAF_FILES[@]}"; }

echo "Planned order for $BLOCK:"
for i in "${!LEAF_FILES[@]}"; do printf '   %2d. %s\n' "$((i+1))" "${LEAF_FILES[$i]}"; done

# --- image preflight -----------------------------------------------------------
if [ "$SKIP_IMAGE_CHECK" -eq 0 ]; then
    echo "Checking serving images exist on Docker Hub..."
    for img in $(grep -h '^\s*image:' "$WORKDIR"/*.yaml | awk '{print $2}' | sort -u); do
        repo="${img%%:*}"; tag="${img#*:}"
        curl -sfL "https://hub.docker.com/v2/repositories/$repo/tags/$tag" >/dev/null \
            && echo "   OK  $img" \
            || die "image $img not found on Docker Hub (fix the CSV version, or --skip-image-check if the registry API is unreachable)"
    done
fi

[ "$DRY_RUN" -eq 1 ] && { echo "Dry run: nothing submitted. YAMLs left in $WORKDIR"; exit 0; }

# --- sequential submission -------------------------------------------------------
PIN_NODE=""
# Repeats must stay on the block's node: recover the pin from the manifest.
if [ -n "$ONLY" ] && [ "$NO_PIN" -eq 0 ] && [ -f "$MANIFEST" ]; then
    PIN_NODE="$(grep -o '"node":"[^"]*"' "$MANIFEST" | tail -1 | cut -d'"' -f4)"
    [ -n "$PIN_NODE" ] && echo "   repeats pinned to recorded node $PIN_NODE"
fi

CURRENT_JOB=""
trap 'echo; echo "Interrupted. In-flight job: ${CURRENT_JOB:-none} — it is STILL RUNNING in the cluster; kubectl -n '"$NAMESPACE"' delete job <name> to stop it, or let it finish and note it has no manifest line."; exit 130' INT TERM

manifest_line() {  # order job job_uid yaml node image_id submitted_at outcome
    printf '{"study_id":"controlled-variation-2026","block":"%s","planned_order":%s,"job":"%s","job_uid":"%s","yaml":"%s","node":"%s","image_id":"%s","moe_cap_ref":"%s","teasbench_commit":"%s","submitted_at":"%s","finished_at":"%s","outcome":"%s"}\n' \
        "$BLOCK" "$1" "$2" "$3" "$4" "$5" "$6" "$MOE_CAP_REF" "$TEASBENCH_COMMIT" \
        "$7" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$8" >> "$MANIFEST"
}

run_leaf() {
    local order="$1" yaml="$WORKDIR/$2"
    local submitted_at job job_uid node image_id outcome
    submitted_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    if [ -n "$PIN_NODE" ]; then
        grep -q 'kubernetes.io/hostname' "$yaml" || \
            sed -i "/nvidia.com\/gpu.product/a\\        kubernetes.io/hostname: $PIN_NODE" "$yaml"
    fi

    local create_out
    if ! create_out="$(kubectl -n $NAMESPACE create -f "$yaml")"; then
        manifest_line "$order" "" "" "$(basename "$yaml")" "" "" "$submitted_at" "create-failed"
        echo "   [$order] kubectl create FAILED for $(basename "$yaml")"
        return 1
    fi
    job="$(echo "$create_out" | awk '{print $1}' | xargs basename)"
    CURRENT_JOB="$job"
    job_uid="$(kubectl -n $NAMESPACE get job "$job" -o jsonpath='{.metadata.uid}' 2>/dev/null)"
    echo "   [$order] submitted $job"
    # The in-pod runner copies (then removes) its own YAML from the shared
    # job-configs dir into the run's output -- same contract as submit_job.sh.
    cp "$yaml" "$JOB_CONFIGS_DIR/$job.yaml"

    # Poll to completion. The timeout clock starts when the pod is Running,
    # so kueue queue wait doesn't count against the leaf.
    node=""; image_id=""
    local deadline="" phase complete failed
    while :; do
        phase="$(kubectl -n $NAMESPACE get pods -l job-name="$job" \
                -o jsonpath='{.items[0].status.phase}' 2>/dev/null)"
        node="$(kubectl -n $NAMESPACE get pods -l job-name="$job" \
                -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null)"
        image_id="$(kubectl -n $NAMESPACE get pods -l job-name="$job" \
                -o jsonpath='{.items[0].status.containerStatuses[0].imageID}' 2>/dev/null)"
        [ -z "$deadline" ] && [ "$phase" = "Running" ] \
            && deadline=$(( $(date +%s) + LEAF_TIMEOUT_HOURS * 3600 ))
        complete="$(kubectl -n $NAMESPACE get job "$job" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)"
        failed="$(kubectl -n $NAMESPACE get job "$job" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)"
        if [ "$complete" = "True" ]; then outcome=complete; break; fi
        if [ "$failed" = "True" ]; then outcome=failed; break; fi
        if [ -n "$deadline" ] && [ "$(date +%s)" -gt "$deadline" ]; then
            outcome=timeout
            # Kill it so it can't share the node with (and contaminate) the
            # next leaf, or publish results the manifest calls timed-out.
            kubectl -n $NAMESPACE delete job "$job" --wait=false >/dev/null 2>&1
            break
        fi
        sleep 60
    done
    CURRENT_JOB=""

    if [ -z "$PIN_NODE" ] && [ "$NO_PIN" -eq 0 ] && [ -n "$node" ]; then
        PIN_NODE="$node"
        echo "   [$order] block pinned to node $PIN_NODE"
    fi

    manifest_line "$order" "$job" "$job_uid" "$(basename "$yaml")" "$node" "$image_id" "$submitted_at" "$outcome"
    echo "   [$order] $job -> $outcome (node $node)"
    [ "$outcome" = "complete" ]
}

RAN=0
FAILED_ORDERS=()
for i in "${!LEAF_FILES[@]}"; do
    order=$((i+1))
    if [ -n "$ONLY" ] && ! [[ ",$ONLY," == *",$order,"* ]]; then continue; fi
    RAN=$((RAN+1))
    run_leaf "$order" "${LEAF_FILES[$i]}" || FAILED_ORDERS+=("$order")
done
[ "$RAN" -gt 0 ] || die "--only $ONLY matched no leaves"

# --- summary ---------------------------------------------------------------------
echo "== block $BLOCK done ($RAN leaves) =="
echo "   manifest: $MANIFEST"
if [ "${#FAILED_ORDERS[@]}" -gt 0 ]; then
    echo "   FAILED/TIMED-OUT leaves: ${FAILED_ORDERS[*]}"
    echo "   Repeat them with: $0 $BLOCK --only $(IFS=,; echo "${FAILED_ORDERS[*]}")"
    echo "   (failed attempts stay in $RESULTS_REPO by design -- do not delete them)"
    exit 1
fi
echo "   all leaves complete. Results are under moe/eidf/**/study-$(echo "$BLOCK" | tr '[:upper:]' '[:lower:]')/ in $RESULTS_REPO."
