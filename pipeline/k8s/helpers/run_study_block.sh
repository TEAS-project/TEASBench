#!/bin/bash
# Launch ONE block of the controlled repeatability / engine-build study on
# EIDF, from a login node. Six fresh blocks (E1..E6), each 12 runs:
# {vllm 0.16.0, vllm 0.21.0, sglang 0.5.9, sglang 0.5.12.post1} x
# {gsm8k, arena-hard, longbench_v1}, gpt-oss-120b, batch-default;
# E1-E3 run on A100x2 and E4-E6 run on H100x2
# (experiments/replication-study-eidf.csv).
#
#   ./run_study_block.sh E1              # run block E1 (12 leaves, sequential)
#   ./run_study_block.sh E1 --dry-run    # generate + preflight only
#   ./run_study_block.sh E1 --only 5,6   # repeat specific leaves of a block
#   ./run_study_block.sh E1 --only 5 --reconcile-identity
#   Suggested block sequence: E1, E4, E2, E5, E3, E6
#
# Options: --preflight-evidence FILE --leaf-timeout-hours N
#          --reconcile-identity (one --only leaf with an ambiguous named Job)
# Dry-run only: --no-pin --skip-image-check --results-repo NAME
# A non-dry E1 requires a validated four-record A100x2 LongBench compatibility
# manifest. See study_guard.py validate-preflight --help for the evidence fields.
#
# Leaves run one at a time in a fixed, balanced order; leaves 2-12 are pinned
# to the node leaf 1 landed on. MoE-CAP and the TEASBench checkout are pinned
# study-wide on first use (state in ~/.teas-replication-study-x2). Run inside
# tmux/screen: a block takes many hours. Use the suggested block sequence and
# spread execution over >= 3 dates. A failed leaf is logged for repeat via
# --only and its results are kept.

set -uo pipefail

# EIDF site facts, mirroring pipeline/configs/sites/eidf.yaml (study is EIDF-only).
NAMESPACE=eidf230ns
JOB_CONFIGS_DIR="${STUDY_JOB_CONFIGS_DIR:-/eidfs/eidf230/shared/gpu-service/job-configs}"
MOE_CAP_REPO=https://github.com/Auto-CAP/MoE-CAP.git
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CSV="$REPO_ROOT/experiments/replication-study-eidf.csv"
GUARD="$SCRIPT_DIR/study_guard.py"
STATE_DIR="${STUDY_STATE_DIR:-$HOME/.teas-replication-study-x2}"

RESULTS_REPO=TEAS_Development_Results_Private
DRY_RUN=0
NO_PIN=0
SKIP_IMAGE_CHECK=0
ONLY=""
PREFLIGHT_EVIDENCE=""
LEAF_TIMEOUT_HOURS=12
CLEANUP_TIMEOUT_SECONDS=600
RECONCILE_IDENTITY=0

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
        --preflight-evidence) need_val "$@"; PREFLIGHT_EVIDENCE="$2"; shift ;;
        --reconcile-identity) RECONCILE_IDENTITY=1 ;;
        --results-repo) need_val "$@"; RESULTS_REPO="$2"; shift ;;
        --leaf-timeout-hours) need_val "$@"; LEAF_TIMEOUT_HOURS="$2"; shift ;;
        *) echo "ERROR: unknown option $1"; usage ;;
    esac
    shift
done
[[ "$LEAF_TIMEOUT_HOURS" =~ ^[0-9]{1,3}$ ]] \
    || die "--leaf-timeout-hours must be a decimal integer from 1 to 168"
LEAF_TIMEOUT_HOURS=$((10#$LEAF_TIMEOUT_HOURS))
[ "$LEAF_TIMEOUT_HOURS" -ge 1 ] && [ "$LEAF_TIMEOUT_HOURS" -le 168 ] \
    || die "--leaf-timeout-hours must be a decimal integer from 1 to 168"
if [ -n "$ONLY" ]; then
    [[ "$ONLY" =~ ^(1[0-2]|[1-9])(,(1[0-2]|[1-9]))*$ ]] \
        || die "--only expects comma-separated leaf numbers 1-12, e.g. --only 5,6"
fi
if [ "$RECONCILE_IDENTITY" -eq 1 ]; then
    [ "$DRY_RUN" -eq 0 ] \
        || die "--reconcile-identity is a non-dry recovery operation"
    [ -n "$ONLY" ] && [[ "$ONLY" != *,* ]] \
        || die "--reconcile-identity requires exactly one leaf via --only"
fi

case "$BLOCK" in
    E1|E2|E3) EXPECTED_GPU=A100 ;;
    E4|E5|E6) EXPECTED_GPU=H100 ;;
esac
[ -f "$CSV" ] || die "study CSV not found: $CSV"
python3 "$GUARD" validate-csv --csv "$CSV" \
    || die "study CSV is outside the frozen 72-coordinate allowlist"
BAD_SHAPE="$(awk -F, -v blk="$BLOCK" -v gpu="$EXPECTED_GPU" '
    NR > 1 && $10 == blk && ($7 != gpu || $8 != 2) {
        print "line " NR ": " $0
    }' "$CSV")"
[ -z "$BAD_SHAPE" ] || die "$BLOCK must contain only ${EXPECTED_GPU}x2 rows:
$BAD_SHAPE"

if [ "$DRY_RUN" -eq 0 ]; then
    [ "$NO_PIN" -eq 0 ] || die "--no-pin is allowed only with --dry-run"
    [ "$SKIP_IMAGE_CHECK" -eq 0 ] \
        || die "--skip-image-check is allowed only with --dry-run"
    [ "$RESULTS_REPO" = "TEAS_Development_Results_Private" ] \
        || die "study runs must write to TEAS_Development_Results_Private"
fi

WORKTREE_STATUS="$(git -C "$REPO_ROOT" status --porcelain)" \
    || die "could not inspect the TEASBench working tree"
if [ -n "$WORKTREE_STATUS" ]; then
    [ "$DRY_RUN" -eq 1 ] \
        || die "TEASBench working tree is dirty; commit the exact launch code before a study submission"
    echo "WARNING: TEASBench working tree is dirty; dry-run output is verification-only."
fi

mkdir -p "$STATE_DIR" || die "could not create study state directory $STATE_DIR"
LOCK_DIR="$STATE_DIR/.launcher.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    die "study launcher lock is held at $LOCK_DIR; inspect the recorded owner before recovery"
fi
release_study_lock() {
    rm -f "$LOCK_DIR/owner"
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap release_study_lock EXIT
printf 'pid=%s\nhost=%s\nstarted_at=%s\n' "$$" "$(hostname)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/owner" \
    || die "could not record study launcher lock ownership"

MANIFEST="$STATE_DIR/manifest-$BLOCK.jsonl"
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/teas-study-$BLOCK-XXXXXX")" \
    || die "could not create temporary launch directory"
IMAGE_PIN_FILE="$STATE_DIR/image-digests.tsv"
PREFLIGHT_EVIDENCE="${PREFLIGHT_EVIDENCE:-$STATE_DIR/a100x2-compatibility-preflight.jsonl}"

# A first launch must follow the frozen interleaved schedule and may start only
# after all 12 predecessor leaves have a successful manifest record. --only is
# a repeat path: it requires an existing manifest, recovers its last non-empty
# node, and rejects any block that has already recorded more than one node.
PIN_NODE=""
if [ "$DRY_RUN" -eq 0 ]; then
    if [ -n "$ONLY" ]; then
        [ -s "$MANIFEST" ] || die "--only requires an existing non-empty manifest for $BLOCK"
        PIN_NODE="$(python3 "$GUARD" manifest-node \
            --manifest "$MANIFEST" --block "$BLOCK")" \
            || die "cannot recover a unique node for $BLOCK repeats"
        [ -n "$PIN_NODE" ] && echo "   repeats pinned to recorded node $PIN_NODE"
    else
        [ ! -s "$MANIFEST" ] \
            || die "$BLOCK already has manifest records; repeat specific leaves with --only"
        case "$BLOCK" in
            E1) REQUIRED_BLOCKS=""; LATER_BLOCKS="E4 E2 E5 E3 E6" ;;
            E4) REQUIRED_BLOCKS="E1"; LATER_BLOCKS="E2 E5 E3 E6" ;;
            E2) REQUIRED_BLOCKS="E1 E4"; LATER_BLOCKS="E5 E3 E6" ;;
            E5) REQUIRED_BLOCKS="E1 E4 E2"; LATER_BLOCKS="E3 E6" ;;
            E3) REQUIRED_BLOCKS="E1 E4 E2 E5"; LATER_BLOCKS="E6" ;;
            E6) REQUIRED_BLOCKS="E1 E4 E2 E5 E3"; LATER_BLOCKS="" ;;
        esac
        for predecessor in $REQUIRED_BLOCKS; do
            PREDECESSOR_MANIFEST="$STATE_DIR/manifest-$predecessor.jsonl"
            python3 "$GUARD" block-complete \
                --manifest "$PREDECESSOR_MANIFEST" --block "$predecessor" \
                || die "$BLOCK is blocked until $predecessor has 12 successful leaves"
        done
        for later in $LATER_BLOCKS; do
            [ ! -s "$STATE_DIR/manifest-$later.jsonl" ] \
                || die "state is out of order: $later has records before first launch of $BLOCK"
        done
    fi
fi

echo "== replication study block $BLOCK =="
echo "   repo:       $REPO_ROOT"
echo "   state dir:  $STATE_DIR"
echo "   manifest:   $MANIFEST"
echo "   results to: $RESULTS_REPO"

if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ "$DRY_RUN" -eq 0 ]; then
    echo "WARNING: not inside tmux/screen; a block takes many hours and dies with your ssh session."
fi

# --- study-wide pins: MoE-CAP commit + TEASBench checkout ---------------------
REF_FILE="$STATE_DIR/moe_cap_ref"
if [ ! -s "$REF_FILE" ]; then
    echo "Resolving MoE-CAP main once for the whole study..."
    RESOLVED_MOE_CAP_REF="$(git ls-remote "$MOE_CAP_REPO" refs/heads/main | cut -f1)"
    [[ "$RESOLVED_MOE_CAP_REF" =~ ^[0-9a-f]{40}$ ]] \
        || die "could not resolve a full MoE-CAP main commit"
    python3 "$GUARD" write-state --path "$REF_FILE" --value "$RESOLVED_MOE_CAP_REF" \
        || die "could not atomically freeze the MoE-CAP commit"
fi
MOE_CAP_REF="$(cat "$REF_FILE")"
[[ "$MOE_CAP_REF" =~ ^[0-9a-f]{40}$ ]] || die "invalid frozen MoE-CAP commit"
echo "   MoE-CAP pinned at $MOE_CAP_REF"

TEASBENCH_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    || die "could not resolve the TEASBench commit"
[[ "$TEASBENCH_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid TEASBench commit"
COMMIT_FILE="$STATE_DIR/teasbench_commit"
if [ -s "$COMMIT_FILE" ] && [ "$(cat "$COMMIT_FILE")" != "$TEASBENCH_COMMIT" ]; then
    die "TEASBench is at $TEASBENCH_COMMIT but the study started at $(cat "$COMMIT_FILE").
       Preserve $STATE_DIR. Check out $(cat "$COMMIT_FILE") to continue. If no jobs were
       submitted, move the state directory aside before restarting; if any submission
       exists, restarting requires a new study identity and an explicit study revision."
fi
if [ "$DRY_RUN" -eq 0 ]; then
    python3 "$GUARD" write-state --path "$COMMIT_FILE" --value "$TEASBENCH_COMMIT" \
        || die "could not freeze the TEASBench commit in $COMMIT_FILE"
fi

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
LEAF_FILES=()
while IFS= read -r leaf_file; do
    [ -n "$leaf_file" ] && LEAF_FILES+=("$leaf_file")
done < <(grep -o 'wrote .*\.yaml' "$GEN_LOG" | awk '{print $2}')
[ "${#LEAF_FILES[@]}" -eq 12 ] || { cat "$GEN_LOG"; die "expected 12 YAMLs, got ${#LEAF_FILES[@]}"; }

echo "Planned order for $BLOCK:"
for i in "${!LEAF_FILES[@]}"; do printf '   %2d. %s\n' "$((i+1))" "${LEAF_FILES[$i]}"; done

# --- immutable serving-image preflight ----------------------------------------
if [ "$SKIP_IMAGE_CHECK" -eq 0 ]; then
    echo "Resolving serving-image tags to immutable Docker Hub digests..."
    IMAGE_TAGS_FILE="$WORKDIR/image-tags.txt"
    RESOLVED_IMAGES_FILE="$WORKDIR/resolved-images.tsv"
    grep -h '^[[:space:]]*image:' "$WORKDIR"/*.yaml \
        | awk '{print $2}' | sort -u > "$IMAGE_TAGS_FILE"
    [ "$(wc -l < "$IMAGE_TAGS_FILE" | tr -d ' ')" -eq 4 ] \
        || die "expected exactly four serving-image tags"
    : > "$RESOLVED_IMAGES_FILE"
    while IFS= read -r img; do
        repo="${img%:*}"; tag="${img##*:}"
        tag_json="$(curl -fsSL \
            "https://hub.docker.com/v2/repositories/$repo/tags/$tag")" \
            || die "could not resolve Docker Hub tag $img"
        digest="$(printf '%s' "$tag_json" | "$PY" -c \
            'import json,sys; print(json.load(sys.stdin).get("digest", ""))')" \
            || die "invalid Docker Hub response for $img"
        [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] \
            || die "Docker Hub returned an invalid digest for $img: $digest"
        printf '%s\t%s\n' "$img" "$digest" >> "$RESOLVED_IMAGES_FILE"
    done < "$IMAGE_TAGS_FILE"

    YAML_PATHS=()
    for leaf_file in "${LEAF_FILES[@]}"; do
        YAML_PATHS+=("$WORKDIR/$leaf_file")
    done
    if [ "$DRY_RUN" -eq 0 ]; then
        "$PY" "$GUARD" pin-images --pin-file "$IMAGE_PIN_FILE" \
            --resolved-file "$RESOLVED_IMAGES_FILE" --persist \
            "${YAML_PATHS[@]}" \
            || die "serving-image digest freeze failed"
    else
        "$PY" "$GUARD" pin-images --pin-file "$IMAGE_PIN_FILE" \
            --resolved-file "$RESOLVED_IMAGES_FILE" \
            "${YAML_PATHS[@]}" \
            || die "serving-image digest preflight failed"
    fi
fi

[ "$DRY_RUN" -eq 1 ] && { echo "Dry run: nothing submitted. YAMLs left in $WORKDIR"; exit 0; }

# The A100x2 stratum cannot begin until all four exact engine/build images have
# completed an excluded 256-sample LongBench compatibility run on the frozen
# commits and digests. The validator hash-binds the metadata and metrics files
# and writes a durable validation record into this fresh study state.
if [ "$BLOCK" = "E1" ]; then
    "$PY" "$GUARD" validate-preflight \
        --manifest "$PREFLIGHT_EVIDENCE" \
        --image-pins "$IMAGE_PIN_FILE" \
        --teasbench-commit "$TEASBENCH_COMMIT" \
        --moe-cap-ref "$MOE_CAP_REF" \
        --record "$STATE_DIR/a100x2-compatibility-preflight.validated.json" \
        || die "E1 blocked: A100x2 compatibility-preflight evidence is incomplete or invalid"
fi

# --- sequential submission -------------------------------------------------------
CURRENT_JOB=""
trap 'echo; echo "Interrupted. In-flight job: ${CURRENT_JOB:-none}. Its durable submitted record is in '"$MANIFEST"'; resume it with --only rather than submitting a duplicate."; exit 130' INT TERM

append_manifest() {  # order job uid yaml node image submitted outcome receipt sha output yaml_path yaml_sha
    local finished_at=""
    [ "$8" = "submitted" ] || finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$PY" "$GUARD" append-manifest \
        --manifest "$MANIFEST" --block "$BLOCK" --order "$1" \
        --job "$2" --job-uid "$3" --yaml "$4" --node "$5" --image-id "$6" \
        --yaml-path "${12:-}" --yaml-sha256 "${13:-}" \
        --moe-cap-ref "$MOE_CAP_REF" --teasbench-commit "$TEASBENCH_COMMIT" \
        --submitted-at "$7" --finished-at "$finished_at" --outcome "$8" \
        --receipt-path "${9:-}" --receipt-sha256 "${10:-}" \
        --output-path "${11:-}"
}

valid_job_uid() {
    [[ "$1" =~ ^[0-9a-f][0-9a-f-]{7,}$ ]]
}

valid_job_name() {
    [ -n "$1" ] && [ "${#1}" -le 63 ] \
        && [[ "$1" =~ ^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$ ]]
}

cleanup_exact_job() {
    local job="$1" job_uid="$2" reason="$3"
    local current_uid pod_records pod_name pod_uid deadline
    current_uid="$(kubectl -n "$NAMESPACE" get job "$job" \
        --ignore-not-found -o jsonpath='{.metadata.uid}' 2>/dev/null)" || {
            echo "ERROR: cannot verify $job identity before $reason cleanup" >&2
            return 1
        }
    if [ -n "$current_uid" ] && [ "$current_uid" != "$job_uid" ]; then
        echo "ERROR: refusing to delete replacement job $job (expected UID $job_uid, got $current_uid)" >&2
        return 1
    fi
    if [ "$current_uid" = "$job_uid" ]; then
        kubectl -n "$NAMESPACE" delete job "$job" --wait=false \
            --cascade=foreground >/dev/null 2>&1 || {
                echo "ERROR: deletion request failed for exact job $job/$job_uid" >&2
                return 1
            }
    fi

    deadline=$(( $(date +%s) + CLEANUP_TIMEOUT_SECONDS ))
    while :; do
        current_uid="$(kubectl -n "$NAMESPACE" get job "$job" \
            --ignore-not-found -o jsonpath='{.metadata.uid}' 2>/dev/null)" || {
                echo "ERROR: cannot verify deletion of $job" >&2
                return 1
            }
        if [ -n "$current_uid" ] && [ "$current_uid" != "$job_uid" ]; then
            echo "ERROR: job name $job was replaced during cleanup (UID $current_uid)" >&2
            return 1
        fi
        pod_records="$(kubectl -n "$NAMESPACE" get pods -l job-name="$job" \
            -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.ownerReferences[?(@.kind=="Job")].uid}{"\n"}{end}' \
            2>/dev/null)" || {
                echo "ERROR: cannot verify pod cleanup for $job/$job_uid" >&2
                return 1
            }
        while IFS=$'\t' read -r pod_name pod_uid; do
            [ -z "$pod_name" ] && continue
            if [ "$pod_uid" != "$job_uid" ]; then
                echo "ERROR: pod $pod_name for $job has unexpected owner UID ${pod_uid:-none}; refusing unsafe cleanup" >&2
                return 1
            fi
        done <<< "$pod_records"
        if [ -z "$current_uid" ] && [ -z "$pod_records" ]; then
            echo "   cleanup confirmed for $job/$job_uid ($reason)"
            return 0
        fi
        if [ "$(date +%s)" -gt "$deadline" ]; then
            echo "ERROR: cleanup timed out for $job/$job_uid ($reason)" >&2
            return 1
        fi
        sleep 5
    done
}

confirm_reconciled_absence() {
    local job="$1" expected_uid="$2"
    local current_uid pod_records pod_name pod_uid
    current_uid="$(kubectl -n "$NAMESPACE" get job "$job" \
        --ignore-not-found -o jsonpath='{.metadata.uid}' 2>/dev/null)" || {
            echo "ERROR: cannot confirm reconciled absence of named job $job" >&2
            return 1
        }
    if [ -n "$current_uid" ]; then
        if valid_job_uid "$expected_uid" && [ "$current_uid" != "$expected_uid" ]; then
            echo "ERROR: named job $job was replaced (expected UID $expected_uid, got $current_uid)" >&2
        else
            echo "ERROR: named job $job still exists during absence reconciliation" >&2
        fi
        return 1
    fi
    pod_records="$(kubectl -n "$NAMESPACE" get pods -l job-name="$job" \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.ownerReferences[?(@.kind=="Job")].uid}{"\n"}{end}' \
        2>/dev/null)" || {
            echo "ERROR: cannot confirm pod absence for named job $job" >&2
            return 1
        }
    while IFS=$'\t' read -r pod_name pod_uid; do
        [ -z "$pod_name" ] && continue
        if valid_job_uid "$expected_uid" && [ "$pod_uid" != "$expected_uid" ]; then
            echo "ERROR: pod $pod_name has replacement owner UID ${pod_uid:-none}; refusing reconciliation" >&2
        else
            echo "ERROR: pod $pod_name for named job $job still exists; refusing reconciliation" >&2
        fi
        return 1
    done <<< "$pod_records"
    echo "   confirmed named job and pods absent for $job"
}

restore_recovered_job_config() {
    local job="$1" yaml_path="$2" expected_sha="$3"
    local destination="$JOB_CONFIGS_DIR/$job.yaml" complete failed actual_sha
    complete="$(kubectl -n "$NAMESPACE" get job "$job" \
        -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)"
    failed="$(kubectl -n "$NAMESPACE" get job "$job" \
        -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)"
    if [ "$complete" = "True" ] || [ "$failed" = "True" ]; then
        if [ -e "$destination" ] && ! rm -f "$destination"; then
            echo "ERROR: could not remove stale shared config for terminal job $job" >&2
            return 1
        fi
        echo "   recovered job $job is already terminal; no shared config restored"
        return 0
    fi
    if ! cp "$yaml_path" "$destination"; then
        rm -f "$destination" 2>/dev/null || true
        echo "ERROR: could not restore frozen shared config for recovered job $job" >&2
        return 1
    fi
    actual_sha="$(sha256sum "$destination" 2>/dev/null | awk '{print $1}')"
    if [ "$actual_sha" != "$expected_sha" ]; then
        rm -f "$destination" 2>/dev/null || true
        echo "ERROR: restored shared config hash mismatch for recovered job $job" >&2
        return 1
    fi
    echo "   restored frozen shared config for recovered job $job"
}

pin_yaml_to_block_node() {
    local yaml="$1"
    if [ -n "$PIN_NODE" ]; then
        grep -q 'kubernetes.io/hostname' "$yaml" || \
            sed -i "/nvidia.com\/gpu.product/a\\        kubernetes.io/hostname: $PIN_NODE" "$yaml"
    fi
}

monitor_leaf() {
    local order="$1" yaml_name="$2" yaml_path="$3" yaml_sha="$4"
    local job="$5" job_uid="$6" submitted_at="$7"
    local node="" image_id="" outcome="" deadline="" phase complete failed current_uid
    local receipt_message receipt_file validation_out receipt_sha output_path
    CURRENT_JOB="$job"

    # The timeout clock starts when the pod is Running, so kueue queue wait
    # does not count against the leaf. A resumed submitted record monitors the
    # exact Kubernetes UID; a vanished/replaced Job is never silently rerun.
    while :; do
        current_uid="$(kubectl -n "$NAMESPACE" get job "$job" \
            -o jsonpath='{.metadata.uid}' 2>/dev/null)"
        if [ -z "$current_uid" ] || [ "$current_uid" != "$job_uid" ]; then
            outcome=missing-job
            break
        fi
        phase="$(kubectl -n "$NAMESPACE" get pods -l job-name="$job" \
                -o jsonpath='{.items[0].status.phase}' 2>/dev/null)"
        node="$(kubectl -n "$NAMESPACE" get pods -l job-name="$job" \
                -o jsonpath='{.items[0].spec.nodeName}' 2>/dev/null)"
        image_id="$(kubectl -n "$NAMESPACE" get pods -l job-name="$job" \
                -o jsonpath='{.items[0].status.containerStatuses[0].imageID}' 2>/dev/null)"
        [ -z "$deadline" ] && [ "$phase" = "Running" ] \
            && deadline=$(( $(date +%s) + LEAF_TIMEOUT_HOURS * 3600 ))
        complete="$(kubectl -n "$NAMESPACE" get job "$job" \
            -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)"
        failed="$(kubectl -n "$NAMESPACE" get job "$job" \
            -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)"
        if [ "$complete" = "True" ]; then
            receipt_message="$(kubectl -n "$NAMESPACE" get pods -l job-name="$job" \
                -o jsonpath='{.items[0].status.containerStatuses[0].state.terminated.message}' \
                2>/dev/null)"
            receipt_file="$STATE_DIR/receipts/$job.json"
            if [ -n "$node" ] && [ -n "$receipt_message" ] \
                && "$PY" "$GUARD" write-state \
                    --path "$receipt_file" --value "$receipt_message" \
                && validation_out="$("$PY" "$GUARD" validate-receipt \
                    --receipt "$receipt_file" --block "$BLOCK" --order "$order" \
                    --job "$job" --job-uid "$job_uid" --node "$node" \
                    --image-id "$image_id" --job-yaml "$yaml_path" \
                    --teasbench-commit "$TEASBENCH_COMMIT" \
                    --moe-cap-ref "$MOE_CAP_REF")"; then
                receipt_sha="${validation_out%%$'\t'*}"
                output_path="${validation_out#*$'\t'}"
                outcome=complete
            else
                outcome=validation-failed
                receipt_file=""
                receipt_sha=""
                output_path=""
            fi
            break
        fi
        if [ "$failed" = "True" ]; then outcome=failed; break; fi
        if [ -n "$deadline" ] && [ "$(date +%s)" -gt "$deadline" ]; then
            outcome=timeout
            break
        fi
        sleep 60
    done
    CURRENT_JOB=""

    if [ -z "$PIN_NODE" ] && [ "$NO_PIN" -eq 0 ] && [ -n "$node" ]; then
        PIN_NODE="$node"
        echo "   [$order] block pinned to node $PIN_NODE"
    fi
    if ! append_manifest "$order" "$job" "$job_uid" "$yaml_name" "$node" "$image_id" \
        "$submitted_at" "$outcome" "${receipt_file:-}" "${receipt_sha:-}" \
        "${output_path:-}" "$yaml_path" "$yaml_sha"; then
        cleanup_exact_job "$job" "$job_uid" "manifest-write failure" \
            || die "manifest write and exact cleanup both failed for $job/$job_uid"
        die "could not append $outcome outcome to $MANIFEST; exact job cleanup completed"
    fi
    if [ "$outcome" = "timeout" ] || [ "$outcome" = "failed" ] \
        || [ "$outcome" = "missing-job" ]; then
        cleanup_exact_job "$job" "$job_uid" "$outcome" \
            || die "exact cleanup failed for $job/$job_uid after $outcome"
        append_manifest "$order" "$job" "$job_uid" "$yaml_name" "$node" "$image_id" \
            "$submitted_at" "cleanup-confirmed" "" "" "" "$yaml_path" "$yaml_sha" \
            || die "exact cleanup completed for $job/$job_uid but its durable confirmation failed"
    fi
    echo "   [$order] $job -> $outcome (node $node)"
    [ "$outcome" = "complete" ]
}

run_leaf() {
    local order="$1" yaml="$WORKDIR/$2"
    local submitted_at job job_uid create_out prepared yaml_path yaml_sha generate_name extra
    submitted_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    pin_yaml_to_block_node "$yaml"

    prepared="$("$PY" "$GUARD" prepare-job --yaml "$yaml" \
        --state-dir "$STATE_DIR/submitted-yamls")" \
        || die "could not prepare immutable Job YAML for $BLOCK/$order"
    IFS=$'\t' read -r yaml_path yaml_sha extra <<< "$prepared"
    [ -z "$extra" ] && [ -s "$yaml_path" ] && [[ "$yaml_sha" =~ ^[0-9a-f]{64}$ ]] \
        || die "job preparation returned invalid durable YAML evidence for $BLOCK/$order"
    generate_name="$(sed -n 's/^  generateName: //p' "$yaml_path")"

    if ! create_out="$(kubectl -n "$NAMESPACE" create -f "$yaml_path" \
        -o 'jsonpath={.metadata.name}{"\t"}{.metadata.uid}')"; then
        append_manifest "$order" "" "" "$(basename "$yaml")" "" "" \
            "$submitted_at" "create-failed" "" "" "" "$yaml_path" "$yaml_sha" \
            || true
        die "create failed before Kubernetes returned the generated Job identity; inspect the namespace before any retry"
    fi
    if [[ "$create_out" == *$'\n'* ]]; then
        append_manifest "$order" "" "" "$(basename "$yaml")" "" "" \
            "$submitted_at" "identity-unknown" "" "" "" "$yaml_path" "$yaml_sha" \
            || true
        die "Kubernetes returned multiple Job identity records; inspect the namespace before any retry"
    fi
    IFS=$'\t' read -r job job_uid extra <<< "$create_out"
    if [ -n "$extra" ] || ! valid_job_name "$job" \
        || [[ "$job" != "$generate_name"* ]] || ! valid_job_uid "$job_uid"; then
        append_manifest "$order" "" "" "$(basename "$yaml")" "" "" \
            "$submitted_at" "identity-unknown" "" "" "" "$yaml_path" "$yaml_sha" \
            || true
        die "Kubernetes returned an invalid generated Job name/UID; inspect the namespace before any retry"
    fi
    CURRENT_JOB="$job"
    if ! append_manifest "$order" "$job" "$job_uid" "$(basename "$yaml")" "" "" \
        "$submitted_at" "submitted" "" "" "" "$yaml_path" "$yaml_sha"; then
        cleanup_exact_job "$job" "$job_uid" "submitted-manifest failure" \
            || die "submitted $job but both manifest write and exact cleanup failed"
        die "submitted $job but could not record it; exact cleanup completed"
    fi
    echo "   [$order] submitted $job"
    # The in-pod runner copies (then removes) its own YAML from the shared
    # job-configs dir into the run's output -- same contract as submit_job.sh.
    if ! cp "$yaml_path" "$JOB_CONFIGS_DIR/$job.yaml"; then
        if ! append_manifest "$order" "$job" "$job_uid" "$(basename "$yaml")" "" "" \
            "$submitted_at" "config-copy-failed" "" "" "" "$yaml_path" "$yaml_sha" \
            ; then
            cleanup_exact_job "$job" "$job_uid" "config-copy manifest failure" \
                || die "config-copy manifest write and exact cleanup both failed"
            die "could not append config-copy failure; exact cleanup completed"
        fi
        cleanup_exact_job "$job" "$job_uid" "config-copy failure" \
            || die "exact cleanup failed for $job/$job_uid after config-copy failure"
        append_manifest "$order" "$job" "$job_uid" "$(basename "$yaml")" "" "" \
            "$submitted_at" "cleanup-confirmed" "" "" "" "$yaml_path" "$yaml_sha" \
            || die "config-copy cleanup completed but its durable confirmation failed"
        CURRENT_JOB=""
        return 1
    fi

    monitor_leaf "$order" "$(basename "$yaml")" "$yaml_path" "$yaml_sha" \
        "$job" "$job_uid" "$submitted_at"
}

RAN=0
FAILED_ORDERS=()
for i in "${!LEAF_FILES[@]}"; do
    order=$((i+1))
    if [ -n "$ONLY" ] && ! [[ ",$ONLY," == *",$order,"* ]]; then continue; fi
    RAN=$((RAN+1))
    if [ -n "$ONLY" ]; then
        if [ "$RECONCILE_IDENTITY" -eq 1 ]; then
            repeat_action="$("$PY" "$GUARD" repeat-action \
                --manifest "$MANIFEST" --block "$BLOCK" --order "$order" \
                --reconcile)" \
                || die "cannot determine safe repeat action for $BLOCK/$order"
        else
            repeat_action="$("$PY" "$GUARD" repeat-action \
                --manifest "$MANIFEST" --block "$BLOCK" --order "$order")" \
                || die "cannot determine safe repeat action for $BLOCK/$order"
        fi
        IFS=$'\t' read -r action prior_job prior_uid prior_yaml prior_submitted \
            prior_yaml_path prior_yaml_sha \
            <<< "$repeat_action"
        if [ "$action" = "resume" ]; then
            [ "$prior_yaml" = "${LEAF_FILES[$i]}" ] \
                || die "$BLOCK/$order submitted YAML differs from the frozen coordinate"
            echo "   [$order] resuming submitted job $prior_job"
            monitor_leaf "$order" "$prior_yaml" "$prior_yaml_path" \
                "$prior_yaml_sha" "$prior_job" "$prior_uid" "$prior_submitted" \
                || FAILED_ORDERS+=("$order")
            continue
        fi
        if [ "$action" = "reconcile" ]; then
            [ "$prior_yaml" = "${LEAF_FILES[$i]}" ] \
                || die "$BLOCK/$order ambiguous YAML differs from the frozen coordinate"
            recovered_uid="$(kubectl -n "$NAMESPACE" get job "$prior_job" \
                --ignore-not-found -o jsonpath='{.metadata.uid}' 2>/dev/null)" \
                || die "could not explicitly reconcile named job $prior_job"
            if [ -z "$recovered_uid" ]; then
                confirm_reconciled_absence "$prior_job" "$prior_uid" \
                    || die "named job $prior_job or its prior pods are not absent"
                confirmation_uid=""
                valid_job_uid "$prior_uid" && confirmation_uid="$prior_uid"
                append_manifest "$order" "$prior_job" "$confirmation_uid" "$prior_yaml" "" "" \
                    "$prior_submitted" "cleanup-confirmed" "" "" "" \
                    "$prior_yaml_path" "$prior_yaml_sha" \
                    || die "could not durably record confirmed absence of named job $prior_job"
                echo "   [$order] cleanup confirmed for absent named job $prior_job; submitting a fresh identity"
            elif valid_job_uid "$recovered_uid"; then
                if valid_job_uid "$prior_uid" && [ "$recovered_uid" != "$prior_uid" ]; then
                    die "named job $prior_job was replaced during reconciliation"
                fi
                restore_recovered_job_config "$prior_job" "$prior_yaml_path" \
                    "$prior_yaml_sha" \
                    || die "could not safely restore shared config for recovered job $prior_job"
                append_manifest "$order" "$prior_job" "$recovered_uid" "$prior_yaml" "" "" \
                    "$prior_submitted" "submitted" "" "" "" \
                    "$prior_yaml_path" "$prior_yaml_sha" \
                    || die "could not record recovered identity for named job $prior_job"
                echo "   [$order] recovered named job $prior_job/$recovered_uid; resuming it"
                monitor_leaf "$order" "$prior_yaml" "$prior_yaml_path" \
                    "$prior_yaml_sha" "$prior_job" "$recovered_uid" "$prior_submitted" \
                    || FAILED_ORDERS+=("$order")
                continue
            else
                die "named job $prior_job returned an invalid UID during reconciliation"
            fi
        fi
    fi
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
