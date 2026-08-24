#!/bin/bash
# Optionally generate, run, collect, and validate four excluded A100x2
# LongBench compatibility diagnostics. These are not an E1 admission requirement.
#
#   ./pipeline/k8s/helpers/run_study_preflight.sh
#   ./pipeline/k8s/helpers/run_study_preflight.sh --dry-run --skip-image-check

set -uo pipefail

NAMESPACE=eidf230ns
JOB_CONFIGS_DIR="${STUDY_JOB_CONFIGS_DIR:-/eidfs/eidf230/shared/gpu-service/job-configs}"
MOE_CAP_REPO=https://github.com/Auto-CAP/MoE-CAP.git
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CSV="$REPO_ROOT/experiments/replication-study-eidf.csv"
GUARD="$SCRIPT_DIR/study_guard.py"
COLLECTOR_YAML="$SCRIPT_DIR/study_preflight_collector.yaml"
STATE_DIR="${STUDY_STATE_DIR:-$HOME/.teas-replication-study-x2}"
RESULTS_REPO=TEAS_Development_Results_Private
DRY_RUN=0
SKIP_IMAGE_CHECK=0
LEAF_TIMEOUT_HOURS=12

die() { echo "ERROR: $*" >&2; exit 1; }
need_val() { [ $# -ge 2 ] || die "$1 needs a value"; }
valid_job_name() {
    [ -n "$1" ] && [ "${#1}" -le 63 ] \
        && [[ "$1" =~ ^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$ ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --skip-image-check) SKIP_IMAGE_CHECK=1 ;;
        --leaf-timeout-hours) need_val "$@"; LEAF_TIMEOUT_HOURS="$2"; shift ;;
        *) die "unknown option $1" ;;
    esac
    shift
done
[[ "$LEAF_TIMEOUT_HOURS" =~ ^[0-9]{1,3}$ ]] \
    || die "--leaf-timeout-hours must be a decimal integer from 1 to 168"
LEAF_TIMEOUT_HOURS=$((10#$LEAF_TIMEOUT_HOURS))
[ "$LEAF_TIMEOUT_HOURS" -ge 1 ] && [ "$LEAF_TIMEOUT_HOURS" -le 168 ] \
    || die "--leaf-timeout-hours must be a decimal integer from 1 to 168"
[ "$DRY_RUN" -eq 1 ] || [ "$SKIP_IMAGE_CHECK" -eq 0 ] \
    || die "--skip-image-check is allowed only with --dry-run"

WORKTREE_STATUS="$(git -C "$REPO_ROOT" status --porcelain)" \
    || die "could not inspect the TEASBench working tree"
if [ -n "$WORKTREE_STATUS" ]; then
    [ "$DRY_RUN" -eq 1 ] \
        || die "TEASBench working tree is dirty; commit the exact preflight code first"
    echo "WARNING: dirty working tree; dry-run output is verification-only."
fi

mkdir -p "$STATE_DIR" || die "could not create study state directory $STATE_DIR"
LOCK_DIR="$STATE_DIR/.launcher.lock"
mkdir "$LOCK_DIR" 2>/dev/null \
    || die "study launcher lock is held at $LOCK_DIR"
COLLECTOR_POD=""
release_resources() {
    if [ -n "$COLLECTOR_POD" ]; then
        kubectl -n "$NAMESPACE" delete pod "$COLLECTOR_POD" \
            --wait=false >/dev/null 2>&1 || true
    fi
    rm -f "$LOCK_DIR/owner"
    rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap release_resources EXIT
printf 'pid=%s\nhost=%s\nstarted_at=%s\n' "$$" "$(hostname)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/owner" \
    || die "could not record launcher lock ownership"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/teas-preflight-XXXXXX")" \
    || die "could not create preflight working directory"
IMAGE_PIN_FILE="$STATE_DIR/image-digests.tsv"
FINAL_MANIFEST="$STATE_DIR/a100x2-compatibility-preflight.jsonl"
VALIDATION_RECORD="$STATE_DIR/a100x2-compatibility-preflight.validated.json"
WORK_MANIFEST="$WORKDIR/a100x2-compatibility-preflight.jsonl"

PY=python3
if ! "$PY" -c "import pandas, yaml" 2>/dev/null; then
    VENV="$STATE_DIR/venv"
    [ -x "$VENV/bin/python3" ] || {
        echo "Creating venv (pandas, pyyaml)..."
        python3 -m venv "$VENV" && "$VENV/bin/pip" -q install pandas pyyaml
    }
    PY="$VENV/bin/python3"
fi

REF_FILE="$STATE_DIR/moe_cap_ref"
if [ ! -s "$REF_FILE" ]; then
    echo "Resolving MoE-CAP main once for the study..."
    RESOLVED_MOE_CAP_REF="$(git ls-remote "$MOE_CAP_REPO" refs/heads/main | cut -f1)"
    [[ "$RESOLVED_MOE_CAP_REF" =~ ^[0-9a-f]{40}$ ]] \
        || die "could not resolve a full MoE-CAP main commit"
    "$PY" "$GUARD" write-state --path "$REF_FILE" --value "$RESOLVED_MOE_CAP_REF" \
        || die "could not freeze the MoE-CAP commit"
fi
MOE_CAP_REF="$(cat "$REF_FILE")"
[[ "$MOE_CAP_REF" =~ ^[0-9a-f]{40}$ ]] || die "invalid frozen MoE-CAP commit"

TEASBENCH_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    || die "could not resolve the TEASBench commit"
[[ "$TEASBENCH_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "invalid TEASBench commit"
COMMIT_FILE="$STATE_DIR/teasbench_commit"
if [ -s "$COMMIT_FILE" ] && [ "$(cat "$COMMIT_FILE")" != "$TEASBENCH_COMMIT" ]; then
    die "study state is pinned to $(cat "$COMMIT_FILE"), not $TEASBENCH_COMMIT"
fi
if [ "$DRY_RUN" -eq 0 ]; then
    "$PY" "$GUARD" write-state --path "$COMMIT_FILE" --value "$TEASBENCH_COMMIT" \
        || die "could not freeze the TEASBench commit"
fi

PREFLIGHT_CSV="$WORKDIR/preflight.csv"
awk -F, -v ref="$MOE_CAP_REF" 'BEGIN {OFS=","}
    NR == 1 {print $0,"compatibility_preflight"; next}
    $10 == "E1" && $5 == "longbench_v1" {$12=ref; print $0,"true"}' \
    "$CSV" > "$PREFLIGHT_CSV"
[ "$(wc -l < "$PREFLIGHT_CSV" | tr -d ' ')" -eq 5 ] \
    || die "expected exactly four E1 LongBench preflight recipes"

echo "Generating four excluded A100x2 compatibility preflights..."
GEN_LOG="$WORKDIR/generate.log"
(cd "$REPO_ROOT/pipeline" && \
    "$PY" generate.py --csv_file="$PREFLIGHT_CSV" --target_dir="$WORKDIR" \
        --results_repo="$RESULTS_REPO") > "$GEN_LOG" 2>&1 \
    || { cat "$GEN_LOG"; die "preflight generation failed"; }
YAMLS=()
while IFS= read -r yaml_name; do
    [ -n "$yaml_name" ] && YAMLS+=("$WORKDIR/$yaml_name")
done < <(grep -o 'wrote .*\.yaml' "$GEN_LOG" | awk '{print $2}')
[ "${#YAMLS[@]}" -eq 4 ] || { cat "$GEN_LOG"; die "expected four YAMLs"; }

if [ "$SKIP_IMAGE_CHECK" -eq 0 ]; then
    TAGS="$WORKDIR/image-tags.txt"
    RESOLVED="$WORKDIR/resolved-images.tsv"
    grep -h '^[[:space:]]*image:' "${YAMLS[@]}" | awk '{print $2}' | sort -u > "$TAGS"
    [ "$(wc -l < "$TAGS" | tr -d ' ')" -eq 4 ] \
        || die "expected four serving-image tags"
    : > "$RESOLVED"
    while IFS= read -r image; do
        repo="${image%:*}"; tag="${image##*:}"
        response="$(curl -fsSL "https://hub.docker.com/v2/repositories/$repo/tags/$tag")" \
            || die "could not resolve Docker Hub tag $image"
        digest="$(printf '%s' "$response" | "$PY" -c \
            'import json,sys; print(json.load(sys.stdin).get("digest", ""))')"
        [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] \
            || die "Docker Hub returned an invalid digest for $image"
        printf '%s\t%s\n' "$image" "$digest" >> "$RESOLVED"
    done < "$TAGS"
    if [ "$DRY_RUN" -eq 1 ]; then
        "$PY" "$GUARD" pin-images --pin-file "$IMAGE_PIN_FILE" \
            --resolved-file "$RESOLVED" "${YAMLS[@]}" \
            || die "preflight image validation failed"
    else
        "$PY" "$GUARD" pin-images --pin-file "$IMAGE_PIN_FILE" \
            --resolved-file "$RESOLVED" --persist "${YAMLS[@]}" \
            || die "preflight image freeze failed"
    fi
fi

echo "Planned preflights:"
for yaml_path in "${YAMLS[@]}"; do echo "   $(basename "$yaml_path")"; done
[ "$DRY_RUN" -eq 1 ] && {
    echo "Dry run: four YAMLs generated; nothing submitted. Files: $WORKDIR"
    exit 0
}

ENGINES=(vllm vllm sglang sglang)
VERSIONS=(0.16.0 0.21.0 0.5.9 0.5.12.post1)
CAPTURE_ROOT="$STATE_DIR/compatibility-preflight/$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$CAPTURE_ROOT" || die "could not create preflight capture directory"
COLLECTOR_OUT="$(kubectl -n "$NAMESPACE" create -f "$COLLECTOR_YAML" -o name)" \
    || die "could not create the develop-PVC evidence collector"
COLLECTOR_POD="${COLLECTOR_OUT#pod/}"
[ -n "$COLLECTOR_POD" ] && [ "$COLLECTOR_POD" != "$COLLECTOR_OUT" ] \
    || die "collector creation returned an invalid pod name"
kubectl -n "$NAMESPACE" wait --for=condition=Ready "pod/$COLLECTOR_POD" \
    --timeout=120s >/dev/null \
    || die "develop-PVC evidence collector did not become ready"

for index in 0 1 2 3; do
    yaml_path="${YAMLS[$index]}"
    prepared="$("$PY" "$GUARD" prepare-job --yaml "$yaml_path" \
        --state-dir "$STATE_DIR/preflight-submitted-yamls")" \
        || die "could not prepare preflight $((index + 1))"
    IFS=$'\t' read -r submitted_yaml submitted_sha extra <<< "$prepared"
    [ -z "$extra" ] && [ -s "$submitted_yaml" ] \
        && [[ "$submitted_sha" =~ ^[0-9a-f]{64}$ ]] \
        || die "preflight preparation returned invalid durable YAML evidence"
    generate_name="$(sed -n 's/^  generateName: //p' "$submitted_yaml")"
    create_out="$(kubectl -n "$NAMESPACE" create -f "$submitted_yaml" \
        -o 'jsonpath={.metadata.name}{"\t"}{.metadata.uid}')" \
        || die "could not submit preflight $((index + 1))"
    [[ "$create_out" != *$'\n'* ]] \
        || die "preflight $((index + 1)) returned multiple identity records"
    IFS=$'\t' read -r job job_uid extra <<< "$create_out"
    [ -z "$extra" ] && valid_job_name "$job" && [[ "$job" == "$generate_name"* ]] \
        || die "preflight $((index + 1)) returned an invalid generated Job name"
    [[ "$job_uid" =~ ^[0-9a-f][0-9a-f-]{7,}$ ]] \
        || die "preflight $job returned an invalid UID"
    if ! cp "$submitted_yaml" "$JOB_CONFIGS_DIR/$job.yaml"; then
        kubectl -n "$NAMESPACE" delete job "$job" --wait=false >/dev/null 2>&1 || true
        die "could not install shared config for preflight $job"
    fi
    echo "   [$((index + 1))/4] submitted $job"

    deadline=""; pod=""; outcome=""
    while :; do
        current_uid="$(kubectl -n "$NAMESPACE" get job "$job" \
            -o jsonpath='{.metadata.uid}' 2>/dev/null)"
        [ "$current_uid" = "$job_uid" ] || die "preflight job $job vanished or was replaced"
        pod="$(kubectl -n "$NAMESPACE" get pods -l job-name="$job" \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)"
        phase="$(kubectl -n "$NAMESPACE" get pods -l job-name="$job" \
            -o jsonpath='{.items[0].status.phase}' 2>/dev/null)"
        [ -z "$deadline" ] && [ "$phase" = "Running" ] \
            && deadline=$(( $(date +%s) + LEAF_TIMEOUT_HOURS * 3600 ))
        complete="$(kubectl -n "$NAMESPACE" get job "$job" \
            -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)"
        failed="$(kubectl -n "$NAMESPACE" get job "$job" \
            -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)"
        if [ "$complete" = True ]; then outcome=complete; break; fi
        if [ "$failed" = True ]; then outcome=failed; break; fi
        if [ -n "$deadline" ] && [ "$(date +%s)" -gt "$deadline" ]; then
            outcome=timeout
            kubectl -n "$NAMESPACE" delete job "$job" --wait=false >/dev/null 2>&1 || true
            break
        fi
        sleep 60
    done
    [ "$outcome" = complete ] || die "preflight $job ended $outcome"
    [ -n "$pod" ] || die "could not identify completed preflight pod for $job"

    receipt_message="$(kubectl -n "$NAMESPACE" get pod "$pod" \
        -o jsonpath='{.status.containerStatuses[0].state.terminated.message}' 2>/dev/null)"
    receipt="$WORKDIR/receipt-$index.json"
    "$PY" "$GUARD" write-state --path "$receipt" --value "$receipt_message" \
        || die "preflight $job has no valid terminal evidence"
    pod_artifact_dir="$("$PY" "$GUARD" capture-preflight \
        --receipt "$receipt" --job "$job" --job-uid "$job_uid")" \
        || die "preflight $job returned invalid terminal evidence"
    local_artifact_dir="$CAPTURE_ROOT/${ENGINES[$index]}-${VERSIONS[$index]}"
    mkdir -p "$local_artifact_dir" || die "could not create local artifact directory"
    for artifact in metadata.json metrics.json; do
        kubectl -n "$NAMESPACE" exec "$COLLECTOR_POD" -- \
            cat "$pod_artifact_dir/$artifact" > "$local_artifact_dir/$artifact" \
            || die "could not collect $artifact through the develop-PVC helper"
        [ -s "$local_artifact_dir/$artifact" ] \
            || die "collector returned an empty $artifact for preflight $job"
    done
    cp "$submitted_yaml" "$local_artifact_dir/preflight-job.yaml" \
        || die "could not capture submitted YAML for preflight $job"
    "$PY" "$GUARD" capture-preflight --receipt "$receipt" \
        --job "$job" --job-uid "$job_uid" --manifest "$WORK_MANIFEST" \
        --artifact-dir "$local_artifact_dir" >/dev/null \
        || die "could not append evidence for preflight $job"
    echo "   [$((index + 1))/4] scientifically complete and published"
done

"$PY" "$GUARD" validate-preflight --manifest "$WORK_MANIFEST" \
    --image-pins "$IMAGE_PIN_FILE" --teasbench-commit "$TEASBENCH_COMMIT" \
    --moe-cap-ref "$MOE_CAP_REF" \
    || die "collected preflight evidence did not validate"
"$PY" "$GUARD" write-state --path "$FINAL_MANIFEST" \
    --value "$(cat "$WORK_MANIFEST")" \
    || die "could not persist validated preflight manifest"
"$PY" "$GUARD" validate-preflight --manifest "$FINAL_MANIFEST" \
    --image-pins "$IMAGE_PIN_FILE" --teasbench-commit "$TEASBENCH_COMMIT" \
    --moe-cap-ref "$MOE_CAP_REF" --record "$VALIDATION_RECORD" \
    || die "persisted preflight evidence did not revalidate"

echo "A100x2 compatibility preflight complete."
echo "   manifest: $FINAL_MANIFEST"
echo "   validation: $VALIDATION_RECORD"
