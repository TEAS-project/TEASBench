#!/bin/bash
# Runs only inside controlled-variation study jobs, after development publish.

study_fail() {
    echo "STUDY VALIDATION FAILED: $*" >&2
    exit 1
}

[ "${STUDY_SERVER_OK:-0}" = 1 ] || study_fail "server did not become ready"
[ "${STUDY_CLIENT_OK:-0}" = 1 ] || study_fail "client did not finish successfully"
[ "${STUDY_ENRICH_OK:-0}" = 1 ] || study_fail "metadata enrichment failed"
[ "${STUDY_PUBLISH_OK:-0}" = 1 ] || study_fail "development publication failed"
[ -n "${PVC_RUN_OUTPUT_DIR:-}" ] && [ -d "$PVC_RUN_OUTPUT_DIR" ] \
    || study_fail "PVC output directory is missing"

for artifact in metadata.json metrics.json detailed_results.jsonl output_data.jsonl \
                timings.json pip_freeze.txt; do
    [ -s "$PVC_RUN_OUTPUT_DIR/$artifact" ] || study_fail "missing artifact $artifact"
done

study_yamls=("$PVC_RUN_OUTPUT_DIR"/*.yaml "$PVC_RUN_OUTPUT_DIR"/*.yml)
existing_yamls=()
for candidate in "${study_yamls[@]}"; do
    [ -f "$candidate" ] && existing_yamls+=("$candidate")
done
[ "${#existing_yamls[@]}" -eq 1 ] \
    || study_fail "expected exactly one launch YAML, got ${#existing_yamls[@]}"
JOB_YAML="${existing_yamls[0]}"
IMAGE_REF="$(sed -nE 's/^[[:space:]]*image:[[:space:]]*(.+@sha256:[0-9a-f]{64})[[:space:]]*$/\1/p' "$JOB_YAML")"
[[ "$IMAGE_REF" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]] \
    || study_fail "launch YAML is not digest-pinned exactly once"
[ "$(grep -Ec '^[[:space:]]*image:' "$JOB_YAML")" -eq 1 ] \
    || study_fail "launch YAML must contain exactly one serving image"

expert_files=("$PVC_RUN_OUTPUT_DIR"/expert_distribution_record*.jsonl)
EXPERT_OK=0
for candidate in "${expert_files[@]}"; do
    [ -s "$candidate" ] && EXPERT_OK=1
done
if [ "$STUDY_ENGINE" = "sglang" ]; then
    [ "$EXPERT_OK" -eq 1 ] || study_fail "missing SGLang expert-distribution artifact"
fi

jq -e '.quality.total == 256 and .quality.attempted == 256 and
       .quality.served == 256 and .quality.completed == 256' \
    "$PVC_RUN_OUTPUT_DIR/metrics.json" >/dev/null \
    || study_fail "metrics quality counts are not all 256"

if [ "${STUDY_PREFLIGHT:-0}" = 1 ]; then
    jq -e --arg node "$k8s_node_name" --arg job "$k8s_job_name" \
          --arg uid "$k8s_job_uid" --arg engine "$STUDY_ENGINE" \
          --arg version "$STUDY_ENGINE_VERSION" --arg teas "$TEASBENCH_COMMIT" \
          --arg moe "$MOE_CAP_COMMIT" --arg image_ref "$IMAGE_REF" '
        .model_config.model_name == "unsloth/gpt-oss-120b" and
        .hardware.num_gpus == 2 and
        .hardware.gpu_type == "NVIDIA-A100-SXM4-80GB" and
        .system_environment.inference_engine == $engine and
        .system_environment.inference_engine_version == $version and
        .system_environment.teasbench_commit == $teas and
        .system_environment.moe_cap_commit == $moe and
        .compatibility_preflight.dataset == "longbench_v1" and
        .compatibility_preflight.num_samples == 256 and
        .compatibility_preflight.batch_size == "default" and
        .compatibility_preflight.gpu == "A100" and
        .compatibility_preflight.num_gpu == 2 and
        .compatibility_preflight.node == $node and
        .compatibility_preflight.job_name == $job and
        .compatibility_preflight.job_uid == $uid and
        .compatibility_preflight.image_ref == $image_ref and
        (.compatibility_preflight.gpu_uuids | length == 2 and
         (unique | length == 2) and all(.[]; length > 0))' \
        "$PVC_RUN_OUTPUT_DIR/metadata.json" >/dev/null \
        || study_fail "metadata does not match the excluded compatibility preflight"
else
    jq -e --arg study_id "$STUDY_ID" --arg block "$STUDY_BLOCK" \
      --argjson order "$STUDY_ORDER" --arg node "$k8s_node_name" \
      --arg job "$k8s_job_name" --arg uid "$k8s_job_uid" \
      --arg dataset "$STUDY_DATASET" --arg engine "$STUDY_ENGINE" \
      --arg version "$STUDY_ENGINE_VERSION" --arg gpu "$STUDY_GPU" \
      --arg gpu_product "$STUDY_GPU_PRODUCT" \
      --arg teas "$TEASBENCH_COMMIT" --arg moe "$MOE_CAP_COMMIT" '
    .model_config.model_name == "unsloth/gpt-oss-120b" and
    .hardware.num_gpus == 2 and
    .system_environment.inference_engine == $engine and
    .system_environment.inference_engine_version == $version and
    .system_environment.teasbench_commit == $teas and
    .system_environment.moe_cap_commit == $moe and
    .study.study_id == $study_id and .study.block_id == $block and
    .study.planned_order == $order and .study.node == $node and
    .study.job_name == $job and .study.job_uid == $uid and
    .study.dataset == $dataset and .study.num_samples == 256 and
    .study.gpu == $gpu and .study.num_gpu == 2 and
    .study.batch_size == "default" and
    .study.gpu_product == $gpu_product and
    (.study.gpu_uuids | split(",") | map(select(length > 0)) |
     length == 2 and (unique | length == 2))' \
    "$PVC_RUN_OUTPUT_DIR/metadata.json" >/dev/null \
    || study_fail "metadata does not match the frozen study coordinate"
fi

if [ "$STUDY_DATASET" = "arena-hard" ]; then
    jq -e '.study.arena_baseline_sha256 | test("^[0-9a-f]{64}$")' \
        "$PVC_RUN_OUTPUT_DIR/metadata.json" >/dev/null \
        || study_fail "Arena baseline hash is missing"
fi

METADATA_SHA="$(sha256sum "$PVC_RUN_OUTPUT_DIR/metadata.json" | cut -d' ' -f1)" \
    || study_fail "could not hash metadata.json"
METRICS_SHA="$(sha256sum "$PVC_RUN_OUTPUT_DIR/metrics.json" | cut -d' ' -f1)" \
    || study_fail "could not hash metrics.json"
JOB_YAML_SHA="$(sha256sum "$JOB_YAML" | cut -d' ' -f1)" \
    || study_fail "could not hash launch YAML"
DETAILED_SHA="$(sha256sum "$PVC_RUN_OUTPUT_DIR/detailed_results.jsonl" | cut -d' ' -f1)" \
    || study_fail "could not hash detailed_results.jsonl"
OUTPUT_SHA="$(sha256sum "$PVC_RUN_OUTPUT_DIR/output_data.jsonl" | cut -d' ' -f1)" \
    || study_fail "could not hash output_data.jsonl"
TIMINGS_SHA="$(sha256sum "$PVC_RUN_OUTPUT_DIR/timings.json" | cut -d' ' -f1)" \
    || study_fail "could not hash timings.json"
PIP_SHA="$(sha256sum "$PVC_RUN_OUTPUT_DIR/pip_freeze.txt" | cut -d' ' -f1)" \
    || study_fail "could not hash pip_freeze.txt"
EXPERT_BUNDLE_SHA=""
if [ "$STUDY_ENGINE" = "sglang" ]; then
    EXPERT_BUNDLE_SHA="$({ for candidate in "${expert_files[@]}"; do
            [ -s "$candidate" ] && sha256sum "$candidate"
        done; } | sort | sha256sum | cut -d' ' -f1)" \
        || study_fail "could not hash SGLang expert-distribution artifacts"
fi
RECEIPT="$PVC_RUN_OUTPUT_DIR/study-validation-receipt.json"
RECEIPT_TMP="$RECEIPT.tmp"

if [ "${STUDY_PREFLIGHT:-0}" = 1 ]; then
    RECEIPT="$PVC_RUN_OUTPUT_DIR/preflight-validation-receipt.json"
    RECEIPT_TMP="$RECEIPT.tmp"
    COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    jq -c -n --arg study_id "$STUDY_ID" --arg job "$k8s_job_name" \
          --arg job_uid "$k8s_job_uid" --arg node "$k8s_node_name" \
          --arg artifact_dir "$PVC_RUN_OUTPUT_DIR" --arg teas "$TEASBENCH_COMMIT" \
          --arg moe_ref "$PREFLIGHT_MOE_CAP_REF" --arg image_tag "$STUDY_IMAGE_TAG" \
          --arg image_ref "$IMAGE_REF" --arg metadata_sha "$METADATA_SHA" \
          --arg metrics_sha "$METRICS_SHA" --arg yaml_sha "$JOB_YAML_SHA" \
          --arg engine "$STUDY_ENGINE" --arg version "$STUDY_ENGINE_VERSION" \
          --arg completed_at "$COMPLETED_AT" --arg gpu_uuids "$GPU_UUIDS" '
        {study_id: $study_id, kind: "excluded-compatibility-preflight",
         inference_engine: $engine, engine_version: $version,
         gpu: "A100", num_gpu: 2, dataset: "longbench_v1", num_samples: 256,
         batch_size: "default", outcome: "complete", teasbench_commit: $teas,
         moe_cap_ref: $moe_ref, image_tag: $image_tag, image_ref: $image_ref,
         job_uid: $job_uid, job_name: $job, node: $node,
         gpu_uuids: ($gpu_uuids | split(",") | map(select(length > 0))),
         completed_at: $completed_at, artifact_dir: $artifact_dir,
         metadata_sha256: $metadata_sha, metrics_sha256: $metrics_sha,
         job_yaml_sha256: $yaml_sha}' > "$RECEIPT_TMP" \
        || study_fail "could not construct preflight evidence receipt"
    mv "$RECEIPT_TMP" "$RECEIPT" \
        || study_fail "could not finalize preflight evidence receipt"
    STUDY_TERMINATION_LOG="${STUDY_TERMINATION_LOG:-/dev/termination-log}"
    cp "$RECEIPT" "$STUDY_TERMINATION_LOG" \
        || study_fail "could not publish preflight evidence to the pod status"
    echo "Preflight evidence receipt written to $RECEIPT"
    exit 0
fi

jq -n --arg study_id "$STUDY_ID" --arg block "$STUDY_BLOCK" \
      --argjson order "$STUDY_ORDER" --arg job "$k8s_job_name" \
      --arg job_uid "$k8s_job_uid" --arg node "$k8s_node_name" \
      --arg output_path "$PVC_RUN_OUTPUT_DIR" --arg publish_path "$PUBLISH_SUBDIR" \
      --arg metadata_sha "$METADATA_SHA" --arg metrics_sha "$METRICS_SHA" \
      --arg yaml_sha "$JOB_YAML_SHA" --arg teas "$TEASBENCH_COMMIT" \
      --arg moe "$MOE_CAP_COMMIT" --arg image_ref "$IMAGE_REF" \
      --arg detailed_sha "$DETAILED_SHA" --arg output_sha "$OUTPUT_SHA" \
      --arg timings_sha "$TIMINGS_SHA" --arg pip_sha "$PIP_SHA" \
      --arg expert_sha "$EXPERT_BUNDLE_SHA" --arg engine "$STUDY_ENGINE" \
      --arg version "$STUDY_ENGINE_VERSION" --arg dataset "$STUDY_DATASET" '
    {receipt_version: 1, status: "validated", study_id: $study_id,
     block: $block, planned_order: $order, job: $job, job_uid: $job_uid,
     node: $node, output_path: $output_path, publish_path: $publish_path,
     publication: "development", inference_engine: $engine,
     engine_version: $version, dataset: $dataset, teasbench_commit: $teas,
     moe_cap_commit: $moe, image_ref: $image_ref, metadata_sha256: $metadata_sha,
     metrics_sha256: $metrics_sha, job_yaml_sha256: $yaml_sha,
     artifact_sha256: ({metadata: $metadata_sha, metrics: $metrics_sha,
       launch_yaml: $yaml_sha, detailed_results: $detailed_sha,
       output_data: $output_sha, timings: $timings_sha, pip_freeze: $pip_sha}
       + if $engine == "sglang" then
           {expert_distribution_bundle: $expert_sha}
         else {} end),
     quality: {total: 256, attempted: 256, served: 256, completed: 256}}' \
    > "$RECEIPT_TMP" || study_fail "could not construct validation receipt"
mv "$RECEIPT_TMP" "$RECEIPT" || study_fail "could not finalize validation receipt"

STUDY_TERMINATION_LOG="${STUDY_TERMINATION_LOG:-/dev/termination-log}"
cp "$RECEIPT" "$STUDY_TERMINATION_LOG" \
    || study_fail "could not publish validation receipt to the pod status"
echo "Study validation receipt written to $RECEIPT"
exit 0
