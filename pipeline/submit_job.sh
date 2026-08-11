#!/bin/bash
#
# Submit a generated Job YAML, and keep a copy of what was submitted.
#
#     bash submit_job.sh out/<run>.yaml [site]
#
# Namespace and archive directory come from the site profile in
# configs/sites/<site>.yaml (default eidf, or $TEASBENCH_SITE), so this script
# names no cluster of its own. Override either with TEASBENCH_K8S_NAMESPACE /
# JOB_CONFIGS_DIR in the environment.

job_yaml=$1
SITE="${2:-${TEASBENCH_SITE:-eidf}}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_FILE="$HERE/configs/sites/$SITE.yaml"
[ -f "$SITE_FILE" ] || { echo "ERROR: no site profile at $SITE_FILE" >&2; exit 1; }

# Read the two values we need. Plain grep rather than a YAML parser: these are
# top-level scalars, and this script must work on a login node with nothing
# installed beyond kubectl and bash.
site_value() {
    sed -n "s/^$1:[[:space:]]*//p" "$SITE_FILE" | head -1 \
        | sed -e 's/[[:space:]]*#.*$//' -e 's/["'\'']//g' -e 's/[[:space:]]*$//'
}

NAMESPACE="${TEASBENCH_K8S_NAMESPACE:-$(site_value namespace)}"
JOB_CONFIGS_DIR="${JOB_CONFIGS_DIR:-$(site_value job_configs_dir)}"

[ -n "$NAMESPACE" ] || { echo "ERROR: no namespace in $SITE_FILE" >&2; exit 1; }

job_create_response=$(kubectl -n "$NAMESPACE" create -f $job_yaml)
job_creation_exit_code=$?

echo $job_create_response

if [ $job_creation_exit_code -eq 0 ]; then
    job_name=`echo $job_create_response | awk '{print $1}' | xargs basename`
    if [ -n "$JOB_CONFIGS_DIR" ]; then
        echo "Copying job yaml to ${JOB_CONFIGS_DIR}/${job_name}.yaml"
        cp $job_yaml ${JOB_CONFIGS_DIR}/${job_name}.yaml
    fi
    echo $job_name >> submitted_jobs.log
fi


