#!/bin/bash

# Takes two input arguments:
# 1: the .yaml file for a job (experiment)
# 2: the name of a k8s job that has the results PVC mounted
#    (e.g. an instance of ../../eidf_script/pvc_access.yaml)
#
# Copies the job .yaml to where its results will end up
# and submits the job
#

yaml=$1

set_output_dir_cmd=`grep "RUN_OUTPUT_DIR=" $yaml`
TEAS_OUTPUT_DIR=/mnt/develop/outputs
eval $set_output_dir_cmd

PVC_ACCESS_JOB=$2
PVC_ACCESS_POD=`kubectl -n eidf230ns get pods --selector=job-name=${PVC_ACCESS_JOB} -o name | xargs basename`

# Create directory path in results pvc if it doesn't exist yet
kubectl -n eidf230ns exec $PVC_ACCESS_POD -- mkdir -p $RUN_OUTPUT_DIR

# Copy the yaml
echo "Copying" $yaml "to" $RUN_OUTPUT_DIR
kubectl cp $yaml eidf230ns/${PVC_ACCESS_POD}:${RUN_OUTPUT_DIR}/${yaml}

# Submit the job and capture the generated name
JOB_NAME=$(basename `kubectl -n eidf230ns create -f $yaml`)
echo "Submitted job $JOB_NAME"

# Copy the job name to the results directory for tracking
kubectl -n eidf230ns exec $PVC_ACCESS_POD -- /bin/bash -c "echo ${JOB_NAME} > $RUN_OUTPUT_DIR/job_name"


