#!/bin/bash

job_yaml=$1

JOB_CONFIGS_DIR=/eidfs/eidf230/shared/gpu-service/job-configs

job_create_response=$(kubectl -n eidf230ns create -f $job_yaml)
job_creation_exit_code=$?

echo $job_create_response

if [ $job_creation_exit_code -eq 0 ]; then
    job_name=`echo $job_create_response | awk '{print $1}' | xargs basename`
    echo "Copying job yaml to ${JOB_CONFIGS_DIR}/${job_name}.yaml"
    cp $job_yaml ${JOB_CONFIGS_DIR}/${job_name}.yaml
fi


