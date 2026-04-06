#!/bin/bash

JOB=$1

echo "Description of job $JOB"


kubectl -n eidf230ns describe jobs $JOB
