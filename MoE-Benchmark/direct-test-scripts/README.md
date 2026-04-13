# Generate Sweep

This repository contains a benchmark sweep to evaluate MoE-Benchmark performance on [EIDF](https://edinburgh-international-data-facility.ed.ac.uk/). The contained [k8s configurations](configs) are created with the provided Python generator based on the parameters specified in [data](data).

- check out the table of experiments as csv
- generate yaml configs from csv
- launch jobs on eidf
- inspect generated performance data

## Experiments

The list of experiments with parameters to be found in [data/experiments.csv](data/experiments.csv).

**Note: data/smoke_test.csv allows for a rapid test of the setup**

## Generate k8s configs

Run [generate.sh](generate.sh) or generate.py directly to generate k8s config files. Yaml configurations will be written to `--target_dir`:

```
#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install pandas pyyaml re

python3 generate.py --csv_file=data/experiments.csv --target_dir=configs --inference-engine=sglang --results_repo=
```


## Create k8s jobs on EIDF

Before submitting an experiment job, make sure a pvc-access-helper job is running.
We need this to save the yaml file generated for the job alongside results for future
reference in the relevant PVC. 

To check if any pvc-access-helper job is running, run `kubectl -n eidf230ns get jobs | grep pvc-access-helper`.

If there is none, create it using `kubectl -n eidf230ns create -f ../../eidf_scripts/pvc_access.yaml`

Create a single experiment job using e.g.: `./submit_job.sh sglang-gpt-oss-20b-gsm8k-ns1-a100x1.yaml pvc-access-helper-qt7w





