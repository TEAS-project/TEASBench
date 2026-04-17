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

Run [generate.sh](generate.sh) or generate.py directly to generate k8s config files.
Yaml configurations will be written to current working directory by default, can specify different using `--target_dir`:

```
#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install pandas pyyaml re

python3 generate.py --csv_file=data/smoke_tests.csv
```


## Create k8s jobs on EIDF

Create a single experiment job using e.g.: `kubectl -n eidf230ns create -f sglang-gpt-oss-20b-gsm8k-ns1-a100x1-bs1.yaml`

To delete both the job itself and the associated configmap (used to save the job yaml in outputs):  `kubectl -n eidf230ns delete -f sglang-gpt-oss-20b-gsm8k-ns1-a100x1-bs1.yaml`







