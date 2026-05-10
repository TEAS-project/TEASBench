#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install pandas pyyaml

python3 generate.py \
    --csv_file=../experiments/smoke_test.csv \
    --target_dir=./ \


