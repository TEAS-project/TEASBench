#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install pandas pyyaml re

python3 generate.py \
    --csv_file=data/smoke_test.csv \
    --target_dir=./ \
    --inference_engine=$1

