#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install pandas

python3 generate.py \
    --csv_file=data/experiments.csv \
    --target_dir=configs
