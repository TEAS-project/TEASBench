#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install pandas

python3 generate_loop.py \
    --csv_file=data/experiments.csv \
    --target_dir=configs_loop \
    --inference_engine=$1
