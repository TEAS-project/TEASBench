#!/bin/bash

source venv/bin/activate
python3 collect_results.py \
	--experiments_csv="data/experiments.csv" \
	--results_dir="MoE-CAP-outputs"