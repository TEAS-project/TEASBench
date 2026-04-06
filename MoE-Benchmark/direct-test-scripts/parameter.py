#!/bin/python3

import argparse
import pandas as pd
from utils import GPU_MAP, TOKEN_LENGTH_MAP


def main(experiments_csv, parameter_name, rowId):

    df = pd.read_csv(experiments_csv)

    x = df[parameter_name][rowId]

    if "token" in parameter_name : # 4K
        x = TOKEN_LENGTH_MAP[x]

    print(x)

    return x


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Path to experiments CSV file")
    parser.add_argument("--parameter_name", type=str, required=True, help="The parameter to be extracted for a given experiment")
    parser.add_argument("--experiment_id", type=int, required=True, help="The row in the csv file corresponding to the experiment to be executed")
    args = parser.parse_args()

    main(args.csv_file, args.parameter_name, args.experiment_id)


