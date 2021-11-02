import os
import logging
import argparse
import pandas as pd
from get_data import read_config, get_data


def load_data(config_path):
    data = get_data(config_path)
    config = read_config(config_path)
    load_data_path = config["load_data"]["raw_dataset_csv"]
    new_cols = [col.replace(' ','_') for col in data.columns]
    data.to_csv(load_data_path,sep=",", index=False, header=new_cols)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    load_data(config_path=parsed_args.config)