import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from get_data import read_config

def split_data(config_path):
    config = read_config(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    
    raw_data = pd.read_csv(raw_data_path, sep=",", encoding="utf-8")
    
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    test_size = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    train, test = train_test_split(raw_data,test_size=test_size,random_state=random_state)
    train.to_csv(train_data_path,sep=",",index=False,encoding="utf-8")
    test.to_csv(test_data_path,sep=",",index=False,encoding="utf-8")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    split_data(config_path=parsed_args.config)