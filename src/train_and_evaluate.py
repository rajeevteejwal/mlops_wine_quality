import os
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse
import numpy as np
import json
import joblib
from get_data import read_config

def evaluate_metrics(actual, pred):
    r2 = r2_score(actual,pred)
    mae = mean_absolute_error(actual,pred)
    rmse = np.sqrt(mean_squared_error(actual,pred))
    return r2, rmse, mae


def train_and_evaluate(config_path):
    config = read_config(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    output_col = config["base"]["target_col"]
    random_state = config["base"]["random_state"]
    train_dataset = pd.read_csv(train_data_path,sep=",", encoding="utf-8")
    test_dataset = pd.read_csv(test_data_path,sep=",", encoding="utf-8")
    y_train =  train_dataset[[output_col]]
    x_train = train_dataset.drop([output_col],axis=1)
    y_test =  test_dataset[[output_col]]
    x_test = test_dataset.drop([output_col],axis=1)
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    lr = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=random_state)
    lr.fit(x_train,y_train)

    prediction = lr.predict(x_test)
    r2, rmse, mae = evaluate_metrics(y_test,prediction)

    print(f"ElasticNet model (alpha: {alpha}, l1_ratio: {l1_ratio}")
    print(f" RMSE: {rmse}")
    print(f" MAE: {mae}")
    print(f" R2 Score: {r2}")

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file,"w") as f:
        scores = {
            "r2":r2,
            "rmse":rmse,
            "mae":mae
        }
        json.dump(scores,f,indent=4)

    with open(params_file,"w") as f:
        params = {
            "alpha":alpha,
            "l1_ratio":l1_ratio
        }
        json.dump(params,f,indent=4)

    model_dir = config["model_dir"]
    os.makedirs(model_dir,exist_ok=True)
    model_path = os.path.join(model_dir,"model.joblib")
    joblib.dump(lr,model_path)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)

