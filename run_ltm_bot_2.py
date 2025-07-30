# First, let's fix and clean up the original script into a working, structured Python script.
# This script:
# - Loads data
# - Normalizes features/targets using a Preprocess() function
# - Sends them to a GraphQL endpoint for training via fit_LTM_Bot()

import numpy as np
import pandas as pd
import requests
import json

from sklearn.preprocessing import StandardScaler
from api.preprocessing import feature_1_denormalize, Preprocess
import requests
import json



timesteps = 10  # or whatever you're using

def sanitize_array(arr):
    if isinstance(arr, np.ndarray):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
        return arr.tolist()
    return arr
def reshape_for_lstm(data, timesteps):
    result = []
    for i in range(len(data) - timesteps):
        result.append(data[i:i + timesteps])
    return np.array(result, dtype='float64')
def fit_LTM_Bot(bot_config):
    query = """
        query TrainLTM(
            $X_train: JSON!
            $y_train: JSON!
            $X_test: JSON!
            $y_test: JSON!
            $predict: Boolean
            $timesteps: Int
            $model_case_version_main_target_code: Int
            $algorithm_configurations: JSON
            $iteration: Int
            $pretreatment_attrs: JSON
        ) {
            trainLTM(
                X_train: $X_train
                y_train: $y_train
                X_test: $X_test
                y_test: $y_test
                predict: $predict
                timesteps: $timesteps
                model_case_version_main_target_code: $model_case_version_main_target_code
                algorithm_configurations: $algorithm_configurations
                iteration: $iteration
                pretreatment_attrs: $pretreatment_attrs
            ) {
                success
                error
                main_target_predictions_train
                main_target_columns
            }
        }
    """

    variables = {
        "X_train": bot_config['X_train'],  # ✅ pass list directly
        "y_train": bot_config['y_train'],
        "X_test": bot_config['X_test'],
        "y_test": bot_config['y_test'],
        "predict": True,
        "timesteps": bot_config['timesteps'],
        "model_case_version_main_target_code": bot_config['model_case_version_main_target_code'],
        "algorithm_configurations": bot_config.get('algorithm_configurations', {}),
        "iteration": bot_config.get('iteration', 1),
        "pretreatment_attrs": bot_config.get('pretreatment_attrs', {})
    }

    url = bot_config.get('endpoint', 'http://localhost:4000/graphql')

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"query": query, "variables": variables}
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Query failed: {response.status_code} - {response.text}")


# --- Main Block ---
if __name__ == '__main__':
    # --- Load Data ---
    pd.set_option("display.max_columns", None)
    df = pd.read_csv("my_data.csv")
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)

    features = ['OPCP', 'HPCP', 'LPCP', 'CPCP', 'ACPCP']
    target = 'MPN5P'
    df = df[features + [target]]

    # --- Train/Test Split ---
    train_end = '4/14/2021'
    train = df.loc[:train_end, :]
    test = df.loc[train_end:, :].drop(train_end, axis=0)

    X_train = train.drop(target, axis=1)
    y_train = train[target]
    X_test = test.drop(target, axis=1)
    y_test = test[target]

    # --- Normalize using Preprocess function ---
    X_train_np = X_train.to_numpy(dtype='float64')
    X_test_np = X_test.to_numpy(dtype='float64')
    y_train_np = y_train.to_numpy(dtype='float64')
    y_test_np = y_test.to_numpy(dtype='float64')




    X_train_norm, X_test_norm, y_train_norm, y_test_norm, scaler_y_1, scaler_y_2 = Preprocess(
        X_train_np, X_test_np, y_train_np, y_test_np
    )
    # --- Do Somthing I really don't Know it ---
    X_train_seq = reshape_for_lstm(X_train_norm, timesteps)
    X_test_seq = reshape_for_lstm(X_test_norm, timesteps)
    # For targets: pick the value after each X sequence
    y_train_seq = y_train_norm[timesteps:]
    y_test_seq = y_test_norm[timesteps:]



    # --- Sanitize for JSON serialization ---
    X_train_json = sanitize_array(X_train_seq)
    X_test_json = sanitize_array(X_test_seq)
    y_train_json = sanitize_array(y_train_seq)
    y_test_json = sanitize_array(y_test_seq)


    # --- Prepare bot_config ---
    bot_config = {
        "X_train": X_train_json,
        "y_train": y_train_json,
        "X_test": X_test_json,
        "y_test": y_test_json,
        "predict": True,
        "timesteps": timesteps,
        "model_case_version_main_target_code": 1,
        "algorithm_configurations": {
            "layers": 3,
            "units": 128,
            "layer_1": 64,
            "layer_2": 64,
            "layer_3": 32,
            "layer_4": 32,
            "layer_5": 16,
            "attenuated_padding_value": 0.0,
            "optimizer": "adam"
        },
        "iteration": 100,
        "pretreatment_attrs": {"normalize": True},
        "endpoint": "http://127.0.0.1:4000/graphql"
    }

    # --- Call fit function ---
    fit_response = fit_LTM_Bot(bot_config)

    if not fit_response['data']['trainLTM']['success']:
        print("❌ Training failed:")
        print(fit_response['data']['trainLTM']['error'])
    else:
        predictions = fit_response['data']['trainLTM']['main_target_predictions_train']
        columns = fit_response['data']['trainLTM']['main_target_columns']
        print("✅ Predictions (train):")
        for row in predictions[:10]:
            print(row)
