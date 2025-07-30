import numpy as np
import requests
import json
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler

from api.preprocessing import feature_1_denormalize, Preprocess







def fit_LTM_Bot(bot_config):
    # Sanitize inputs
    bot_config['X_train'] = sanitize_array(bot_config['X_train'])
    bot_config['y_train'] = sanitize_array(bot_config['y_train'])
    bot_config['X_test'] = sanitize_array(bot_config['X_test'])
    bot_config['y_test'] = sanitize_array(bot_config['y_test'])

    # GraphQL query
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
        "X_train": bot_config['X_train'],
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

    # Replace with your actual endpoint
    url = "http://192.168.0.5:4000/graphql"

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"query": query, "variables": variables}
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Query failed: {response.status_code} - {response.text}")
def sanitize_array(arr):
    if isinstance(arr, np.ndarray):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
        return arr.tolist()
    return arr
def predict_LTM_Bot(bot_config, model_path):
    query = f"""
        query {{
            predictLTMBot(
                X_test: {bot_config['X_test']}
                timesteps: {bot_config['timesteps']}
                model_path: {json.dumps(model_path)}
            ){{
                success,
                error,
                predictions
            }}
        }}
    """
    endpoint = bot_config['endpoint']
    headers = {
        'Accept-Encoding': 'gzip, deflate, br',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Connection': 'keep-alive',
        'DNT': '1'
    }

    response = requests.post(endpoint, json={"query": query}, headers=headers).json()
    return response



pd.set_option.max_columns = None

df = pd.read_csv("my_data.csv")
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

features = ['OPCP', 'HPCP', 'LPCP', 'CPCP', 'ACPCP']
target = 'MPN5P'

df = df[features + [target]]


train_end = '4/14/2021'
train = df.loc[:train_end,:]
test = df.loc[train_end:,:].drop(train_end,axis=0)


X_train = train.drop("MPN5P", axis=1)
y_train = train["MPN5P"]
X_test = test.drop("MPN5P", axis=1)
y_test = test["MPN5P"]

train_index = X_train.index
test_index = X_test.index

scaler_features = StandardScaler()
scaler_target = StandardScaler()

# scale features
X_train = pd.DataFrame(scaler_features.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler_features.transform(X_test), columns=X_test.columns, index=X_test.index)
# scale target
y_train = pd.Series(scaler_target.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index)
y_test = pd.Series(scaler_target.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index)

X_train = X_train.to_numpy(dtype='float64')
y_train = y_train.to_numpy(dtype='float64')
X_test = X_test.to_numpy(dtype='float64')
y_test = y_test.to_numpy(dtype='float64')

timesteps = 3
model_case_version_main_target_code = 5
iteration = 2
apply_hyperparameter_tuning = False
apply_hyperparameter_tuning_json = json.dumps(apply_hyperparameter_tuning)

X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized , scaler_y_1, scaler_y_2 = Preprocess(X_train,X_test,y_train,y_test)
# y_original_train = feature_1_denormalize(y_train_normalized, scaler_y_1, scaler_y_2)


X_train_normalized = sanitize_array(X_train_normalized)
y_train_normalized = sanitize_array(y_train_normalized)
X_test_normalized = sanitize_array(X_test_normalized)
y_test_normalized = sanitize_array(y_test_normalized)


# bot_config = {
#     'X_train': X_train_normalized,
#     'y_train': y_train_normalized,
#     'X_test': X_test_normalized,
#     'y_test': y_test_normalized,
#     'timesteps': timesteps,
#     'apply_hyperparameter_tuning': apply_hyperparameter_tuning_json,
#     'model_case_version_main_target_code': model_case_version_main_target_code,
#     'iteration': iteration,
#     'model_name': 'MPN5P_LTM_3_timesteps',
#     'endpoint': 'http://192.168.0.5:4000/graphql'
# }
bot_config = {
    "X_train": sanitize_array(X_train),
    "y_train": sanitize_array(y_train),
    "X_test": sanitize_array(X_test),
    "y_test": sanitize_array(y_test),
    "predict": True,
    "timesteps": 3,
    "model_case_version_main_target_code": 1,
    "algorithm_configurations": {"layers": 3, "units": 128},
    "iteration": 100,
    "pretreatment_attrs": {"normalize": True}
}


if __name__ == '__main__':
    fit_response = fit_LTM_Bot(bot_config)

    if not fit_response['data']['trainLTM']['success']:
        print(fit_response['data']['trainLTM']['error'])
        print(fit_response['data']['trainLTM'])
    else:
        predictions = fit_response['data']['trainLTM']['main_target_predictions_train']
        columns = fit_response['data']['trainLTM']['main_target_columns']
        print("Predictions (train):")
        for row in predictions[:10]:
            print(row)