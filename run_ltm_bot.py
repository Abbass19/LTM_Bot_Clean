import requests
import pandas as pd
from datetime import datetime
import json

import pandas as pd
import psycopg2
import psycopg2.extras

from sklearn.preprocessing import StandardScaler


pd.set_option.max_columns = None

def data_extractor(asset,cols=['DCP','DNCP','OPCP','HPCP','LPCP','CPCP','ACPCP','VTCP','MPN5P']):
    # The credentials to conect to the database
    hostname = 'database-1.ctzm0hf7fhri.eu-central-1.rds.amazonaws.com'
    database = 'dyDATA_new'
    username = 'postgres'
    pwd = 'Proc2023awsrdspostgresql'
    port_id = 5432
    conn = None
    asset_script="SELECT * FROM "+'\"'+"ASSET_"+asset+'\"'+".features_targets_input_view WHERE features_targets_input_view."+'\"'+"cleaned_raw_features_environment_PK"+'\"'+ "= 4"
    # Here we select the active financial asset from the financial asset list table
    try:
        with psycopg2.connect(
            host = hostname,
            dbname = database,
            user = username,
            password = pwd,
            port = port_id
        ) as conn:
            dataframe = pd.read_sql(asset_script,conn)
    except Exception as error:
        conn.close()
        return error
    finally:
        if conn is not None:
            conn.close()
    dataframe = dataframe.filter(regex='|'.join(cols),axis=1)
    
    for i,j in zip(cols,dataframe.columns):
        dataframe.rename(columns={j:i},inplace=True)
    print(dataframe.tail())

    return dataframe

df = data_extractor(asset='MSFT')
df['DCP'] = pd.to_datetime(df['DCP'])
df.set_index('DCP', inplace=True) 

features = ['OPCP', 'HPCP', 'LPCP', 'CPCP', 'ACPCP', 'VTCP'] 
target = 'MPN5P'

df = df[features + [target]]

train_start = '2000-01-01'
train_end = '2000-01-31'
test_start = '2000-02-03'
# test_end = datetime.today()
test_end = '2000-02-13'

train = df.loc[(df.index >= train_start) & (df.index <= train_end)]
test = df.loc[(df.index >= test_start) & (df.index <= test_end)]

X_train = train.drop("MPN5P", axis=1)
y_train = train["MPN5P"]
X_test = test.drop("MPN5P", axis=1)
y_test = test["MPN5P"]

train_index = X_train.index
test_index = X_test.index

scaler_features = StandardScaler()
scaler_target = StandardScaler()

#scale features
X_train = pd.DataFrame(scaler_features.fit_transform(X_train),columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler_features.transform(X_test), columns=X_test.columns, index=X_test.index)
#scale target
y_train = pd.Series(scaler_target.fit_transform(y_train.values.reshape(-1,1)).flatten(), index=y_train.index)
y_test = pd.Series(scaler_target.transform(y_test.values.reshape(-1,1)).flatten(), index=y_test.index)

X_train = X_train.to_numpy(dtype='float64')
y_train = y_train.to_numpy(dtype='float64')
X_test = X_test.to_numpy(dtype='float64')
y_test = y_test.to_numpy(dtype='float64')

timesteps = 3
model_case_version_main_target_code = 5
iteration = 2
apply_hyperparameter_tuning = False
apply_hyperparameter_tuning_json = json.dumps(apply_hyperparameter_tuning)

bot_config = {
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    'timesteps': timesteps,
    'apply_hyperparameter_tuning': apply_hyperparameter_tuning_json,
    'model_case_version_main_target_code': model_case_version_main_target_code,
    'iteration': iteration,
    'model_name': 'MPN5P_LTM_3_timesteps',
    'endpoint': 'http://localhost:8000/graphql'
}

def fit_LTM_Bot(bot_config):
    query = f"""
        query {{
            fitLTMBot(
                X_train: {bot_config['X_train']}
                y_train: {bot_config['y_train']}
                timesteps: {bot_config['timesteps']}
                apply_hyperparameter_tuning: {bot_config['apply_hyperparameter_tuning']}
                model_case_version_main_target_code: {bot_config['model_case_version_main_target_code']}
                iteration: {bot_config['iteration']}
                model_name: {json.dumps(bot_config['model_name'])}
            ){{
                success,
                error,
                model_path
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

    response = requests.post(endpoint,json={"query":query},headers=headers).json()
    return response

def predict_LTM_Bot(bot_config,model_path):
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

    response = requests.post(endpoint,json={"query":query},headers=headers).json()
    return response

if __name__ == '__main__':
    case = 'MSFT'
    dataset_start_date = pd.Timestamp('2020-01-01')
    targets = ['MPN5P']
    model_case_version_main_target_code = '5'

    fit_response=fit_LTM_Bot(bot_config)
    if not fit_response['data']['fitLTMBot']['success']:
        print(fit_response['data']['fitLTMBot']['error'])
        print(fit_response['data']['fitLTMBot'])
    else:
        model_path = fit_response['data']['fitLTMBot']['model_path']
        print(model_path)
        predict_response = predict_LTM_Bot(bot_config,model_path)
        if not predict_response['data']['predictLTMBot']['success']:
            print(predict_response['data']['predictLTMBot']['error'])
            print(predict_response['data']['predictLTMBot'])
        else:
            print('Predictions: \n')
            print(predict_response['data']['predictLTMBot']['predictions'][:10])
