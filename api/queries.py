import pandas as pd
import numpy as np
import json
import sys

from scipy.stats import linregress
from joblib import dump, load
from keras.models import load_model
from keras.optimizers import Adam

from api import settings
from api.utils import *

pd.options.display.max_columns = None


def resolve_fitLongShortTermMemory(
    obj,
    info,
    train,
    predict,
    timesteps,
    apply_hyperparameter_tuning,
    model_case_version_main_target_code,
    iteration,
    model_name,
    features,
    targets,
    learning_rate=0.0005,
    batch_size=64,
    n_epochs=100,
    attenuated_padding_value=1,
    test=None
):
    try:
        # convert json data to dataframe
        train = pd.DataFrame(json.loads(train))
        if test is not None:
             test = pd.DataFrame(json.loads(test))
        else:
            test = pd.DataFrame()
        # FTP implementation
        # concat train and test sets
        features_and_targets = features + targets
        train = train.reset_index()[features_and_targets]
        print(train.head())
        if not test.empty:
            test = test.reset_index()[features_and_targets]
            print(test.head())
        #call the FTP bot
        print('\n')
        print('-' * 25, f'FTP STARTED TO RUN FOR TRAIN SET', 25 * '-', '\n')
        values = json.dumps(train.values.tolist())
        index = json.dumps(train.index.to_list())
        columns = json.dumps(train.columns.to_list())

        ftp_response_train = execute_FTP(values=values,index=index,columns=columns).json()
        if not ftp_response_train['data']['featuresTargetsPretreatment']['success']:
            response = {
                'success':False,
                'error': ftp_response_train['data']['featuresTargetsPretreatment']['error']
                }
            return response
        pretreatment_object = json.loads(ftp_response_train['data']['featuresTargetsPretreatment']['pretreatment_info'])
        # exract scaler attributes for each target
        targets_pretreatment_attrs = {key: pretreatment_object[key] for key in targets if key in pretreatment_object}
        train = pd.DataFrame(
                data = ftp_response_train['data']['featuresTargetsPretreatment']['pretreated_values'],
                columns=ftp_response_train['data']['featuresTargetsPretreatment']['columns'],
                index=ftp_response_train['data']['featuresTargetsPretreatment']['index']
            )
        print(train.head())

        print('-' * 25, f'FTP STARTED TO RUN FOR TEST SET', 25 * '-', '\n')
        values = json.dumps(test[features_and_targets].values.tolist())
        index = json.dumps(test.index.to_list())
        columns = json.dumps(features_and_targets)
        ftp_response_test = execute_FTP(values, index, columns,json.dumps(pretreatment_object)).json()
        if not ftp_response_test['data']['featuresTargetsPretreatment']['success']:
            response = {
                'success':False,
                'error': ftp_response_test['data']['featuresTargetsPretreatment']['error']
                }
            return response
        
        test = pd.DataFrame(
            data = ftp_response_test['data']['featuresTargetsPretreatment']['pretreated_values'],
            columns = ftp_response_test['data']['featuresTargetsPretreatment']['columns'],
            index = ftp_response_test['data']['featuresTargetsPretreatment']['index'],
        )
        print(test.head())
        # Create X_train, y_train, X_test, y_test
        X_train = train[features].to_numpy('float64')
        y_train = train[targets].to_numpy('float64')
        X_test = test[features].to_numpy('float64')
        y_test = test[targets].to_numpy('float64')
        # create sequental data
        X_train, y_train = building_data_sequences(X_train, y_train, timesteps)
        X_test, y_test = building_data_sequences(X_test, y_test, timesteps)
        print('X_train shape: ', X_train.shape)
        print('y_train shape: ', y_train[0].shape)
        print('X_test shape: ', X_test.shape)
        print('y_test shape: ', y_test[0].shape)
        # define the input parameters
        input_shape = ((X_train).shape[1], (X_train).shape[2])
        optimizer = Adam(learning_rate=learning_rate)
        # return the model
        model = compile_model(
            input_shape,
            iteration,
            model_case_version_main_target_code,
            optimizer,
            attenuated_padding_value,
        )
        print("Input shape: \n", X_train.shape)
        print("\n")
        print("Model summary: \n", model.summary())
        #fit with test data if provided
        if X_test is not None:
            history = model.fit(
                X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=2, validation_data = (X_test,y_test)
            )
        else:
            history = model.fit(
                X_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=2
            )
        # run the train predictions if predict true
        if predict:
            train_predictions = model.predict(X_train).flatten().tolist()
            print('train_predictions', train_predictions[:10])
            # run the test predictions if provided if predict true
            if X_test is not None:
                test_predictions = model.predict(X_test).flatten().tolist()
                print('test_predictions', test_predictions[:10])
            else:
                test_predictions = None
        model_path = settings.MODELS / model_name
        #save trained model
        model.save(model_path)

    except Exception as error:
        return {
            "success": False,
            "error": error
        }

    response = {"success": True,
                 "error": None, 
                 "model_path": json.dumps(str(model_path)),
                "train_predictions": train_predictions,
                "test_predictions": test_predictions}
    return response


def resolve_consumeLongShortTermMemory(obj, info):
    pass
