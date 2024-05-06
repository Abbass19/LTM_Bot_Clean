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

'''Function for making sequences (blocks) of test and train data'''
def building_data_sequences(data_X,data_Y, timesteps): #timesteps means how many days we consider for each block

    X=[]
    y_MPNxP = []
    for i in range(len(data_X)-timesteps+1):  #how it works: every timesteps (e.g. 10 days) a block is constituted and for each block data and true values are stored
        X.append(data_X[i:(i+timesteps),:])
        y_MPNxP.append(data_Y[i+timesteps-1])
    return np.array(X), np.array(y_MPNxP)

def compile_model(
        twoexp_nodes_number_layer_1,
        twoexp_nodes_number_layer_2,
        twoexp_nodes_number_layer_3,
        twoexp_nodes_number_layer_4,
        twoexp_nodes_number_layer_5,
        attenuated_padding_value,
        optimizer,
        model_case_version_main_target_code,
        input_shape,
        iteration=1,
        *args,
        **kwargs
):
    tf.keras.backend.clear_session()
    model=tf.keras.Sequential()
    '''The layers of the model (see case_version_cat Tab)'''
    model.add(LSTM(2**twoexp_nodes_number_layer_1,input_shape=input_shape,return_sequences=True, name=f'prediction_lstm_0_for_iteration_{iteration}'))
    model.add(LSTM(2**twoexp_nodes_number_layer_2, return_sequences=True,name = f'prediction_lstm_1_for_iteration_{iteration}'))
    model.add(LSTM(2**twoexp_nodes_number_layer_3,name = f'prediction_lstm_2_for_iteration_{iteration}'))
    model.add(Dense(2**twoexp_nodes_number_layer_4, name = f'prediction_dense_0_for_iteration_{iteration}'))
    model.add(Dense(int(model_case_version_main_target_code)+1, name = f'prediction_dense_1_for_iteration_{iteration}'))

    model.compile(optimizer = optimizer, loss = custom_loss_function(attenuated_padding_value))

    return model

def train_predict_save_model(compiled_model,X_train,y_train,X_test,y_test,predict,batch_size, n_epochs, test_mode=False,save_model=True):
    if not test_mode:
        compiled_model.fit(
            X_train,
            y_train,
            batch_size = batch_size,
            epochs = n_epochs,
            verbose=2,
            validation_data=(X_test,y_test)
        )
    if save_model:
        compiled_model.save(settings.MODELS / 'lstm_test_model.h5')
    train_predictions = compiled_model.predict(X_train)
    return train_predictions

def load_model_and_predict(model_path, X_train, X_test):
    model = load_model(model_path, compile=False)
    train_predictions = model.predict(X_train)
    return train_predictions

def resolve_trainLTM(
    obj,
    info,
    X_train,
    y_train,
    X_test,
    y_test,
    predict,
    timesteps,
    model_case_version_main_target_code,
    algorithm_configurations,
    iteration
):
    try:
        # print('predict', predict)
        # print('model_case_version_main_target_code', model_case_version_main_target_code)
        # print('algorithm_configurations',algorithm_configurations)
        X_train = pd.DataFrame(json.loads(X_train)).to_numpy(dtype='float64')
        y_train = pd.DataFrame(json.loads(y_train)).to_numpy(dtype='float64')
        X_test = pd.DataFrame(json.loads(X_test)).to_numpy(dtype='float64')
        y_test = pd.DataFrame(json.loads(y_test)).to_numpy(dtype='float64')
        # print('X_train shape: ', X_train.shape)
        # print('y_train shape:', y_train.shape)
        # print('X_test shape: ', X_train.shape)
        # print('y_test shape:', y_train.shape)
        # print('\n')
        X_train, y_train = building_data_sequences(X_train, y_train,timesteps)
        X_test,y_test = building_data_sequences(X_test,y_test,timesteps)
        print("-" * 25, f"SEQUENTIAL DATA SHAPES", 25 * "-", "\n")
        print('X_train shape: ', X_train.shape)
        print('y_train shape:', y_train.shape)
        print('X_test shape: ', X_train.shape)
        print('y_test shape:', y_train.shape)
        input_shape = ((X_train).shape[1], (X_train).shape[2])
        use_pretrained_model = True
        if use_pretrained_model:
            train_predictions = load_model_and_predict(settings.MODELS / 'lstm_test_model.h5', X_train,X_test)
            main_target_predictions_train = np.concatenate([a[:1] for a in train_predictions])
            main_target_predictions_train = main_target_predictions_train.tolist()
        else:
            print("-" * 25, f"COMPILE LTM WITH ITS PARAMETERS", 25 * "-", "\n")
            algorithm_configurations = json.loads(algorithm_configurations)
            print(algorithm_configurations)
            ### Compile LSTM Model
            ltm_model = compile_model(
                model_case_version_main_target_code=model_case_version_main_target_code,
                input_shape=input_shape,
                **algorithm_configurations["compile_parameters"],
            )
            if iteration == 0:
                print(ltm_model.summary())
            print("-" * 25, f"TRAINING... ITERATION:{iteration}", 25 * "-", "\n")
            train_predictions = train_predict_save_model(
                compiled_model=ltm_model,
                X_train = X_train,
                y_train = y_train,
                X_test=X_test,
                y_test=y_test,
                predict = predict,
                test_mode = True,
                save_model=True,
                **algorithm_configurations['fit_parameters']
            )
            main_target_predictions_train = np.concatenate([a[:1] for a in train_predictions])
            main_target_predictions_train = main_target_predictions_train.tolist()
        #print('train_predictions: ',train_predictions)

    except Exception as error:
        response = {"success": False, "error": error}
        return response

    response = {
        "success": True,
        "error": None,
        "main_target_predictions_train":main_target_predictions_train
    }

    return response


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
    algorithm_configurations,
    test=None,
):
    try:
        ltm_configurations = json.loads(algorithm_configurations)['LTM']

        # convert json data to dataframe
        train = pd.DataFrame(json.loads(train))
        if test is not None:
            test = pd.DataFrame(json.loads(test))
        else:
            test = pd.DataFrame()
            test_predictions = None
        # FTP implementation
        features_and_targets = features + targets
        train = train.reset_index()[features_and_targets]
        print(train.head())
        if not test.empty:
            test = test.reset_index()[features_and_targets]
            print(test.head())
        # call the FTP bot
        print("\n")
        print("-" * 25, f"FTP STARTED TO RUN FOR TRAIN SET", 25 * "-", "\n")
        values = json.dumps(train.values.tolist())
        index = json.dumps(train.index.to_list())
        columns = json.dumps(train.columns.to_list())

        ftp_response_train = execute_FTP(
            values=values, index=index, columns=columns
        ).json()
        if not ftp_response_train["data"]["featuresTargetsPretreatment"]["success"]:
            response = {
                "success": False,
                "error": ftp_response_train["data"]["featuresTargetsPretreatment"][
                    "error"
                ],
            }
            return response
        pretreatment_object = json.loads(
            ftp_response_train["data"]["featuresTargetsPretreatment"][
                "pretreatment_info"
            ]
        )
        # exract scaler attributes for each target
        targets_pretreatment_attrs = {
            key: pretreatment_object[key]
            for key in targets
            if key in pretreatment_object
        }
        train = pd.DataFrame(
            data=ftp_response_train["data"]["featuresTargetsPretreatment"][
                "pretreated_values"
            ],
            columns=ftp_response_train["data"]["featuresTargetsPretreatment"][
                "columns"
            ],
            index=ftp_response_train["data"]["featuresTargetsPretreatment"]["index"],
        )
        print(train.head())
        # Create X_train, y_train
        X_train = train[features].to_numpy("float64")
        y_train = train[targets].to_numpy("float64")
        # create sequental data
        X_train, y_train = building_data_sequences(X_train, y_train, timesteps)
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train[0].shape)

        # if condition must be added here
        if not test.empty:
            print("-" * 25, f"FTP STARTED TO RUN FOR TEST SET", 25 * "-", "\n")
            values = json.dumps(test[features_and_targets].values.tolist())
            index = json.dumps(test.index.to_list())
            columns = json.dumps(features_and_targets)
            ftp_response_test = execute_FTP(
                values, index, columns, json.dumps(pretreatment_object)
            ).json()
            if not ftp_response_test["data"]["featuresTargetsPretreatment"]["success"]:
                response = {
                    "success": False,
                    "error": ftp_response_test["data"]["featuresTargetsPretreatment"][
                        "error"
                    ],
                }
                return response

            test = pd.DataFrame(
                data=ftp_response_test["data"]["featuresTargetsPretreatment"][
                    "pretreated_values"
                ],
                columns=ftp_response_test["data"]["featuresTargetsPretreatment"]["columns"],
                index=ftp_response_test["data"]["featuresTargetsPretreatment"]["index"],
            )
            print(test.head())
            X_test = test[features].to_numpy("float64")
            y_test = test[targets].to_numpy("float64")
            X_test, y_test = building_data_sequences(X_test, y_test, timesteps)
            print("X_test shape: ", X_test.shape)
            print("y_test shape: ", y_test[0].shape)
        
        # define the input parameters
        input_shape = ((X_train).shape[1], (X_train).shape[2])
        optimizer = Adam(learning_rate=ltm_configurations['learning_rate'])
        # return the model
        model = compile_model(
            input_shape,
            iteration,
            model_case_version_main_target_code,
            optimizer,
            ltm_configurations['attenuated_padding_value']
        )
        print("Input shape: \n", X_train.shape)
        print("\n")
        print("Model summary: \n", model.summary())
        # fit with test data if provided
        if not test.empty:
            history = model.fit(
                X_train,
                y_train,
                batch_size=ltm_configurations['batch_size'],
                epochs=ltm_configurations['n_epochs'],
                verbose=2,
                validation_data=(X_test, y_test),
            )
        else:
            history = model.fit(
                X_train, y_train, batch_size=ltm_configurations['batch_size'], epochs=ltm_configurations['n_epochs'], verbose=2
            )
        model_path = settings.MODELS / model_name
        # save trained model
        model.save(model_path)
        # run the train predictions if predict true
        if predict:
            train_predictions = model.predict(X_train)
            print("train_predictions", train_predictions.shape)
            # run the test predictions if provided if predict true
            if not test.empty:
                test_predictions = model.predict(X_test)
                print("test_predictions", test_predictions.shape)
            else:
                test_predictions = None
        # CALL TPT BOT FOR TRAIN PREDICTIONS
        tpt_response_train = execute_TPT(
            values=train_predictions.tolist(),
            index=np.arange(0, train_predictions.shape[0]).tolist(),
            columns=json.dumps(targets),
            pretreatment_attrs=json.dumps(targets_pretreatment_attrs),
        )
        if not tpt_response_train["data"]["targetsPostTreatment"]["success"]:
            response = {
                "success": False,
                "error": tpt_response_train["data"]["targetsPostTreatment"]["error"],
            }
            return response

        train_predictions = np.array(tpt_response_train["data"]["targetsPostTreatment"][
            "postreated_values"
        ])
        print('post treated train predicitons shape: ', train_predictions.shape)
        train_predictions = [a[:1] for a in train_predictions]
        train_predictions = np.concatenate(train_predictions).tolist()
        print('first column train predictions: ', train_predictions[:10])
        # CALL TPT BOT FOR TEST PREDICTIONS
        if not test.empty:
            tpt_response_test = execute_TPT(
                values=test_predictions.tolist(),
                index=np.arange(0, test_predictions.shape[0]).tolist(),
                columns=json.dumps(targets),
                pretreatment_attrs=json.dumps(targets_pretreatment_attrs),
            )
            if not tpt_response_test["data"]["targetsPostTreatment"]["success"]:
                response = {
                    "success": False,
                    "error": tpt_response_test["data"]["targetsPostTreatment"]["error"],
                }
                return response
            test_predictions = np.array(tpt_response_test["data"]["targetsPostTreatment"][
                "postreated_values"
            ])
            print('post treated test predicitons shape: ', test_predictions.shape)
            # continue with returning post treated first
            test_predictions = [a[:1] for a in test_predictions]
            test_predictions = np.concatenate(test_predictions)
            print('first column test predictions: ', test_predictions[:10])

    except Exception as error:
        return {"success": False, "error": error}

    response = {
        "success": True,
        "error": None,
        "model_path": json.dumps(str(model_path)),
        "train_predictions": train_predictions,
        "test_predictions": test_predictions
    }
    return response


def resolve_consumeLongShortTermMemory(obj, info):
    pass
