import pandas as pd
import numpy as np
import json
import requests
import sys

from scipy.stats import linregress
from joblib import dump, load
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM 
from keras.models import load_model
from keras.optimizers import Adam

from api import settings
#from api.utils import *

pd.options.display.max_columns = None

'''Function for making sequences (blocks) of test and train data'''
def building_data_sequences(data_X,data_Y, timesteps): #timesteps means how many days we consider for each block
    X=[]
    y_MPNxP = []
    for i in range(len(data_X)-timesteps+1):  #how it works: every timesteps (e.g. 10 days) a block is constituted and for each block data and true values are stored
        X.append(data_X[i:(i+timesteps),:])
        y_MPNxP.append(data_Y[i+timesteps-1])
    return np.array(X), np.array(y_MPNxP)

def custom_loss_function(attenuated_padding_value):

  def padding_loss_function(y_true, y_pred):

    y_pred = tf.multiply(y_pred, attenuated_padding_value) #this is the multiplication between the predictions and the attenuated_padding_value

    squared_difference = tf.square(y_true - y_pred)

    return tf.reduce_mean(squared_difference, axis=-1) #mse

  return padding_loss_function

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

def train_and_save_model(compiled_model,X_train,y_train,X_test,y_test,batch_size, n_epochs, test_mode=False,save_model=True):
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
    return compiled_model

def load_pretreained_model(model_path):
    model = load_model(model_path, compile=False)
    return model

def make_query(query,endpoint):
    headers = {
        'Accept-Encoding': 'gzip, deflate, br',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Connection': 'keep-alive',
        'DNT': '1'
    }

    response = requests.post(endpoint,json={"query":query},headers=headers)
    return response

def call_TPT(predictions,target_columns,pretreatment_attrs=None):
    predictions_df = pd.DataFrame(predictions,columns=target_columns)
    pretreatment_attrs = json.dumps(pretreatment_attrs)
    values = predictions_df.values.tolist()
    index = predictions_df.index.tolist()
    query = f"""
        query {{
            targetsPostTreatment(
                values: {values}, 
                index: {index},
                columns: {json.dumps(target_columns)},
                pretreatment_attrs: {pretreatment_attrs}) {{
                    success,
                    error,
                    postreated_values,
                    index,
                    columns
                }}
        }}
    """
    tpt_response = make_query(query=query,endpoint=settings.TPT_HOST).json()
    return tpt_response

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
    iteration,
    pretreatment_attrs
):
    try:
        X_train = pd.DataFrame(json.loads(X_train))
        y_train = pd.DataFrame(json.loads(y_train))
        X_test = pd.DataFrame(json.loads(X_test))
        y_test = pd.DataFrame(json.loads(y_test))
        target_columns = list(y_train.columns)
        X_train, y_train = building_data_sequences(X_train.to_numpy(dtype='float64'), y_train.to_numpy(dtype='float64'),timesteps)
        X_test,y_test = building_data_sequences(X_test.to_numpy(dtype='float64'),y_test.to_numpy(dtype='float64'),timesteps)
        print("-" * 25, f"SEQUENTIAL DATA SHAPES", 25 * "-", "\n")
        print('X_train shape: ', X_train.shape)
        print('y_train shape:', y_train.shape)
        print('X_test shape: ', X_train.shape)
        print('y_test shape:', y_train.shape)
        input_shape = ((X_train).shape[1], (X_train).shape[2])
        use_pretrained_model = True
        if use_pretrained_model:
            print("-" * 25, f"LOADING THE MODEL", 25 * "-", "\n")
            trained_model = load_model(settings.MODELS / 'lstm_test_model.h5', compile=False)
            # main_target_predictions_train = np.concatenate([a[:1] for a in train_predictions])
            # main_target_predictions_train = main_target_predictions_train.tolist()
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
            trained_model = train_and_save_model(
                compiled_model=ltm_model,
                X_train = X_train,
                y_train = y_train,
                X_test=X_test,
                y_test=y_test,
                test_mode = True,
                save_model=True,
                **algorithm_configurations['fit_parameters']
            )

        print("-" * 25, f"RUNNING... PREDICTIONS:", 25 * "-", "\n")
        train_predictions = trained_model.predict(X_train)
        print("-" * 25, f"CALLING TPT ON PREDICTIONS", 25 * "-", "\n")
        tpt_response = call_TPT(train_predictions,target_columns,pretreatment_attrs)
        if not tpt_response['data']['targetsPostTreatment']['success']:
            error = tpt_response['data']['targetsPostTreatment']['error']
            response = {"success": False, "error": error}
            return response
        train_predictions = pd.DataFrame(
            tpt_response['data']['targetsPostTreatment']['postreated_values'],
            columns=tpt_response['data']['targetsPostTreatment']['columns'],
            index = tpt_response['data']['targetsPostTreatment']['index'],
        )
        main_target_predictions_train = train_predictions.iloc[:,:1] # 1 will be replaced by dynamic variable depeding on number of main targets
        main_target_columns = main_target_predictions_train.columns.tolist()
        main_target_predictions_train = main_target_predictions_train.values.tolist()
        print('main_target_predictions_train', main_target_predictions_train[:15])

    except Exception as error:
        response = {"success": False, "error": error}
        return response

    response = {
        "success": True,
        "error": None,
        "main_target_predictions_train":main_target_predictions_train,
        "main_target_columns": main_target_columns
    }

    return response