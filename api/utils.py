import numpy as np
import json
import requests

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM 

from .settings import FTP_HOST,TPT_HOST

def building_data_sequences(data_X, data_Y, timesteps):
    #generate data sequence with timesteps
    X, y_MPNxP = [], [] 
    for i in range(len(data_X) - timesteps + 1):
        sequence = data_X[i:(i + timesteps), :]
        target = data_Y[i + timesteps - 1]
        X.append(sequence)
        y_MPNxP.append(target)
    return np.array(X), [np.array(y_MPNxP)]


def custom_loss_function(attenuated_padding_value):

  def padding_loss_function(y_true, y_pred):

    y_pred = tf.multiply(y_pred, attenuated_padding_value) #this is the multiplication between the predictions and the attenuated_padding_value

    squared_difference = tf.square(y_true - y_pred)

    return tf.reduce_mean(squared_difference, axis=-1) #mse

  return padding_loss_function


def compile_model(input_shape, iteration, model_case_version_main_target_code, optimizer, attenuated_padding_value):
    twoexp_nodes_number_layer_1 = 7
    twoexp_nodes_number_layer_2 = 10
    twoexp_nodes_number_layer_3 = 7
    twoexp_nodes_number_layer_4 = 6
    twoexp_nodes_number_layer_5 = 0

    tf.keras.backend.clear_session()
    model= tf.keras.Sequential()
    '''The layers of the model (see case_version_cat Tab)'''
    model.add(LSTM(2**twoexp_nodes_number_layer_1,input_shape=input_shape,return_sequences=True, name=f'prediction_lstm_0_for_iteration_{iteration}'))
    model.add(LSTM(2**twoexp_nodes_number_layer_2, return_sequences=True,name = f'prediction_lstm_1_for_iteration_{iteration}'))
    model.add(LSTM(2**twoexp_nodes_number_layer_3,name = f'prediction_lstm_2_for_iteration_{iteration}'))
    model.add(Dense(2**twoexp_nodes_number_layer_4, name = f'prediction_dense_0_for_iteration_{iteration}'))
    model.add(Dense(int(model_case_version_main_target_code)+1, name = f'prediction_dense_1_for_iteration_{iteration}'))

    model.compile(optimizer = optimizer, loss = custom_loss_function(attenuated_padding_value))

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

def execute_FTP(values,index,columns,pretreatment_attrs=None):
    pretreatment_attrs = json.dumps(pretreatment_attrs)
    query = f"""
        query {{
            featuresTargetsPretreatment(
                values: {values}, 
                index: {index},
                columns: {columns},
                pretreatment_attrs: {pretreatment_attrs}) {{
                    success,
                    error,
                    pretreated_values
                    index,
                    columns,
                    pretreatment_info
                }}
        }}
    """

    ftp_response = make_query(query=query,endpoint=FTP_HOST)
    return ftp_response

def execute_TPT(values, index, columns, pretreatment_attrs=None):
    pretreatment_attrs = json.dumps(pretreatment_attrs)
    query = f"""
        query {{
            targetsPostTreatment(
                values: {values}, 
                index: {index},
                columns: {columns},
                pretreatment_attrs: {pretreatment_attrs}) {{
                    success,
                    error,
                    postreated_values,
                    index,
                    columns
                }}
        }}
    """

    tpt_response = make_query(query=query,endpoint=TPT_HOST).json()
    return tpt_response