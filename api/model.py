import psycopg2.extras
from psycopg2.extensions import AsIs

import pandas as pd 
import numpy as np 
import optuna
from joblib import dump, load

from scipy.stats import linregress 

import tensorflow as tf
from keras.backend import clear_session
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM 
from tensorflow.keras.optimizers import Adam

#generate data sequence with timesteps
def building_data_sequences(data_X, data_Y, timesteps):
    X, y_MPNxP = [], [] 
    
    for i in range(len(data_X) - timesteps + 1):
        sequence = data_X[i:(i + timesteps), :]
        target = data_Y[i + timesteps - 1]
        
        X.append(sequence)
        y_MPNxP.append(target)
    
    return np.array(X), [np.array(y_MPNxP)]

#generate data sequence for test data
def building_test_data_sequences(data_X, timesteps):
    X = [] 
    
    for i in range(len(data_X) - timesteps + 1):
        sequence = data_X[i:(i + timesteps), :]
        X.append(sequence)
    
    return np.array(X)

#evaluation metric
def sir_parameters(x,y):#(actual_value, predicted value)
    analytical_params = linregress(x, y)
    slope = analytical_params.slope
    intercept = analytical_params.intercept
    rvalue = analytical_params.rvalue
    x_trend_line = slope*x + intercept
    avg_trend_line_distance = np.mean(np.abs(x_trend_line-y))

    return slope, intercept, rvalue**2, avg_trend_line_distance
'''
#custom loss function
def custom_loss_function(attenuated_padding_value):

  def padding_loss_function(y_true, y_pred):
    y_pred = tf.multiply(y_pred, attenuated_padding_value) #this is the multiplication between the predictions and the attenuated_padding_value
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1) #mse

  return padding_loss_function
'''
def fitLTMBot(X_train, y_train, apply_hyperparameter_tuning, model_case_version_main_target_code, iteration):

    n_epochs = 100
    batch = 64

    '''These are the exponent used to define the number of nodes for each layer'''
    twoexp_nodes_number_layer_1 = 7 #7
    twoexp_nodes_number_layer_2 = 7 #10
    twoexp_nodes_number_layer_3 = 8 #7
    twoexp_nodes_number_layer_4 = 9 #6
    twoexp_nodes_number_layer_5 = 0 #0

    lr=0.0005 #learning rate
    optimizer = Adam(learning_rate = lr)
    attenuated_padding_value = 1

    input_shape = ((X_train).shape[1], (X_train).shape[2])
    
    try:
        if(apply_hyperparameter_tuning==True):

            model = 'Not Available'
        
        else:

            model= tf.keras.Sequential()
            model.add(LSTM(2**twoexp_nodes_number_layer_1,input_shape=input_shape,return_sequences=True, name=f'prediction_lstm_0_for_iteration_{iteration}'))
            model.add(LSTM(2**twoexp_nodes_number_layer_2, return_sequences=True,name = f'prediction_lstm_1_for_iteration_{iteration}'))
            model.add(LSTM(2**twoexp_nodes_number_layer_3,name = f'prediction_lstm_2_for_iteration_{iteration}'))
            model.add(Dense(2**twoexp_nodes_number_layer_4, name = f'prediction_dense_0_for_iteration_{iteration}'))
            model.add(Dense(int(model_case_version_main_target_code)+1, name = f'prediction_dense_1_for_iteration_{iteration}'))

            clear_session()
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch, verbose=1)

    except Exception as errors:
        print(errors)
    
    return model