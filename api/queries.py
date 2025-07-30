from tensorflow.keras.layers import Dense, LSTM
from keras.models import load_model
from api import settings
import tensorflow as tf
import pandas as pd
import numpy as np
import requests
import json



pd.options.display.max_columns = None

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
        layer_1, layer_2, layer_3, layer_4, layer_5, attenuated_padding_value,
        optimizer, model_case_version_main_target_code, input_shape, num_targets, iteration=1, *args, **kwargs):
    tf.keras.backend.clear_session()

    model=tf.keras.Sequential()
    model.add(LSTM(2 ** layer_1, input_shape=input_shape, return_sequences=True, name=f'prediction_lstm_0_for_iteration_{iteration}'))
    model.add(LSTM(2 ** layer_2, return_sequences=True, name =f'prediction_lstm_1_for_iteration_{iteration}'))
    model.add(LSTM(2 ** layer_3, name =f'prediction_lstm_2_for_iteration_{iteration}'))
    model.add(Dense(2 ** layer_4, name =f'prediction_dense_0_for_iteration_{iteration}'))
    model.add(Dense(num_targets, name = f'prediction_dense_1_for_iteration_{iteration}'))
    model.compile(optimizer = optimizer, loss = custom_loss_function(attenuated_padding_value))

    return model

def train_and_save_model(compiled_model, X_train, y_train, X_test, y_test,
                         batch_size, n_epochs, test_mode=False,
                         save_model=True, model_save_path=None):

    if not test_mode:
        compiled_model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=1,
            validation_data=(X_test, y_test)
        )

    if save_model and model_save_path:
        compiled_model.save(model_save_path)

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

def call_TPT(predictions, target_columns, pretreatment_attrs=None):
    predictions_df = pd.DataFrame(predictions, columns=target_columns)
    pretreatment_attrs = pretreatment_attrs or {}

    values = predictions_df.values.tolist()
    index = predictions_df.index.tolist()
    columns = target_columns

    query = """
        query TargetsPostTreatment(
            $values: JSON,
            $index: JSON,
            $columns: JSON,
            $pretreatment_attrs: JSON
        ) {
            targetsPostTreatment(
                values: $values,
                index: $index,
                columns: $columns,
                pretreatment_attrs: $pretreatment_attrs
            ) {
                success
                error
                postreated_values
                index
                columns
            }
        }
    """

    variables = {
        "values": values,
        "index": index,
        "columns": columns,
        "pretreatment_attrs": pretreatment_attrs,
    }

    tpt_response = make_query(query=query, variables=variables, endpoint=settings.TPT_HOST).json()
    return tpt_response

def resolve_trainLTM(obj, info, X_train, y_train, X_test, y_test, predict, timesteps,
                     model_case_version_main_target_code, algorithm_configurations,
                     iteration, pretreatment_attrs):

    try:
        # Parse inputs
        X_train = pd.DataFrame(X_train)g
        y_train = pd.DataFrame(y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.DataFrame(y_test)
        target_columns = list(y_train.columns)

        # Reshape for LSTM input
        X_train, y_train = building_data_sequences(X_train.to_numpy(dtype='float64'),
                                                   y_train.to_numpy(dtype='float64'), timesteps)
        X_test, y_test = building_data_sequences(X_test.to_numpy(dtype='float64'),
                                                 y_test.to_numpy(dtype='float64'), timesteps)

        input_shape = (X_train.shape[1], X_train.shape[2])
        num_targets = y_train.shape[-1]
        use_pretrained_model = False

        model_name = f"lstm_model_iter_{iteration}.h5"
        model_path = settings.MODELS / model_name

        if use_pretrained_model:
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            trained_model = load_model(model_path, compile=False)
        else:
            print("-" * 25, f"COMPILE LTM WITH ITS PARAMETERS", "-" * 25, "\n")
            algorithm_configurations = json.loads(algorithm_configurations)
            print(algorithm_configurations)

            ltm_model = compile_model(
                model_case_version_main_target_code=model_case_version_main_target_code,
                input_shape=input_shape,
                num_targets=num_targets,
                iteration=iteration,
                **algorithm_configurations["compile_parameters"],
            )

            if iteration == 0:
                print(ltm_model.summary())

            trained_model = train_and_save_model(
                compiled_model=ltm_model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                test_mode=False,
                save_model=True,
                model_save_path=model_path,
                **algorithm_configurations["fit_parameters"]
            )

        predictions = trained_model.predict(X_train)

        tpt_response = call_TPT(predictions, target_columns, pretreatment_attrs)

        if not tpt_response['data']['targetsPostTreatment']['success']:
            error = tpt_response['data']['targetsPostTreatment']['error']
            return {"success": False, "error": error}

        predictions_df = pd.DataFrame(
            tpt_response['data']['targetsPostTreatment']['postreated_values'],
            columns=tpt_response['data']['targetsPostTreatment']['columns'],
            index=tpt_response['data']['targetsPostTreatment']['index']
        )

        main_target_predictions_train = predictions_df.iloc[:, :1]
        main_target_columns = main_target_predictions_train.columns.tolist()
        main_target_predictions_train = main_target_predictions_train.values.tolist()

    except Exception as error:
        return {"success": False, "error": str(error)}

    return {
        "success": True,
        "error": None,
        "main_target_predictions_train": main_target_predictions_train,
        "main_target_columns": main_target_columns
    }