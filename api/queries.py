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


def resolve_fitLongShortTermMemory(
    obj,
    info,
    X_train,
    y_train,
    predict,
    timesteps,
    apply_hyperparameter_tuning,
    model_case_version_main_target_code,
    iteration,
    model_name,
    learning_rate=0.0005,
    batch_size=64,
    n_epochs=100,
    attenuated_padding_value=1,
    X_test=None,
    y_test=None
):
    try:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
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
