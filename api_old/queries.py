import numpy as np
from keras.models import load_model

from api_old import settings
from api_old.model import building_data_sequences, building_test_data_sequences
from api_old.model import fitLTMBot

def resolve_fitLTMBot(obj, info, X_train, y_train, timesteps, apply_hyperparameter_tuning, model_case_version_main_target_code, iteration, model_name):
    try:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train_seq, y_train_seq = building_data_sequences(X_train, y_train, timesteps=timesteps)
        print('\n')
        print('X_train: ',X_train_seq[:10], '\n')
        print('y_train: ',y_train_seq[:10], '\n')

        model = fitLTMBot(X_train_seq, y_train_seq, apply_hyperparameter_tuning, model_case_version_main_target_code, iteration)
        
        model.save(str(settings.MODELS / model_name) + '.h5')
        model.save_weights(str(settings.MODELS / model_name) + '_weights.h5')
        print('Model saved to: \n')
        print(str(settings.MODELS / model_name) + '.h5')
    except Exception as error:
        print('Error: ', error)
        response = {
            'success':False,
            'error':error,
        }
        return response
    
    response = {
        'success':True,
        'error':None,
        'model_path': str(settings.MODELS / model_name) + '.h5'
    }
    return response

def resolve_predictLTMBot(obj,info, X_test, timesteps, model_path):
    X_test = np.array(X_test)
    X_test_seq = building_test_data_sequences(X_test, timesteps)
    try:
        model = load_model(model_path)
        predictions = model.predict(X_test_seq)
        print('Predictions: \n')
        print(predictions)
    except Exception as error:
        print('Error: ', error)
        response = {
            'success':False,
            'error':error,
        }
        return response

    response = {
        'success':True,
        'error':None,
        'predictions': predictions
    }
    return response