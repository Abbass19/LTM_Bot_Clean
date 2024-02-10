import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

import psycopg2
import psycopg2.extras
from psycopg2.extensions import AsIs

from sklearn.preprocessing import RobustScaler
from scipy.stats import linregress #for the slope and the value of Y at X=0 of the linear trend line
import tsmoothie

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM,Conv1D,MaxPooling1D,Flatten,TimeDistributed 


def building_data_sequences(data_X,data_Y, timesteps): #timesteps means how many days we consider for each block
    X=[]
    y_MPNxP = []
    for i in range(len(data_X)-timesteps+1):  #how it works: every timesteps (e.g. 10 days) a block is constituted and for each block data and true values are stored
        X.append(data_X[i:(i+timesteps),:])
        y_MPNxP.append(data_Y[i+timesteps-1])
    return np.array(X), [np.array(y_MPNxP)]

def sir_parameters(x,y): #sir stands for slope, intercept, rvalue (actually there's also the average trend line distance or avg_tld, but it came later)
  analytical_params = linregress(x, y)
  slope = analytical_params.slope
  intercept = analytical_params.intercept
  rvalue = analytical_params.rvalue #pay attention that here we have the correlaton coefficient (so not r2 that is the coefficient of determination)
  y_trend_line = slope*x + intercept #this is computed just for the avg_tld
  avg_trend_line_distance = np.mean(np.abs(y_trend_line - y)/y_trend_line)

  return slope, intercept, rvalue**2, avg_trend_line_distance

def custom_loss_function(attenuated_padding_value):

  def padding_loss_function(y_true, y_pred):

    y_pred = tf.multiply(y_pred, attenuated_padding_value) #this is the multiplication between the predictions and the attenuated_padding_value

    squared_difference = tf.square(y_true - y_pred)

    return tf.reduce_mean(squared_difference, axis=-1) #mse

  return padding_loss_function

def plot_model_history(history: pd.DataFrame, limit_x = [None,None], limit_y = [None,None]):
   ax = history['loss'].plot(label='Train Loss')
   history['val_loss'].plot(ax=ax, label='Validation Loss')

   ax.set_xlim(limit_x)
   ax.set_ylim(limit_y)
   plt.legend()
   plt.show()

def preprare_correction_lstm_table_train(period_day_number, raw_predicted_targets,volume,actual,model_case_version_main_target_code,model_case_version_time_steps):
  correction_lstm_table_temp=pd.DataFrame()
  correction_lstm_table_temp['period_day_number']=period_day_number[int(model_case_version_main_target_code):]
  correction_lstm_table_temp['raw_predicted_targets']=raw_predicted_targets[int(model_case_version_main_target_code):]
  new = (actual/raw_predicted_targets)
  # print(len(new[:-int(model_case_version_main_target_code)]))
  # print(new)
  # sys.exit()
  correction_lstm_table_temp['new_feature'] = new[:-int(model_case_version_main_target_code)]
  # print(correction_lstm_table['new_feature'])
  correction_lstm_table_temp['volume'] = volume[int(model_case_version_main_target_code):]
  correction_lstm_table_temp['actual_raw_predited_ratio']=(actual/raw_predicted_targets)[int(model_case_version_main_target_code):]
  # pd.set_option('display.max_rows', None)
  print(correction_lstm_table_temp)
  pd.reset_option('display.max_rows')
  correction_lstm_table=correction_lstm_table_temp.to_numpy()
  correction_lstm_table = correction_lstm_table[int(model_case_version_main_target_code):]
  robust_scaler_LSTM_features= RobustScaler().fit(correction_lstm_table[:,:4])
  robust_scaler_LSTM_target=RobustScaler().fit(correction_lstm_table[:,4].reshape(-1,1))
  train_dataframe_lstm_features=robust_scaler_LSTM_features.transform(correction_lstm_table[:,:4])
  train_dataframe_lstm_target=robust_scaler_LSTM_target.transform(correction_lstm_table[:,4].reshape(-1,1))
  final_correction_lstm=np.concatenate((train_dataframe_lstm_features,train_dataframe_lstm_target),axis=1)
  return final_correction_lstm,robust_scaler_LSTM_features,robust_scaler_LSTM_target,correction_lstm_table_temp['actual_raw_predited_ratio'][int(model_case_version_time_steps)-1:].to_numpy()

def correction_data_sequences(data, timesteps): #timesteps means how many days we consider for each block
    X=[]
    y_MPNxP = []
    for i in range(len(data)-timesteps+1):  #how it works: every timesteps (e.g. 10 days) a block is constituted and for each block data and true values are stored
        X.append(data[i:(i+timesteps),:4])
        y_MPNxP.append(data[i+timesteps-1,4])

    return np.array(X),np.array(y_MPNxP)

def extract_dohlcav_mpnxp_data(
                    case,
                    dataset_start_date,
                    hostname='database-1.ctzm0hf7fhri.eu-central-1.rds.amazonaws.com',
                    database='dyDATA_new',
                    username='postgres',
                    pwd='Proc2023awsrdspostgresql',
                    port_id='5432',
                    conn=None):
    
    asset_script="SELECT * FROM "+'\"'+"ASSET_"+case+'\"'+".features_targets_input_view WHERE features_targets_input_view."+'\"'+"cleaned_raw_features_environment_PK"+'\"'+ "= 4"
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

    if dataset_start_date:
        dataframe = dataframe.loc[dataframe['cleaned_raw_features_DCP_date_current_period'] >= str(dataset_start_date.date())].reset_index(drop=True)

    return dataframe

def filter_and_rename_columns(dataframe,targets,model_case_version_main_target_code):
    # FILTERING
    filtered_columns_1=list(dataframe.columns[:9])#to filter out the dates columns and features columns
    filtered_columns_2=[x for x in dataframe.columns if  targets[0][3:] in x ]#feature out the main target columns
    
    if model_case_version_main_target_code=='1':
        temp=filtered_columns_2[0]
        temp_2=filtered_columns_2[1]
        filtered_columns_2[0]=filtered_columns_2[2]
        filtered_columns_2[1]=temp
        filtered_columns_2[2]=temp_2

    #to add the last two constant columns to the table
    filtered_columns_3=['calculated_targets_HPN1P','calculated_targets_LPN1P']
    filtered_columns=filtered_columns_1+filtered_columns_2+filtered_columns_3
    print(filtered_columns)
    dataframe=dataframe[filtered_columns]
    #RENAMING COLS
    if model_case_version_main_target_code =='1':
        dataframe.columns = [
            "ID","DCP_date_current_period","DNCP_day_number_current_period","OPCP_open_price_current_period",
            "HPCP_high_price_current_period","LPCP_low_price_current_period",
            "CPCP_close_price_current_period","ACPCP_adjusted_close_price_current_period","VTCP_volume_of_transactions_current_period",
            "MPN"+model_case_version_main_target_code+"P_median_price_next_"+model_case_version_main_target_code+"_periods","HPN"+model_case_version_main_target_code+"P_highest_price_next_"+model_case_version_main_target_code+"_periods","LPN"+model_case_version_main_target_code+"P_lowest_price_next_"+model_case_version_main_target_code+"_periods","HPN1P_high_price_next_1_period",
            "LPN1P_low_price_next_1_period"
        ]

    else:
        dataframe = dataframe.rename(columns={
            "cleaned_raw_features_id":"ID",
            "cleaned_raw_features_DCP_date_current_period": "DCP_date_current_period",
            "calculated_features_DNCP":"DNCP_day_number_current_period",
            "cleaned_raw_features_OPCP_open_price_current_period":"OPCP_open_price_current_period",
            "cleaned_raw_features_HPCP_high_price_current_period":"HPCP_high_price_current_period",
            "cleaned_raw_features_LPCP_low_price_current_period":"LPCP_low_price_current_period",
            "cleaned_raw_features_CPCP_close_price_current_period": "CPCP_close_price_current_period",
            "cleaned_raw_features_ACPCP_adjusted_close_price_current_period":"ACPCP_adjusted_close_price_current_period",
            "cleaned_raw_features_VTCP_volume_of_transactions_current_period":"VTCP_volume_of_transactions_current_period",
            filtered_columns_2[0]:"MPN"+model_case_version_main_target_code+"P_median_price_next_"+model_case_version_main_target_code+"_periods",
            filtered_columns_2[1]:"HPN"+model_case_version_main_target_code+"P_highest_price_next_"+model_case_version_main_target_code+"_periods",
            filtered_columns_2[2]:"LPN"+model_case_version_main_target_code+"P_lowest_price_next_"+model_case_version_main_target_code+"_periods",
            filtered_columns_3[0]:"HPN1P_high_price_next_1_period",
            filtered_columns_3[1]:"LPN1P_low_price_next_1_period",
            })
        
        dataframe['DCP_date_current_period'] = pd.to_datetime(dataframe['DCP_date_current_period']) 
                        
    return dataframe

def get_main_dataframe(dataframe,model_case_version_main_target_code):
    #pay attention here because everytime targets change, also the name of the columns change
    dataframe = dataframe.drop(["ID"], axis=1)

    #pay attention here because everytime targets change, also the name of the columns change
    dataframe = dataframe.rename(columns={"DCP_date_current_period": "DATE",
                            "DNCP_day_number_current_period": "DNCP",
                            "OPCP_open_price_current_period":"OPCP",
                            "HPCP_high_price_current_period":"HPCP",
                            "LPCP_low_price_current_period":"LPCP",
                            "CPCP_close_price_current_period":"CPCP",
                            "ACPCP_adjusted_close_price_current_period": "ACPCP",
                            "VTCP_volume_of_transactions_current_period":"VTCP",
                            "MPN"+model_case_version_main_target_code+"P_median_price_next_"+model_case_version_main_target_code+"_periods":"MPN"+model_case_version_main_target_code+"P",
                            "HPN"+model_case_version_main_target_code+"P_highest_price_next_"+model_case_version_main_target_code+"_periods":"HPN"+model_case_version_main_target_code+"P",
                            "LPN"+model_case_version_main_target_code+"P_lowest_price_next_"+model_case_version_main_target_code+"_periods":"LPN"+model_case_version_main_target_code+"P",
                            'HPN1P_high_price_next_1_period':'hpn1p',
                            'LPN1P_low_price_next_1_period':'lpn1p'})

    #dataframe = dataframe.set_index('DATE')
    #dataframe.index = pd.to_datetime(dataframe.index)
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])

    return dataframe

def new_target_column(dataframe,target_code , shift_back_period):
  prev_target = dataframe[target_code]
  new_target = prev_target[:-shift_back_period]
  first_dates_handling = [0] * shift_back_period
  new_target=np.concatenate((first_dates_handling,new_target))
  return new_target

def feature_engineering(dataframe,base_target_code,base_target_column_index,model_case_version_main_target_code,targets):
    new_target_index = base_target_column_index
    for i in range(int(model_case_version_main_target_code)):
        new_target_code = 'MPN-' + str(i+1) + 'P'
        dataframe.insert(new_target_index+1,new_target_code,new_target_column(dataframe,base_target_code,i+1))
        new_target_index = new_target_index + 1
    
    targets.append(new_target_code)

    # '''Ratio Transformation for features'''
    dataframe.insert(7,'OPCP_Ratio',dataframe['OPCP']/dataframe['CPCP'])
    dataframe.insert(8,'HPCP_Ratio',dataframe['HPCP']/dataframe['CPCP'])
    dataframe.insert(9,'LPCP_Ratio',dataframe['LPCP']/dataframe['CPCP'])
    dataframe.insert(10,'ACPCP_Ratio',dataframe['ACPCP']/dataframe['CPCP'])
    dataframe.insert(dataframe.columns.get_loc('MPN'+model_case_version_main_target_code+'P') ,'MPN'+ model_case_version_main_target_code +'P_Ratio',dataframe['MPN'+ model_case_version_main_target_code +'P'].shift(5)/dataframe['CPCP'].shift(5))
    dataframe.insert(dataframe.columns.get_loc('HPN'+model_case_version_main_target_code+'P'),'HPN'+ model_case_version_main_target_code +'P_Ratio',dataframe['HPN'+ model_case_version_main_target_code +'P']/dataframe['CPCP'])
    dataframe.insert(dataframe.columns.get_loc('LPN'+model_case_version_main_target_code+'P'),'LPN'+ model_case_version_main_target_code +'P_Ratio',dataframe['LPN'+ model_case_version_main_target_code +'P']/dataframe['CPCP'])
    dataframe.insert(dataframe.columns.get_loc('hpn1p')+1,'hpn1p_Ratio',dataframe['hpn1p']/dataframe['CPCP'])
    dataframe.insert(dataframe.columns.get_loc('lpn1p')+1,'lpn1p_Ratio',dataframe['lpn1p']/dataframe['CPCP'])

    dataframe = dataframe.iloc[int(model_case_version_main_target_code):]

    return dataframe

def mask_train_test_validation(dataframe,
                               dataset_start_date,
                                train_end_date,
                                test_end_date,
                                validation_end_date,
                                model_case_version_main_target_code):
    # DEFINE TRAIN MASK
    train_period = str(dataset_start_date.date()) + '_' + str(train_end_date.date())

    train_end_idx = dataframe.index[dataframe['DCP_date_current_period']==train_end_date].values[0]
    train_end_new_idx = train_end_idx-int(model_case_version_main_target_code)
    train_new_end_date=dataframe.loc[train_end_idx,'DCP_date_current_period']

    train_mask = (dataframe['DCP_date_current_period'] <= train_end_date)#to select data for training
    #prediction_mask = (dataframe['DCP_date_current_period'] <= validation_end_date)#to select data for prediction
    print('Train size:', sum(train_mask))
    #print('Prediction size:', sum(prediction_mask))

    # DEFINE TEST AND VALIDATION PERIOD IF TEST END DATE PROVIDED
    if test_end_date:
        test_start_date = (train_end_date + pd.Timedelta(days=1))
        test_mask = (dataframe['DCP_date_current_period'] <= test_end_date)
        print('Test size:', sum(test_mask))

        validation_start_date = (test_end_date + pd.Timedelta(days=1))
        validation_mask = (dataframe['DCP_date_current_period'] <= validation_end_date)
        print('Validation mask:', sum(validation_mask))

        return (train_mask,test_mask,validation_mask,
                train_end_idx,train_end_new_idx,train_new_end_date,
                test_start_date,validation_start_date)
    else:
        test_mask=None
        test_start_date=None
        validation_start_date = (train_end_date + pd.Timedelta(days=1))
        validation_mask = (dataframe['DCP_date_current_period'] <= validation_end_date)
        print('Validation mask:', sum(validation_mask))

    return (train_mask,test_mask,validation_mask,
            train_end_idx,train_end_new_idx,train_new_end_date,
            test_start_date,validation_start_date)


def kalman_filter(dataframe):
    kalman_smoother=tsmoothie.KalmanSmoother(component='level_trend',  component_noise={'level':0.1, 'trend':0.1})
    kalman_smoother.smooth(dataframe)
    dataframe = pd.DataFrame(kalman_smoother.smooth_data,columns=dataframe.columns,index=dataframe.index)
    return dataframe
    
def apply_pretreatment(train_df,test_df,validation_df,features_start_index,features_stop_index,model_case_version_main_target_code):
    features = list(train_df.iloc[:,features_start_index:features_stop_index].columns)
    targets = list(train_df.iloc[:,features_stop_index:features_stop_index + int(model_case_version_main_target_code) + 1].columns)

    features_and_targets = features + targets

    print('Features and Targets: ', features_and_targets)
    print('Features List: ', features)
    print('Targets List: ',targets)

    train_dates_np_arr = train_df['DATE'].values
    validation_dates_np_arr = validation_df['DATE'].values

    if isinstance(test_df,pd.DataFrame):
        test_dates_np_arr = test_df['DATE'].values
    else:
        test_dates_np_arr=None

    train_df = train_df.set_index('DATE')
    validation_df = validation_df.set_index('DATE')

    if isinstance(test_df,pd.DataFrame):
        test_df = test_df.set_index('DATE')

    #log transform
    train_df = np.log(train_df[features_and_targets])
    validation_df = np.log(validation_df[features_and_targets])
    if isinstance(test_df,pd.DataFrame):
        test_df = np.log(test_df[features_and_targets])

    #Kalman filter
    train_df = kalman_filter(train_df[features_and_targets])
    validation_df = kalman_filter(validation_df[features_and_targets])
    if isinstance(test_df,pd.DataFrame):
        test_df = kalman_filter(test_df[features_and_targets])

    #Robust scaling
    # Fit transform train
    robust_scaler_features = RobustScaler().fit(train_df[features])
    robust_scaler_targets = RobustScaler().fit(train_df[targets])

    train_df_features = pd.DataFrame(robust_scaler_features.transform(train_df[features]),
                columns=train_df[features].columns, index=train_df.index)


    train_df_targets = pd.DataFrame(robust_scaler_targets.transform(train_df[targets]),
                                    columns=train_df[targets].columns, index=train_df.index)

    
    validation_df_features = pd.DataFrame(robust_scaler_features.transform(validation_df[features]),
                             columns=validation_df[features].columns, index=validation_df.index)

    validation_df_targets = pd.DataFrame(robust_scaler_targets.transform(validation_df[targets]),
                             columns=validation_df[targets].columns, index=validation_df.index)

    if isinstance(test_df,pd.DataFrame):
        test_df_features = pd.DataFrame(robust_scaler_features.transform(test_df[features]),
                                columns=test_df[features].columns, index=test_df.index)

        test_df_targets = pd.DataFrame(robust_scaler_targets.transform(test_df[targets]),
                                columns=test_df[targets].columns, index=test_df.index)  
    else:
        test_df_features = None
        test_df_targets = None

    return (train_df,
            test_df,
            validation_df,
            train_df_features,
            train_df_targets,
            test_df_features,
            test_df_targets,
            validation_df_features,
            validation_df_targets,
            train_dates_np_arr,
            validation_dates_np_arr,
            test_dates_np_arr,
            robust_scaler_features,
            robust_scaler_targets,
            features,
            targets,
            features_and_targets)


def model_cnn_lstm_v1(input_shape,optimizer,attenuated_padding_value,
                      twoexp_nodes_number_layer_1,twoexp_nodes_number_layer_2,
                      twoexp_nodes_number_layer_3,twoexp_nodes_number_layer_4,
                      model_case_version_main_target_code):
    model = Sequential([
        Conv1D(filters=64,kernel_size=4,activation='relu',input_shape=input_shape),
        MaxPooling1D(),
        LSTM(2**twoexp_nodes_number_layer_1,return_sequences=True),
        LSTM(2**twoexp_nodes_number_layer_2,return_sequences=True),
        LSTM(2**twoexp_nodes_number_layer_3),
        Dense(2**twoexp_nodes_number_layer_4),
        Dense(int(model_case_version_main_target_code)+1)
    ])

    model.compile(optimizer=optimizer, loss=custom_loss_function(attenuated_padding_value))
    return model


# Obtain Predictions
def consume_model(model,features,scaler):
    predictions = model.predict(features)
    predictions = scaler.inverse_transform(predictions) #convert prediction first by inverting the Robust scaler transformation and then the e_logarithmic one.
    predictions = np.exp(predictions)

    predictions = np.concatenate([a[:1] for a in predictions])
    
    return predictions

def calculate_analytical_parameters(actual,predicted):
    x = actual[:len(predicted)]
    y = predicted
    analytical_parameters = {}

    analytical_params = linregress(x, y)

    slope = analytical_params.slope
    intercept = analytical_params.intercept
    rvalue = analytical_params.rvalue
    y_trend_line = slope*x + intercept #this is computed just for the avg_tld
    avg_trend_line_distance = np.mean(np.abs(y_trend_line - y)/y_trend_line)

    analytical_parameters['slope'] = slope
    analytical_parameters['intercept'] = intercept
    analytical_parameters['rvalue'] = rvalue
    analytical_parameters['dispersion'] = avg_trend_line_distance
    #analytical_parameters['mse'] = mean_squared_error(x,y) 
    #analytical_parameters['rmse'] = np.sqrt(mean_squared_error(x,y))
    #analytical_parameters['mape'] = mean_absolute_percentage_error(x,y)
    
    return analytical_parameters

def calculate_position_day_number_split(dataframe,mask,model_case_version_time_steps_integer,main_target_code_integer):
    dncp = dataframe[mask]['DNCP_day_number_current_period'][model_case_version_time_steps_integer-1+main_target_code_integer:]
    dncp = dncp.astype(int).to_numpy()
    span_dncp = dncp[-1] - dncp[0] +1
    positions_day_number=dncp-dncp[0]+1
    print(positions_day_number.shape)

    return positions_day_number

def fit_vertical_correction(train_actual,train_predictions,day_number_index,url,headers):
    query = f"""
        query {{
            fitVerticalCorrection(
                train_actual: {train_actual},
                train_predictions: {train_predictions},
                day_number_index: {day_number_index}
            ) {{
                success,
                error,
                vertical_padding_correction_factor,
                corrected_train_raw_targets_ratios,
                train_vertical_corrected_targets,
                train_raw_targets_ratio
            }}
        }}
    """

    response = requests.post(url,headers=headers,json={'query':query}).json()
    return response

def fit_swing_correction(train_actual,
                         train_vertical_corrected_targets,
                         day_number_index,
                         vertical_padding_correction_factor,
                         train_raw_targets_ratio,
                         headers,url):
    query = f'''
        query {{
            fitSwingCorrection(
                train_actual: {train_actual},
                train_vertical_corrected_targets: {train_vertical_corrected_targets},
                day_number_index: {day_number_index},
                vertical_padding_correction_factor: {vertical_padding_correction_factor},
                train_raw_targets_ratio: {train_raw_targets_ratio},
            ) {{
                swing_padding_correction_factors,
                swing_targets_ratios,
                swing_predicted_targets
            }}
        }}
        '''

    response = requests.post(url,headers=headers,json={'query':query}).json()
    return response

def fit_horizontal_correction_regression(swing_predicted_targets,train_actual,day_number_index,url,headers):
    query = f"""
        query {{
            fitHorizontalPaddingCorrectionRegression (
                    swing_predicted_train_targets: {swing_predicted_targets},
                    train_actual: {train_actual},
                    day_number_index: {day_number_index}
            ) {{
                success,
                error,
                model_path
            }}
        }} 
    """

    response = requests.post(url,headers=headers,json={'query':query}).json()
    return response

def apply_horizontal_correction(actual,swing_predicted_targets,day_number_index,horizontal_correction_model_path,url,headers):
    horizontal_correction_model_path = json.dumps(horizontal_correction_model_path)
    query = f"""
        query{{
            applyHorizontalPaddingCorrection (
                actual:{actual},
                swing_predicted_targets:{swing_predicted_targets},
                day_number_index: {day_number_index},
                horizontal_correction_model_path: {horizontal_correction_model_path}
            ) {{
                success,
                error,
                horizontal_corrected_swing_targets,
                horizontal_correction_model_rmse_error
            }}
        }}
    """
    response = requests.post(url,headers=headers,json={'query':query}).json()
    return response