import pandas as pd
import matplotlib.pyplot as plt
from api.preprocessing import Preprocess, feature_1_denormalize

plt.rcParams['font.family'] = 'Segoe UI Emoji'  # or 'Noto Color Emoji' or any emoji-supporting font

pd.set_option.max_columns = None

df = pd.read_csv("my_data.csv")
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

features = ['OPCP', 'HPCP', 'LPCP', 'CPCP', 'ACPCP']
target = 'MPN5P'

df = df[features + [target]]


train_end = '4/14/2021'
train = df.loc[:train_end,:]
test = df.loc[train_end:,:].drop(train_end,axis=0)


X_train = train.drop("MPN5P", axis=1)
y_train = train["MPN5P"]
X_test = test.drop("MPN5P", axis=1)
y_test = test["MPN5P"]


X_train = X_train.to_numpy(dtype='float64')
y_train = y_train.to_numpy(dtype='float64')
X_test = X_test.to_numpy(dtype='float64')
y_test = y_test.to_numpy(dtype='float64')

#Feature 1 : OPCP
X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized , scaler_y_1, scaler_y_2 = Preprocess(X_train,X_test,y_train,y_test, True)


# y_original_train = feature_1_denormalize(y_train_normalized, scaler_y_1, scaler_y_2)


