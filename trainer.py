from model_builder import rul_LSTM
from data_ops import arr_generator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

np.random.seed(7)

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
path = r"C:\Users\Dennis Pkemoi\Desktop\College Education\2020 NESBE Research internship\Methods and work\Prognostics\Bearing_Dataset\3rd_test.xlsx"

# Desired time sequence to look back on
steps = 50
# Reading and scaling the in the training data
df = pd.read_excel(path)
col_norm = df.columns.difference(['Datetime', 'cycles'], sort=False)
x_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
y_df = y_scaler.fit_transform(df[['cycles']])
train_df = pd.DataFrame(x_scaler.fit_transform(df[col_norm]), columns=col_norm, index=df.index)
y_df = pd.DataFrame(y_df, columns=['cycles'], index=df.index)
train_df = train_df.join(y_df)
print(train_df.head())
# Taking the last 50 points for data modelling
lim = list(train_df['cycles'][-steps:])
# Data Modelling
cols = df.columns.to_list()[2:]
cols = [cols[i:i + 11] for i in range(0, len(cols), 11)]
for i in range(len(cols)):
    train_df.plot(x='cycles', y=cols[i], subplots=True, xlim=[lim[0], lim[steps-1]], figsize=(20, 20))
    # plt.savefig('images/bearing{0}.pdf'.format(i+1))
# plt.show()

