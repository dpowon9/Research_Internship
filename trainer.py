from model_builder import rul_LSTM
from Data_scaler import arr_generator
import pandas as pd
from sklearn import preprocessing

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
path = r"C:\Users\Dennis Pkemoi\Desktop\College Education\2020 NESBE Research internship\Methods and work\Prognostics\Bearing_Dataset\3rd_test.xlsx"
path2 = r"C:\Users\Dennis Pkemoi\Desktop\College Education\2020 NESBE Research internship\Methods and work\Prognostics\Bearing_Dataset\2nd_test.xlsx"

steps = 50
batch = 32
df = pd.read_excel(path)
col_norm = df.columns.difference(['Datetime'])
scale = preprocessing.MinMaxScaler()
train_df = pd.DataFrame(scale.fit_transform(df[col_norm]), columns=col_norm, index=df.index)
print(train_df)
train_gen = arr_generator(train_df, steps, batch, 'cycles', is_train=True)
print(train_gen[0][0].shape)
model = rul_LSTM(steps, train_gen[0][0].shape[2], 1)
print(model.summary())
model.fit(train_gen, epochs=5)

