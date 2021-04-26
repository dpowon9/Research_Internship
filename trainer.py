import keras
from model_builder import Bi_LSTM
from data_ops import arr_generator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error, mean_squared_log_error

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
'''
for i in range(len(cols)):
    train_df.plot(x='cycles', y=cols[i], subplots=True, xlim=[lim[0], lim[steps-1]], figsize=(20, 20))
    plt.savefig('Train_metrics/Data_Visualization/bearing{0}.pdf'.format(i+1))
plt.show()
'''
# Setting up training requirements
X_train, Y_train = arr_generator(train_df, 50, 'cycles')
model = Bi_LSTM(50, X_train.shape[2], 1)
call = keras.callbacks.ModelCheckpoint(filepath='Models/Bi_LSTM.hdf5', verbose=1, save_best_only=True)
print(model.summary())
# Training
model.fit(X_train, Y_train, epochs=20, batch_size=100, validation_split=0.15, verbose=1, callbacks=call)
# Getting metrics
ep_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
plt.figure(1)
plt.plot(range(len(ep_loss)), ep_loss)
plt.plot(range(len(ep_loss)), val_loss)
plt.legend(['Training loss', 'Val loss'])
plt.xlabel('epoch')
plt.ylabel('mean_squared_logarithmic_error')
plt.title('Train mean_squared_logarithmic_error vs epoch')
plt.savefig('Train_metrics/loss.pdf')
# Evaluation
best_model = keras.models.load_model('Models/Bi_LSTM.hdf5')
performance = best_model.evaluate(X_train, Y_train, batch_size=100)
y_pred = best_model.predict(X_train)
y_unscaled = y_scaler.inverse_transform(y_pred)
y_true = y_scaler.inverse_transform(Y_train.reshape(1, -1))
y_true = np.array(y_true).flatten()
final = pd.DataFrame({'Truth': y_true, 'predicted': np.array(y_unscaled).flatten()}, index=np.arange(len(y_pred)))
final.to_excel('Train_metrics/train_res.xlsx', index=False)
plt.figure(2)
plt.plot(y_true, y_true)
plt.plot(y_true, np.array(y_unscaled).flatten())
plt.legend(['Ground Truth RUL', 'Predicted RUL'])
plt.xlabel('RUL in minutes')
plt.ylabel('RUL in minutes')
plt.title('GT vs Predicted')
plt.savefig('Train_metrics/GT_v_Pred.pdf')
# Model metrics
mse = mean_squared_error(Y_train, y_pred)
rmse = np.sqrt(mse)
r2_coef = r2_score(Y_train, y_pred)
exp_var = explained_variance_score(Y_train, y_pred)
err = max_error(Y_train, y_pred)
log_err = mean_squared_log_error(Y_train, y_pred)
scores = [mse, rmse, r2_coef, exp_var, err, log_err]
col = ['mse', 'rmse', 'r2 score', 'evs', 'max error', 'msle']
final_res = pd.DataFrame([scores], columns=col)
print(final_res)
final_res.to_csv('Train_metrics/eval.csv', index=False)
