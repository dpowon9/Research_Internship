import keras
from model_builder import Bi_LSTM
from data_ops import arr_generator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import precision_score, confusion_matrix, recall_score, f1_score
from sklearn.utils import class_weight
import warnings

warnings.filterwarnings('ignore')
np.random.seed(7)

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
path = r"C:\Users\Dennis Pkemoi\Desktop\College Education\2020 NESBE Research internship\Methods and work\Prognostics\Bearing_Dataset\3rd_test.xlsx"

# Desired time sequence to look back on
steps = 50
# Reading and scaling the in the training data
df = pd.read_excel(path)
df['cycles'] = df['cycles'].div(60)
col_norm = df.columns.difference(['Datetime', 'cycles'], sort=False)
print(df.head())
x_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
train_df = pd.DataFrame(x_scaler.fit_transform(df[col_norm]), columns=col_norm, index=df.index)
train_df = train_df.join(df['cycles'])
minutes_1 = 1000/60
train_df['label'] = np.where(train_df['cycles'] <= minutes_1, 1, 0)
# Taking the last 50 points for data modelling
lim = list(train_df['cycles'][-steps:])
# Data Modelling
cols = df.columns.to_list()[2:]
cols = [cols[i:i + 11] for i in range(0, len(cols), 11)]
'''
for i in range(len(cols)):
    test_df.plot(x='cycles', y=cols[i], subplots=True, xlim=[lim[0], lim[steps-1]], figsize=(20, 20))
    plt.savefig('Train_metrics/Data_Visualization/bearing{0}.pdf'.format(i+1))
plt.show()
'''
train_df.drop(columns=['cycles'], axis=1, inplace=True)
print(train_df.head())

# Setting up training requirements
X_train, Y_train = arr_generator(train_df, 50, 'label')
model = Bi_LSTM(50, X_train.shape[2], 3)
call = keras.callbacks.ModelCheckpoint(filepath='Models/Bi_LSTM.hdf5', monitor='val_loss', mode='min', verbose=1,
                                       save_best_only=True)
print(model.summary())

# Training
weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
print('Using weights, class 0: %f and class 1: %f' % (weights[0], weights[1]))
weights = {0: weights[0], 1: weights[1]}

model.fit(X_train, Y_train, epochs=10, batch_size=10, validation_data=(X_train, Y_train), verbose=1,
          callbacks=call, use_multiprocessing=True, class_weight=weights)
# Getting metrics
ep_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
plt.figure(1)
plt.plot(range(len(ep_loss)), ep_loss)
plt.plot(range(len(ep_loss)), val_loss)
plt.legend(['Training loss', 'Val loss'])
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Train loss vs epoch')
plt.savefig('Train_metrics/loss.pdf')

# Evaluation
best_model = keras.models.load_model('Models/Bi_LSTM.hdf5')
performance = best_model.evaluate(X_train, Y_train, batch_size=10)
y_pred = best_model.predict_classes(X_train)
y_true = Y_train
final = pd.DataFrame({'Truth': y_true, 'predicted': np.array(y_pred).flatten()}, index=np.arange(len(y_pred)))
final.to_excel('Train_metrics/train_res.xlsx', index=False)
# Model metrics
precision = precision_score(Y_train, y_pred, average='weighted')
recall = recall_score(Y_train, y_pred, average='weighted')
fscore = f1_score(Y_train, y_pred)
scores = [precision, recall, fscore]
col = ['Precision', 'Recall', 'F1 Score']
final_res = pd.DataFrame([scores], columns=col)
print(final_res)
cf = confusion_matrix(Y_train, y_pred)
cfd = pd.DataFrame(cf, columns=['Pred-0', 'Pred-1'], index=['GT-0', 'GT-1'])
print(cfd)
final_res.to_csv('Train_metrics/eval.csv', index=False)
cfd.to_csv('Train_metrics/Conf_Mat.csv')
