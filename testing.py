import keras
from data_ops import arr_generator
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, confusion_matrix, recall_score, f1_score
import warnings

np.random.seed(20)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

path = r"C:\Users\Dennis Pkemoi\Desktop\College Education\2020 NESBE Research internship\Methods and work\Prognostics\Bearing_Dataset\2nd_test.xlsx"
# Desired time sequence to look back on
steps = 50
# Reading and scaling the in the training data
df = pd.read_excel(path)
df['cycles'] = df['cycles'].div(60)
col_norm = df.columns.difference(['Datetime', 'cycles'], sort=False)
print(df.head())
x_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
test_df = pd.DataFrame(x_scaler.fit_transform(df[col_norm]), columns=col_norm, index=df.index)
test_df = test_df.join(df['cycles'])
minutes_1 = 1000/60
test_df['label'] = np.where(test_df['cycles'] <= minutes_1, 1, 0)

lim = list(test_df['cycles'][-steps:])
# Data Modelling
cols = df.columns.to_list()[2:]
cols = [cols[i:i + 11] for i in range(0, len(cols), 11)]
'''
for i in range(len(cols)):
    test_df.plot(x='cycles', y=cols[i], subplots=True, xlim=[lim[0], lim[steps - 1]], figsize=(20, 20))
    plt.savefig('Test_metrics/Data_Visualization/bearing{0}.pdf'.format(i+1))
'''
test_df.drop(columns=['cycles'], axis=1, inplace=True)
print(test_df.head())

# Setting up training requirements
X_test, Y_test = arr_generator(test_df, 50, 'label')
best_model = keras.models.load_model('Models/Bi_LSTM.hdf5')
performance = best_model.evaluate(X_test, Y_test, batch_size=10)
y_pred = best_model.predict_classes(X_test)
y_true = Y_test
final = pd.DataFrame({'Truth': y_true, 'predicted': np.array(y_pred).flatten()}, index=np.arange(len(y_pred)))
final.to_excel('Test_metrics/test_res.xlsx', index=False)
# Model metrics
precision = precision_score(Y_test, y_pred, average='weighted')
recall = recall_score(Y_test, y_pred, average='weighted')
fscore = f1_score(Y_test, y_pred)
scores = [precision, recall, fscore]
col = ['Precision', 'Recall', 'F1 Score']
final_res = pd.DataFrame([scores], columns=col)
print(final_res)
cf = confusion_matrix(Y_test, y_pred)
cfd = pd.DataFrame(cf, columns=['Pred-0', 'Pred-1'], index=['GT-0', 'GT-1'])
print(cfd)
final_res.to_csv('Test_metrics/eval.csv', index=False)
cfd.to_csv('Test_metrics/Conf_Mat.csv')
