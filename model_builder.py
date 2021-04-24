from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def rul_LSTM(seq_length, features, out_length):
    """
    :param seq_length: lookback
    :param features: Number of features
    :param out_length: Output dimensions
    :return: Model
    """
    model = Sequential()
    model.add(LSTM(input_shape=(seq_length, features), units=100, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=out_length, activation='sigmoid'))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
    return model
