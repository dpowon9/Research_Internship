import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten


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
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mean_squared_logarithmic_error'])
    return model


def Bi_LSTM(seq_length, features, out_length):
    """
    :param seq_length: lookback
    :param features: Number of features
    :param out_length: Output dimensions
    :return: Model
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=(seq_length, features)))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True, dropout=0.4, recurrent_dropout=0.2)))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dense(units=out_length, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])
    return model
