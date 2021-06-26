import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Flatten


def Bi_LSTM(seq_length, features, out_length):
    """
    :param seq_length: lookback
    :param features: Number of features
    :param out_length: Output dimensions
    :return: Model
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1), input_shape=(seq_length, features)))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True, dropout=0.4, recurrent_dropout=0.1)))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True, dropout=0.4, recurrent_dropout=0.1)))
    model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dense(units=out_length, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])
    return model
