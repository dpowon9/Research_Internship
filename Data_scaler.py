import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator


def arr_generator(df, steps, batch, y_col, is_train=False, is_test=False):
    """
    :param is_test: generator called for testing
    :param is_train: generator called for training
    :param y_col: String: name of the y column in the dataframe
    :param df: Pandas dataframe
    :param steps: Time steps to look back on
    :param batch: batch size
    :return: a keras generator object to be passed into model.fit_generator()
    """
    x = df[df.columns.difference([y_col])]
    y = df[y_col]
    x, y = x.to_numpy(), y.to_numpy()
    if is_train:
        generator = TimeseriesGenerator(x, y, length=steps, batch_size=batch)
        return generator
    elif is_test:
        generator = TimeseriesGenerator(x, np.zeros(len(x)), length=steps, batch_size=batch)
        return generator
    else:
        raise Exception('Input must be either "train" or "test"')
