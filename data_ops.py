import numpy as np


def arr_generator(df, steps, y_col):
    """
    :param y_col: String: name of the y column in the dataframe
    :param df: Pandas dataframe
    :param steps: Time steps to look back on
    :return: 3D shaped x array and 1D y array for LSTM input
    """
    x = df[df.columns.difference([y_col])]
    y = df[y_col]
    x, y = x.to_numpy(), y.to_numpy()
    print('Dimensions of input X Data: ', x.shape)
    print('Dimensions of input Y Data: ', y.shape)
    Xt, Yt = [], []
    for i in range(len(x) - steps):
        v = x[i:i + steps, :]
        Xt.append(v)
        Yt.append(y[i + steps])
    Xt, Yt = np.array(Xt), np.array(Yt)
    print('Dimensions of output X Data: ', Xt.shape)
    print('Dimensions of output Y Data: ', Yt.shape)
    return Xt, Yt
