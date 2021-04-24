from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import os
import fileinput
import time

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
path = r"C:\Users\Dennis Pkemoi\Desktop\College Education\2020 NESBE Research internship\Methods and work\Prognostics\Bearing_Dataset\4th_test\txt"
save_dir = r"C:\Users\Dennis Pkemoi\Desktop\College Education\2020 NESBE Research internship\Methods and work\Prognostics\Bearing_Dataset/3rd_test.xlsx"


def dir_iterator(path_to_file, channels=4):
    col = ['Ch{}'.format(i + 1) for i in range(channels)]
    data = []
    lin = fileinput.input(files=path_to_file)
    for line in lin:
        lines = [float(i) for i in line.strip().split()]
        data.append(lines)
    df = pd.DataFrame(data, columns=col)
    return df


def array_gen(df):
    return df.to_numpy()


def feat_extract(arr):
    peak = np.amax(np.abs(arr))
    rms = np.sqrt(np.mean(arr ** 2))
    mean = np.mean(arr)
    median = np.median(arr)
    variance = np.var(arr)
    std = np.std(arr)
    kurt = kurtosis(arr)
    sk = skew(arr)
    crest_fac = peak / rms
    impulse_factor = peak / np.abs(mean)
    shape = rms / np.abs(mean)
    clearance_factor = peak / np.square(np.mean(np.sqrt(np.abs(arr))))
    return rms, mean, median, variance, std, kurt, sk, crest_fac, impulse_factor, shape, clearance_factor


def cgen(string, c):
    arr = []
    for i in range(c):
        arr += [m + '{}'.format(i + 1) for m in string]
    return arr


def rul_calc(date):
    date_arr = [datetime.strptime(d, '%Y.%m.%d.%H.%M.%S') for d in date]
    diff = np.linspace(0, (date_arr[-1] - date_arr[0]).total_seconds() / 60, len(date_arr))
    diff = diff[::-1]
    dateframe = pd.DataFrame({'Datetime': date_arr, 'cycles': diff})
    return dateframe


def final_form(Dataset):
    vals = ['rms', 'mean', 'median', 'variance', 'std', 'kurt', 'skew', 'crest_fac', 'impulse_factor', 'shape',
            'clearance_factor']
    cols = cgen(vals, 4)
    start = time.time()
    dat_arr = os.listdir(Dataset)
    frame = rul_calc(dat_arr)
    feat = []
    for n in dat_arr:
        df2 = dir_iterator(Dataset + '/' + n)
        arr1 = np.array([feat_extract(array_gen(df2[str(i)])) for i in df2.columns]).flatten()
        feat.append(arr1)
    feat = pd.DataFrame(feat, columns=cols)
    final_df = pd.merge(frame, feat, how='inner', left_index=True, right_index=True)
    end = time.time()
    print("Time elapsed: %s" % (end - start))
    return final_df


# df = final_form(path)
# print(df)
# df.to_excel(save_dir, index=False)

