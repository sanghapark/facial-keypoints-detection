import os
import numpy as np
import matplotlib.pyplot as plt


from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

def load_data(cols=None, test=False):
    """
    Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Important that the files are in a `data` directory
    """  
    FTRAIN = '../../data/training.csv'
    FTEST  = '../../data/test.csv'
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load dataframes

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if not test:
        df = df[df.columns[cols]] if cols != None else df
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1) # return each images as 96 x 96 x 1

    if not test:  # only FTRAIN has target columns
        Y = df[df.columns[:-1]].values
        Y = (Y - 48) / 48  # scale target coordinates to [-1, 1]
        X, Y = shuffle(X, Y, random_state=42)  # shuffle train data
        Y = Y.astype(np.float32)
    else:
        Y = None

    return X, Y


