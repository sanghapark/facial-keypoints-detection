import numpy as np
import os
from pandas.io.parsers import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

FTRAIN = '../data/training.csv'
FTEST  = '../data/test.csv'

def load_train_data(cols=None):
    """
    Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Important that the files are in a `data` directory
    """  
    df = read_csv(os.path.expanduser(FTRAIN))  # load dataframes

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))


    df = df[df.columns[cols]] if cols != None else df
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    X = X.reshape(-1, 1, 96, 96) # return each images as 96 x 96 x 1

    Y = df[df.columns[:-1]].values
    Y = (Y - 48) / 48  # scale target coordinates to [-1, 1]
    X, Y = shuffle(X, Y, random_state=42)  # shuffle train data
    Y = Y.astype(np.float32)

    return X, Y

def load_test_data(cols=None):
    df = read_csv(FTEST)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    if cols != None:
        df = df[df.columns[cols]]
    df = df.dropna()
    X = np.vstack(df['Image'].values) / 255.
    X = X.reshape(-1, 1, 96, 96)
    X = X.astype(np.float32)
    return X


def load_train_data_and_split(cols, size=0.2):
    X, Y = load_train_data(cols)
    return  train_test_split(X, Y, test_size=size)
