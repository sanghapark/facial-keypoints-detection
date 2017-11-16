import numpy as np
import os
from pandas.io.parsers import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle


def load_train_data(filepath, cols=None):
    df = read_csv(os.path.expanduser(filepath))  # load dataframes

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))


    df = df[cols] if cols != None else df
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1) # return each images as 96 x 96 x 1

    Y = df[df.columns[:-1]].values
    Y = (Y - 48) / 48  # scale target coordinates to [-1, 1]
    X, Y = shuffle(X, Y, random_state=42)  # shuffle train data
    Y = Y.astype(np.float32)

    return X, Y

def load_test_data(filepath, cols=None):
    df = read_csv(filepath)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df = df[cols] if cols != None else df
    df = df.dropna()
    X = np.vstack(df['Image'].values) / 255.
    X = X.reshape(-1, 96, 96, 1)
    X = X.astype(np.float32)
    return X


def load_train_data_and_split(filepath, cols, size=0.2):
    X, Y = load_train_data(filepath, cols)
    return  train_test_split(X, Y, test_size=size)
