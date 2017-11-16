import numpy as np
import keras.backend as K

def rmse(target, y_predicted):
    rmse = np.sqrt(K.mean(K.square(y_predicted - target))) * 48
    return rmse