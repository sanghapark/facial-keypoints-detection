import keras.backend as K

def rmse(target, y_predicted):
    rmse = K.sqrt(K.mean(K.square(y_predicted - target))) * 48
    return rmse
