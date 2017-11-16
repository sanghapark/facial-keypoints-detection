from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from utils.constant import *

def create_cnn(n_output, activation, last_activation):
    input_shape = (COLOR_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
    model = Sequential()
    model.add(Convolution2D(64, (3, 3), padding='same', activation=activation, input_shape=input_shape))
    model.add(Convolution2D(64, (3, 3), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(128, (3, 3), padding='same', activation=activation))
    model.add(Convolution2D(128, (3, 3), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(256, (3, 3), padding='same', activation=activation))
    model.add(Convolution2D(256, (3, 3), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Convolution2D(512, (3, 3), padding='same', activation=activation))
    model.add(Convolution2D(512, (3, 3), padding='same', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(n_output, activation=last_activation))

    print(model.summary())
    return model
