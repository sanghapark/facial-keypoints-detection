from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from utils.constant import *


# Batch Normalization, Ensemble 참고
# https://github.com/alexander-rakhlin/kaggle_otto/blob/master/keras.ensemble.py


def create_cnn(n_output, activation, last_activation):
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, COLOR_CHANNEL)
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


def create_cnn2(n_output, activation, last_activation):
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, COLOR_CHANNEL)
    model = Sequential([
        Convolution2D(128, 3, 3, border_mode='valid', input_shape=input_shape),
        Activation(activation),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        Convolution2D(256, 2, 2),
        Activation(activation),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Convolution2D(512, 2, 2),
        Activation(activation),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512),
        Activation(activation),
        Dropout(0.5),
        Dense(512),
        Activation(last_activation),

        Dense(n_output),
    ])
    print(model.summary())
    return model

def create_cnn2(n_output, activation, last_activation):
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, COLOR_CHANNEL)
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, border_mode='same', init='he_normal', input_shape=input_shape, dim_ordering='tf'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    model.add(Convolution2D(36, 5, 5))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    model.add(Convolution2D(48, 5, 5))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation(activation))
    model.add(GlobalAveragePooling2D());

    model.add(Dense(500, activation=activation))
    model.add(Dense(90, activation=last_activation))
    model.add(Dense(n_output))

    print(model.summary())
    return model