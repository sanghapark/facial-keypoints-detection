from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from utils.constant import *


# Batch Normalization, Ensemble 참고
# https://github.com/alexander-rakhlin/kaggle_otto/blob/master/keras.ensemble.py

# 최고가 대략 4점대, 오래걸림
# LeakyReLU: 별로다 처음에도 나아지질 않닸다. elu도 나아 지지 않는다.
# relu가 가장 나았다.
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


"""
    dataset01
        cnn_for_dataset01_relu_tanh_adam.h5
            min val RMSE = 2.1115, epoch: 180

        cnn_for_dataset01_elu_tahn_adam.h5
            min val RMSE = 2.0817  epoch: 170 epoch부터 validation rmse가 24취솓다가 4점대 유지 시작 (오버피팅)

        cnn_for_dataset01_elu_tanh_rmsprop.h5
            RMSE arount 2.4~2.5 epoch: 300

        cnn_for_dataset01_elu_tahn_rmsprop_drop.h5
            min val RMSE = 2.0759 epoch 300
        
        cnn_for_dataset01_elu_tahn_adam_drop.h5
            min val RMSE = 2.0817  epoch: 180 epoch부터 validation rmse가 7취솓다가 4점대 유지 시작 (오버피팅)

        cnn_for_dataset01_selu_tahn_adam_drop.h5
            min val RMSE = 4에서 계속 머뭄

        cnn_for_dataset01_leakyrulu_tanh_adam_drop.h5
            epoch 180에 2.1까지 갔다가 다시 2.4까지 올라와서 얼리스탑
            
    dataset02
        cnn_for_dataset02_elu_tanh_adam_drop.ht
            RMSE 1.5~1.6 after 350 epochs
"""
def create_cnn2(n_output, activation, last_activation):
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, COLOR_CHANNEL)
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, border_mode='same', init='he_normal', input_shape=input_shape, dim_ordering='tf'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Convolution2D(36, 5, 5))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Convolution2D(48, 5, 5))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation(activation))
    model.add(GlobalAveragePooling2D());
    model.add(Dropout(0.3))

    model.add(Dense(500, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(90, activation=last_activation))
    model.add(Dense(n_output))

    print(model.summary())
    return model


def create_cnn3(n_output, activation, last_activation):
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, COLOR_CHANNEL)
    model = Sequential([
        BatchNormalization(input_shape=input_shape),
        Convolution2D(128, 3, 3, border_mode='valid', kernel_initializer='glorot_normal', input_shape=input_shape),
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
