import os
import datetime as dt
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.load import load_train_data_and_split
from utils.metric import rmse
from utils.data_augment_generator import DataAugmentGenerator
from utils.cnn import *
from utils.constant import *


def train(filepath, cols, flip_indices, optimizer, epochs):
    X_train, X_valid, Y_train, Y_valid = load_train_data_and_split(FILEPATH_TRAIN, cols, VALIDATION_RATIO)
    n_output = Y_train.shape[1]
    model = create_cnn2(n_output, ACTIVATION, LAST_ACTIVATION)
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', 
                                  min_delta=0,
                                  patience=50,
                                  verbose=0,
                                  mode='min')
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])
    model.load_weights(filepath) if os.path.exists(filepath) else None
    generator = DataAugmentGenerator(X_train,
                                     Y_train,
                                     batchsize=BATCH_SIZE,
                                     flip_indices=flip_indices,
                                     flip_ratio=FLIP_RATIO,
                                     rotate_ratio=ROTATE_RATIO,
                                     contrast_ratio=CONTRAST_RATIO)
    model.fit_generator(generator.generate(batchsize=BATCH_SIZE, flip=FLIP, rotate=ROTATE, contrast=CONTRAST),
                               steps_per_epoch=int(generator.size_train/BATCH_SIZE),
                               epochs=epochs,
                               verbose=1,
                               callbacks=[checkpoint, earlystopping],
                               validation_data=[X_valid, Y_valid])
    model.save_weights(filepath)
    print('Weights are saved as {}'.format(filepath))


# Dataset01에 대한 앙상블 만들기
for i in range(10):
    filepath = 'cnn2_dataset01_{:02}.h5'.format(i)
    optimizer = RMSprop(0.001, 0.9, 1e-8, 0)
    train(filepath, COLS01, FLIP_INDICES01, optimizer, EPOCHS01)

# Dataset02에 대한 앙상블 만들기
for i in range(20):
    filepath = 'cnn2_dataset02_{:02}.h5'.format(i)
    optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    train(filepath, COLS02, FLIP_INDICES02, optimizer, EPOCHS02)
