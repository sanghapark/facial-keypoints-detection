import os
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
from utils.load import load_train_data_and_split
from utils.metric import rmse
from utils.data_augment_generator import DataAugmentGenerator
# from utils.visualize import plot_error_metric_history
from utils.cnn import *
from utils.constant import *


# (2155, 23)
COLS = [
    'left_eye_inner_corner_x',      'left_eye_inner_corner_y',
    'left_eye_outer_corner_x',      'left_eye_outer_corner_y',
    'right_eye_inner_corner_x',     'right_eye_inner_corner_y',
    'right_eye_outer_corner_x',     'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x',     'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x',     'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x',    'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x',    'right_eyebrow_outer_end_y',
    'mouth_left_corner_x',          'mouth_left_corner_y',
    'mouth_right_corner_x',         'mouth_right_corner_y',
    'mouth_center_top_lip_x',       'mouth_center_top_lip_y',
    'Image'
]
FLIP_INDICES = [(0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15), (16, 18), (17, 19)]


WEIGHTS_FILE_NAME = 'cnn_for_dataset02.h5'
BATCH_SIZE = 128
EPOCHS = 300
VALIDATION_RATIO = 0.2

ACTIVATION = 'elu'
LAST_ACTIVATION = 'tanh'

FLIP = True
ROTATE = True
CONTRAST = True
FLIP_RATIO = 0.5
ROTATE_RATIO = 0.5
CONTRAST_RATIO = 0.5

metrics = [rmse]

# optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
optimizer = RMSprop(0.001, 0.9, 1e-8, 0)


print('=== LOADING DATA ===')

X_train, X_valid, Y_train, Y_valid = load_train_data_and_split(FILEPATH_TRAIN, COLS, VALIDATION_RATIO)
n_output = Y_train.shape[1]

print('=== BUILDING CNN ===')
model = create_cnn2(n_output, ACTIVATION, LAST_ACTIVATION)


print('=== COMPILING ===')

# Save the model after every epoch
checkpoint = ModelCheckpoint(WEIGHTS_FILE_NAME,
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')

# Stop training when a monitored quantity has stopped improving.
earlystopping = EarlyStopping(monitor='val_loss', 
                             min_delta=0, 
                             patience=50, 
                             verbose=0, 
                             mode='min')

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=metrics)

print('=== LOADING PRETRAINED WEIGHTS ===')


model.load_weights(WEIGHTS_FILE_NAME) if os.path.exists(WEIGHTS_FILE_NAME) else None

generator = DataAugmentGenerator(X_train,
                                 Y_train,
                                 batchsize=BATCH_SIZE,
                                 flip_indices=FLIP_INDICES,
                                 flip_ratio=FLIP_RATIO,
                                 rotate_ratio=ROTATE_RATIO,
                                 contrast_ratio=CONTRAST_RATIO)

print('=== TRAINING ===')

hist = model.fit_generator(generator.generate(batchsize=BATCH_SIZE, flip=FLIP, rotate=ROTATE, contrast=CONTRAST),
                    steps_per_epoch=int(generator.size_train/BATCH_SIZE),
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=[checkpoint, earlystopping],
                    validation_data=[X_valid, Y_valid])

print('=== FINISHED TRAINING ===')

print('=== SAVING WEIGHTS ===')
model.save_weights(WEIGHTS_FILE_NAME)
print('Weights are saved as {}'.format(WEIGHTS_FILE_NAME))


# Plotting
# plot_error_metric_history(hist.history)