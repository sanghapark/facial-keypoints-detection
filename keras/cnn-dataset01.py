import os
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
from utils.load import load_train_data_and_split
from utils.metric import rmse
from utils.data_augment_generator import DataAugmentGenerator
from utils.visualize import plot_error_metric_history
from utils.cnn import create_cnn

# (7000, 9)
COLS = [
    'left_eye_center_x',            'left_eye_center_y',
    'right_eye_center_x',           'right_eye_center_y',
    'nose_tip_x',                   'nose_tip_y',
    'mouth_center_bottom_lip_x',    'mouth_center_bottom_lip_y',
    'Image'
]
FLIP_INDICES = [(0, 2), (1, 3)]

WEIGHTS_FILE_NAME = 'cnn_for_dataset01.h5'
BATCH_SIZE = 100
EPOCHS = 500
VALIDATION_RATIO = 0.1

ACTIVATION = 'relu'
LAST_ACTIVATION = 'tanh'

FLIP = True
ROTATE = True
CONTRAST = True
FLIP_RATIO = 0.5
ROTATE_RATIO = 0.5
CONTRAST_RATIO = 0.5

metrics = [rmse]

optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


print('=== LOADING DATA ===')

X_train, X_valid, Y_train, Y_valid = load_train_data_and_split(COLS, VALIDATION_RATIO)
n_output = Y_train.shape[1]

print('=== BUILDING CNN ===')
model = create_cnn(n_output, ACTIVATION, LAST_ACTIVATION)


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
                    steps_per_epoch=generator.size_train,
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