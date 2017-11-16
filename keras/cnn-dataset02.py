from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
from utils.load import load_data

weights_file_name = 'model_for_dataset02.h5'


# (2155, 23)
cols = [
    'left_eye_inner_corner_x',
    'left_eye_inner_corner_y',
    'left_eye_outer_corner_x',
    'left_eye_outer_corner_y',
    'right_eye_inner_corner_x',
    'right_eye_inner_corner_y',
    'right_eye_outer_corner_x',
    'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x',
    'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x',
    'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x',
    'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x',
    'right_eyebrow_outer_end_y',
    'mouth_left_corner_x',
    'mouth_left_corner_y',
    'mouth_right_corner_x',
    'mouth_right_corner_y',
    'mouth_center_top_lip_x',
    'mouth_center_top_lip_y',
    'Image'
]
flip_indices_dataset02 = [(0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15), (16, 18), (17, 19)]


# Load training set
X_train, y_train = load_data(cols=cols, test=False)

# Load testing set
X_test, _ = load_data(test=True)

model = Sequential()
model.add(Convolution2D(64, (3, 3), padding='same', activation='relu', input_shape=(96, 96, 1)))
model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(22, activation='tanh'))

model.summary()


from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])

## TODO: Train the model
hist = model.fit(X_train, y_train, batch_size=256, verbose=2, epochs=100, validation_split=0.2, shuffle=True)

## TODO: Save the model as model.h5
model.save(weights_file_name)