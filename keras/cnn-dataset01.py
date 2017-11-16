from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from utils.load import load_data

# (7000, 9)
cols = [
    'left_eye_center_x',
    'left_eye_center_y',
    'right_eye_center_x',
    'right_eye_center_y',
    'nose_tip_x',
    'nose_tip_y',
    'mouth_center_bottom_lip_x',
    'mouth_center_bottom_lip_y',
    'Image'
]

# Load training set
X_train, y_train = load_data(cols=cols, test=False)
print("X_train.shape == {}".format(X_train.shape))
print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

# Load testing set
X_test, _ = load_data(test=True)
print("X_test.shape == {}".format(X_test.shape))

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
model.add(Dense(8, activation='tanh'))

model.summary()




adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])

## TODO: Train the model
hist = model.fit(X_train, y_train, batch_size=256, verbose=2, epochs=100, validation_split=0.2, shuffle=True)

## TODO: Save the model as model.h5
model.save('model_for_dataset01.h5')