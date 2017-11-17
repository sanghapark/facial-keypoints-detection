import os
import datetime as dt
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.load import load_train_data_and_split
from utils.metric import rmse
from utils.data_augment_generator import DataAugmentGenerator
from utils.cnn import create_cnn2
from utils.constant import *
from utils.models import save_model
from utils.loss_history import LossHistory

datetime = dt.datetime.now().strftime("%Y%m%d_%H%M")
modelname = 'model_{}'.format(datetime)
if not os.path.exists('models/{}'.format(modelname)):
    os.makedirs('models/{}'.format(modelname))

def train(model, cnnname, submodelpath, cols, flip_indices, optimizer, epochs):
    X_train, X_valid, Y_train, Y_valid = load_train_data_and_split(FILEPATH_TRAIN, cols, VALIDATION_RATIO)

    weightfile = os.path.join(submodelpath, cnnname + '.h5')
    histfile   = os.path.join(submodelpath, cnnname + '.csv')

    history = LossHistory(histfile)
    checkpoint    = ModelCheckpoint(weightfile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min')
    
    model.load_weights(weightfile) if os.path.exists(weightfile) else None
    generator = DataAugmentGenerator(X_train, Y_train, BATCH_SIZE, flip_indices, FLIP_RATIO, ROTATE_RATIO, CONTRAST_RATIO)
    model.fit_generator(generator.generate(BATCH_SIZE, FLIP, ROTATE, CONTRAST), 
                        steps_per_epoch=int(generator.size_train/BATCH_SIZE),
                        epochs=epochs,
                        verbose=1,
                        callbacks=[checkpoint, earlystopping, history],
                        validation_data=[X_valid, Y_valid])
    model.save_weights(weightfile)
    print('Weights and Loss History are saved as {} and {}'.format(weightfile, histfile))


# Dataset01에 대한 앙상블 만들기
submodelpath = 'models/{}/{}'.format(modelname, 'cnn2_dataset01')
if not os.path.exists(submodelpath):
    os.makedirs(submodelpath)
model = create_cnn2(8, ACTIVATION, LAST_ACTIVATION)
optimizer = RMSprop(0.001, 0.9, 1e-8, 0)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])
save_model(submodelpath, "cnn2_dataset01", model)
for i in range(10):
    cnnname = 'cnn2_dataset01_{:02}'.format(i)
    train(model, cnnname, submodelpath, COLS01, FLIP_INDICES01, optimizer, EPOCHS01)


# Dataset02에 대한 앙상블 만들기
submodelpath = 'models/{}/{}'.format(modelname, 'cnn2_dataset02')
if not os.path.exists(submodelpath):
    os.makedirs(submodelpath)
model = create_cnn2(22, ACTIVATION, LAST_ACTIVATION)
optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])
save_model(submodelpath, "cnn2_dataset02", model)
for i in range(20):
    cnnname = 'cnn2_dataset02_{:02}'.format(i)
    train(model, cnnname, submodelpath, COLS02, FLIP_INDICES02, optimizer, EPOCHS02)
