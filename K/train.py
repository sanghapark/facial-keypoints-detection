import os
import pickle
import datetime as dt
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.load import load_train_data_and_split
from utils.metric import rmse
from utils.data_augment_generator import DataAugmentGenerator
from utils.cnn import *
from utils.constant import *
from utils.models import save_pretrained_model_with_weight

datetime = dt.datetime.now().strftime("%Y%m%d_%H%M")
modelname = 'model_{}'.format(datetime)
if not os.path.exists('models/{}'.format(modelname)):
    os.makedirs('models/{}'.format(modelname))

def train(model, cnnname, submodelpath, cols, flip_indices, optimizer, epochs):
    X_train, X_valid, Y_train, Y_valid = load_train_data_and_split(FILEPATH_TRAIN, cols, VALIDATION_RATIO)

    weightfile = cnnname + '.h5'
    weightfile = os.path.join(submodelpath, cnnfile)
    checkpoint = ModelCheckpoint(weightfile, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min')
    
    model.load_weights(weightfile) if os.path.exists(weightfile) else None
    generator = DataAugmentGenerator(X_train, Y_train, batchsize=BATCH_SIZE, flip_indices=flip_indices, flip_ratio=FLIP_RATIO, rotate_ratio=ROTATE_RATIO, contrast_ratio=CONTRAST_RATIO)
    model.fit_generator(generator.generate(batchsize=BATCH_SIZE, flip=FLIP, rotate=ROTATE, contrast=CONTRAST), steps_per_epoch=int(generator.size_train/BATCH_SIZE), epochs=epochs, verbose=1, callbacks=[checkpoint, earlystopping], validation_data=[X_valid, Y_valid])
    
    histfile = os.path.join(submodelpath, cnnname + '.history')
    with open(histfile, 'wb') as f:
        pickle.dump(model.history, f)
    model.save_weights(weightfile)
    print('Weights and Loss History are saved as {} and {}'.format(weightfile, histfile))


# Dataset01에 대한 앙상블 만들기
submodelpath = 'models/{}/{}'.format(modelname, 'cnn2_dataset01')
if not os.path.exists(submodelpath):
    os.makedirs(submodelpath)
model = create_cnn2(8, ACTIVATION, LAST_ACTIVATION)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])
save_pretrained_model_with_weight(submodelpath, "cnn2_dataset01", model)
for i in range(10):
    cnnname = 'cnn2_dataset01_{:02}'.format(i)
    optimizer = RMSprop(0.001, 0.9, 1e-8, 0)
    train(model, cnnname, submodelpath, COLS01, FLIP_INDICES01, optimizer, EPOCHS01)



# Dataset02에 대한 앙상블 만들기
submodelpath = 'models/{}/{}'.format(modelname, 'cnn2_dataset02')
if not os.path.exists(submodelpath):
    os.makedirs(submodelpath)
model = create_cnn2(22, ACTIVATION, LAST_ACTIVATION)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])
save_pretrained_model_with_weight(submodelpath, "cnn2_dataset02", model)
for i in range(20):
    cnnname = 'cnn2_dataset02_{:02}'.format(i)
    optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    train(model, cnnname, submodelpath, COLS02, FLIP_INDICES02, optimizer, EPOCHS02)
