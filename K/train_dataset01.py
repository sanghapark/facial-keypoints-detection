import os
import datetime as dt
from keras.optimizers import RMSprop, Adam, SGD, Nadam, Adamax, TFOptimizer
from utils.metric import rmse
from utils.cnn import create_cnn2, train
from utils.constant import *
from utils.models import save_model


datetime = dt.datetime.now().strftime("%Y%m%d_%H%M")
modelname = 'model_{}'.format(datetime)
if not os.path.exists('models/{}'.format(modelname)):
    os.makedirs('models/{}'.format(modelname))


# Dataset01에 대한 앙상블 만들기
submodelpath = 'models/{}/{}'.format(modelname, 'cnn2_dataset01')
if not os.path.exists(submodelpath):
    os.makedirs(submodelpath)
for i in range(50):
    # model = create_cnn2(8, 'he_uniform', ACTIVATION, LAST_ACTIVATION)
    model = create_cnn2(8, 'he_normal', 'relu', 'relu')

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilson=1e-8, decay=0)
    # optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = Nadam()

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])
    if i == 0:
        save_model(submodelpath, "cnn2_dataset01", model)
    cnnname = 'cnn2_dataset01_{:02}'.format(i)
    train(model, cnnname, submodelpath, COLS01, FLIP_INDICES01, optimizer, EPOCHS01, 256)
