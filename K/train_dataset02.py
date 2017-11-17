import os
import datetime as dt
from keras.optimizers import RMSprop, Adam
from utils.metric import rmse
from utils.cnn import create_cnn2, train
from utils.constant import *
from utils.models import save_model, reset_model


datetime = dt.datetime.now().strftime("%Y%m%d_%H%M")
modelname = 'model_{}'.format(datetime)
if not os.path.exists('models/{}'.format(modelname)):
    os.makedirs('models/{}'.format(modelname))


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
    reset_model(model)
