import pandas as pd
from keras.callbacks import Callback

class LossHistory(Callback):

    def __init__(self):
        print('LossHistory init')
        self.losses = pd.DataFrame(columns=['loss', 'val_loss', 'rmse', 'val_rmse'])

    def on_training_begin(self, logs={}):
        print('on_training_begin')
        self.losses = pd.DataFrame(columns=['loss', 'val_loss', 'rmse', 'val_rmse'])


    def on_epoch_begin(self, epoch, logs=None):
        print('on_epoch_begin')

    def on_epoch_end(self, epoch, logs=None):
        print('\n')
        print('='*100)
        print('on_epoch_end\n')
        print(logs)
        print('\n')
        print(logs.get('val_rmse'))
        print('\n')
        print('='*100)
        print('\n')
        # cols = ['loss', 'val_loss', 'rmse', 'val_rmse']
        # metrics = [
        #     logs.get('loss'),
        #     logs.get('val_loss'),
        #     logs.get('rmse'),
        #     logs.get('val_rmse')
        # ]
        # self.losses.append(pd.DataFrame(metrics, columns=cols))


    def on_train_end(self, logs=None):
        print('on_train_end')
