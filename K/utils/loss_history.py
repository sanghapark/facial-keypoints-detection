import pandas as pd
from keras.callbacks import Callback

class LossHistory(Callback):
    def on_training_begin(self, logs={}):
        print('on_training_begin')
        self.losses = pd.DataFrame(columns=['loss', 'val_loss', 'rmse', 'val_rmse'])

    def on_batch_end(self, batch, logs=None):
        print('on_batch_end')
        cols = ['loss', 'val_loss', 'rmse', 'val_rmse']
        metrics = [
            logs.get('loss'),
            logs.get('val_loss'),
            logs.get('rmse'),
            logs.get('val_rmse')
        ]
        self.losses.append(pd.DataFrame(metrics, columns=cols))


    def on_epoch_begin(self, epoch, logs=None):
        print('on_epoch_begin')

    def on_epoch_end(self, epoch, logs=None):
        print('on_epoch_end')

    def on_batch_begin(self, batch, logs=None):
        print('on_batch_begin')

    def on_train_end(self, logs=None):
        print('on_train_end')
