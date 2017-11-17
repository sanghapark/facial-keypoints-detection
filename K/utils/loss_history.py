import pandas as pd
from keras.callbacks import Callback

class LossHistory(Callback):

    def __init__(self):
        self.losses = pd.DataFrame(columns=['loss', 'val_loss', 'rmse', 'val_rmse'])

    def on_epoch_end(self, epoch, logs=None):
        cols = ['loss', 'val_loss', 'rmse', 'val_rmse']
        metrics = [[
            logs.get('loss'),
            logs.get('val_loss'),
            logs.get('rmse'),
            logs.get('val_rmse')
        ]]
        self.losses.append(pd.DataFrame(metrics, columns=cols))