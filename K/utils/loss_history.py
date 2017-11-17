import pandas as pd
from keras.callbacks import Callback

class LossHistory(Callback):

    def on_training_begin(self, logs={}):
        self.losses = pd.DataFrame(columns=['loss', 'val_loss', 'rmse', 'val_rmse'])

    def on_batch_end(self, batch, logs=None):
        cols = ['loss', 'val_loss', 'rmse', 'val_rmse']
        metrics = [
            logs.get('loss'),
            logs.get('val_loss'),
            logs.get('rmse'),
            logs.get('val_rmse')
        ]
        self.losses.append(pd.DataFrame(metrics, columns=cols))
