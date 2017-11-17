import pandas as pd
from keras.callbacks import Callback

class LossHistory(Callback):

    def __init__(self, modelfile):
        self.file = open(modelfile, 'a+')
        self.file.write(','.join(['epoch', 'loss', 'val_loss', 'rmse', 'val_rmse'])+'\n')

    def __del__(self):
        self.file.close()

    def on_epoch_end(self, epoch, logs=None):
        row = "{},{},{},{},{}\n".format(epoch, logs.get('loss'), logs.get('val_loss'), logs.get('rmse'), logs.get('val_rmse'))
        self.file.write(row)
