import pandas as pd
from keras.callbacks import Callback

class LossHistory(Callback):

    def __init__(self, modelfile):
        self.modelfile = modelfile
        file = open(modelfile, 'a+')
        file.write(','.join(['epoch', 'loss', 'val_loss', 'rmse', 'val_rmse']))

    def on_epoch_end(self, epoch, logs=None):
        file = open(self.modelfile, 'a+')
        row = "{},{},{},{},{}".format(epoch, logs.get('loss'), logs.get('val_loss'), logs.get('rmse'), logs.get('val_rmse'))
        file.write(row)
