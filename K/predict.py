import os
import matplotlib.pyplot as plt
from utils.cnn import create_cnn2
from utils.load import load_test_data
from utils.models import load_pretrained_models_with_weights
from utils.constant import FILEPATH_TEST

ACTIVATION = 'elu'
LAST_ACTIVATION = 'tanh'

model_path = 'models/ensemble01/'

X_test = load_test_data(FILEPATH_TEST)

def load_models(name, n_output):
    models = []
    i = 0
    while True:
        filepath = '{}_{:02}'.format(name, i)
        model = create_cnn2(n_output, ACTIVATION, LAST_ACTIVATION)
        if os.path.exists(filepath):
            model.load_weights(filepath)
        else:
            break
        models.append(model)
    return models

def plot_models(models):
    for model in models:
        plt.plot(model.history['rmse'])
        plt.plot()



models01 = load_models('cnn2_dataset01', 8)
models02 = load_models('cnn2_dataset02', 22)

print(len(models01), len(models02))
