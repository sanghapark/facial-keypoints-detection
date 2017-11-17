import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import model_from_yaml

def load_models_with_weights(model_name):
    path = './models/{}/weights'.format(model_name)
    weights = [join(path, w) for w in listdir(path) if isfile(join(path, w))]
    models = []
    for idx, w in enumerate(weights):
        model_path = './models/{}/{}.yaml'.format(model_name, model_name)
        yaml_file = open(model_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        model.load_weights(w)
        models.append(model)
    return models

models01 = load_models_with_weights('cnn2_dataset01')
models02 = load_models_with_weights('cnn2_dataset02')