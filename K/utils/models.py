import os
from os import listdir
from os.path import isfile, join, dirname, abspath
from keras.models import model_from_yaml
import keras.backend as K

def load_models_with_weights(model_name):
    models = []
    submodel_dirs = []
    parent_path = dirname(dirname(abspath(__file__)))
    for item in listdir('{}/models/{}'.format(parent_path, model_name)):
        if not item.startswith('.') and os.path.isdir('{}/models/{}'.format(parent_path, model_name)):
            submodel_dirs.append(item)
    for submodel in submodel_dirs:
        path = '{}/models/{}/{}'.format(parent_path, model_name, submodel)
        weights = [f for f in listdir(path) if f.endswith('.h5')]

        submodels = []
        for idx, w in enumerate(weights):
            w = '{}/models/{}/{}/{}'.format(parent_path, model_name, submodel, w)
            model_path = '{}/models/{}/{}/{}.yaml'.format(parent_path, model_name, submodel, submodel)
            yaml_file = open(model_path, 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            model = model_from_yaml(loaded_model_yaml)
            model.load_weights(w)
            submodels.append(model)
        models.append(submodels)
    return models

def save_model(dir, modelname, model):
    model.save(dir + '/' + modelname + '.model')
    model_yaml = model.to_yaml()
    with open(dir + '/' + "{}.yaml".format(modelname), "w") as yaml_file:
        yaml_file.write(model_yaml)


def reset_model(model):
    session = K.get_session()
    for layer in model.layers: 
        for v in layer.__dict__:
            v_arg = getattr(layer,v)
            if hasattr(v_arg,'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
                print('reinitializing layer {}.{}\n'.format(layer.name, v))