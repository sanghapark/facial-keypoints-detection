from os import listdir
from os.path import isfile, join
from keras.models import model_from_yaml
import keras.backend as K

def load_models_with_weights(model_name, submodel_name):
    path = './models/{}/'.format(model_name)
    weights = [join(path, w) for w in listdir(path) if isfile(join(path, w) and w.endswith('.h5'))]
    models = []
    for idx, w in enumerate(weights):
        model_path = './models/{}/{}.yaml'.format(model_name, submodel_name)
        yaml_file = open(model_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        model.load_weights(w)
        models.append(model)
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