from os import listdir
from os.path import isfile, join
from keras.models import model_from_yaml

def load_pretrained_models_with_weights(model_name):
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

def save_pretrained_model(dir, modelname, model):
    model.save(dir + '/' + modelname + '.model')
    model_yaml = model.to_yaml()
    with open(dir + '/' + "{}.yaml".format(modelname), "w") as yaml_file:
        yaml_file.write(model_yaml)
