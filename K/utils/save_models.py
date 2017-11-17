from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from utils.cnn import create_cnn2
from utils.metric import rmse


model = create_cnn2(8, 'elu', 'tanh')
optimizer = RMSprop(0.001, 0.9, 1e-8, 0)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])
model.save('cnn2_dataset01.model')

model_yaml = model.to_yaml()
with open("cnn2_dataset01.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)


model = create_cnn2(22, 'elu', 'tanh')
optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])
model.save('cnn2_dataset02.model')


model_yaml = model.to_yaml()
with open("cnn2_dataset02.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)