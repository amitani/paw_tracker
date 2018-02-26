from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Conv2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.applications.vgg16 import VGG16

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from data_generator import DataGenerator, frames

import numpy as np

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    idx_train = []
    idx_test = []

    test_files = [0]
    #train_files = [i for i in range(0,len(frames)) if i not in test_files]
    train_files = [1]
    for i in train_files:
        for j in range(0,frames[i]):
            idx_train.append([i, j])
    for i in test_files:
        for j in range(0,frames[i]):
            idx_test.append([i, j])
    print('prepared data')
    return idx_train, idx_test

def create_model(idx_train, idx_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    vgg = VGG16(False)
    pop_layers = {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}}
    trainable_layers = {{choice([0, 1, 2, 3, 4, 5])}}
    #pop_layers = {{choice([5])}}
    #trainable_layers = {{choice([2])}}
    print(pop_layers)
    print(trainable_layers)
    for i in range(0,pop_layers):
        vgg.layers.pop()
    for layer in vgg.layers[0:-1-trainable_layers]:
        layer.trainable = False
    vgg.summary()

    inputs = Input(shape=(144,288,3))
    x = inputs
    x = vgg(x)
    x = Conv2D(16, (1,1))(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dense(64)(x)
    x = Dense(2,activation = 'linear')(x)
    model = Model(input=inputs,output=x)
    model.summary()

    lr = {{choice([1E-4, 5E-5, 2E-5, 1E-5, 5E-6, 2E-6, 1E-6])}}
    #lr_e = {{uniform(-4,-4)}}
    print(lr)
    optim = {{choice([RMSprop, Adam, SGD])}}(lr = lr);
    #optim = {{choice([RMSprop])}}(lr = 10**lr_e);
    print(optim)
    model.compile(loss='mse', metrics=[],
                  optimizer=optim)

    batch_size={{choice([32, 64, 128])}}
    #batch_size={{choice([128])}}
    to_aug = {{choice([True, False])}}
    if to_aug:
        aug_param = {'angle':10,'scale':0.05,'aspect':0.05,'shift':0.05}
    else:
        aug_param = []
    train_generator = DataGenerator(idx_train,batch_size,
        params = aug_param)
    test_generator = DataGenerator(idx_test,batch_size,
        params = [])
    train_steps = np.ceil(len(idx_train)/batch_size)
    test_steps = np.ceil(len(idx_test)/batch_size)
    model.fit_generator(train_generator.get_generator()(),
            steps_per_epoch = train_steps,
            epochs=10, verbose=1)
    score = model.evaluate_generator(test_generator.get_generator()(),
            steps = test_steps)
    print('Test score:', score)
    return {'loss': score, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials())
    best_model.summary()
    print('Best run:', best_run)
