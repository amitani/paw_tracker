from __future__ import print_function

import numpy as np

import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Conv2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.applications.vgg16 import VGG16
from keras import backend as K

from hyperopt import Trials, STATUS_OK, tpe, STATUS_FAIL
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from data_generator import DataGenerator, frames, validation_set, training_set

import sys

def acc15(y_true, y_pred):
    return K.mean(K.less(K.sum(K.square(y_pred - y_true),axis=1), 225), axis=-1)

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """
    idx_train = []
    idx_test = []

    test_files = validation_set #[0]
    train_files = training_set #[i for i in range(0,len(frames)) if i not in test_files]
    #test_files = [3]
    #train_files = [4]

    print(test_files)
    print(train_files)
    #train_files = [1]
    for i in train_files:
        for j in range(0,frames[i]):
            idx_train.append([i, j])
    for i in test_files:
        for j in range(0,frames[i]):
            idx_test.append([i, j])
    print('prepared data')
    print(len(idx_train))
    print(len(idx_test))
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

    def print_num_parameters(model):
        trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))

    vgg = VGG16(False, input_shape = (144,288,3))
    pop_layers = 2 # {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}}
    trainable_layers = 2 #{{choice([0, 1, 2, 3, 4])}}
    for i in range(0,pop_layers):
        vgg.layers.pop()
    if trainable_layers>0:
        for layer in vgg.layers[0:-trainable_layers]:
            layer.trainable = False
    else:
        for layer in vgg.layers:
            layer.trainable = False
    print_num_parameters(vgg)
    x = vgg.layers[-1].output
    inputs = vgg.layers[0].input
    #    inputs = Input(shape=(144,288,3))
    #    x = inputs
    #    x = vgg(x)
    #    for layer in vgg.layers:
    #        x = layer(x)

    conv_dim = 64#{{choice([16, 32, 64])}}
    x = Conv2D(conv_dim, (1,1))(x)
    x = Flatten()(x)
    fc_dim = 32#{{choice([32, 64, 128, 256])}}
    fc_layers = 2#{{choice([1,2,3])}}
    for l in range(0,fc_layers):
        x = Dense(fc_dim)(x)
    x = Dense(2,activation = 'linear')(x)
    model = Model(input=inputs,output=x)
    print_num_parameters(model)

    lr = 1E-4#{{choice([1E-4, 5E-5, 2E-5, 1E-5, 5E-6])}}
    optim = Adam(lr=lr)#{{choice([RMSprop, Adam, SGD])}}(lr = lr);
    model.compile(loss='mse', metrics=[acc15], optimizer=optim)
    model.summary()

    batch_size=32#{{choice([32, 64, 128])}}
    to_aug = True#{{choice([True, False])}}
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
    n_epoch = 20#5#{{choice([2,3,4,5])}}

    for i in range(0,n_epoch):
        h = model.fit_generator(train_generator.get_generator()(),
                steps_per_epoch = train_steps,
                epochs=1, verbose=1)
        score = model.evaluate_generator(test_generator.get_generator()(),
                steps = test_steps)
        print(score)

    try:
        loss = h.history['loss'][-1]
    except:
        loss = 2001
    if np.isnan(score):
        score = 2000
    print({'train_loss': loss, 'test_score': score})

    if score >= 2000:
        return  score, STATUS_FAIL, model
    else:
        return  score, STATUS_OK, model

if __name__ == '__main__':
    idx_train, idx_test = data()
    score, status, best_model = create_model(idx_train, idx_test)
    best_model.summary()
    print('valid_loss:', score)
    best_model.save('best_model_180701.h5')
