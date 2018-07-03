from __future__ import print_function

import numpy as np

import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.layers import Input, Conv2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.applications.vgg16 import VGG16
from keras import backend as K

from hyperopt import Trials, STATUS_OK, tpe, STATUS_FAIL
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from data_generator import DataGenerator, names, frames, validation_set, training_set

import sys

model=load_model('best_model.h5')

test_files = validation_set #[0]
batch_size = 128
for i in test_files:
    idx_test = []
    for j in range(0,frames[i]):
        idx_test.append([i, j])
    print(names[i])
    test_generator = DataGenerator(idx_test,batch_size,params = [])
    test_steps = np.ceil(len(idx_test)/batch_size)
    score = model.evaluate_generator(test_generator.get_generator()(),
        steps = test_steps)
    print(score)
