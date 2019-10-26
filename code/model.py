import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras import backend as K

#import matplotlib.pyplot as plt
#from scipy.stats import norm


### Cross-validation
# Get new training and testing indexes 
def crossValidate(data, kfold, num = 1, seed = 0):
    # data  - pandas DataFrame
    # kfold - int, total number of folds
    # num   - which number fold number to use
    # seed  - for setting the random seed (this should be the same
    #         when iterating through num)

    if num < 1 or num > kfold:
        raise ValueError("num must be between 1 and kfold")

    # Randomize all the indexes
    rand_index = data.sample(frac = 1.0, random_state = seed).index
    n = data.shape[0]

    # Ranges to use
    bounds = [i for i in range(0, n, round(n/kfold))[0:kfold]]
    bounds.append(n)

    # Get new test and train sets
    test_ind = rand_index[bounds[num-1]:bounds[num]]
    train_ind = pd.Index(set(rand_index).difference(set(test_ind)))

    return train_ind, test_ind


### The neural network classifier
def buildNeuralNetwork(x, y):
    # x - the input variables
    # y - the target variable

    # Define the neural network architecture
    inputs = Input(shape=(x.shape[1],))
    layer = Dense(units = 32, activation = 'relu')(inputs)
    layer = Dense(units = 32, activation = 'relu')(layer)
    output = Dense(units = 1, activation = 'linear')(layer)
    model = Model(inputs = inputs, outputs = output)

    # Compile the model
    model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fit the model
    model.fit(x, y, epochs = 100)

    return model


def calcAccuracy(model, x, y):
    # Compute AUC, sensitivity, specificity, confusion matrix
    pass


def plots(model, x, y):
    pass










