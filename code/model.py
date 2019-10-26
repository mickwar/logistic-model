### crossValidate()
### Randomly splits the training data into new "train" and "test" sets
###
### buildNeuralNetwork()
### Model set up
###
### calcAccuracy()
### Computes accuracy, AUC, and other statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras import backend as K

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
def buildNeuralNetwork(x, y, epochs = 10):
    # x - the input variables
    # y - the target variable

    # Define the neural network architecture
    inputs = Input(shape=(x.shape[1],))
    layer = Dense(units = 32, activation = 'relu')(inputs)
    layer = Dense(units = 32, activation = 'relu')(layer)
    output = Dense(units = 1, activation = 'sigmoid')(layer)
    model = Model(inputs = inputs, outputs = output)

    # Compile the model
    model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fit the model
    model.fit(x, y, epochs = epochs)

    return model


def calcAccuracy(y, pred_y):
    # Compute AUC, sensitivity, specificity
    # Determine optimal threshold

    threshold = np.arange(0.001, 0.999, 0.001)
    tpr = [0] * len(threshold)
    tnr = [0] * len(threshold)
    ppv = [0] * len(threshold)
    npv = [0] * len(threshold)
    for i in range(len(threshold)):
        new_y = np.squeeze((pred_y > threshold[i]) * 1)
        tpr[i] = np.mean(new_y[y == 1] == 1)   # True positive rate (sensitivity, recall)
        tnr[i] = np.mean(new_y[y == 0] == 0)   # True negative rate (specificity)
        ppv[i] = np.mean(y[new_y == 1] == 1)   # Positive predictive value (precision)
        npv[i] = np.mean(y[new_y == 0] == 0)   # Negative predictive value

    # AUC
    fpr = [1 - x for x in tnr]  # false positive rate
    auc = 0
    for i in range(len(threshold) - 1):
        auc += ((tpr[i+1] + tpr[i]) / 2) * (fpr[i] - fpr[i+1])


    # Optimal threshold (based off point along ROC curve which is furthest
    # from the bottom right corner)
    area = [0] * len(threshold)
    for i in range(len(threshold)):
        area[i] = (1-fpr[i]) * tpr[i]

    index = np.where(area == np.max(area))[0][0]

    # Another optimization approach might be to look at the cost of making
    # errors in our predictions. This could be done by minimizing a function
    # such as:
    #   f(t) = FPR(t) * C1 + FNR(t) * C2
    # where FPR is the false positive rate, FNR is the false negative rate
    # and C1 and C2 are costs associated with making each of those decisions.
    # If false positive costs $100 and a false negative costs $50, then we
    # might do:
    #costs = [100, 50]
    #optim = [0]*len(threshold)
    #for i in range(len(threshold)):
    #    optim[i] = (1-tpr[i]) * costs[0] + (1-tnr[i]) * costs[1]
    #
    #index = np.where(area == np.min(optim))[0][0]

    optimal_threshold = threshold[index]

    # Accuracy
    P = sum(y == 1)
    N = sum(y == 0)
    acc = (tpr[index] * P  + tnr[index] * N) / (P + N)

    # F-1 score (harmonic mean of precision and sensitivity)
    F1 = 2 * ppv[index] * tpr[index] / (ppv[index] + tpr[index])

    return (acc, auc, F1, optimal_threshold, \
        tpr[index], tnr[index], ppv[index], npv[index])

