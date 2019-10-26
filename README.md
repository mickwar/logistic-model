# Application Processing

## Introduction

The goal of this project is predict a binary variable called `TARGET` as accurately as possible.

Given (1) that accuracy is our goal and (2) we have a sizable data set (in number of rows and columns) we will use a binary classifier neural network for our model.

The code is split into several files, which are tied together by the [`code/full.py`](code/full.py) script.

## Data summary

Code for data exploration can be found in [`code/explore.py`](code/explore.py).

Training data consists of 125000 rows with 122 columns (about 67 Mb), including `ID` and `TARGET`.
The objective is to make predictions for the binary variable called `TARGET`.

The test data consists of 48744 rows with 121 columns (`ID` is included, `TARGET` is omitted).

The input data consists of 120 columns (no `TARGET` and `ID` columns) of which
- 76 are numerical variables
- 44 are categorical

Even though about two-thirds of the categorical variables contain numbers, they need to be treated as categorical.

Of the 125000 rows in the training data
- Only 3461 contain no missing values.
- Most rows have between 5 and 49 missing values.
- The largest number of missing values in a single row is 61.

Expanding the data set with indicator variables (for missing values and multi-class categorical variables) left us with 312 columns.
After the variable selection we had 235 columns, most of which were from the indicator variables.
Details for this expansion and contraction are given in the next section.

There were no major issues---such as typos---found in the data which warranted extensive cleaning.

## Data processing

Data processing takes place in two steps:
1. Modify the numerical and categorical variables. Found in [`code/transformations.py`](code/transformations.py).
2. Do a variable selection. Found in [`code/variable_selection.py`](code/variable_selection.py).

Since there are so many variables, it would be too time consuming to go through each one and decide which transformation (whether scaling or one-hot encoding) needs to be done.
So, based on a set of criteria, we automatically determine the transformation to apply.

Similarly, we want to automatically determine which variables to keep and which to discard.

### Numerical variables

Every numerical variable (either discrete or continuous) is scaled to have mean 0 and standard deviation 1.

We then check the following:
- If the standard deviation of a variable was 0, then that variable was removed.
Such a variable adds no value.
- If a strictly positive variable has a range (i.e. difference between the maximum and the minimum) which spans at least 10 standard deviations, then the logarithm is applied first.
This will help deal with long-tailed distributions.
- If there are missing values, then an indicator (dummy) variable is created.
In the dummy variable, a 1 indicates missing and a 0 indicates not missing.
The missing values in the original variable are set to 0.

In this data set, many of the numerical variables had at least a handful of missing values.

### Categorical variables

For categorical variables, we need to make sure there aren't too many unique classes and that each class has a sufficient number of instances to justify fitting a model.

For each variable, we do the following:
- Classes having fewer than 500 occurrences (arbitrarily chosen) were re-labeled into a new class called `SMALL_GROUP`.
- If the combined class of `SMALL_GROUP` is still below 500, then all those classes are set to `NaN`, the place holder for missing values.
- Indicator variables are created for each class in the variable where a `1` means class presence and `0` means no class presence.
- If a class is `NaN`, then a `0` is found for each indicator variable, indicating no classes.

### Variable Selection

We do a rough variable selection using importance measures from an XGBoost model:
- Fit the model to the transformed training data
- Calculate importance scores
- Sort the scores in descending order
- Take a cumulative sum of the scores and divide them all by the total
- Keep the variables which contribute to 95% of the total importance

This might not be the most theoretically sound, but it's a quick, automatic way of eliminating unnecessary variables.

## Model

Model architecture and accuracy calculations are found in [`code/model.py`](code/model.py).

The model is a feed-forward neural network with binary cross-entropy for our loss function.
We use two hidden layers with 32 neurons each.
Each hidden layer uses a RELU activation function.
The output layer uses a sigmoid activation to produce a probability.

We do `K=10` fold cross-validation to make sure we aren't overfitting.

At 10 epochs, models took about two minutes to fit.

## Results

For each fold in the cross-validation, we see similar calculations for accuracy, AUC, optimal threshold, and positive predictive value.
(See comments in `code/full.py` to details.)
This suggests we are not overfitting to the training data.

We convert the predicted probabilities to binary variables by checking whether the probability exceeds some threshold.
The optimal threshold is calculated using the ROC curve and is determined to be the value which maximizes the true negative rate times the true positive rate.
Each k-fold taken together would suggest we use a threshold of 0.09 to maximize this quantity for future predictions.

Our baseline positive predictive value is 8.1%.
Meaning if we predict every row to be a 1, we would be right about 8.1% of the time (assuming our test set is comparable enough to the training set).

The neural network had an average positive predictive value of about 15.9%, a substantial improvement over the baseline.

Predictions are found under `predictions/`.

## Using the virtual environment

Initialize the Python virtual environment in the command line with

```bash
virtualenv env --python=python3.6   # Creates the environment with a Python 3.6 binary
source env/bin/activate             # Activates the environment
pip install -r requirements.txt     # Installs the packages found in requirements.txt
```
This will install packages such as Keras, Numpy, and Pandas into a self-contained environment, avoiding any conflicts with globally installed packages.
The environment can also be safely deleted by removing the `env` directory.

When doing any work with the code, be sure to run
```bash
source env/bin/activate
```
so that the necessary packages will be available.
This enables the virtual environment.

Exit the virtual environment with
```bash
deactivate
```
