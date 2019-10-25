# Application Processing

## Introduction

The goal of this project is predict a binary variable called `TARGET` as accurately as possible.

Given (1) that accuracy is our goal and (2) we have a sizable data set (in number of rows and columns) we will use a binary classifier neural network for our model.

## Data summary

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

## Data processing

The data need to be prepared to be used in a model.
This means that categorical variables need to be one-hot encoded and numerical variables may need to be transformed.
There may also be the need to remove certain rows from the data set.

Since there are so many variables, it would be too time consuming to go through each one and decide which transformation (whether scaling or one-hot encoding) needs to be done.
So, based on a set of criteria, we automatically determine the transformation to apply.

This method has the disadvantage of retaining useless variables (i.e. they don't contribute to the overall prediction), but with a fast enough model that point is moot.
It sure beats cherry-picking every variable.

The goal here is to create a transformation on the training set that would allow us to safely fit a model.

Since the test set needs to be treated as though we have not seen it, the transformations we use must be based solely on the training set.
This presents a number of problems, especially with the categorical variables:
- Does the test set contain classes not seen in the training set?
- Does the test set contain an instance of every class in the training set?
- How should missing values be treated?

We need to ensure that both training and test sets have the same number of columns post-transformation.

### Numerical variables

Every numerical variable (either discrete or continuous) is scaled to have mean 0 and standard deviation 1.

If the standard deviation of a variable was 0, then that variable was removed.
Such a variable adds no value.

If a strictly positive variable has a range (i.e. difference between the maximum and the minimum) which spans at least 10 standard deviations, then the logarithm is applied first.
This will help deal with long-tailed distributions.

If there are missing values, then an indicator (dummy) variable is created.
In the dummy variable, a 1 indicates missing and a 0 indicates not missing.
The missing values in the original variable are set to 0.

In this data set, many of the categorical variables had at least a handful of missing values.

### Categorical variables

Computational problems with fitting the model may arise if a categorical variable has classes which appear infrequently.
That is, there may be be singularities in our data, possibly leaving us (wrongly) with near zero standard errors.
Accordingly, we may unknowingly underestimate or overestimate our target probability.

Another issue that may arise is if a categorical variable is heavily skewed toward `TARGET == 0` or `TARGET == 1` when the class frequency is too low.
This didn't appear to be a major issue in the training set.

To counter these issues, classes having fewer than 500 occurrences were re-labeled into a new class called `SMALL_GROUP`.
If the combined class of `SMALL_GROUP` is still below 500, then all those classes are set to `NaN`, the place holder for missing values.

From here, indicator variables are created for each class in the variable where a `1` means class presence and `0` means no class presence.
If a class is `NaN`, then a `0` is found for each indicator variable, inidicating no classes.

## Procedure

Cross-validation on the training set.
Split the training set into `k=10` subsets.
For each subset, fit a model to the nine other subsets and make predictions on the held-out subset.
The predictions can be compared with the actual `TARGET` so care can be taken to not overfit the model.

## Model

## Results

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
