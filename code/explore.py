# Import packages
import pandas as pd
import numpy as np

# Read in the data
train = pd.read_csv("../data/application_train.csv")
test = pd.read_csv("../data/application_test.csv")

# Get numerical and categorical columns
columns_numerical = train.select_dtypes(include = 'number').keys().tolist()
columns_numerical = columns_numerical[2:]   # Remove ID and TARGET

columns_categorical = train.select_dtypes(include = 'object').keys().tolist()

# Anything with FLAG or RATING needs to be categorical, not numerical
tmp_rm = []
for c in columns_numerical:
    if c.find('FLAG') >= 0 or c.find('RATING') >= 0:
        tmp_rm.append(c)
        columns_categorical.append(c)

for t in tmp_rm:
    columns_numerical.remove(t)

del tmp_rm



# Count missing values
train.isna().sum(1).describe()      # Summary of nans
np.sum(train.isna().sum(1) == 0)    # Number of zero nan rows

### An in-depth look at the categorical variables
# Things to look out for:
#   Classes with very few frequencies -> combine if possible, or may need to remove rows
#   Too many classes -> combine several small classes
#   Missing values -> a zero in all relevant one-hot encoded columns
#   Heavily lopsided classes -> could skew the model to always predict 0 or 1 when the class is present
def summarize_categorical_variables(data, columns = None):
    # Try to get categorical columns
    if columns is None:
        columns = data.select_dtypes(include = 'object').keys().tolist()

    for c in columns:
        print("--- Summary for {} ---".format(c))
        print(data[c].describe())
        print("--- Frequencies ---")
        print(data[c].value_counts())
        
        try:
            data['TARGET']
            print("--- Cross-tabulation ---")
            print(pd.crosstab(data[c], data['TARGET']))
        except:
            pass

        print()

summarize_categorical_variables(train)
#summarize_categorical_variables(test)


### Function to help create the one-hot encoding transformation
# Dictionary for mapping category values
def get_categorical_map(data, min_frequency, columns = None):

    # The out['normal'] dictionary determines which classes in each variable
    # should be mapped to a dummy variable.
    # The out['small'] dictionary determines which classes should be re-labeled
    # as 'SMALL_GROUP'
    # If a class is not found in either, then nothing happens (that class is
    # effectively in a 'None' class
    out = {}
    out['normal'] = {}  # Unaffected classes (normal mapping)
    out['small'] = {}   # Low frequency classes (combined to Other)
    out['remove'] = {}  # For single-class categories

    # Try to get categorical columns
    if columns is None:
        columns = data.select_dtypes(include = 'object').keys().tolist()

    for c in columns:
        # Remove single-class categories
        out['remove'][c] = False
        try:
            if len(np.unique(data[c])) == 1:
                out['remove'][c] = True
        except:
            pass

        # Get all classes not having `min_frequency` occurrences
        ind = data[c].value_counts() >= min_frequency
        mapping = data[c].value_counts().keys()[ind]
        out['normal'][c] = [m for m in mapping]

        # See if grouping those classes will exceed `min_frequency`
        # If so, these are placed in the 'SMALL_GROUP' class
        # Otherwise, just making an empty list for that key-value pair
        other = data[c].value_counts().keys()[~ind]
        if np.sum(data[c].value_counts()[~ind]) >= min_frequency:
            out['small'][c] = [o for o in other]
        else:
            out['small'][c] = []

    return out


### Wrapper for using Pandas' get_dummies method
# This does the transformation for the categorical variables
def make_dummy_variables(data, mapping, columns = None):

    # Try to get categorical columns
    if columns is None:
        columns = data.select_dtypes(include = 'object').keys().tolist()
    else:
        columns = columns.copy()

    for c in columns:
        if mapping['remove'][c] is True:
            data = data.drop(c, 1)
            columns.remove(c)
            continue
            # Need to continue, else we'll get errors

        # Set classes not in either 'normal' or 'small' to NaN
        data.loc[~data[c].isin(mapping['normal'][c] + mapping['small'][c]), c] = np.nan

        # Set classes in 'small' to 'SMALL_GROUP'
        for m in mapping['small'][c]:
            data.loc[data[c] == m, c] = "SMALL_GROUP"

    return pd.get_dummies(data, columns = columns)


# The reason we do this weird one-hot encoding stuff is because we need to do
# the same transformation to the test set as we do to the training set. Some
# classes in the test set may not be present, so these should be appropriately
# assigned to no dummy variable. Otherwise, we might be training on data we
# haven't seen.
#
# We also want to make sure we have enough observations in each category so our
# model training doesn't break. Too few occurrences of a class could lead to
# singularity problems. I arbitrarily chose the minimum class size to be 500.

categorical_mapping = get_categorical_map(train, 500, columns_categorical)

train = make_dummy_variables(train, categorical_mapping, columns_categorical)
test = make_dummy_variables(test, categorical_mapping, columns_categorical)


### A look at the numerical variables

# Check if we should do a log-transform first:
#   Minimum value is greater than 0
#   Maximum value is more than Mean + 5 * SD (or something)
# Then always standardize


# Look at correlations between them, make dummy variables for the NaNs (still
# keep an eye on frequency)

# Find which columns can straight up be removed

# Standardize the numeric variables to mean 0 and sd 1.
# May need to take the log of certain variables


# Remove the ID column

# Separate the TARGET varible

# Need to check whether test set contains categories not found in
# the training set

# AUC

# Predictive accuracy (sensitivity, specificity, true/false positive, true/false negative)
# Confusion matrix
