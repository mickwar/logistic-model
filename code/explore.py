# Import packages
import pandas as pd
import numpy as np

# Read in the data
train = pd.read_csv("../data/application_train.csv")
test = pd.read_csv("../data/application_test.csv")

# Get numerical and categorical columns
columns_numeric = train.select_dtypes(include = 'number').keys()
columns_categorical = train.select_dtypes(include = 'object').keys()

# Count missing values
train.isna().sum(1).describe()      # Summary of nans
np.sum(train.isna().sum(1) == 0)    # Number of zero nan rows

# Find which columns can straight up be removed
# Categorical variables first
for c in columns_categorical:
    print("--- Summary for {} ---".format(c))
    print(train[c].describe())
    print("--- Frequencies ---")
    print(train[c].value_counts())
    print("--- Cross-tabulation ---")
    print(pd.crosstab(train[c], train['TARGET']))
    print()

# Notes:
#   CODE_GENDER: a single "XNA" category (remove row)
#   NAME_TYPE_SUITE: low frequency categories
#       "Group of people" ~ 100
#       "Other_A" ~ 300
#       "Other_B" ~ 700
#       (Combine these categories)


# Standardize the numeric variables to mean 0 and sd 1.
# May need to take the log of certain variables

# One-hot encode the categorical variables
# Look at frequncies of the categories and determine if the data
# is sufficient to model with (i.e. not too many categories)

# Remove the ID column

# Separate the TARGET varible

# Need to check whether test set contains categories not found in
# the training set

