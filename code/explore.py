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

### An in-depth look at the categorical variables
# Things to look out for:
#   Categories with very few frequencies -> combine if possible, or may need to remove rows
#   Too many categories -> combine several small categories
#   Missing values -> convert to "Other"/"None" category (i.e. a zero in all relevant one-hot encoded columns)
#   Heavily lopsided categories -> could skew the model to always predict 0 or 1 when the category is present
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
#       There are also about 500 nans to make into "None" category
#   NAME_INCOME_TYPE: several low frequency categories adding up to 19 occurrences.
#       Remove these categories, treat as "None"/"Other"
#   NAME_EDUCATION_TYPE: Heavily lopsided on low frequency category
#       All 70 Acedemic degrees had target 0. Removing category for now,
#       treating as "None"/"Other" (if it looks like a problem for the model,
#       consider removing the rows altogether)
#   NAME_FAMILY_STATUS
#       Remove single "Unknown" row
#   OCCUPATION_TYPE
#       Lots of categories and about 40000 missing
#       Safe to combine low frequency categories into "Other"
#   ORGANIZATION_TYPE
#       Similar deal with OCCUPATION_TYPE, but first deal with all
#       those "Type X"s. (e.g. there are 13 different industry types, but not
#       all have low frequency). Maybe still just combine low frequencies?
#   FONDKAPREMONT_MODE: lots of nans
#   HOUSETYPE_MODE: about half nans
#   WALLSMATERIAL_MODE: about half nans
#   EMERGENCYSTATE_MODE: about half nans

### One-hot encoding
# Dictionary for mapping category values
def get_categorical_map(data, min_frequency, columns = None):

    # The out['normal'] dictionary determines which classes in each variable
    # should be mapped to a dummy variable.
    # The out['small'] dictionary determines which classes should be re-labeled
    # as 'SMALL_GROUP'
    # If a class is not found in either, then nothing happens (that class is
    # effectively in a 'None' category
    out = {}
    out['normal'] = {}  # Unaffected categories (normal mapping)
    out['small'] = {}   # Low frequency categories (combined to Other)

    # Try to get categorical columns
    if columns is None:
        columns = data.select_dtypes(include = 'object').keys()

    for c in columns:
        # Get all classes not having `min_frequency` occurrences
        ind = data[c].value_counts() >= min_frequency
        mapping = data[c].value_counts().keys()[ind]
        out['normal'][c] = [m for m in mapping]

        # See if grouping those classes will exceed `min_frequency`
        # If so, these are placed in the 'other' category
        # Otherwise, just making an empty list for that key-value pair
        other = data[c].value_counts().keys()[~ind]
        if np.sum(data[c].value_counts()[~ind]) >= min_frequency:
            out['small'][c] = [o for o in other]
        else:
            out['small'][c] = []

    return out


# Wrapper for using pandas' get_dummies method
def make_dummy_variables(data, mapping, columns = None):

    # Try to get categorical columns
    if columns is None:
        columns = data.select_dtypes(include = 'object').keys()

    for c in columns:
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

mapping = get_categorical_map(train, 500)

train = make_dummy_variables(train, mapping)
test = make_dummy_variables(test, mapping)

# Probably should do the same thing during the cross-validation.




# Find which columns can straight up be removed

# Standardize the numeric variables to mean 0 and sd 1.
# May need to take the log of certain variables


# Remove the ID column

# Separate the TARGET varible

# Need to check whether test set contains categories not found in
# the training set

