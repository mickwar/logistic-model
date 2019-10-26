### Two functions: getTransform() and doTransform()
###
### The training data is passed to getTransform() and both training and test data
### are passed to doTransform(). This will help keep things organized so we know
### exactly what we are doing to the raw data to get it ready for model fitting
###
### Also, with the two separate functions, we are sure to apply the same
### transformation to the test set as to the training set, but the transformation
### is only based on the training set.

import pandas as pd
import numpy as np

# Function for getting transformations
def getTransform(data, min_unique, min_frequency):
    # data          - a pandas data frame
    # min_unique    - float/int, if numerical values has as most min_unique unique values, it becomes categorical
    # min_frequency - float/int, if a class has fewer than min_frequency occurrences, that class gets combined
    #                 with others like it into a new class called "SMALL_GROUP"

    ### Set up
    # Dictionary to be passed to doTransform()
    transform = {}
    transform['numerical'] = {}
    transform['categorical'] = {}

    # Get numerical and categorical columns
    numerical = data.select_dtypes(include = 'number').keys().tolist()
    categorical = data.select_dtypes(include = 'object').keys().tolist()

    # Determine which numerical variables (if any) are actually categorical
    # (Pandas won't pick up on it if numerical values are used in placed of strings)
    tmp_rm = []
    for c in numerical:
        if len(pd.unique(data[c])) <= min_unique:
            tmp_rm.append(c)
            categorical.append(c)

    for t in tmp_rm:
        numerical.remove(t)

    del tmp_rm

    transform['numerical']['columns'] = numerical
    transform['categorical']['columns'] = categorical

    ### Categorical processing
    # Combine low frequency classes into new class called "SMALL_GROUP"
    transform['categorical']['normal'] = {} # Unaffected classes (normal mapping)
    transform['categorical']['small'] = {}  # Low frequency classes (combined to Other)
    transform['categorical']['remove'] = {} # For single-class categories

    for v in categorical:
        # Remove single-class categories
        transform['categorical']['remove'][v] = False
        if len(pd.unique(data[v])) == 1:
            transform['categorical']['remove'][v] = True

        # Get all classes not having `min_frequency` occurrences
        ind = data[v].value_counts() >= min_frequency
        mapping = data[v].value_counts().keys()[ind]
        transform['categorical']['normal'][v] = [m for m in mapping]

        # See if grouping those classes will exceed `min_frequency`
        # If so, these are placed in the 'SMALL_GROUP' class
        # Otherwise, just making an empty list for that key-value pair
        other = data[v].value_counts().keys()[~ind]
        transform['categorical']['small'][v] = []
        if np.sum(data[v].value_counts()[~ind]) >= min_frequency:
            transform['categorical']['small'][v] = [o for o in other]


    ### Numerical processing
    # Numerical variables will be standardized to mean 0 and standard deviation 1.
    # Some might be have the logarithm taken first, if terms and conditions apply.

    # NOTE: Another valid approach is to scale the variables each to [0, 1]
    # In that case, still attempt to do the log-transform with the same check,
    # but store min and max instead
    
    transform['numerical']['log'] = {}
    transform['numerical']['mean'] = {}
    transform['numerical']['sd'] = {}   # If sd == 0, the variable is removed
    transform['numerical']['dummy'] = {}

    # Get mean, standard deviation, and ranges of numerical variables
    # If range covers more than 10 standard deviations and is strictly positive,
    # then take the log transform first
    for v in numerical:
        transform['numerical']['mean'][v] = np.mean(data[v])
        transform['numerical']['sd'][v] = np.mean(data[v])
        # Check range and sd
        transform['numerical']['log'][v] = False
        if np.min(data[v]) > 0 and transform['numerical']['sd'][v] > 0:
            if np.max(data[v]) - np.min(data[v]) > 10 * transform['numerical']['sd'][v]:
                transform['numerical']['log'][v] = True
                # Re-compute mean and sd for the log-transformed variable
                # (still going to do the standardization)
                transform['numerical']['mean'][v] = np.mean(np.log(data[v]))
                transform['numerical']['sd'][v] = np.mean(np.log(data[v]))

        # Check if any NaNs, need to make indicators if so
        transform['numerical']['dummy'][v] = True if any(data[v].isna()) else False

    return transform



# Function for carrying out the transformations
def doTransform(data, transform):

    col_rm = []

    ### Check for extraneous columns
    # If columns are in data which are not in either numerical or categorical,
    # remove those columns
    for v in set(data.keys()).difference(set(transform['numerical']['columns'] + transform['categorical']['columns'])):
        col_rm.append(v)


    ### Numerical processing
    for v in transform['numerical']['columns']:
        if transform['numerical']['sd'] == 0:
            col_rm.append(v)
            continue
            # Don't need to keep processing, move to next iteration

        # Take the log
        if transform['numerical']['log'][v] is True:
            data[v] = np.log(data[v])

        # Set to mean 0, sd 1
        data[v] = (data[v] - transform['numerical']['mean'][v]) / transform['numerical']['sd'][v]

        # Create indicator variables for missing values
        if transform['numerical']['dummy'][v] is True:
            ind = data[v].isna()
            print("{}: Making indicator for missing values".format(v))
            data.loc[ind, v] = 0
            new = pd.DataFrame({"_".join([v, "NaN"]) : (ind) * 1})
            data = pd.concat([data, new], 1)


    ### Categorical processing
    for v in transform['categorical']['columns']:
        if transform['categorical']['remove'][v] is True:
            col_rm.append(v)
            continue

        # Set classes not in either 'normal' or 'small' to NaN
        ind = ~data[v].isin(transform['categorical']['normal'][v] + transform['categorical']['small'][v])
        if sum(ind) > 0:
            print("{}: Setting unseen classes to NaN".format(v))
            data.loc[ind, v] = np.nan

        # Set classes in 'small' to 'SMALL_GROUP'
        if len(transform['categorical']['small'][v]) > 0:
            print("{}: Re-labeling low frequency classes to SMALL_GROUP".format(v))
            for m in transform['categorical']['small'][v]:
                data.loc[data[v] == m, v] = "SMALL_GROUP"

        # Create the indicator variables
        for l in transform['categorical']['normal'][v] + transform['categorical']['small'][v]:
            new = pd.DataFrame({"_".join([v, str(l)]) : (data[v] == l) * 1})
            data = pd.concat([data, new], 1)

        # NOTE: A better option is to use pd.get_dummies(), but I need to make
        # sure that there are still the correct number of columns when creating
        # indicator variables for the test set

        # Remove original column
        data = data.drop(v, 1)


    ### Remove certain columns
    for v in col_rm:
        print("{}: Removing variable".format(v))
        data = data.drop(v, 1)

    return data

