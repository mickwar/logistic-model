import pandas as pd
import numpy as np

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
        # Basic summary statistics (count, uniq, most frequent)
        print("--- Summary for {} ---".format(c))
        print(data[c].describe())
        print("--- Frequencies ---")
        print(data[c].value_counts())
        
        # Cross-tabulation between variable and TARGET
        try:
            data['TARGET']
            print("--- Cross-tabulation ---")
            print(pd.crosstab(data[c], data['TARGET']))
        except:
            pass

        print()


### A description for the numerical variables
def summarize_numerical_variables(data, columns = None):
    # Try to get numerical columns
    if columns is None:
        columns = data.select_dtypes(include = 'number').keys().tolist()

    for c in columns:
        # Mean, std, some quantiles, number of missing
        print("--- Summary for {} ---".format(c))
        print(data[c].describe())
        print()



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


# Loop through each variable and print a summary
# Look for anything weird like typos, outliers
summarize_categorical_variables(train)
summarize_numerical_variables(train)


#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.pairplot(dataset, hue='Class')
#sns.heatmap(dataset.corr(), annot=True)

# Variable selection?
# Should have used xgboost to get importance measures
