from transformations import *
from model import *
from variable_selection import *


### Read in the data
train = pd.read_csv("../data/application_train.csv")
test = pd.read_csv("../data/application_test.csv")

### Remove IDs and TARGET columns
train_id = train['SK_ID_CURR']
target = train['TARGET']
train = train.drop('SK_ID_CURR', 1)
train = train.drop('TARGET', 1)

test_id = test['SK_ID_CURR']
test = test.drop('SK_ID_CURR', 1)

### Make the transformations
# This includes the one-hot encoding for categorical variables
# and standardization for numeric variables
transform = getTransform(train, 3, 500)

# Transformed data
train = doTransform(train, transform)
test = doTransform(test, transform)


### Do the variable selection
# NOTE: Ideally, this should be part of the transformation. Something to think
# about for future projects
var_keep = doVariableSelection(train, target, alpha = 0.95, num_round = 20)
joined_var_keep = '\t'.join(var_keep)

# Get which variables to keep
# Need to work with the original variable names and also keep every column
# for the categorical variable, even if only parts were found to be important
# enough to keep.
new_numerical = [v for v in transform['numerical']['columns'] if v in joined_var_keep]
new_categorical = [v for v in transform['categorical']['columns'] if v in joined_var_keep]
new_variables = new_numerical + new_categorical

keep = []
for k in train.keys():
    for v in new_variables:
        if v in k:
            keep.append(k)

# Drop the unimportant columns
train = train.loc[:, keep]
test = test.loc[:, keep]


### Do the cross-validation
# This is to make sure we're not drastically overfitting to the training data.
# Things should be fine (i.e. to fit to the full data set) if such values as
# accuracy, AUC, and optimal threshold are similar across the k-folds.
#
# The cross-validation can also help tune the model so we pick the right number
# of layers and neurons.

kfold = 10
acc, auc, F1, optimal_threshold, tpr, tnr, ppv, npv = [[0] * kfold for _ in range(8)]

for num in range(1, kfold + 1):
    new_train, new_test = crossValidate(train, kfold, num, 0)

    # Split training set into new "train" and "test" sets
    cv_train_x = train.loc[new_train]
    cv_train_y = target.loc[new_train]
    cv_test_x = train.loc[new_test]
    cv_test_y = target.loc[new_test]

    # Fit the model
    model = buildNeuralNetwork(cv_train_x, cv_train_y, epochs = 10)

    # Predict on the hold out sample
    pred_y = model.predict(cv_test_x)

    # Compute some statistics
    acc[num-1], auc[num-1], F1[num-1], optimal_threshold[num-1], \
        tpr[num-1], tnr[num-1], ppv[num-1], npv[num-1] = calcAccuracy(cv_test_y, pred_y)



### Baselines
np.mean(target == 0) #  91.9%
np.mean(target == 1) #   8.1%

# If we predict everything to be 0, we'll be right 91.9% of the time.
# We want our negative predictive value (npv) to be greater than 91.9%.

# If we predict everything to be 1, we'll be right 8.1% of the time.
# We want our positive predictive value (ppv) to be greater than 8.1%.

acc                 # [0.69, 0.71, 0.68, 0.69, 0.67, 0.66, 0.69, 0.69, 0.66, 0.67]
auc                 # [0.74, 0.74, 0.73, 0.74, 0.73, 0.73, 0.72, 0.74, 0.73, 0.72]
F1                  # [0.25, 0.26, 0.26, 0.25, 0.25, 0.26, 0.25, 0.26, 0.24, 0.24]
optimal_threshold   # [0.09, 0.10, 0.08, 0.10, 0.07, 0.09, 0.08, 0.09, 0.08, 0.06]
tpr                 # [0.65, 0.64, 0.67, 0.67, 0.69, 0.70, 0.65, 0.66, 0.70, 0.65]
tnr                 # [0.69, 0.71, 0.68, 0.70, 0.66, 0.65, 0.70, 0.70, 0.65, 0.67]
ppv                 # [0.15, 0.16, 0.16, 0.16, 0.15, 0.16, 0.16, 0.16, 0.15, 0.15]
npv                 # [0.95, 0.95, 0.95, 0.96, 0.96, 0.95, 0.95, 0.95, 0.96, 0.95]

np.mean(ppv)
np.mean(optimal_threshold)


### Ready to fit the model with all training samples
# Fit the model
model = buildNeuralNetwork(train, target, epochs = 20)

# Predict on the test set
pred_y = model.predict(test)

# Export the predictions to a CSV
output = pd.DataFrame({"TARGET" : np.squeeze(pred_y)}, index = test_id)
output.to_csv("../predictions/submission.csv")
