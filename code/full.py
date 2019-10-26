from transformations import *
from model import *


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

acc                 # [0.67, 0.65, 0.73, 0.70, 0.65, 0.66, 0.67, 0.69, 0.66, 0.65]
auc                 # [0.73, 0.74, 0.72, 0.74, 0.72, 0.73, 0.72, 0.73, 0.73, 0.72]
F1                  # [0.24, 0.24, 0.27, 0.26, 0.24, 0.26, 0.25, 0.26, 0.25, 0.24]
optimal_threshold   # [0.08, 0.05, 0.08, 0.07, 0.11, 0.07, 0.08, 0.09, 0.07, 0.06]
tpr                 # [0.67, 0.72, 0.61, 0.65, 0.69, 0.70, 0.67, 0.67, 0.70, 0.68]
tnr                 # [0.67, 0.64, 0.74, 0.71, 0.65, 0.66, 0.67, 0.69, 0.65, 0.65]
ppv                 # [0.15, 0.14, 0.17, 0.16, 0.14, 0.16, 0.15, 0.16, 0.15, 0.14]
npv                 # [0.96, 0.96, 0.95, 0.96, 0.96, 0.95, 0.95, 0.95, 0.96, 0.95]



### Ready to fit the model with all training samples
# Fit the model
model = buildNeuralNetwork(train, target, epochs = 20)

# Predict on the test set
pred_y = model.predict(test)

# Export the predictions to a CSV
output = pd.DataFrame({"TARGET" : np.squeeze(pred_y)}, index = test_id)
output.to_csv("../predictions/submission.csv")
