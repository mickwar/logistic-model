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

train = doTransform(train, transform)
test = doTransform(test, transform)

new_train, new_test = crossValidate(train, 10, 1, 0)

cv_train_x = train.loc[new_train]
cv_train_y = target.loc[new_train]
cv_test_x = train.loc[new_test]
cv_test_y = target.loc[new_test]
