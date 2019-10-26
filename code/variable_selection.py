### Function for doing automatic variable selection

import xgboost as xgb
import numpy as np
from operator import itemgetter


# Fits an XGBoost model
# Gets the the largest variables which contribute to alpha% of the importance
def doVariableSelection(data, target, alpha = 0.95, num_round = 10, params = None):

    # Default parameters
    if params is None:
        params = {'max_depth' : 10,
                  'eta' : 1,
                  'verbosity' : 2,
                  'objective' : 'binary:logistic'}

    # The data matrix
    dtrain = xgb.DMatrix(data, label = target)

    # Train the model
    bst = xgb.train(params, dtrain, num_round)

    # Compute importance scores
    importance = bst.get_score()

    # Sort them in descending order
    sorted_importance = sorted(importance.items(), key = itemgetter(1), reverse = True)
    sort_key = []
    sort_val = []
    for k, v in sorted_importance:
        sort_key.append(k)
        sort_val.append(v)


    # Since the scores are sorted, we need the highest index which gets us to
    # the alpha% contribution
    index = np.cumsum(sort_val) / np.sum(sort_val) <= alpha

    # Return the variable names to keep
    return sort_key[0:np.max(np.where(index))]

