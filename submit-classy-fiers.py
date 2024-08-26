import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to train your model using training CRPs
    # X_train has 32 columns containing the challeenge bits
    # y_train contains the responses

    # THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
    # If you do not wish to use a bias term, set it to 0
    logr = sklearn.linear_model.LogisticRegression(C=100) 
    X_train_trans = my_map(X_train)
    y_train = np.where( y_train > 0, 1, -1 )
    logr.fit(X_train_trans, y_train)
    w = logr.coef_.reshape( (528,) )
    b = logr.intercept_
    return w, b

    #return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to create features.
    # It is likely that my_fit will internally call my_map to create features for train points

    feat = []
    f = []
    ones_column = np.ones((X.shape[0], 1))  # Column vector of 1s
    X = np.hstack((X, ones_column))
    for j in range(31, -1, -1):
        X[:, j] = (1 - 2 * X[:, j]) * X[:, j + 1]
    for j in range(0, 32):
        for k in range(j + 1, 32):
            f.append(X[:, j] * X[:, k])
    for i in range(0, 32):
        f.append(X[:, i])
    feat.append(f)

    return np.squeeze(np.array(feat)).T
