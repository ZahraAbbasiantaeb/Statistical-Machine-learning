import numpy as np
from sklearn.impute import KNNImputer
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("/dataset03.csv",  na_values='?')

data = np.array(data)

imputer = KNNImputer(n_neighbors=3, weights="uniform")

data_X = data[:, 0:16]

data_Y = data[:, 16]

data_X = imputer.fit_transform(data_X[:, 0:16])


def partition(X, Y):

    one = []
    two = []

    prior_B = 0
    prior_M = 0

    for i in range(0, len(X)):

        if Y[i] == 'B':
            one.append(X[i])
            prior_B +=1

        else:
            two.append(X[i])
            prior_M += 1

    prior_B/=len(X)
    prior_M/=len(X)

    return one, two, prior_B, prior_M


def create_PDF(data_X, data_Y):

    class_B = []
    class_M = []


    row, col = np.shape(data_X)

    for i in range(0, col):

        one, two, prior_B, prior_M = partition(data_X[:,i],data_Y )
        class_B.append((np.mean(one), np.var(one)))
        class_M.append((np.mean(two), np.var(two)))

    return class_B, class_M, prior_B, prior_M


def bayesian_model(X, class_B, class_M, prior_B, prior_M):

    tmp_B = np.log(prior_B)
    tmp_M = np.log(prior_M)

    # tmp_B = 0
    # tmp_M = 0

    i = 0

    for mean, var in class_B:
        tmp_B += np.log(scipy.stats.norm(mean, var).pdf(X[i])+1)
        i += 1

    i = 0
    for mean, var in class_M:
        tmp_M += np.log(scipy.stats.norm(mean, var).pdf(X[i])+1)
        i += 1

    if(tmp_M>=tmp_B):

        return 'M'

    return 'B'


def apply_bayesian(X, class_B, class_M, prior_B, prior_M ):

    pred = []

    for row in X:
        pred.append(bayesian_model(row, class_B, class_M, prior_B, prior_M ))

    return pred


def k_fold_cross_validation_bayesian(K, data_X, data_Y):

    kf = KFold(n_splits=K)
    res = []

    for train_index, test_index in kf.split(data_X):

        X_train, X_test = data_X[train_index], data_X[test_index]
        y_train, y_test = data_Y[train_index], data_Y[test_index]

        class_B, class_M, prior_B, prior_M = create_PDF(X_train, y_train)
        pred = apply_bayesian(X_test, class_B, class_M, prior_B, prior_M )

        res.append((pred, y_test))

    acc = 0

    for elem in res:
        acc += accuracy_score(elem[0], elem[1])

    return acc/K


