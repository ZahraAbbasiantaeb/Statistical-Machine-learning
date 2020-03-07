# Part B
import matplotlib.pyplot as plt

from q1.functions import get_r2_numpy_manual, get_RSS
from q2.functions import train_Y, train_X, test_Y, test_X
from sklearn.linear_model import Lasso
import numpy as np

lasso_train = []
lasso_test = []

lambdas = np.arange(0, 5, 0.001)
best_lambda  = 0
best_lasso = np.inf

def get_lasso(pred, actual, coef, lambda_):

    lasso_val = np.sum([elem ** 2 for elem in pred - actual]) + np.sum([lambda_ * np.abs(B) for B in coef])

    return lasso_val


for lambda_ in lambdas:

    lasso_reg = Lasso(normalize=True,  alpha=lambda_, fit_intercept=False)

    lasso_reg.fit(train_X, train_Y)

    coef = lasso_reg.coef_

    y_pred_lass =lasso_reg.predict(train_X)

    lasso_val = get_lasso(y_pred_lass, train_Y, coef, lambda_)

    lasso_train.append(lasso_val)

    y_pred_lass = lasso_reg.predict(test_X)

    lasso_val = get_lasso(y_pred_lass , test_Y, coef, lambda_)

    lasso_test.append(lasso_val)

    if(lasso_val < best_lasso):

        best_lambda = lambda_
        best_lasso = lasso_val

plt.plot(lambdas, lasso_train)
plt.xlabel('lambda')
plt.ylabel('lasso value')
plt.title('train dataset')
plt.show()

plt.plot(lambdas, lasso_test)
plt.xlabel('lambda')
plt.ylabel('lasso value')
plt.title('test dataset')
plt.show()

print(best_lambda)

lambda_ = best_lambda

lasso_reg = Lasso(normalize=True,  alpha=lambda_, fit_intercept=False)

lasso_reg.fit(train_X, train_Y)

coef = lasso_reg.coef_

y_pred_lass =lasso_reg.predict(train_X)
print(get_r2_numpy_manual(y_pred_lass, train_Y))

RSS_train = get_RSS(y_pred_lass,train_Y )

y_pred_lass_test =lasso_reg.predict(test_X)

RSS_test = get_RSS(y_pred_lass_test, test_Y)

print(RSS_test, RSS_train)
print(get_r2_numpy_manual(y_pred_lass_test, test_Y))
print(get_r2_numpy_manual(y_pred_lass, train_Y))