from sklearn.linear_model import Lasso
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

imputer = KNNImputer(n_neighbors=3, weights="uniform")

nan = np.nan

x = np.loadtxt(open("/dataset02.csv", "rb"), delimiter=",", skiprows=1)

X_2 = np.array(x).astype("float")

np.place(X_2, X_2 == 0, nan)

X = imputer.fit_transform(X_2)

train_X = X[0:230, 0:6]
train_Y = X [0:230, 6]

test_X = X[230:273, 0:6]
test_Y = X[230:273, 6]

# # Part A
# #
# # fig, ax = plt.subplots(3, 2)
# # for i in range(0, 6):
# #
# #     # plt.scatter(X[:, i], X[:, 6])
# #     ax[int(i/2), i%2].scatter(x[:,i], x[:,6])
# #     ax[int(i/2), i%2].set_title('feature ' + str(i+1))
# #
# # fig.show()
# #
# # fig, ax = plt.subplots(3, 2)
# # for i in range(0, 6):
# #
# #     ax[int(i/2), i%2].scatter(X[:, i], X[:, 6])
# #     ax[int(i/2), i%2].set_title('feature ' + str(i+1))
# #
# # fig.show()
#
#
# # Part B
#
# train_X = X[0:230, 0:6]
# train_Y = X [0:230, 6]
#
# test_X = X[230:273, 0:6]
# test_Y = X[230:273, 6]
#
# lasso_train = []
# lasso_test = []
#
# lambdas = np.arange(0, 5, 0.001)
# best_lambda  = 0
# best_lasso = np.inf
#
# for lambda_ in lambdas:
#
#
#     lasso_reg = Lasso(normalize=True,  alpha=lambda_, fit_intercept=False)
#
#     lasso_reg.fit(train_X, train_Y)
#
#     coef = lasso_reg.coef_
#
#     y_pred_lass =lasso_reg.predict(train_X)
#
#     lasso_val = np.sum([ elem**2 for elem in y_pred_lass-train_Y]) + np.sum([lambda_ * B for B in coef])
#
#     lasso_train.append(lasso_val)
#
#     y_pred_lass = lasso_reg.predict(test_X)
#
#     lasso_val = np.sum([elem ** 2 for elem in y_pred_lass - test_Y]) + np.sum([lambda_ * B for B in coef])
#
#     lasso_test.append(lasso_val)
#
#     if(lasso_val<best_lasso):
#
#         best_lambda = lambda_
#         best_lasso = lasso_val
#
# plt.plot(lambdas, lasso_train)
# plt.xlabel('lambda')
# plt.ylabel('lasso value')
# plt.title('train dataset')
# plt.show()
#
# plt.plot(lambdas, lasso_test)
# plt.xlabel('lambda')
# plt.ylabel('lasso value')
# plt.title('test dataset')
# plt.show()
#
# print(best_lambda)
#
# #  Part C
#
# # print RSS
#
# # lambda_ = best_lambda
# #
# # lasso_reg = Lasso(normalize=True,  alpha=lambda_, fit_intercept=False)
# #
# # lasso_reg.fit(train_X, train_Y)
# #
# # coef = lasso_reg.coef_
# #
# # y_pred_lass =lasso_reg.predict(train_X)
# # print(get_r2_numpy_manual(y_pred_lass, train_Y))
# #
# # RSS_train = np.sum([ elem**2 for elem in y_pred_lass-train_Y])
# #
# # y_pred_lass_test =lasso_reg.predict(test_X)
# #
# # RSS_test = np.sum([ elem**2 for elem in y_pred_lass_test-test_Y])
# #
# # print(RSS_test, RSS_train)
# # print(get_r2_numpy_manual(y_pred_lass_test, test_Y))
# # print(get_r2_numpy_manual(y_pred_lass, train_Y))
#
# # Part D
# #
# # x = np.loadtxt(open("./dataset/Dataset2_Unlabeled.csv", "rb"), delimiter=",")
# #
# # unlabeled_data = np.array(x).astype("float")
# #
# # print(np.shape(unlabeled_data))
# #
# # np.place(unlabeled_data, unlabeled_data == 0, nan)
# #
# # unlabeled_data = imputer.fit_transform(unlabeled_data)
# #
# # pred_unlabeled = lasso_reg.predict(unlabeled_data)
# #
# # f = open("dataset02_mylabel.csv","w+")
# #
# # for num in pred_unlabeled:
# #
# #     f.write(str(num)+'\n')
# #
# # f.close()
# #
# #
# # Part E
#
# x = np.loadtxt(open("./dataset/Dataset2_extended.csv", "rb"), delimiter=",")
#
# extended_data = np.array(x).astype("float")
#
# print(np.shape(extended_data))
#
# np.place(extended_data, extended_data == 0, nan)
#
# extended_data = imputer.fit_transform(extended_data)
#
# lasso_reg = Lasso(normalize=True,  alpha=0.001, fit_intercept=False)
#
# lasso_reg.fit(extended_data[:,0:6], extended_data[:,6])
#
# print(lasso_reg.coef_)