from q3.functions import k_fold_cross_validation_bayesian, data_X, data_Y

K = 10

print(k_fold_cross_validation_bayesian(K, data_X[:,[0, 1]], data_Y))
print(k_fold_cross_validation_bayesian(K, data_X[:,[0, 1,5]], data_Y))
print(k_fold_cross_validation_bayesian(K, data_X[:, [0, 1, 5, 10]], data_Y))
print(k_fold_cross_validation_bayesian(K, data_X[:, [0, 1, 5, 6, 10]], data_Y))

# for i in range(0, 15):
#     print(k_fold_cross_validation_bayesian(K, data_X[:, [i]], data_Y))

# for i in range(1, 15):
#     print(k_fold_cross_validation_bayesian(K, data_X[:, [0, i]], data_Y))

# for i in range(2, 15):
#     if(not i in [5, 10]):
#         print(k_fold_cross_validation_bayesian(K, data_X[:, [0, 1, 5, 10, i]], data_Y))


