from q1.functions import multiple_regression, dataset, evaluate_multiple_reg


# print('RSS for feature 7 and 8:')
#
# Beta_hat = multiple_regression(dataset[0:500, [6,7]], dataset[0:500, 8])
#
# RSS, sigma_2, log_likelihood, AIC, R_2 = evaluate_multiple_reg(dataset[500:600, [6,7]], dataset[500:600, 8], Beta_hat)
#
# print(RSS)

for i in [0,1,2,3,4,5,7]:

    Beta_hat = multiple_regression(dataset[0:500,[6,i]], dataset[0:500, 8])

    RSS, sigma_2, log_likelihood, AIC, R_2 = evaluate_multiple_reg(dataset[500:600,[6,i]], dataset[500:600, 8], Beta_hat)

    print('for test set: ')
    print(RSS, sigma_2, AIC, R_2)


    RSS, sigma_2, log_likelihood, AIC, R_2 = evaluate_multiple_reg(dataset[0:500,[6,i]], dataset[0:500, 8], Beta_hat)

    print('for train set: ')
    print(RSS, sigma_2, AIC, R_2)


