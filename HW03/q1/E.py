from q1.functions import multiple_regression, dataset, variance_beta, evaluate_multiple_reg, leave_one_out

Beta_hat = multiple_regression(dataset[0:500,0:8], dataset[0:500, 8])
RSS, sigma_2, log_likelihood, AIC, R_2 = evaluate_multiple_reg(dataset[0:500,0:8], dataset[0:500, 8], Beta_hat)
print(variance_beta(dataset[0:500,0:8], sigma_2))

print(RSS, sigma_2, log_likelihood, AIC, R_2)
print(leave_one_out(dataset[:,0:8], dataset[:, 8]))