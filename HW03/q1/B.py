from q1.functions import find_regression_param, dataset, plot_data_with_regression_line, calculate_RSS

for i in range(0, 8):

    Beta_0, Beta_1 = find_regression_param(dataset[0:500, i], dataset[0:500, 8])
    title = 'feature ' + str(i+1)
    print(Beta_0, Beta_1)
    plot_data_with_regression_line(dataset[:, i], dataset[:, 8],Beta_0, Beta_1, title)
    RSS, sigma_2, se_B_0, se_B_1, R_2 = calculate_RSS(dataset[0:500, i], dataset[0:500, 8],Beta_0, Beta_1,)
    RSS_test, _, _, _, R_2_test = calculate_RSS(dataset[500:600, i], dataset[500:600, 8], Beta_0, Beta_1, )
    print(RSS, sigma_2, se_B_0, se_B_1, R_2, RSS_test, R_2_test)
