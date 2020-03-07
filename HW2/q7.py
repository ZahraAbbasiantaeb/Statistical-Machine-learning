import numpy as np
from scipy.stats import norm

# Wald test
from q4 import get_random_samples

sample_1 = [225, 262, 217, 240, 230, 229, 235, 217]
sample_2 = [209, 205, 196, 210, 202, 207, 224, 223, 220, 201]

n_1 = len(sample_1)
n_2 = len(sample_2)

mean_2 = np.mean(sample_2)
mean_1 = np.mean(sample_1)

var_2 = np.var(sample_2)
var_1 = np.var(sample_1)

teta_hat = mean_2 - mean_1

teta_zero = 0

standard_error = np.sqrt( (var_1/n_1) + (var_2/n_2) )

W = (np.abs(teta_hat - teta_zero)) / standard_error

print('teta hat is: ')
print(teta_hat)

# P_value
print('P-Value is: ')

P_value = 2*norm.cdf(-1*W)

print(P_value)

# Confidence interval

simulations = 1000

z_alpha = norm.ppf(1 - (0.03/2))

res = []

for i in range (0,simulations):

    data_1 = get_random_samples(sample_1, 8)
    data_2 = get_random_samples(sample_2, 10)

    res.append(np.mean(data_2) - np.mean(data_1))

std = np.sqrt(np.var(res))

print('Confidence Interval:')
print(teta_hat - std * z_alpha )
print(teta_hat + std * z_alpha )



# part B

t_obs = np.abs(teta_hat)

B = 10000
All = [225, 262, 217, 240, 230, 229, 235, 217, 209, 205, 196, 210, 202, 207, 224, 223, 220, 201]
counter = 0

for i in range(0, B):

    tmp = np.random.permutation(All)

    T = np.abs(np.mean(tmp[0:10]) - np.mean(tmp[10:18]))

    if T > t_obs:

        counter +=1

print('Permutation test: ')
print(counter/B)