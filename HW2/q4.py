import random
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

n = 100
mu = 5
sigma = 1

# Part A and B

samples = np.random.normal(mu, sigma, n)
samples = list(samples)
main_mean = np.mean(samples)

B = 10000
mean_b = []
teta = []

def get_random_samples(data, n):

    res = []

    for i in range(0, n):
        res.append(random.choice(data))

    return res


for i in range(0, B):

    data_b = get_random_samples(samples, n)
    tmp_mean = np.mean(data_b)
    mean_b.append(tmp_mean)
    teta.append(np.exp(tmp_mean))

mean_b = np.sort(mean_b)
teta = np.sort(teta)

print(main_mean)

se_hat = np.sqrt(np.var(teta))
print(se_hat)

tetaHat = np.exp(main_mean)
print(tetaHat)

alpha = 0.06
alpha_2 = 1-(alpha/2)

Z_alpha = norm.ppf(1- (alpha/2))

L_normal = tetaHat - se_hat*Z_alpha
U_normal = tetaHat + se_hat*Z_alpha

print('94 percent confidence interval gaussian:')
print(L_normal)
print(U_normal)

print('94 percent confidence interval percentile:')
L_percentile = teta[25]
U_percentile =teta[1000-25]

print(L_percentile)
print(U_percentile)

print('94 percent confidence interval pivotal:')
L_pivotal = 2*tetaHat - teta[1000-25]
U_pivotal = 2*tetaHat - teta[25]
print(L_pivotal)
print(U_pivotal)

# Part C
print(teta)

plt.hist(teta, bins=100)
plt.ylabel('Probability')

Y = []
for i in range(1, B):
    Y.append(np.exp(np.mean(np.random.normal(mu, sigma, n))))

plt.hist(Y, bins=100)
plt.legend(['bootstrap', 'true'])
plt.show()

