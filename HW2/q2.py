import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from q4 import get_random_samples

N= 100

samples = np.random.exponential(2, N)

mean_hat = np.mean(samples)

print('estimated lambda: ')
print(mean_hat)

means = []

for i in range(0, 1000):
    tmp = get_random_samples(samples, N)
    means.append(np.mean(tmp))

variance = np.var(means)
means= np.sort(means)

alpha = 0.03
confidence_left_97 = mean_hat - np.sqrt(variance) * scipy.stats.norm.ppf(1 - (alpha / 2))
confidence_right_97 = mean_hat + np.sqrt(variance) * scipy.stats.norm.ppf(1 - (alpha / 2))

print(confidence_left_97)
print(confidence_right_97)

print(means[15])
print(means[1000-15])


alpha = 0.07
confidence_left_93 = mean_hat - np.sqrt(variance) * scipy.stats.norm.ppf(1 - (alpha / 2))
confidence_right_93 = mean_hat + np.sqrt(variance) * scipy.stats.norm.ppf(1 - (alpha / 2))

print(confidence_left_93)
print(confidence_right_93)
print('***')
print(means[35])
print(means[1000-35])

plt.hist(means)
plt.show()



N = 100

for simulations in  [10000, 1000]:

    tmp_var_93 = 0
    tmp_var_97 = 0

    for i in range(0, simulations):

        tmp = np.random.exponential(2, N)
        estimated_mean = np.mean(tmp)

        if estimated_mean >= confidence_left_93 and estimated_mean <= confidence_right_93:
            tmp_var_93 += 1

        if estimated_mean >= confidence_left_97 and estimated_mean <= confidence_right_97:
            tmp_var_97 += 1



    print('how often 93%: ')
    print(tmp_var_93 / simulations)

    print('how often 97%: ')
    print(tmp_var_97 / simulations)


