import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from q4 import get_random_samples
from scipy.stats import norm

n = 100
mu = 5
sigma = 1

# a, b
samples = np.random.normal(mu, sigma, n)
samples = np.sort(samples)

s = scipy.stats.norm(np.mean(samples), np.sqrt(1/n)).pdf(samples)
plt.plot(samples, s)
plt.title('Founded Posterior Density')
plt.show()

# c
s = np.random.normal(np.mean(samples), np.sqrt(1/n), 500)

plt.hist(s, bins=100)
plt.title('Simulated Posterior Density')
plt.show()

# d Founded Posterior

theta = []

for i in range(0, 300):
    theta.append(i+1.0)

theta = np.array(theta)

new_samples = (1/theta)*scipy.stats.norm(np.mean(samples),
                                           np.sqrt(1/n)).pdf(np.log(theta))

plt.plot(theta, new_samples)
plt.title('Founded Posterior Density')
plt.show()

# Simulated Posterior

simulations = 1000

teta_hat = []

data = np.random.normal(mu, sigma, 10000)

for i in range(0, simulations):

    sample = get_random_samples(data, n)
    teta_hat.append(np.exp(np.mean(sample)))

plt.hist(teta_hat)
plt.title('Simulated Posterior Density')
plt.show()

# e
simulations = 100
samples = np.random.normal(mu, sigma, n)
teta = []
teta_hat = np.exp(np.mean(samples))

for i in range (0, simulations):
    tmp = get_random_samples(data, 100)
    teta.append(np.exp(np.mean(tmp)))

std = np.sqrt(np.var(teta))

alpha = 0.03

Z_alpha = norm.ppf(1- (alpha/2))

print('Confidence Interval:')
print(teta_hat - std * Z_alpha)
print(teta_hat + std * Z_alpha)


alpha = 0.07

Z_alpha = norm.ppf(1 - (alpha/2))

print('Confidence Interval:')
print(teta_hat - std * Z_alpha)
print(teta_hat + std * Z_alpha)