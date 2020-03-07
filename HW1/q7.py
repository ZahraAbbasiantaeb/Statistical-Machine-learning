import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

index = [1,100,1000,10000]
mu = 0
X = [np.arange(-5, 5, 0.01).tolist(),
     np.arange(-0.5,  0.5, 0.01).tolist(),
     np.arange(-0.05, 0.05, 0.001).tolist(),
     np.arange(-0.005, 0.005, 0.0001).tolist()]

for i in range(0, 4):
    n = index[i]
    sigma = 1 / n
    dist = scipy.stats.norm(mu, sigma)
    res = dist.cdf(X[i])
    plt.plot(X[i], res)
    plt.show()

