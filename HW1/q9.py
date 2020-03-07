import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

n = 10000


X = np.array(np.random.normal(0, 1, n))

mean_ = []

for i in range(1, n):
    mean_.append(np.mean(X[0:i]))

plt.plot(mean_)
plt.show()


X = scipy.stats.cauchy.rvs(0, 1, n)

mean_ = []

for i in range(1, n):
    mean_.append(np.mean(X[0:i]))

plt.plot(mean_)
plt.show()