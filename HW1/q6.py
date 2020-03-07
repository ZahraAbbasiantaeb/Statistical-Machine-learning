import numpy as np
import pylab as plt
import warnings
from scipy.stats import binom

n = 100
p = 0.3
observation = 1000

X1 = np.random.binomial(n, p, observation)
print(X1)

p = 0.5
X2 = np.random.binomial(n, p, observation)

n = 200
X3 = np.random.binomial(n, p, observation)


Y1 = X1 + X2

Y2 = X2 + X3


plt.hist(X1, bins='auto')
plt.title('X1')
plt.show()

plt.hist(X2, bins='auto')
plt.title('X2')
plt.show()

plt.hist(X3, bins='auto')
plt.title('X3')
plt.show()

plt.hist(Y1, bins='auto')
plt.title('Y1')
plt.show()

plt.hist(Y2, bins='auto')
plt.title('Y2')
plt.show()

