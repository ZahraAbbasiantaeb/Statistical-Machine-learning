import numpy as np
import matplotlib.pyplot as plt

n = 100

X = np.random.uniform(0, 1, n)

Y = np.random.uniform(0, 1, n)

Z = X-Y

plt.hist(Z, 40, density=True)

plt.show()

print(np.mean(Z))
print(np.var(Z))
print(max(Z))
print(min(Z))

print('*********')
Z = X/Y
plt.hist(Z, 40, density=True)

plt.show()

print(np.mean(Z))
print(np.var(Z))
print(max(Z))
print(min(Z))

