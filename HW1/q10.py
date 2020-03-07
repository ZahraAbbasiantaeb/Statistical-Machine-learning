import numpy as np
import matplotlib.pyplot as plt

n = 1000

s = np.random.poisson(6, n)

plt.hist(s, 15, normed=True)

plt.show()

res = []

for i in range(0, n):

    num = []

    for i in range(0, 100):
        num.append(np.random.choice(np.arange(0, 2), p=[0.94, 0.06]))

    res.append(np.sum(num))

plt.hist(res, 15, normed=True)
plt.show()

print(np.mean(res))
print(np.mean(s))