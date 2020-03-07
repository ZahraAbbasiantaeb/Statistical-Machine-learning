import numpy as np
import matplotlib.pyplot as plt

mean = [0,0]
cov = [[1, 1/2], [1/2, 1/3]]
x, y =  np.random.multivariate_normal(mean, cov, 50000).T

plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()

print(np.mean(x))
print(np.mean(y))

print(np.cov(x,y))