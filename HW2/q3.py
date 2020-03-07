import numpy as np
import matplotlib.pyplot as plt
from q1 import get_CDF

n = 100

data = np.random.exponential(5, size=n)

print('estimated parameter:')

print(np.mean(data))

# Part A

X, Y = get_CDF(data)

plt.plot(X, Y)
plt.title('Emprical Distribution Function')
plt.show()

# Part B


plug_mean = np.mean(data)
print(plug_mean)

plug_var1 = sum((data-plug_mean)**2)/n
print(plug_var1)

plug_var2 = sum((data-plug_mean)**2)/(n-1)
print(plug_var2)

plug_skewness = (sum((data-plug_mean)**3)/n)/(np.sqrt(plug_var1)**3)

print(plug_skewness)