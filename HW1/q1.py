from random import random
import matplotlib.pyplot as plt

n = 1000

count = 0

prob = 0.03

distribution = []

for i in range(0, n):

    if random() < prob:
        count += 1

    distribution.append(count/(i+1))

plt.plot(distribution)

plt.xlabel('n')

plt.ylabel('prob')

plt.show()