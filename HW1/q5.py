import numpy as np


mu, sigma = 3, 16

n = 1000

simulations = 1000

#  A

res = []

for i in range(0, simulations):

    s = np.random.normal(mu, sigma, n)

    count = 0

    for elem in s:

        if elem < 7:

            count += 1

    res.append(count/n)

print(np.mean(res))


# B

res = []

for i in range(0, simulations):

    s = np.random.normal(mu, sigma, n)

    count = 0

    for elem in s:

        if elem > -2:

            count += 1

    res.append(count/n)

print(np.mean(res))

# C

res = []

for i in range(0, simulations):

    s = np.random.normal(mu, sigma, n)

    sorted = np.sort(s)

    id = int (n*0.95)

    res.append(sorted[id])

print(np.mean(res))


# D

res = []

for i in range(0, simulations):

    s = np.random.normal(mu, sigma, n)

    count = 0

    for elem in s:

        if elem > -2 and elem < 2:

            count += 1

    res.append(count/n)

print(np.mean(res))

# E

res = []

for i in range(0, simulations):

    s = np.random.normal(mu, sigma, n)

    count = 0

    for elem in s:

        if elem > 0 and elem < 4:

            count += 1

    res.append(count/n)

print(np.mean(res))

# F

res = []

for i in range(0, simulations):

    s = np.random.normal(mu, sigma, n)

    count = 0

    for elem in s:

        if elem > -32 and elem < 32:

            count += 1

    res.append(count/n)

print(np.mean(res))