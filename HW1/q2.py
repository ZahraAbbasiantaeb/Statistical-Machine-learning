from random import random

n = 10

count = 0

prob = 0.3

for i in range(0, 1000):

    for i in range(0, n):

        res = random()

        if res < prob:
            count += 1

print(n*prob)

print(count/1000)