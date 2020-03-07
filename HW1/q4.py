from random import randint, random

n = 1000

count = 0

for i in range(0, n):

    prize = randint(1, 3)

    choice = randint(1, 3)

    if prize != choice:
        count += 1

print(count/n)

count = 0

for i in range(0, n):

    prize = randint(1, 3)

    choice = randint(1, 3)

    if prize == choice:
        count += 1

print(count / n)