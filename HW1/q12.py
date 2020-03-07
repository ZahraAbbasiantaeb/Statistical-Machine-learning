import math
import numpy as np

n = 100
p1 = 0.3
p2 = 0.5
eps = 0.2

Num = 10000

bound_ = 2*math.exp(-2*n*(eps**2))
print('boundary is: ')
print(bound_)

#  part A for p1
count = 0

bound1 = (p1*(1-p1))/(n*(eps**2))
print('boundary of p1 is: ')
print(bound1)

for i in range(0, Num):
    num = []

    for j in range(0, n):
        num.append(np.random.choice(np.arange(0, 2), p=[1-p1, p1]))

    X_bar = np.sum(num)/n

    if(abs(X_bar-p1)>=eps):
        count += 1

print(count/Num)
print(count)

# Part B for P2

count = 0

bound2 = (p2*(1-p2))/(n*(eps**2))
print('boundary of p2 is: ')
print(bound2)

for i in range(0, Num):
    num = []

    for j in range(0, n):
        num.append(np.random.choice(np.arange(0, 2), p=[1-p2, p2]))

    X_bar = np.sum(num)/n

    if(abs(X_bar-p2)>=eps):
        count += 1

print(count/Num)
print(count)