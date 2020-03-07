import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from q4 import get_random_samples


def find_occurance(data, X):

    count = 0

    for elem in data:
        if elem == X:
            count += 1

    return count


def get_upper_lower(X, Y):

    U = 0
    L = 0

    for i in range(0, len(Y)):

        if (X[i]<=4.3):
            L = i

        if (X[i]<=4.9):
            U = i

    return Y[L], Y[U]


with open('/Earthquakes.txt', 'r') as fp:
    lines = fp.readlines()

data = []

for line in lines:

    mag = float(line.split(',')[4])
    data.append(mag)


def get_CDF(data):

    bins = []

    set_ = set(data)
    for elem in set_:
        bins.append(elem)

    bins = np.array(bins)
    bins = np.sort(bins)
    list_ = []

    size = len(data)

    for bin in bins:
        list_.append([bin, find_occurance(data, bin) / size])

    prev = 0
    index = 0

    X = []
    Y = []

    for elem in list_:
        X.append(elem[0])
        Y.append(elem[1] + prev)
        prev += elem[1]
        index += 1

    return X,Y


# Part A and B

X, Y = get_CDF(data)

alpha = 0.05

size = len(data)

error = np.sqrt((np.log(2/alpha)) / (2*size))

print('Error for 95% confidence interval of cdf is: ')
print(error)

plt.plot(X, Y)
plt.plot(X, Y + error)
plt.plot(X, Y - error)
plt.legend(['F_hat', 'lower_bound', 'upper_bound'])
plt.show()

A, B = get_upper_lower(X, Y)

# Part C
res = []

for i in range(0, 1000):

    arr = get_random_samples(data, size)
    X, Y = get_CDF(arr)
    L, U = get_upper_lower(X, Y)
    res.append((U-L))


res = np.sort(res)

Z_alpha = norm.ppf( 1- (alpha/2))

Se = np.std(res)

result = B - A

print('Confidence interval of F(9.4) - F(4.3) is: ')
print(result - Z_alpha * Se)
print(result + Z_alpha * Se)