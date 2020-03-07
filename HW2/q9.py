import random
import numpy as np
from scipy.stats import norm


def get_random_samples(data, n):

    res = []

    for i in range(0, n):
        res.append(random.choice(data))

    return np.array(res)


old_placebo = [8406, 2342, 8187, 8459, 4795, 3516, 4796, 10238]

new_old = [-1200, 2601, -2705, 1982, -1290, 351, -638, -2719]

data = [[8406, -1200], [2342,2601], [8187, -2705], [8459, 1982],
         [4795, -1290], [3516, 351], [4796, -638], [10238,-2719]]

B = 1000

teta_hat = np.mean(new_old) / np.mean(old_placebo)


estimations = []

for i in range(0, B):

    res = get_random_samples(data, len(data))
    teta = np.mean(res[:,1]) / np.mean(res[:,0])
    estimations.append(teta)


standard_error = np.sqrt(np.var(estimations))

alpha = 0.05

print('Confidence Interval is: ')
print(teta_hat - norm.ppf(1- (alpha/2)) * standard_error)
print(teta_hat + norm.ppf(1- (alpha/2)) * standard_error)


