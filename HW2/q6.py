import numpy as np
from scipy.stats import norm

n = 50
alpha = 0.05
lambde_zero = 1

def test():

    samples = np.random.poisson(lambde_zero, n)

    lambda_hat = np.mean(samples)

    se_hat = np.sqrt(np.var(samples)/n)

    W = (lambda_hat - lambde_zero)/se_hat


    if(np.abs(W) > norm.ppf(1- (alpha/2))):
        # print('reject H0')
        return 1

    else:
        # print('retain H0')
        return 0

    return 0

tmp = 0

simulations = 1000

for i in range(0, simulations):
    tmp +=  test()

print(tmp/simulations)