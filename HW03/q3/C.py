from q3.functions import k_fold_cross_validation_bayesian, data_X, data_Y, partition
import matplotlib.pyplot as plt
K = 10

for i in range(0, 15):

    one, two, _,_ = partition(data_X[:,i],data_Y )
    plt.hist(one, bins = 100)
    plt.hist(two, bins = 100)
    plt.title('feature: '+ str(i))
    plt.show()

print(k_fold_cross_validation_bayesian(K, data_X, data_Y))