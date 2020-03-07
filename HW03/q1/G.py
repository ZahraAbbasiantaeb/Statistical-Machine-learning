import numpy as np
from q1.functions import multiple_regression, dataset, evaluate_multiple_reg
import matplotlib.pyplot as plt

train = []
test = []
x = []
total = [ ]
rang = np.arange(100, 600, 25)

for i in rang:

    beta_hat = multiple_regression(dataset[0:i,[0,1,2,3,5,6]], dataset[0:i, 8])
    RSS_train,_,_,_,_ = evaluate_multiple_reg(dataset[0:i , [0, 1, 2, 3, 5, 6]], dataset[0:i, 8], beta_hat)
    RSS_test,_,_,_,_ = evaluate_multiple_reg(dataset[i:600,[0,1,2,3,5,6]], dataset[i:600, 8], beta_hat)
    train.append(RSS_train)
    test.append(RSS_test)
    total.append(RSS_test+RSS_train)
    x.append(i)

print(train)
print(test)

plt.plot(x, train)
plt.plot(x, test)
plt.plot(x, total)
plt.legend(['train', 'test', 'total dataset'])
plt.show()