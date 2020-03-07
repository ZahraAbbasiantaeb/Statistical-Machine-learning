import numpy as np
import matplotlib.pyplot as plt
from q1.functions import leave_one_out, dataset


def backward_feat_remove(features):

    cost = np.inf
    feat = features[0]

    for i in range(0, len(features)):

        new_feat = np.delete(features, i)

        RSS = leave_one_out(dataset[:, new_feat], dataset[:, 8])

        if (RSS < cost):

            cost = RSS
            feat = i

    return feat, cost


features = [0,1,2,3,4,5,6,7]

RSS = leave_one_out(dataset[:, 0:8], dataset[:, 8])

print(RSS)

x = [0]
y = [RSS]

for i in range(0,7):

    feat, RSS = backward_feat_remove(features)
    print(feat, RSS)

    x.append(i+1)
    y.append(RSS)
    features = np.delete(features, feat)

    print(features)
    print()
    print('#####')
    print()

print(y)
plt.plot(x, y)
plt.xlabel('Step')
plt.ylabel('RSS')
plt.show()