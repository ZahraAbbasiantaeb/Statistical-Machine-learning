from q1.functions import find_feature_AIC
import matplotlib.pyplot as plt
import numpy as np

features = [6]
RSS_test_arr = [487.275897]
RSS_train_arr = [21599.558]

for i in range(0, 7):

    features_new , RSS, R_2, RSS_train, R_2_train = find_feature_AIC(features)
    print(features_new , RSS, R_2, RSS_train, R_2_train)

    features.append(features_new)
    print(features)
    RSS_train_arr.append(RSS_train)
    RSS_test_arr.append(RSS)

plt.xlabel('feature count')
plt.ylabel('RSS')

plt.plot(np.arange(0, 8, 1), RSS_test_arr)
plt.plot(np.arange(0, 8, 1), RSS_train_arr)
plt.legend(['test', 'train'])
plt.show()