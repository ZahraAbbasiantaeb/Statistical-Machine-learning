import matplotlib.pyplot as plt

# Part A
from q2.functions import x, X

fig, ax = plt.subplots(3, 2)
for i in range(0, 6):

    # plt.scatter(X[:, i], X[:, 6])
    ax[int(i/2), i%2].scatter(x[:,i], x[:,6])
    ax[int(i/2), i%2].set_title('feature ' + str(i+1))

fig.show()

fig, ax = plt.subplots(3, 2)
for i in range(0, 6):

    ax[int(i/2), i%2].scatter(X[:, i], X[:, 6])
    ax[int(i/2), i%2].set_title('feature ' + str(i+1))

fig.show()