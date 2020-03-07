from q3.functions import data_X, data_Y
import matplotlib.pyplot as plt

def plot_scatter_of_classes():
    plt.subplot(621)
    plt.scatter(data_X[:, 0], data_Y)
    plt.xlabel('feature '+str(1))

    plt.subplot(622)
    plt.scatter(data_X[:, 1], data_Y)
    plt.xlabel('feature '+str(2))

    plt.subplot(623)
    plt.scatter(data_X[:, 2], data_Y)
    plt.xlabel('feature '+str(3))

    plt.subplot(624)
    plt.scatter(data_X[:, 3], data_Y)
    plt.xlabel('feature '+str(4))

    plt.subplot(625)
    plt.scatter(data_X[:, 4], data_Y)
    plt.xlabel('feature '+str(5))

    plt.subplot(626)
    plt.scatter(data_X[:, 5], data_Y)
    plt.xlabel('feature '+str(6))

    plt.subplot(627)
    plt.scatter(data_X[:, 6], data_Y)
    plt.xlabel('feature '+str(7))

    plt.subplot(628)
    plt.scatter(data_X[:, 7], data_Y)
    plt.xlabel('feature '+str(8))

    plt.subplot(629)
    plt.scatter(data_X[:, 8], data_Y)
    plt.xlabel('feature '+str(9))

    plt.show()

    plt.subplot(321)
    plt.scatter(data_X[:, 9], data_Y)
    plt.xlabel('feature '+str(10))

    plt.subplot(322)
    plt.scatter(data_X[:, 10], data_Y)
    plt.xlabel('feature '+str(11))

    plt.subplot(323)
    plt.scatter(data_X[:, 11], data_Y)
    plt.xlabel('feature '+str(12))

    plt.subplot(324)
    plt.scatter(data_X[:, 12], data_Y)
    plt.xlabel('feature '+str(13))

    plt.subplot(325)
    plt.scatter(data_X[:, 13], data_Y)
    plt.xlabel('feature '+str(14))

    plt.subplot(326)
    plt.scatter(data_X[:, 14], data_Y)
    plt.xlabel('feature '+str(15))

    plt.show()

    return


plot_scatter_of_classes()