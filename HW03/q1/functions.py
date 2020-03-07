import numpy
import numpy as np
import matplotlib.pyplot as plt

x =numpy.loadtxt(open("/Dataset01.csv", "rb"), delimiter=",", skiprows=1)


dataset = numpy.array(x).astype("float")


def get_r2_numpy_manual(x, y):

    # zx = (x-np.mean(x))
    y_bar = np.mean(y)

    r = 1 - (get_RSS(x, y) / np.sum([(y-y_bar)**2 for y in y]))

    return r


def get_RSS(X, Y):

    RSS = np.sum([(x - y) ** 2 for x, y in zip(X, Y)])

    return RSS


def scatter_features():

    for i in range(0, 8):
        plt.scatter(dataset[:, i], dataset[:, 8])
        plt.xlabel('feature ' + str(i + 1))
        plt.ylabel('feature 9')
        plt.show()


    return


def find_regression_param(X, Y):

    x_bar = np.mean(X)
    y_bar = np.mean(Y)

    Beta_1 = np.sum([(x_i-x_bar) * (y_i-y_bar) for x_i, y_i in zip(X, Y)]) /\
    np.sum([(x_i-x_bar) **2 for x_i in X])

    Beta_0 = y_bar - Beta_1*x_bar


    return Beta_0, Beta_1


def plot_data_with_regression_line(X, Y, B_0, B_1, label):

    x = np.linspace(np.min(X), np.max(X) + 2, 1000)

    y = B_1*x+B_0

    plt.scatter(X, Y)

    plt.plot(x, y, c='red')

    plt.title(label)

    plt.show()

    return


def calculate_RSS(X, Y, B0, B1):

    # RSS = np.sum([ (B1 * x + B0 - y)**2 for x, y in zip(X, Y) ]

    y_pred = [(B1 * x + B0 ) for x in X ]

    RSS = get_RSS(y_pred, Y)

    N = len(X)

    sigma_2 = RSS/(N-2)

    x_bar = np.mean(X)

    S_X_2 = np.sum([(x-x_bar)**2 for x in X])/N

    se_B_1 = np.sqrt(sigma_2 / (N * S_X_2))

    se_B_0 = (se_B_1 * np.sqrt(np.sum([x**2 for x in X]))) / np.sqrt(N)

    R_2 = get_r2_numpy_manual(y_pred,Y)

    return RSS, sigma_2, se_B_0, se_B_1, R_2


def multiple_regression(X, Y):

    Beta_hat = np.dot(np.dot(numpy.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)

    return Beta_hat


def evaluate_multiple_reg(X, Y, Beta_hat):

    n, k = np.shape(X)

    res = np.dot(X, Beta_hat)

    res = Y-res

    RSS = np.sum([x**2 for x in res])

    sigma_2 = RSS/(n-k)

    log_likelihood = -n * np.log(np.sqrt(sigma_2)) - (RSS/ (2*sigma_2))

    AIC = log_likelihood - k

    R_2 = get_r2_numpy_manual(np.dot(X, Beta_hat), Y)

    return RSS, sigma_2, log_likelihood, AIC, R_2


def leave_one_out(X, Y):

    n, k = np.shape(X)
    RSS = 0

    for i in range(0, n):

        X_ = np.delete(X, (i), axis=0)
        Y_ = np.delete(Y, (i), axis=0)

        Beta_hat = np.dot(np.dot(numpy.linalg.inv(np.dot(np.transpose(X_), X_)), np.transpose(X_)), Y_)

        res = np.dot(X[i], Beta_hat)

        res = (Y[i] - res)**2

        RSS += res

    return RSS


def find_feature_AIC(features):

    tmp = -10000000
    index = 0
    RSS_best = 0
    best_R_2 = 0
    RSS_train_best = 0
    R_2_train_best = 0

    for i in range(0, 8):

        if not (i in features):

            tmp_feat = np.append(features, i)
            Beta_hat = multiple_regression(dataset[0:500, tmp_feat], dataset[0:500, 8])

            RSS, _, _, _, R_2 = evaluate_multiple_reg(dataset[500:600, tmp_feat], dataset[500:600, 8], Beta_hat)
            RSS_train, _, _, AIC, R_2_train = evaluate_multiple_reg(dataset[0:500, tmp_feat], dataset[0:500, 8], Beta_hat)

            if AIC > tmp:

                tmp = AIC

                index = i

                RSS_best = RSS
                best_R_2 = R_2
                RSS_train_best = RSS_train
                R_2_train_best = R_2_train

    return index, RSS_best, best_R_2, RSS_train_best, R_2_train_best


def variance_beta(X, sigma_2):

    tmp = sigma_2 * np.linalg.inv(np.dot(np.transpose(X), X))

    return tmp

