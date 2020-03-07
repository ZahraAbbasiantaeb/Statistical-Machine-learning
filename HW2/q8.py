import numpy as np


def create_train_test(mu, sigma):

    train = np.random.multivariate_normal(mu, sigma, 50)

    test = np.random.multivariate_normal(mu, sigma, 50)

    return train, test


def find_elements_in_range(begin, end, data):

    count = 0

    for elem in data:

        if(elem[0]>= begin[0] and elem[0]<= end[0] and elem[1]>=begin[1] and elem[1]<= end[1]):

            count += 1

    return count


def find_bin(num, interval):

    for i in range(0, len(interval)):

        if num[0] >= interval[i][0][0] and num[0] <= interval[i][1][0] and num[1] >= interval[i][0][1] and num[1] <= interval[i][1][1]:
            return i

    return 0


def create_bin_dist(data, bin, begin, end):


    X_length = (end[0] - begin[0]) / bin
    Y_length = (end[1] - begin[1]) / bin

    bin_size = [X_length, Y_length]

    interval = []
    pdf = []

    for i in range(0, bin):

        for j in range(0, bin):

            pdf.append(find_elements_in_range([begin[0]+ i* bin_size[0], begin[1]+ j* bin_size[1]],
                                              [begin[0]+ (i+1)* bin_size[0], begin[1]+ (j+1)* bin_size[1]]
                                              , data))
            interval.append(([begin[0]+ i* bin_size[0], begin[1]+ j* bin_size[1]],
                                          [begin[0]+ (i+1)* bin_size[0], begin[1]+ (j+1)* bin_size[1]]))

    return interval, pdf


def predic(indexes, datasets):

    confusion_mat = np.zeros((3, 3))

    for i in range(0, len(datasets)):
        dataset = datasets[i]

        for num in dataset:

            index = find_bin(num, intervals)

            label = 2

            if (pdf_1[index] >= pdf_2[index] and pdf_1[index] >= pdf_3[index]):
                label = 0

            if (pdf_2[index] >= pdf_1[index] and pdf_2[index] >= pdf_3[index]):
                label = 1

            confusion_mat[indexes[i], label] += 1

    return confusion_mat


mu, sigma = [0, 0], [[1, 0.5],
                     [0.5, 1]]

train_1, test_1 = create_train_test(mu, sigma)

mu, sigma = [2, 2], [[1, 0.4],
                     [0.4, 1]]

train_2, test_2 = create_train_test(mu, sigma)

mu, sigma = [4, 1], [[1, 0.3],
                     [0.3, 1]]

train_3, test_3 = create_train_test(mu, sigma)



X_min = np.min([np.min(train_1[:,0]), np.min(train_2[:,0]), np.min(train_3[:,0]),
                np.min(test_1[:, 0]), np.min(test_2[:, 0]), np.min(test_3[:, 0])])

Y_min = np.min([np.min(train_1[:,1]), np.min(train_2[:,1]), np.min(train_3[:,0]),
                np.min(test_1[:, 1]), np.min(test_2[:, 1]), np.min(test_3[:, 0])])

X_max = np.max([np.max(train_1[:,0]), np.max(train_2[:,0]), np.max(train_3[:,0]),
                np.max(test_1[:, 0]), np.max(test_2[:, 0]), np.max(test_3[:, 0])])


Y_max = np.max([np.max(train_1[:,1]), np.max(train_2[:,1]), np.max(train_3[:,1]),
                np.max(test_1[:, 1]), np.max(test_2[:, 1]), np.max(test_3[:, 1])])

bin = 5

begin = [X_min, Y_min]
end = [X_max, Y_max]

intervals, pdf_1 = create_bin_dist(train_1, bin, begin, end)
intervals, pdf_2 = create_bin_dist(train_2, bin, begin, end)
intervals, pdf_3 = create_bin_dist(train_3, bin, begin, end)


confusion_mat = predic([0,1,2], [test_1, test_2, test_3])

print('Test Confusion Matrix')

print(confusion_mat)

confusion_mat = predic([0,1,2], [train_1, train_2, train_3])

print('Train Confusion Matrix')

print(confusion_mat)