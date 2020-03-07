import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from q2.C_D import lasso_reg

nan = np.nan

imputer = KNNImputer(n_neighbors=3, weights="uniform")

x = np.loadtxt(open("/Users/zahra_abasiyan/PycharmProjects/SML/HW03/dataset/Dataset2_Unlabeled.csv", "rb"), delimiter=",")

unlabeled_data = np.array(x).astype("float")

print(np.shape(unlabeled_data))

np.place(unlabeled_data, unlabeled_data == 0, nan)

unlabeled_data = imputer.fit_transform(unlabeled_data)

pred_unlabeled = lasso_reg.predict(unlabeled_data)

f = open("dataset02_mylabel.csv","w+")

for num in pred_unlabeled:

    f.write(str(num)+'\n')

f.close()

