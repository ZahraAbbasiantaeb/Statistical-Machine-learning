import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso

nan = np.nan

imputer = KNNImputer(n_neighbors=3, weights="uniform")

x = np.loadtxt(open("/Users/zahra_abasiyan/PycharmProjects/SML/HW03/dataset/Dataset2_extended.csv", "rb"), delimiter=",")

extended_data = np.array(x).astype("float")

print(np.shape(extended_data))

np.place(extended_data, extended_data == 0, nan)

extended_data = imputer.fit_transform(extended_data)

lasso_reg = Lasso(normalize=True,  alpha=0.001, fit_intercept=False)

lasso_reg.fit(extended_data[:,0:6], extended_data[:,6])

print(lasso_reg.coef_)