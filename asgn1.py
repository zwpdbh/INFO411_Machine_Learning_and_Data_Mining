from sklearn import (datasets, decomposition)
from Online_PCA import OnlinePCA
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


##############


digits = datasets.load_digits(n_class=6)
total_X = digits.data
total_y = digits.target

# task 1: use first 100 data as base, loop through the following data to update covariance
# matrix, then at last to compute eigenvectors

onlinePCA = OnlinePCA(X=total_X[:100])

for i in range(101, total_X.shape[0]):
    onlinePCA.updateCov(total_X[i])

W0, ev0 = onlinePCA.onlinePowerPCA(n_pcs=2)

prj = np.matrix(total_X) * W0[:, 0:2]
onlinePCA.plot_embedding(prj, total_y, "my onlinePCA result")
plt.show()

# compare the eigenvector gained in each loop with the original eigenvector obtained through
# batch mode
#
# X = np.matrix(total_X)
# print X.shape
# W0, ev0 = OnlinePCA.powerPCA(X, n_pcs=2)
# prj = X * W0[:, 0:2]
#
# w0 = W0.T[0]
#
# onlinePCA = OnlinePCA(X=total_X[:100])
# covariances = []
# index = []
# for i in range(101, total_X.shape[0]):
#     onlinePCA.updateCov(total_X[i])
#     cov = onlinePCA.compareWithCurrentEigenvector(w0=w0.T)
#     covariances.append(cov)
#     index.append(i)
#
# plt.plot(index, covariances)
# plt.show()
# cov = onlinePCA.comparisionBetweenTwoEigenvector(w0=w0.T, w=total_X[101])



# '''
# 1. randomly extract 100 data instances that belong to "0" class, use it to initialize PCA
# '''

# datasets_zero = []
# for i in range(total_y.shape[0]):
#     if total_y[i] == 0:
#         datasets_zero.append(total_X[i])
#
# datasets_zero = np.matrix(datasets_zero)
# np.random.shuffle(datasets_zero)
#
# onlinePCA = OnlinePCA(X=datasets_zero[:100])
# initialR = onlinePCA.initialR.copy()
#
# e0, ev0 = onlinePCA.computePCA(n_pcs=1)
# e0 = np.squeeze(np.asarray(e0))
#
# inlier = []
# outlier = []
#
# covs = []
# index = []
# for i in range(total_X.shape[0]):
#     index.append(i)
#     onlinePCA.initialR = initialR.copy()
#     onlinePCA.updateCov(x=total_X[i], gamma=0.05)
#     cov = onlinePCA.compareWithCurrentEigenvector(e0)
#     covs.append(cov)
#
# plt.plot(index, covs)
# plt.show()