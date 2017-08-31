from sklearn import (datasets, decomposition)
from Online_PCA import OnlinePCA
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

##############
# Power method
def power_iterations(R, w):
    while 1:
        w1 = w.copy()
        w = R * w
        w /= LA.norm(w)
        if converged(w, w1):
            break
    return w


# test whether two normalized vectors are the same (or deviation within a threshold)
def converged(w, w1, thres=1e-10):
    converged = False
    corr = w.T * w1
    # print corr
    if abs(corr) > 1 - thres:
        converged = True
    return converged


def powerPCA(data, n_pcs):
    nr, dim = data.shape
    X = np.matrix(data)
    m = np.mean(data, axis=0)
    R = (X - m).T * (X - m) / (nr - 1.0)
    R0 = R.copy()

    # initialize
    w = np.matrix(np.random.rand(dim)).T
    w /= LA.norm(w)
    # iterate for 1st eigenvector
    w = power_iterations(R, w)
    # first eigenvector as first column of W
    W = w
    # iterate for other eigenvectors
    for i in range(1, n_pcs):
        # deflate R
        R -= w * w.T * R
        # initialize and Power iter
        w = np.matrix(np.random.rand(dim)).T
        w /= LA.norm(w)
        w = power_iterations(R, w)
        # attach the new eigenvector to W as a new column
        W = np.c_[W, w]

    # get eigenvalues and save them in the ev array
    y = R0 * W
    ev = LA.norm(y, axis=0) / LA.norm(W, axis=0)
    return W, ev
##############


digits = datasets.load_digits(n_class=6)
total_X = digits.data
total_y = digits.target

# task 1: use first 100 data as base, loop through the following data to update covariance
# matrix, then at last to compute eigenvectors
# onlinePCA = OnlinePCA(X=total_X[:100])
#
# for i in range(101, total_X.shape[0]):
#     onlinePCA.updateCov(total_X[i])
#
# W0, ev0 = onlinePCA.computePCA(n_pcs=2)
#
# prj = np.matrix(total_X) * W0[:, 0:2]
# onlinePCA.plot_embedding(prj, total_y, "my onlinePCA result")
# plt.show()

# compare the eigenvector gained in each loop with the original eigenvector obtained through
# batch mode
# X=np.matrix(total_X)
# m=np.mean(total_X,axis=0)
# X-=m
# R=X.T*X/(total_X.shape[0]-1.0)
# W0,ev0=powerPCA(total_X,n_pcs=2)
# prj=X*W0[:,0:2]
#
# w0 = W0.T[0]
#
# onlinePCA = OnlinePCA(X=total_X[:100])
# covariances = []
# index = []
# for i in range(101, total_X.shape[0]):
#     cov = onlinePCA.comparisionBetweenTwoEigenvector(w0=w0.T, w=total_X[i])
#     covariances.append(cov)
#     index.append(i)
#
# plt.plot(index, covariances)
# plt.show()
# cov = onlinePCA.comparisionBetweenTwoEigenvector(w0=w0.T, w=total_X[101])

'''
1. randomly extract 100 data instances that belong to "0" class, use it to initialize PCA
'''
datasets_zero = []
for i in range(total_y.shape[0]):
    if total_y[i] == 0:
        datasets_zero.append(total_X[i])

datasets_zero = np.matrix(datasets_zero)
np.random.shuffle(datasets_zero)

onlinePCA = OnlinePCA(X=datasets_zero[:100])
W0, ev0 = onlinePCA.computePCA(n_pcs=2)
