import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from PCA import PCA
from sklearn import (datasets, decomposition)

#
# digits = datasets.load_digits(n_class=6)
# total_X = digits.data
# total_y = digits.target
# n_samples, n_features = total_X.shape
#
#
#
# '''task a1'''
# # 1. use the first 100 rows of data to get the covariance matrix
# X = np.matrix(total_X[:100])
# initial_100_mean = np.mean(X, axis=0)
# mean = initial_100_mean.copy()
#
# initial_100_R  = (X - mean).T * (X - mean) / (n_samples * 1.0)
# R = initial_100_R.copy()
#
# # 2. loop through the rest of data, use each x to update R
# for i in range(101, n_samples):
#     gamma = 1 / (i * 1.0)
#     x = np.matrix(total_X[i])
#     mean = (1 - gamma) * mean + gamma * x
#     R = gamma * (x - mean).T * (x - mean) + (1 - gamma) * R
#
# # 3. use current R to compute eigenvectors and visualize the result
# W0, ev0 = PCA.onlinePowerPCA(R=R, n_pcs=2)
# prj=np.matrix(total_X) * W0[:,0:2]
# # PCA.plot_embedding(prj, total_y, "onlinePowerPCA")
# # plt.show()
#
#
#
#
# # '''task a2'''
# corrs = []
# index = []
#
# batch_e0, batch_ev0 = PCA.powerPCA(data=np.matrix(total_X), n_pcs=1)
#
# mean = initial_100_mean.copy()
# R = initial_100_R.copy()
# for i in range(101, n_samples):
#     gamma = 1 / (i * 1.0)
#     x = np.matrix(total_X[i])
#     # update mean and R
#     mean = (1 - gamma) * mean + gamma * x
#     R = gamma * (x - mean).T * (x - mean) + (1 - gamma) * R
#
#     e0, ev0 = PCA.onlinePowerPCA(R=R, n_pcs=1)
#     corr = np.abs((e0.T * batch_e0).item((0, 0)))
#     corrs.append(corr)
#     index.append(i)
#
# # plt.plot(index, corrs)
# # plt.show()


'''task b1'''
from sklearn import (datasets, decomposition)

digits = datasets.load_digits(n_class=6)
total_X = digits.data
total_y = digits.target
n_samples, n_features = total_X.shape

# 1. randomly extract 100 data instances that belong to 0
X_zeros = []
for i in range(n_samples):
    if total_y[i] == 0:
        X_zeros.append(total_X[i])

# This function only shuffles the array along the first axis
np.random.shuffle(X_zeros)
X_zeros = np.matrix(X_zeros[:100])

# 2. compute initial_R, and use batch method to obtain the first eigenvector of X_zeros
initial_mean = np.mean(X_zeros, axis=0)
mean = initial_mean.copy()

initial_R = (X_zeros - mean).T * (X_zeros - mean) / (100 * 1.0)
R = initial_R.copy()

# eigenvector gain from the first 100, "0" label data
e0_zero, ev0_zero = PCA.powerPCA(data=X_zeros, n_pcs=1)

i = 0
gamma = 0.05
threshold = 0.5

def compute_FPR_and_TPR(dataset, initial_R, initial_mean, gamma, threshold):
    inlier = []
    outlier = []
    corrs = []
    index = []

    my_labels = []
    i = 0

    total_X = dataset.copy()
    initial_R = initial_R.copy()
    initial_mean = initial_mean.copy()

    for x in total_X:
        x = np.matrix(x)

        # 3. update R, compute the first eigenvector
        R = initial_R.copy()
        mean = initial_mean.copy()

        mean = (1 - gamma) * mean + gamma * x
        R = gamma * (x - mean).T * (x - mean) + (1 - gamma) * R

        # compute the new eigenvector
        e0, ev0 = PCA.onlinePowerPCA(R=R, n_pcs=1)

        # compute corr
        corr = np.abs((e0_zero.T * e0).item((0, 0)))
        # print total_y[i], corr
        # do the classification
        if corr > threshold:
            my_labels.append('y')
        else:
            my_labels.append('n')
        i += 1

    return PCA.computeFPR(total_y=total_y, my_labels=my_labels), PCA.computeTPR(total_y=total_y, my_labels=my_labels)


# FPRs = np.empty([3, 1])
# TPRs = np.empty([3, 1])
# thresholds = []
# i = 0
# legends = []
# for gamma in np.linspace(0.02, 0.05, 4):
#     FPRs[i].append([])
#     TPRs[i].append([])
#     for threshold in np.linspace(0, 1, 11):
#         print gamma, threshold
#         result = compute_FPR_and_TPR(total_X, initial_R, initial_mean, gamma=0.05, threshold=threshold)
#         FPRs[i].append(result[0])
#         TPRs[i].append(result[1])
#         # legends.append("gamma={}, threshold={}".format(gamma, threshold))
#     plt.plot(FPRs[i], TPRs[i])
#     i += 1

FPRs_1 = []
TPRs_1 = []
for threshold in np.linspace(0, 1, 11):
    result = compute_FPR_and_TPR(total_X, initial_R, initial_mean, gamma=0.05, threshold=threshold)
    FPRs_1.append(result[0])
    TPRs_1.append(result[1])
plt.plot(FPRs_1, TPRs_1)

FPRs_2 = []
TPRs_2 = []
for threshold in np.linspace(0, 1, 11):
    result = compute_FPR_and_TPR(total_X, initial_R, initial_mean, gamma=0.04, threshold=threshold)
    FPRs_2.append(result[0])
    TPRs_2.append(result[1])
plt.plot(FPRs_2, TPRs_2)

FPRs_3 = []
TPRs_3 = []
for threshold in np.linspace(0, 1, 11):
    result = compute_FPR_and_TPR(total_X, initial_R, initial_mean, gamma=0.03, threshold=threshold)
    FPRs_3.append(result[0])
    TPRs_3.append(result[1])
plt.plot(FPRs_3, TPRs_3)

FPRs_4 = []
TPRs_4 = []
for threshold in np.linspace(0, 1, 11):
    result = compute_FPR_and_TPR(total_X, initial_R, initial_mean, gamma=0.02, threshold=threshold)
    FPRs_4.append(result[0])
    TPRs_4.append(result[1])
plt.plot(FPRs_4, TPRs_4)

plt.legend(['gamma = 0.05',
            'gamma = 0.04',
            'gamma = 0.03',
            'gamma = 0.02'])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

from sklearn.metrics import auc
roc_auc_1 = auc(x=FPRs_1, y=TPRs_1)
roc_auc_2 = auc(x=FPRs_2, y=TPRs_2)
roc_auc_3 = auc(x=FPRs_3, y=TPRs_3)
roc_auc_4 = auc(x=FPRs_4, y=TPRs_4)

print roc_auc_1
print roc_auc_2
print roc_auc_3
print roc_auc_4

# FPRs_1 = []
# TPRs_1 = []
# thresholds = []
# for threshold in np.linspace(0.5, 0.95, 10):
#     thresholds.append(threshold)
#     result = compute_FPR_and_TPR(total_X,  initial_R, initial_mean, gamma=0.05, threshold=threshold)
#     FPRs_1.append(result[0])
#     TPRs_1.append(result[1])
#
#
# FPRs_2 = []
# TPRs_2 = []
# thresholds = []
# for threshold in np.linspace(0.5, 0.95, 10):
#     thresholds.append(threshold)
#     result = compute_FPR_and_TPR(total_X,  initial_R, initial_mean, gamma=0.03, threshold=threshold)
#     FPRs_2.append(result[0])
#     TPRs_2.append(result[1])
#
#
# FPRs_3 = []
# TPRs_3 = []
# thresholds = []
# for threshold in np.linspace(0.5, 0.95, 10):
#     thresholds.append(threshold)
#     result = compute_FPR_and_TPR(total_X,  initial_R, initial_mean, gamma=0.01, threshold=threshold)
#     FPRs_3.append(result[0])
#     TPRs_3.append(result[1])
#
# plt.plot(thresholds, FPRs_1)
# plt.plot(thresholds, FPRs_2)
# plt.plot(thresholds, FPRs_3)
# plt.plot(thresholds, TPRs_1)
# plt.plot(thresholds, TPRs_2)
# plt.plot(thresholds, TPRs_3)
#
# plt.xlabel('threshold')
# plt.ylabel('different FPR and TPR')
# plt.legend(['gamma = 0.05, FPR',
#             'gamma = 0.03, FPR',
#             'gamma = 0.01, FPR',
#             'gamma = 0.05, TPR',
#             'gamma = 0.03, TPR',
#             'gamma = 0.01, TPR'])
# plt.show()




