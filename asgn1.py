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

# onlinePCA = OnlinePCA(X=total_X[:100])
#
# for i in range(101, total_X.shape[0]):
#     onlinePCA.updateCov(total_X[i])
#
# W0, ev0 = onlinePCA.onlinePowerPCA(n_pcs=2)
#
# prj = np.matrix(total_X) * W0[:, 0:2]
# onlinePCA.plot_embedding(prj, total_y, "my onlinePCA result")
# plt.show()


# task2: compare the eigenvector gained in each loop with the original eigenvector obtained through
# batch mode
#
# X = np.matrix(total_X)
# print X.shape
# W0, ev0 = OnlinePCA.powerPCA(X, n_pcs=2)
#
# w0 = W0.T[0]
#
# onlinePCA = OnlinePCA(X=total_X[:100])
# corrs = []
# index = []
# for i in range(101, total_X.shape[0]):
#     onlinePCA.updateCov(total_X[i])
#     corr = onlinePCA.computeCorr(w0=w0.T)
#     corrs.append(corr)
#     index.append(i)
#
# plt.plot(index, corrs)
# plt.show()
# cov = onlinePCA.computeCorr(w0=w0.T, w=total_X[101])



'''
task3 randomly extract 100 data instances that belong to "0" class, use it to initialize PCA
'''
# 1 Randomly extract 100 data instances that belong to the "0" class
datasets_zero = []
for i in range(total_y.shape[0]):
    if total_y[i] == 0:
        datasets_zero.append(total_X[i])

datasets_zero = np.matrix(datasets_zero)
np.random.shuffle(datasets_zero)

# 2 construct the initial R matrix, get the first eigenvector from the 100 "0"'s samples
onlinePCA = OnlinePCA(X=datasets_zero[:100])
initialR = onlinePCA.initialR.copy()

e0, ev0 = onlinePCA.onlinePowerPCA(n_pcs=1)

# outlier detection
inlier = []
outlier = []
corrs = []
index = []
my_labels = []

# if the corr > threshold, it will be classified as positive
threshold = 0.5

for i in range(total_X.shape[0]):
    # reset the R to initial R
    onlinePCA.initialR = initialR.copy()
    onlinePCA.updateCov(x=total_X[i], gamma=0.05)
    cov = onlinePCA.computeCorr(w0=e0)
    if cov < threshold:
        my_labels.append('y')
    else:
        my_labels.append('n')
    corrs.append(cov)
    index.append(i)

plt.plot(index, corrs)
# plt.show()


# TPR = TP/(TP+FN)
def computeTPR(total_y, my_labels):
    tp = 0
    fn = 0

    for i in range(total_y.shape[0]):
        if total_y[i] == 0:
            if my_labels[i] == 'y':
                tp += 1
                print "y"
            elif my_labels[i] == 'n':
                print "n"
                fn += 1

    return tp * 1.0 / (tp + fn) * 1.0


# FPR = FP / (FP + TN)
def computeFPR(total_y, my_labels):
    fp = 0
    tn = 0

    for i in range(total_y.shape[0]):
        if total_y[i] != 0:
            if my_labels[i] == 'y':
                fp += 1
            elif my_labels[i] == 'n':
                tn += 1
    return fp * 1.0 / (fp + tn) * 1.0


print computeFPR(total_y=total_y, my_labels=my_labels)
print computeTPR(total_y=total_y, my_labels=my_labels)
