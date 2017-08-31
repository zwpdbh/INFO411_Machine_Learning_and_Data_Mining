import numpy as np
import matplotlib.pylab as plt
import numpy.linalg as LA
import matplotlib.pyplot as plt

'''
The powerPCA method given about carries out PCA ina batch mode. It needs to gather all data
to compute the covariance matrix R, and then the power iterations follow
(along with the deflation of R for each eigenvector obtained).

For online PCA:
First, use a mini-batch, say the first 100 rows of X to obtain the mean vector m, and the covariance matrix R 
in the same way as in the powerPCA() method.

1. use the next row of X as input x, and update the online average m.
2. use x to update R (as an online average of x.T * x)
3. write an onlinePowerPCA(R) method that derives the eigenvectors and eigenvalues of the current R.
4. Repeat the above process until all entries of X are handled.

Following the idea we introduce in online clustering, we can use the exponential weighting to produce
the online average, e.g
m <= r * m + (1 - r)x
where r is a weighting coefficient, 0 < r << 1. The same idea works for updating R as Well.
'''

class OnlinePCA:
    def __init__(self, X):
        self.X = np.matrix(X)
        self.m = np.mean(X, axis=0)
        self.X = self.X - self.m
        self.R = self.X.T * self.X / (self.X.shape[0] - 1.0)
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.initialR = self.R.copy()

    def getCovarianceMatrix(self):
        return self.R

    def updateCov(self, x, gamma=None):
        self.n += 1
        x = np.matrix(x)
        if gamma==None:
            gamma = (1.0 / self.n)
        self.m = self.m + gamma * (x - self.m)
        self.R = self.R + gamma * (x.T * x - self.R)

    def converged(self, w, w1, thres=1e-10):
        convergent = False
        corr = w.T * w1
        if abs(corr) > 1 - thres:
            convergent = True
        return convergent

    def power_iteration(self, R, w):
        while 1:
            w1 = w.copy()
            w = R * w
            w = w / LA.norm(w)
            if self.converged(w, w1):
                break
        return w

    def plot_embedding(self, X, labels, title=None):
        y = labels.copy()
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X1 = (X - x_min) / (x_max - x_min)

        plt.figure()
        for i in range(X.shape[0]):
            plt.text(X1[i, 0], X1[i, 1], str(labels[i]),
                     color=plt.cm.Set1(y[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})

        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    # compute the eigenvectors and eigenvalues based on the current covariance matrix stored as self.R
    def computePCA(self, n_pcs):
        R0 = self.R.copy()
        w = np.matrix(np.random.rand(self.dim)).T
        w = w / LA.norm(w)
        w = self.power_iteration(self.R, w)

        W = w
        for i in range(1, n_pcs):
            self.R = self.R - w * w.T * self.R
            w = np.matrix(np.random.rand(self.dim)).T
            w = w / LA.norm(w)
            w = self.power_iteration(self.R, w)
            W = np.c_[W, w]

        y = R0 * W
        ev = LA.norm(y, axis=0) / LA.norm(W, axis=0)
        return W, ev

    # w0 is the batch mode eigenvector
    def comparisionBetweenTwoEigenvector(self, w0, w, gamma=None):
        self.updateCov(w, gamma)
        # compute the first eigenvector, based on the current updated covariance matrix
        W, ev = self.computePCA(n_pcs=1)
        w = np.squeeze(np.asarray(W))
        print w.shape

        w0 = np.squeeze(np.asarray(w0))
        mean_w0 = np.mean(w0)
        mean_w = np.mean(w)
        s = 0
        for i in range(w.shape[0]):
            s += (w[i] - mean_w) * (w0[i] - mean_w0)
        return s / w.shape[0]