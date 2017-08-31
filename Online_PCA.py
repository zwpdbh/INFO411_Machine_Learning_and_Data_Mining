import numpy as np
import matplotlib.pylab as plt
import numpy.linalg as LA


class OnlinePCA:
    def __init__(self, X):
        self.X = np.matrix(X)
        self.m = np.mean(X, axis=0)
        self.X = self.X - self.m
        self.R = self.X.T * self.X / (self.X.shape[0] - 1.0)
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.initialR = self.R.copy()

    @staticmethod
    def converged(w, w1, thres=1e-10):
        convergent = False
        corr = w.T * w1
        if abs(corr) > 1 - thres:
            convergent = True
        return convergent

    @staticmethod
    def power_iterations(R, w):
        while 1:
            w1 = w.copy()
            w = R * w
            w = w / LA.norm(w)
            if OnlinePCA.converged(w, w1):
                break
        return w

    @staticmethod
    def plot_embedding(X, labels, title=None):
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

    @staticmethod
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
        w = OnlinePCA.power_iterations(R, w)
        # first eigenvector as first column of W
        W = w
        # iterate for other eigenvectors
        for i in range(1, n_pcs):
            # deflate R
            R -= w * w.T * R
            # initialize and Power iter
            w = np.matrix(np.random.rand(dim)).T
            w /= LA.norm(w)
            w = OnlinePCA.power_iterations(R, w)
            # attach the new eigenvector to W as a new column
            W = np.c_[W, w]

        # get eigenvalues and save them in the ev array
        y = R0 * W
        ev = LA.norm(y, axis=0) / LA.norm(W, axis=0)
        return W, ev

    def getCovarianceMatrix(self):
        return self.R

    def updateCov(self, x, gamma=None):
        self.n += 1
        x = np.matrix(x)
        if gamma==None:
            gamma = (1.0 / self.n)
        self.m = self.m + gamma * (x - self.m)
        self.R = self.R + gamma * (x.T * x - self.R)

    # compute the eigenvectors and eigenvalues based on the current covariance matrix stored as self.R
    def onlinePowerPCA(self, n_pcs):
        R0 = self.R.copy()
        w = np.matrix(np.random.rand(self.dim)).T
        w = w / LA.norm(w)
        w = self.power_iterations(self.R, w)

        W = w
        for i in range(1, n_pcs):
            self.R = self.R - w * w.T * self.R
            w = np.matrix(np.random.rand(self.dim)).T
            w = w / LA.norm(w)
            w = self.power_iterations(self.R, w)
            W = np.c_[W, w]

        y = R0 * W
        ev = LA.norm(y, axis=0) / LA.norm(W, axis=0)
        return W, ev

    # compute the correlation between current first eigenvector with w0
    def computeCorr(self, w0):
        # compute the first eigenvector, based on the current updated covariance matrix
        e1, ev = self.onlinePowerPCA(n_pcs=1)

        e1 = np.squeeze(np.asarray(e1))
        w0 = np.squeeze(np.asarray(w0))
        mean_e1 = np.mean(e1)
        mean_w0 = np.mean(w0)
        s = 0
        for i in range(w0.shape[0]):
            s += (e1[i] - mean_e1) * (w0[i] - mean_w0)
        return s / 64

