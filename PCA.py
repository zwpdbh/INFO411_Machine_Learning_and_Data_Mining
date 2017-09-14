import numpy as np
import matplotlib.pylab as plt
import numpy.linalg as LA


class PCA:
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
            if PCA.converged(w, w1):
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
        w = PCA.power_iterations(R, w)
        # first eigenvector as first column of W
        W = w
        # iterate for other eigenvectors
        for i in range(1, n_pcs):
            # deflate R
            R -= w * w.T * R
            # initialize and Power iter
            w = np.matrix(np.random.rand(dim)).T
            w /= LA.norm(w)
            w = PCA.power_iterations(R, w)
            # attach the new eigenvector to W as a new column
            W = np.c_[W, w]

        # get eigenvalues and save them in the ev array
        y = R0 * W
        ev = LA.norm(y, axis=0) / LA.norm(W, axis=0)
        return W, ev

    # using given R to compute eigenvectors and eigenvalues
    @staticmethod
    def onlinePowerPCA(R, n_pcs):
        dim = R.shape[0]
        R0 = R.copy()

        # initialize
        w = np.matrix(np.random.rand(dim)).T
        w /= LA.norm(w)
        # iterate for 1st eigenvector
        w = PCA.power_iterations(R, w)
        # first eigenvector as first column of W
        W = w
        # iterate for other eigenvectors
        for i in range(1, n_pcs):
            # deflate R
            R -= w * w.T * R
            # initialize and Power iter
            w = np.matrix(np.random.rand(dim)).T
            w /= LA.norm(w)
            w = PCA.power_iterations(R, w)
            # attach the new eigenvector to W as a new column
            W = np.c_[W, w]

        # get eigenvalues and save them in the ev array
        y = R0 * W
        ev = LA.norm(y, axis=0) / LA.norm(W, axis=0)
        return W, ev

    # TPR = TP/(TP+FN)
    # compare true label with my labels which come from threshold of corr
    @staticmethod
    def computeTPR(total_y, my_labels):
        tp = 0
        fn = 0

        for i in range(total_y.shape[0]):
            if total_y[i] == 0:
                if my_labels[i] == 'y':
                    tp += 1
                    # print "y"
                elif my_labels[i] == 'n':
                    # print "n"
                    fn += 1

        return tp * 1.0 / (tp + fn) * 1.0

    # FPR = FP / (FP + TN)
    # compare true label with my labels which come from threshold of corr
    @staticmethod
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


