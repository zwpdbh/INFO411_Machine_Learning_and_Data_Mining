import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# Power method
def power_iteration(R, w):
    while 1:
        w1 = w.copy()
        w = R * w
        w = w / LA.norm(w)
        if converged(w, w1):
            break
    return  w


def converged(w, w1, thres=1e-10):
    convergent = False
    corr = w.T * w1
    if abs(corr) > 1 - thres:
        convergent = True
    return convergent


def powerPCA(data, n_pcs):
    nr, dim = data.shape
    X = np.matrix(data)
    m = np.mean(data, axis=0)
    R = (X - m).T * (X - m) / (nr - 1.0)
    R0 = R.copy()

    # initialize
    w = np.matrix(np.random.rand(dim)).T
    w = w / LA.norm(w)

    # literate for 1st eigenvector
    w = power_iteration(R, w)

    # first eigenvector as first column of W
    W = w

    # iterate for other eigenvectors
    for i in range(1, n_pcs):
        # deflate R
        R = R - w * w.T * R
        # initialize and Power iter
        w = np.matrix(np.random.rand(dim)).T
        w = w / LA.norm(w)
        w = power_iteration(R, w)
        # attach the new eigenvector to W as a new column
        W = np.c_[W, w]

    # get eigenvalues and save them in the ev array
    y = R0 * W
    ev = LA.norm(y, axis=0) / LA.norm(W, axis=0)
    return W, ev


# a plot-embedding() function that displays 2-D data points with their corresponding labels
# in different colours.
# scale and visualize the embedding vector
def plot_embedding(X, labels, title = None):
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
    plt.show()
