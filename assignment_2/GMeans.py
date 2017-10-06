import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import anderson
from PIL import Image, ImageDraw, ImageOps
from sklearn.preprocessing import scale
from Tools import Tools
import matplotlib.pyplot as pl
import math
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster, datasets, mixture
from XMeans import XMeans
import matplotlib.cm as cm

# for each cluster do the split
# test if the split cluster fit anderson test
# if so, accept the split, otherwise, not accept
class GMeans:
    def __init__(self, strictlevel=4):
        self.stricklevel = strictlevel
        self.centroids = []
        self.clusters = []

    def fit(self, X):
        self.__recursive_clustering(X)
        self.centroids = np.array(self.centroids)

        return self

    # recursively apply k-means on X, with n-clusters = 2
    # the result is 2 centroids, then apply anderson test on the result
    # if it is greater than before, accept the test. Otherwise, keep original result
    def __recursive_clustering(self, X):
        if len(X) < 2:
            return

        original_X = X.copy()
        X = np.asarray(X)
        centroid = np.mean(X, axis=0)

        km = self.KMeans().split(dataSet=X)
        v = km.c_0 - km.c_1

        # X_prime = self.get_X_prime(X, v)
        X_prime = self.get_X_prime_original(X, v)

        accept_split = self.checkAnderson(X_prime)

        if accept_split:
            self.__recursive_clustering(km.cluster_0)
            self.__recursive_clustering(km.cluster_1)
        else:
            self.centroids.append(centroid)
            self.clusters.append(np.array(original_X))

    def get_X_prime_original(self, X, v):
        # normalize v, need?
        # v = v / np.sqrt(v.T * v)
        X_prime = scale(X.dot(v) / (v.dot(v)))

        return X_prime

    def get_X_prime(self, X, v):
        v = v.reshape((len(v), 1))
        v = np.asmatrix(v)

        X_prime = X * v
        # X_prime is: (n, 1) <class 'numpy.matrixlib.defmatrix.matrix'>
        tmp = []
        for x in X_prime:
            tmp.append(x.item(0))
        return tmp

    def get_X_prime_new(self, X):
        # compute eigenvector of X
        pca = self.PCA(X)
        e = pca.find_the_first_eigen_vector()

    def checkAnderson(self, X):
        X = X - np.mean(X)
        output = anderson(X)
        statistic = output[0]
        critical_value = output[1][self.stricklevel]
        if np.isnan(statistic):
            return False

        # print statistic, critical_value
        if statistic > critical_value:
            return True
        else:
            return False

    # inner class for doing the K-Means clustering
    class KMeans:
        def __init__(self):
            self.centroids = np.array([])
            self.cluster_0 = []
            self.cluster_1 = []
            self.c_0 = None
            self.c_1 = None

        def split(self, dataSet):
            kM = KMeans(init='k-means++', n_clusters=2).fit(dataSet)

            self.c_0 = kM.cluster_centers_[0]
            self.c_1 = kM.cluster_centers_[1]

            index = 0
            for label in kM.labels_:
                if label == 0:
                    self.cluster_0.append(dataSet[index])
                else:
                    self.cluster_1.append(dataSet[index])
                index += 1

            self.cluster_0 = np.array(self.cluster_0)
            self.cluster_1 = np.array(self.cluster_1)
            return self

    class PCA:
        def __init__(self, data):
            self.X = np.matrix(data)
            self.nr = self.X.shape[0]
            self.dim = self.X.shape[1]

            # make it zero meaned:
            self.m = np.mean(self.X, axis=0)
            self.X -= self.m
            self.cov = (self.X.T * self.X) / self.nr

        def find_the_first_eigen_vector(self):
            w = np.matrix(np.ones(self.dim)).T
            w = w / np.sqrt(w.T * w)
            e = None
            for i in range(100):
                e = self.cov * w
                e = e / np.sqrt(e.T * e)
                if np.dot(e.T, w) == 1:
                    break
                else:
                    w = e
            return e



def demo1():
    X, y = make_blobs(n_samples=1000, centers=5, cluster_std=[0.4, 1.5, 1.7, 0.4, 1.3], n_features=2, random_state=0)

    gm = GMeans().fit(X)

    print "found {} clusters".format(len(gm.clusters))
    print "found {} centroids".format(len(gm.centroids))
    Tools.drawCentroids(gm.centroids)
    Tools.drawClusters(gm.clusters)

def demo2():
    img = Image.open("africa.jpg")
    img = ImageOps.fit(img, (200, 100), Image.ANTIALIAS)
    # img.show()

    pix = np.array(img)[:, :, 0:3]
    pix = pix.reshape((-1, 3)).astype(float)

    gm = GMeans().fit(pix)
    print len(gm.centroids)

def demo3():
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=1000, centers=7, n_features=2, random_state=random_state)

    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    gm = GMeans().fit(X_aniso)

    print "found {} clusters".format(len(gm.clusters))
    print "found {} centroids".format(len(gm.centroids))
    Tools.drawCentroids(gm.centroids)
    Tools.drawClusters(gm.clusters)

def demo_XMean():
    random_state = 170
    X, y = datasets.make_blobs(n_samples=1000, centers=7, n_features=2, random_state=random_state)

    xm = XMeans()
    xm = xm.fit(X)

    print "found {} centroids".format(len(xm.cluster_centers_))
    Ys = np.array([[4, 8, 12, 16],
                   [1, 4, 9, 16],
                   [17, 10, 13, 18],
                   [9, 10, 18, 11],
                   [4, 15, 17, 6],
                   [7, 10, 8, 7],
                   [9, 0, 10, 11],
                   [14, 1, 15, 5],
                   [8, 15, 9, 14],
                   [20, 7, 1, 5]])

    colors = cm.rainbow(np.linspace(0, 1, len(Ys)))

    # draw centers
    for c in xm.cluster_centers_:
        pl.plot(c[0], c[1], 'ro')

    # draw each data
    for i in range(len(xm.labels_)):
        p = X[i]
        label = xm.labels_[i]
        pl.scatter(p[0], p[1], s=1, c=colors[label % len(colors)])


if __name__ == '__main__':


    # demo1()
    # demo2()
    # demo3()

    demo_XMean()

    pl.show()

    # list = [[1], [2], [3], [4], [5]]
    # print getWhatIWant(list).shape





