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
        self.index_records = []
        self.X = None
        self.labels = []
        self.index_result = []

    def __recursive_cluster_index(self, indexes):
        if len(indexes) < 2:
            return
        original_indexes = []
        for index in indexes:
            original_indexes.append(index)

        centroid = np.mean(self.X[indexes], axis=0)

        km = self.KMeans().split(dataSet=self.X, index_records=indexes)
        v = km.c_0 - km.c_1

        X_prime = self.get_X_prime_original(self.X[indexes], v)
        accept_split = self.checkAnderson(X_prime)

        if accept_split:
            # km.cluster_0 should be the index_records
            self.__recursive_cluster_index(km.index_records_0)
            self.__recursive_cluster_index(km.index_records_1)
        else:
            self.centroids.append(centroid)
            self.index_result.append(original_indexes)

    def fit(self, X):
        self.X = X.copy()
        for i in range(len(X)):
            self.index_records.append(i)
        # self.index_records = np.asarray(self.index_records)

        self.__recursive_cluster_index(self.index_records)
        self.centroids = np.array(self.centroids)

        # print len(self.centroids), type(self.centroids)
        # print len(self.index_result), type(self.index_result)
        # generate lables
        self.labels = np.zeros(len(self.index_records))
        for label in range(len(self.index_result)):
            for each in self.index_result[label]:
                self.labels[each] = label

        self.labels = np.asarray(self.labels).astype(int)
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
            self.__recursive_clustering(km.index_records_0)
            self.__recursive_clustering(km.index_records_1)
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
            self.index_records_0 = []
            self.index_records_1 = []
            self.c_0 = None
            self.c_1 = None

        def split(self, dataSet, index_records):
            print "\nindex_records length = {}".format(len(index_records))
            print "index_records = {}".format(index_records)
            kM = KMeans(init='k-means++', n_clusters=2).fit(dataSet[index_records])

            self.c_0 = kM.cluster_centers_[0]
            self.c_1 = kM.cluster_centers_[1]

            for index in range(len(index_records)):
                if kM.labels_[index] == 0:
                    self.index_records_0.append(index_records[index])
                else:
                    self.index_records_1.append(index_records[index])

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
    X, y = make_blobs(n_samples=200, centers=5, n_features=2, random_state=100)
    gm = GMeans().fit(X)
    print "found {} centroids".format(len(gm.centroids))
    Tools.draw(X=X, lables=gm.labels, centroids=gm.centroids)

# def demo2():
#     img = Image.open("africa.jpg")
#     img = ImageOps.fit(img, (200, 100), Image.ANTIALIAS)
#     # img.show()
#
#     pix = np.array(img)[:, :, 0:3]
#     pix = pix.reshape((-1, 3)).astype(float)
#
#     gm = GMeans().fit(pix)
#     print len(gm.centroids)

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

    Tools.draw(X=X, lables=xm.labels_, centroids=xm.cluster_centers_)


if __name__ == '__main__':

    random_state = 0

    # Anisotropicly distributed data
    X, y = datasets.make_blobs(n_samples=1000, centers=7, n_features=2, random_state=random_state)
    # transformation = [[0.6, -0.6], [-0.4, 0.8]]
    # X = np.dot(X, transformation)

    gm = GMeans().fit(X)
    Tools.draw(X=X, lables=gm.labels, centroids=gm.centroids, title="G-Mean")

    # pl.figure()
    # xm = XMeans()
    # xm = xm.fit(X)
    # Tools.draw(X=X, lables=xm.labels_, centroids=xm.cluster_centers_, title="X-Mean")

    pl.show()






