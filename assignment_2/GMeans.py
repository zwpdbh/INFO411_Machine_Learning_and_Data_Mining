import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import anderson
from PIL import Image, ImageDraw, ImageOps
from sklearn.preprocessing import scale
from Tools import Tools
import matplotlib.pyplot as pl
import math

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
        # normalize v, need?
        # v = v / np.sqrt(v.T * v)
        v = v.reshape((len(v), 1))
        v = np.asmatrix(v)

        # X_prime has to be the shape of (N, )
        X_prime = self.get_X_prime(X, v)
        accept_split = self.checkAnderson(X_prime)

        if accept_split:
            self.__recursive_clustering(km.cluster_0)
            self.__recursive_clustering(km.cluster_1)
        else:
            self.centroids.append(centroid)
            self.clusters.append(np.array(original_X))

    def get_X_prime(self, X, v):
        X_prime = X * v
        # X_prime is: (n, 1) <class 'numpy.matrixlib.defmatrix.matrix'>
        tmp = []
        for x in X_prime:
            tmp.append(x.item(0))
        return tmp

    def getWhatIWant(data):
        a = []
        for i in data:
            a.append(i[0])
        return np.asarray(a)

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
    # demo one:
    dataSet1 = np.random.randn(100, 2) + 2
    dataSet2 = np.random.randn(100, 2) + 5
    dataSet3 = np.random.randn(200, 2) - 2
    dataSet4 = np.random.randn(50, 2)-5
    dataSet5 = np.random.randn(20, 2) - 10

    total_dataSet = np.vstack((dataSet1, dataSet2))
    total_dataSet = np.vstack((total_dataSet, dataSet3))
    total_dataSet = np.vstack((total_dataSet, dataSet4))
    total_dataSet = np.vstack((total_dataSet, dataSet5))

    total_dataSet = np.random.permutation(total_dataSet)

    gM = GMeans(strictlevel=4).fit(total_dataSet)

    print "found {} clusters".format(len(gM.clusters))
    print "found {} centroids".format(len(gM.centroids))

    # Tools.drawDataSet(total_dataSet, 'g+')
    Tools.drawCentroids(gM.centroids)
    Tools.drawClusters(gM.clusters)

def demo2():
    img = Image.open("africa.jpg")
    img = ImageOps.fit(img, (200, 100), Image.ANTIALIAS)
    # img.show()

    pix = np.array(img)[:, :, 0:3]
    pix = pix.reshape((-1, 3)).astype(float)

    gm = GMeans().fit(pix)
    print len(gm.centroids)




if __name__ == '__main__':


    # demo1()

    demo2()

    # list = [[1], [2], [3], [4], [5]]
    # print getWhatIWant(list).shape

    pl.show()


