import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import anderson
from PIL import Image, ImageDraw, ImageOps
from sklearn.preprocessing import scale
from Tools import Tools
import matplotlib.pyplot as pl

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

        # # if i use eigenvector to initalize the position of centroids
        # e = self.get_prime_vector(X)
        # # use e to create initial centroids
        # initial_c0 = centroid + e

        km = self.KMeans().split(dataSet=X)
        v = km.c_0 - km.c_1

        X_prime = scale(X.dot(v) / (v.dot(v)))
        # use X * e, matrix multiple to map the X into one dimension
        # currently, v = [float, float, float], we need to convert it into
        # the eigen vector form [matrix, matrix, matrix]

        accept_split = self.checkGaussianStatistic(X_prime)

        if accept_split:
            self.__recursive_clustering(km.cluster_0)
            self.__recursive_clustering(km.cluster_1)
        else:
            self.centroids.append(centroid)
            self.clusters.append(np.array(original_X))

    def checkGaussianStatistic(self, X):
        X = X - np.mean(X)
        output = anderson(X)
        statistic = output[0]
        critical_value = output[1][self.stricklevel]

        if statistic > critical_value:
            return True
        else:
            return False

    def get_X_prime(self, X):
        pca = self.PCA(X)
        e = pca.find_the_first_eigen_vector()
        # print "the prime eigenvector = \n{}".format(e)
        return X * e

    def get_prime_vector(self, X):
        pca = self.PCA(X)
        e = pca.find_the_first_eigen_vector()
        return e

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
    dataSet3 = np.random.randn(400, 2) - 2
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

if __name__ == '__main__':
    img = Image.open("africa.jpg")
    img = ImageOps.fit(img, (200, 100), Image.ANTIALIAS)

    # img.show()

    # pix = np.array(img)[:, :, 0:3]
    # pix = pix.reshape((-1, 3)).astype(float)

    # gm = GMeans().fit(pix)
    # print len(gm.centroids)

    # pix_prime = gm.get_X_prime(pix)
    # print "begin computing the anderson statistics:"
    # print "data size is {}".format(pix_prime.shape)
    # output = anderson(pix_prime)
    # print "the anderson statistics is \n{}".format(output[0])
    # print "cricital value = {}".format(output[1])

    demo1()
    pl.show()


