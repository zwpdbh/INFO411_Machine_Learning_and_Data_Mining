import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import anderson
from PIL import Image, ImageDraw, ImageOps
from sklearn.preprocessing import scale

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

    # recursively apply k-means on X, with n-clusters = 2
    # the result is 2 centroids, then apply anderson test on the result
    # if it is greater than before, accept the test. Otherwise, keep original result
    def __recursive_clustering(self, X):
        centroid = np.mean(X, axis=0)
        km = self.KMeans().split(X=X)
        v = km.c_0 - km.c_1

        # this is not working because it is in 3 dimension space, we should
        X_prime = self.get_X_prime(X)
        X_prime = scale(X.dot(v) / (v.dot(v)))
        X_prime = X * v

        # use X * e, matrix multiple to map the X into one dimension
        # currently, v = [float, float, float], we need to convert it into
        # the eigen vector form [matrix, matrix, matrix]


        accept_split = self.checkGaussianStatistic(X_prime)

        if accept_split:
            self.__recursive_clustering(km.cluster_0)
            self.__recursive_clustering(km.cluster_1)
        else:
            self.centroids.append(centroid)

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

    class KMeans:
        def __init__(self):
            self.centroids = np.array([])
            self.cluster_0 = []
            self.cluster_1 = []
            self.c_0 = None
            self.c_1 = None

        def split(self, X):
            km = KMeans(init='k-means++', n_clusters=2).fit(X)
            self.c_0 = km.cluster_centers_[0]
            self.c_1 = km.cluster_centers_[1]

            label = 0
            for label in km.labels_:
                if  label == 0:
                    self.cluster_0.append(X[label])
                else:
                    self.cluster_1.append(X[label])
                label += 1

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


if __name__ == '__main__':
    img = Image.open("africa.jpg")
    img = ImageOps.fit(img, (200, 100), Image.ANTIALIAS)

    # img.show()

    pix = np.array(img)[:, :, 0:3]
    pix = pix.reshape((-1, 3)).astype(float)

    gm = GMeans().fit(pix)
    print len(gm.centroids)

    # pix_prime = gm.get_X_prime(pix)
    # print "begin computing the anderson statistics:"
    # print "data size is {}".format(pix_prime.shape)
    # output = anderson(pix_prime)
    # print "the anderson statistics is \n{}".format(output[0])
    # print "cricital value = {}".format(output[1])
