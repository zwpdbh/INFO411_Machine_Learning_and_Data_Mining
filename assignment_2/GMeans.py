import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import anderson
from Tools import Tools
import matplotlib.pyplot as pl
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster, datasets, mixture
from XMeans import XMeans
from sklearn.metrics import silhouette_score
import numpy.linalg as LA
from sklearn.preprocessing import scale


# for each cluster do the split
# test if the split cluster fit anderson test
# if so, accept the split, otherwise, not accept

# g-means using k-means to split
class GMeans_01:
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
        # no intialize centroid, just use k++
        km = self.KMeans().split(dataSet=self.X, index_records=indexes)
        v = km.c_0 - km.c_1

        X_prime = self.get_X_prime(self.X[indexes], v)
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

    def get_X_prime(self, X, v):
        # normalize v, need?
        v = v / np.sqrt(v.T * v)
        X_prime = scale(X.dot(v) / (v.dot(v)))

        return X_prime

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
            # print "\nindex_records length = {}".format(len(index_records))
            # print "index_records = {}".format(index_records)
            # print "in gm_01, we use k-mean++ to initialize"
            kM = KMeans(n_clusters=2, init='k-means++').fit(dataSet[index_records])

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


# g-means using eigenvector to split
class GMeans_02:
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
        # compute eigen vector and eigen values
        W0, ev0 = self.powerPCA(data=self.X[indexes], n_pcs=1)
        m = W0 * np.sqrt(2 * ev0 / 3.1415926)
        # c_0, c_1: 2 by 1 matrix
        c_0 = np.matrix(centroid).getT() + m
        c_1 = np.matrix(centroid).getT() - m
        km = self.KMeans(c_0, c_1).split(dataSet=self.X, index_records=indexes)
        v = km.c_0 - km.c_1

        X_prime = self.get_X_prime(self.X[indexes], v)

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

    def get_X_prime(self, X, v):
        # normalize v, need?
        v = v / np.sqrt(v.T * v)
        X_prime = scale(X.dot(v) / (v.dot(v)))

        return X_prime

        # X_prime = (X * v).tolist()
        # tmp = []
        # for each in X_prime:
        #     tmp.append(each[0])
        #
        # return tmp

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

    # Power method
    def power_iterations(self, R, w):
        while 1:
            w1 = w.copy()
            w = R * w
            w /= LA.norm(w)
            if self.converged(w, w1):
                break
        return w

        # test whether two normalized vectors are the same (or deviation within a threshold)

    def converged(self, w, w1, thres=1e-10):
        converged = False
        corr = w.T * w1
        # print corr
        if abs(corr) > 1 - thres:
            converged = True
        return converged

    def powerPCA(self, data, n_pcs):
        nr, dim = data.shape
        X = np.matrix(data)
        m = np.mean(data, axis=0)
        R = (X - m).T * (X - m) / (nr - 1.0)
        R0 = R.copy()

        # initialize
        w = np.matrix(np.random.rand(dim)).T
        w /= LA.norm(w)
        # iterate for 1st eigenvector
        w = self.power_iterations(R, w)
        # first eigenvector as first column of W
        W = w
        # iterate for other eigenvectors
        for i in range(1, n_pcs):
            # deflate R
            R -= w * w.T * R
            # initialize and Power iter
            w = np.matrix(np.random.rand(dim)).T
            w /= LA.norm(w)
            w = self.power_iterations(R, w)
            # attach the new eigenvector to W as a new column
            W = np.c_[W, w]

        # get eigenvalues and save them in the ev array
        y = R0 * W
        ev = LA.norm(y, axis=0) / LA.norm(W, axis=0)
        return W, ev

    # inner class for doing the K-Means clustering
    class KMeans:
        def __init__(self, c_0, c_1):
            self.centroids = np.array([])
            self.index_records_0 = []
            self.index_records_1 = []
            self.c_0 = c_0
            self.c_1 = c_1

        def split(self, dataSet, index_records):
            # print "\nindex_records length = {}".format(len(index_records))
            # print "index_records = {}".format(index_records)
            c0 = np.squeeze(np.asarray(self.c_0))
            c1 = np.squeeze(np.asarray(self.c_1))
            initial_centroids = np.stack((c0, c1))
            # print "in gm_02, we use initial_centroids to initialize"
            kM = KMeans(n_clusters=2, init=initial_centroids).fit(dataSet[index_records])

            self.c_0 = kM.cluster_centers_[0]
            self.c_1 = kM.cluster_centers_[1]

            for index in range(len(index_records)):
                if kM.labels_[index] == 0:
                    self.index_records_0.append(index_records[index])
                else:
                    self.index_records_1.append(index_records[index])

            return self


def demo1():
    X, y = make_blobs(n_samples=200, centers=5, n_features=2, random_state=100)
    gm = GMeans_01().fit(X)
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

    gm = GMeans_02().fit(X_aniso)

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


def summary(n_samples, min_n_clusters, max_n_clusters, n_features, random_state=0, n_loops=10):
    gm_01_scores = []
    gm_02_scores = []
    xm_scores = []

    for n_cluster in np.linspace(min_n_clusters, max_n_clusters, max_n_clusters - min_n_clusters + 1):
        total_gm_01_score = 0
        total_gm_02_score = 0
        total_xm_score = 0

        for i in range(n_loops):
            n_cluster = n_cluster.astype(int)
            X, y = datasets.make_blobs(n_samples=n_samples, n_features=n_features, centers=n_cluster,
                                       random_state=random_state)
            gm_01 = GMeans_01().fit(X)
            gm_02 = GMeans_02().fit(X)
            xm = XMeans().fit(X)

            gm_01_score = silhouette_score(X, gm_01.labels, metric='euclidean')
            gm_02_score = silhouette_score(X, gm_02.labels, metric='euclidean')
            xm_score = silhouette_score(X, xm.labels_, metric='euclidean')

            total_gm_01_score += gm_01_score
            total_gm_02_score += gm_02_score
            total_xm_score += xm_score

        total_gm_01_score = total_gm_01_score / (n_loops * 1.0)
        total_gm_02_score = total_gm_02_score / (n_loops * 1.0)
        total_xm_score = total_xm_score / (n_loops * 1.0)

        print "n_samples = {}, n_features = {}, n_cluster = {}, gm_01_score = {}, gm_02_score = {}, xm_score = {}" \
            .format(n_samples, n_features, n_cluster, total_gm_01_score, total_gm_02_score, total_xm_score)

        gm_01_scores.append(total_gm_01_score)
        gm_02_scores.append(total_gm_02_score)
        xm_scores.append(total_xm_score)

    return gm_01_scores, gm_02_scores, xm_scores



def show_different_clusters_with_g_means():
    pl.figure(figsize=(8, 8))

    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # Different variance
    X_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

    pl.subplot(221)
    pl.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred, s=1)
    pl.title("Unequal Variance")

    # Unevenly sized blobs
    X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
    y_pred = KMeans(n_clusters=3,
                    random_state=random_state).fit_predict(X_filtered)

    pl.subplot(222)
    pl.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred, s=1)
    pl.title("Unevenly Sized Blobs")

    # use g-means
    gm = GMeans_01().fit(X_varied)
    pl.subplot(223)
    Tools.draw(X_varied, gm.labels, gm.centroids, "title")

    pl.show()


def plot():
    # run different set of data to draw graph
    # prepare settings
    min_n_cluster = 3
    max_n_cluster = 20
    n_loops = 10
    indexes = []
    for index in np.linspace(min_n_cluster, max_n_cluster, max_n_cluster - min_n_cluster + 1):
        indexes.append(index)


    # sampling
    n_features = 4
    n_samples = 2000
    gm_01, gm_02, xm = summary(n_samples, min_n_cluster, max_n_cluster, n_features=n_features, random_state=0,
                         n_loops=n_loops)

    pl.plot(indexes, gm_01, label="gm_01_n_feature = {}, n_samples = {}".format(n_features, n_samples))
    pl.plot(indexes, gm_02, label="gm_02_n_feature = {}, n_samples = {}".format(n_features, n_samples))
    pl.plot(indexes, xm, '--', label="xm_n_feature = {}, n_samples = {}".format(n_features, n_samples))


    pl.legend()
    pl.show()

if __name__ == '__main__':
    # n_clusters = 3
    # X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=n_clusters, random_state=0)
    # gm = GMeans_01().fit(X)
    # print "n_cluster = {} ".format(n_clusters), silhouette_score(X, gm.labels, metric='euclidean')
    # Tools.draw(X, gm.labels, gm.centroids, title="g-means")
    # pl.show()

    # collect_silhouette_score_for_gmeans()
    # collect_silhouette_score_for_xmeans()
    #
    # show_different_clusters_with_k_means()
    # show_different_clusters_with_g_means()

    plot()




