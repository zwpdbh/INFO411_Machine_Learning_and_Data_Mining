import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import scale
from scipy.stats import anderson
from KMeans import KMeans
from Tools import Tools as tl

class GMeans:
    def __init__(self, dataSet):
        self.X = dataSet.copy()
        centroid = np.mean(self.X, axis=0)
        self.centroids = np.array([centroid])

    @staticmethod
    def kmeans(dataSet, k):
        kM = MiniBatchKMeans(n_clusters=k, init='k-means++')
        kM.fit(X=dataSet)
        return kM.cluster_centers_

    @staticmethod
    def testGaussian(dataSet):
        kM = MiniBatchKMeans(n_clusters=2, init='k-means++')
        kM.fit(dataSet)
        centroids = kM.cluster_centers_

        c1 = centroids[0]
        c2 = centroids[1]
        v = c1 - c2
        X_prime = scale(dataSet.dot(v)) / (v.dot(v))
        GMeans.computeStatisticOfAnderson(X=X_prime)

    @staticmethod
    def plotCentroids(centroids):
        plt.plot(centroids[:, 0], centroids[:, 1], 'ro')

    @staticmethod
    def computeStatisticOfAnderson(X):
        print anderson(X)


if __name__ == '__main__':

    dataSet1 = np.random.randn(100, 2)+2
    dataSet2 = np.random.randn(100, 2)+5

    total_dataSet = np.vstack((dataSet1, dataSet2))
    total_dataSet = np.random.permutation(total_dataSet)

    km = MiniBatchKMeans(2)
    result = km.fit(total_dataSet)
    print result.cluster_centers_

    km = KMeans()
    km.split(dataSet=total_dataSet)
    tl.drawCentroid(km.c_0, symbol='r^')
    tl.drawDataSet(km.cluster_0, 'g+')

    tl.drawCentroid(km.c_1)
    tl.drawDataSet(km.cluster_1, 'bx')

    plt.show()

