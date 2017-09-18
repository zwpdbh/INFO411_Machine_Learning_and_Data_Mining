import numpy as np
from Tools import Tools
from sklearn.cluster import MiniBatchKMeans

class Cluster:
    def __init__(self, dataSet, centroid):
        self.dataSet = dataSet
        self.centroid = centroid

    def getCentroid(self):
        return self.centroid

    def getCluster(self):
        return self.dataSet


class KMeans:
    def __init__(self):
        self.centroids = np.array([])
        self.cluster_0 = []
        self.cluster_1 = []
        self.c_0 = None
        self.c_1 = None

    def split(self, dataSet):
        kM = MiniBatchKMeans(n_clusters=2)
        result = kM.fit(dataSet)

        self.c_0 = result.cluster_centers_[0]
        self.c_1 = result.cluster_centers_[1]

        index = 0
        for label in result.labels_:
            if label == 0:
                self.cluster_0.append(dataSet[index])
            else:
                self.cluster_1.append(dataSet[index])
            index += 1
        self.cluster_0 = np.array(self.cluster_0)
        self.cluster_1 = np.array(self.cluster_1)


