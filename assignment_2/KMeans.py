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


# class KMeans:
#     @staticmethod
#     def computeKMeans(dataSet, k_clusters, n_loop):
#         num_samples = len(dataSet)
#
#         clusters = []
#         # first generate k initial centroids
#         centroids = []
#         for i in range(k_clusters):
#             centroids.append(dataSet[np.random.randint(0, num_samples)])
#
#         # loop specified times
#         for i in range(n_loop):
#             clusters = KMeans.assignSamplesToCentroids(X=dataSet, centroids=centroids)
#             # each around, update centroid for each clusters
#             index = 0
#             for each_cluster in clusters:
#                 new_centroid = np.mean(each_cluster, axis=0)
#                 centroids[index] = new_centroid
#
#         index = 0
#         result = []
#         for each_cluster in clusters:
#             cluster = Cluster(dataSet=each_cluster, centroid=centroids[index])
#             result.append(cluster)
#             index += 1
#
#         return result
#
#     # given X and centroids, assign each x in X to associated clusters
#     @staticmethod
#     def assignSamplesToCentroids(X, centroids):
#         clusters = []
#         for c in centroids:
#             clusters.append([])
#
#         for x in X:
#             min_index = 0
#             min_dist = Tools.distBetween(x, centroids[min_index])
#             # print "min_dist is: {}".format(min_dist)
#             for i in range(len(centroids)):
#                 dist = Tools.distBetween(x, centroids[i])
#                 # print "distance between {} and centroid_{}:{} is {}".format(x, i, centroids[i], dist)
#                 if dist <= min_dist:
#                     min_dist = dist
#                     min_index = i
#             clusters[min_index].append(x)
#
#         # for cluster in clusters:
#         #     print len(cluster)
#
#         return clusters



