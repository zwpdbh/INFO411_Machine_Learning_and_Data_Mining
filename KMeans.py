import numpy as np
import matplotlib.pyplot as plt

# KMeans
class KMeans:
    def __init__(self, dataSet, numOfClusters, iterationCount):
        self.numOfLoop = iterationCount
        self.dataSet = dataSet
        self.numOfClusters = numOfClusters
        self.clusters = []
        self.centroids = []
        self.clustersData = []
        self.numOfData = len(self.dataSet)
        for i in range(self.numOfClusters):
            self.centroids.append(self.dataSet[np.random.randint(0, self.numOfData)])
            self.clusters.append([])
            self.clustersData.append([])


    def distBetween(self, a, b):
        return np.sqrt(abs(a-b))


    def computeKMeans(self):
        # loop specified times
        for i in range(self.numOfLoop):
            self.clusters = []
            for i in range(self.numOfClusters):
                eachCluster = []
                self.clusters.append(eachCluster)

            # compare each data with centroid
            for index in range(self.numOfData):
                # initialize the min_dist is the dist with centroid0
                min_distance = self.distBetween(self.dataSet[index], self.centroids[0])
                cloest_centroid_index = 0
                # find the closest centroid
                for i in range(self.numOfClusters):
                    dist = self.distBetween(self.dataSet[index], self.centroids[i])
                    # update the closest centroid
                    if dist <= min_distance:
                        min_distance = dist
                        cloest_centroid_index = i

                # assign the data to the closest centroid corresponding cluster
                self.clusters[cloest_centroid_index].append(index)

            # each around, update centroid
            for i in range(self.numOfClusters):
                for j in self.clusters[i]:
                    self.clustersData[i].append(self.dataSet[j])
                self.centroids[i] = np.average(self.clustersData[i])

    def getCentroids(self):
        self.computeKMeans()
        return self.centroids

    def getClusters(self):
        self.computeKMeans()
        return self.clusters

    def getClustersData(self):
        self.computeKMeans()
        return self.clustersData

    # def show(self):
    #     self.computeKMeans()
    #     plt.plot(self.dataSet, 'g+')
    #     for centroid in self.centroids:
    #         plt.plot(self.numOfData / 4, centroid, 'r^')
    #     plt.show()