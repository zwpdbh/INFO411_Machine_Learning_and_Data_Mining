# CMeans
import numpy as np
import math

class CMeans:
    def __init__(self, dataSet, numberOfClusters, iterationCount):
        self.dataSet = dataSet
        self.numOfLoop = iterationCount
        self.numOfClusters = numberOfClusters
        self.dataSize = len(self.dataSet)

        # initialization of weights, randomly assign the weight for cluster i = 1, else 0
        index = 0
        self.weightsMatrix = np.zeros(shape=[self.numOfClusters, self.dataSize], dtype=float)
        for each in np.random.randint(low=0, high=self.numOfClusters, size=self.dataSize, dtype=int):
            self.weightsMatrix[each][index] = 1
            index += 1

        self.centroids = np.zeros([self.numOfClusters, 1], dtype=float)


    def distBetween(self, a, b):
        return math.pow(abs(a-b), -2)

    def computeWeightBetween(self, j, i):
        total = 0
        for i in range(self.numOfClusters):
            total += self.distBetween(self.dataSet[j], self.centroids[i])
        return self.distBetween(self.dataSet[j], self.centroids[i]) / total


    def computeCMeans(self):
        for eachLoop in range(self.numOfLoop):
            for i in range(self.numOfClusters):
                self.centroids[i] = sum(np.multiply(self.weightsMatrix[i], self.dataSet)) / sum(self.weightsMatrix[i])

            for j in range(self.dataSize):
                total = 0.0
                for i in range(self.numOfClusters):
                    total += self.distBetween(self.dataSet[j], self.centroids[i])
                for i in range(self.numOfClusters):
                    self.weightsMatrix[i][j] = (self.distBetween(self.dataSet[j], self.centroids[i]) / total)

    def getCentroids(self):
        self.computeCMeans()
        return self.centroids





