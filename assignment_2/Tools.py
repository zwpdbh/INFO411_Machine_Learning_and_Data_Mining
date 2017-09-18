import matplotlib.pyplot as plt
import numpy as np

class Tools:
    @staticmethod
    def drawCentroids(centroids='ro'):
        plt.plot(centroids[:, 0], centroids[:, 1], 'ro')

    @staticmethod
    def drawCentroid(centroid, symbol='ro'):
        plt.plot(centroid[0], centroid[1], symbol)

    @staticmethod
    def drawDataSet(X, symbol):
        plt.plot(X[:, 0], X[:, 1], symbol)

    @staticmethod
    def distBetween(a, b):
        return np.sqrt(np.power(a[0]-b[0], 2) + np.power(a[1] - b[1], 2))
