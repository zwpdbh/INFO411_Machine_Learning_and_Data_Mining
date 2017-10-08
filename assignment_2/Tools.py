import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from random import randint

class Tools:
    @staticmethod
    def drawCentroids(centroids='ro'):
        plt.plot(centroids[:, 0], centroids[:, 1], 'ro')

    @staticmethod
    def drawDataSet(X, symbol):
        plt.plot(X[:, 0], X[:, 1], symbol)

    @staticmethod
    def distBetween(a, b):
        return np.sqrt(np.power(a[0]-b[0], 2) + np.power(a[1] - b[1], 2))

    @staticmethod
    def generateSomeData(start, end, numbers):
        pass

    @staticmethod
    def exchange_binary_image_white_black(img):
        # img is an binary img
        img = img.convert('L')
        pixels = np.array(img)
        threshold = 100

        rows = len(pixels)
        cols = len(pixels[0])

        for i in range(rows):
            for j in range(cols):
                if pixels[i][j] > threshold:
                    pixels[i][j] = 0
                else:
                    pixels[i][j] = 255

        return pixels


    @staticmethod
    def drawClusters(clusters):
        Ys = np.array([[4, 8, 12, 16],
                       [1, 4, 9, 16],
                       [17, 10, 13, 18],
                       [9, 10, 18, 11],
                       [4, 15, 17, 6],
                       [7, 10, 8, 7],
                       [9, 0, 10, 11],
                       [14, 1, 15, 5],
                       [8, 15, 9, 14],
                       [20, 7, 1, 5]])

        colors = cm.rainbow(np.linspace(0, 1, len(Ys)))

        for label in range(len(clusters)):
            for each in clusters[label]:
                plt.scatter(each[0], each[1], s=1, c=colors[label % len(colors)])

    @staticmethod
    def draw(X, lables, centroids=[], title=None, interim_centers=None):
        Ys = np.array([[4, 8, 12, 16],
                       [1, 4, 9, 16],
                       [17, 10, 13, 18],
                       [9, 10, 18, 11],
                       [4, 15, 17, 6],
                       [7, 10, 8, 7],
                       [9, 0, 10, 11],
                       [14, 1, 15, 5],
                       [8, 15, 9, 14],
                       [20, 7, 1, 5]])

        colors = cm.rainbow(np.linspace(0, 1, len(Ys)))
        fig = plt.gcf()
        if title != None:
            fig.canvas.set_window_title(title)

        # draw centroid
        if len(centroids) > 0:
            for c in centroids:
                plt.plot(c[0], c[1], 'ro')

        if interim_centers != None:
            if len(interim_centers) > 0:
                for c in interim_centers:
                    plt.plot(c[0], c[1], 'g*')


        # draw each point
        for i in range(len(lables)):
            p = X[i]
            label = lables[i]
            plt.scatter(p[0], p[1], s=1, c=colors[label % len(colors)])

        if title != None:
            print "use {}, we found {} clusters".format(title, len(centroids))
