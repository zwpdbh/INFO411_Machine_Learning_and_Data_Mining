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


    # @staticmethod
    # def get_clustering_pixel_position_from_image(img):
    #     img = img.convert('L')
    #     pixels = np.array(img)
    #     threshold = 100
    #
    #     for i in range(len(pixels)):
    #         for j in range(len(pixels[0])):
    #             if pixels[i][j]
    #             pass
