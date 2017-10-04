import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import anderson
from KMeans import KMeans
from Tools import Tools as tl
from sklearn.preprocessing import scale
from PIL import Image, ImageDraw, ImageOps
from sklearn import datasets
import matplotlib.pyplot as pl
from scipy.misc import imsave
from XMeans import XMeans

class GMeans:
    def __init__(self, strictLevel):
        self.strickLevel = strictLevel
        self.centroids = []
        self.cluster = []

    # recursive helper method
    def computeGMeans(self, X):
        centroid = np.mean(X, axis=0)

        km = KMeans()
        km.split(dataSet=X)
        v = km.c_0 - km.c_1

        X_prime = scale(X.dot(v) / (v.dot(v)))
        accept_split = GMeans.checkGaussianStatistic(X_prime, self.strickLevel)

        if accept_split:
            self.computeGMeans(km.cluster_0)
            self.computeGMeans(km.cluster_1)
        else:
            self.centroids.append(centroid)

    @staticmethod
    def checkGaussianStatistic(X, strictLevel):
        X = X - np.mean(X)
        output = anderson(X)
        statistic = output[0]
        critical_value = output[1][strictLevel]
        # print statistic, critical_value
        if statistic >= critical_value:
            return True
        else:
            return False

    def fit(self, X):
        self.computeGMeans(X)
        self.centroids = np.array(self.centroids)



if __name__ == '__main__':

    # # demo one:
    dataSet1 = np.random.randn(100, 2)+2
    dataSet2 = np.random.randn(100, 2)+5
    # dataSet3 = np.random.randn(400, 2)-2
    # # dataSet4 = np.random.randn(50, 2)-5
    # dataSet5 = np.random.randn(20, 2)-10
    #
    total_dataSet = np.vstack((dataSet1, dataSet2))
    # total_dataSet = np.vstack((total_dataSet, dataSet3))
    # # total_dataSet = np.vstack((total_dataSet, dataSet4))
    # total_dataSet = np.vstack((total_dataSet, dataSet5))
    #
    # total_dataSet = np.random.permutation(total_dataSet)
    #
    # gM = GMeans(strictLevel=4)
    # gM.fit(total_dataSet)
    #
    # print "found {} clusters".format(len(gM.centroids))
    #
    # tl.drawDataSet(total_dataSet, 'g+')
    # tl.drawCentroids(gM.centroids)
    # plt.show()

    imageString = "/Users/zw/code/INFO411_Machine_Learning_and_Data_Mining/snapshot_of_dilate_mask.png"
    imageString_1 = "/Users/zw/code/INFO411_Machine_Learning_and_Data_Mining/snapshot.png"
    img = Image.open(imageString)


    # exchange image white to black, black to white
    binary_img_narry = tl.exchange_binary_image_white_black(img)
    print len(binary_img_narry)

    moving_object_data_set = []
    for i in range(len(binary_img_narry)):
        for j in range( len(binary_img_narry[0])):
            if binary_img_narry[i][j] == 0:
                moving_object_data_set.append([j, i])


    image_data_set = np.asarray(moving_object_data_set)
    print image_data_set.shape

    gM = GMeans(strictLevel=4)
    gM.fit(image_data_set)
    print "found {} clusters".format(len(gM.centroids))

    x_means = XMeans(random_state=1).fit(image_data_set)
    print len(x_means.cluster_centers_)

    # tl.drawDataSet(total_dataSet, 'g+')
    tl.drawDataSet(image_data_set, 'g+')
    tl.drawCentroids(gM.centroids)
    plt.show()
