from GMeans import GMeans
from XMeans import XMeans
from PIL import Image, ImageDraw, ImageOps
from Tools import Tools as tl
import matplotlib.pyplot as plt
import numpy as np
import time

imageString = "/Users/zw/code/INFO411_Machine_Learning_and_Data_Mining/snapshot_of_dilate_mask.png"
imageString_1 = "/Users/zw/code/INFO411_Machine_Learning_and_Data_Mining/snapshot.png"
img = Image.open(imageString)


# exchange image white to black, black to white
binary_img_narry = tl.exchange_binary_image_white_black(img)
moving_object_data_set = []

for x in range(len(binary_img_narry[0])):
    for y in range(len(binary_img_narry)):
        if binary_img_narry[y][x] == 0:
            moving_object_data_set.append([x, y])

image_data_set = np.asarray(moving_object_data_set)


# Use G-means to do clustering
# gM = GMeans(strictLevel=4)
# gM.fit(image_data_set)
# print "found {} clusters".format(len(gM.centroids))
#
# tl.drawDataSet(image_data_set, 'g+')
# tl.drawCentroids(gM.centroids)
# plt.show()


# Use X-means to do clustering
# x_means = XMeans()
# x_means.fit(image_data_set)
# print x_means.cluster_centers
# tl.drawCentroids(x_means.cluster_centers_)
# plt.show()



N = 10

def test_1(visualize = False):
    x = np.array([np.random.normal(loc, 0.1, 20) for loc in np.repeat([1,2], 2)]).flatten()
    y = np.array([np.random.normal(loc, 0.1, 20) for loc in np.tile([1,2], 2)]).flatten()
    st = time.time()

    # print "x = ", x
    # print "y = ", y
    print "data set = ", np.c_[x, y]

    x_means = XMeans(k_init=2).fit(np.c_[x, y])
    et = time.time() - st

    if visualize:
        print(x_means.labels_)
        print(x_means.cluster_centers_)

        colors = ["g", "b", "c", "m", "y", "b", "w"]
        for label in range(x_means.labels_.max()+1):
            plt.scatter(x[x_means.labels_ == label], y[x_means.labels_ == label], c = colors[label], label = "sample", s = 30)
        plt.scatter(x_means.cluster_centers_[:,0], x_means.cluster_centers_[:,1], c = "r", marker = "+", label = "center", s = 100)
        plt.xlim(0, 3)
        plt.ylim(0, 3)
        plt.title("x-means_test")
        plt.legend()
        plt.grid()
        plt.show()

    return et

def test_on_image(imageString):
    img = Image.open(imageString)

    # exchange image white to black, black to white
    binary_img_narry = tl.exchange_binary_image_white_black(img)
    moving_object_data_set = []

    for x in range(len(binary_img_narry[0])):
        for y in range(len(binary_img_narry)):
            if binary_img_narry[y][x] == 0:
                moving_object_data_set.append([y*1.0, x*1.0])

    image_data_set = np.asarray(moving_object_data_set)
    image_data_set = image_data_set[:3000:]
    st = time.time()
    x = image_data_set[:, 0]
    y = image_data_set[:, 1]

    print "data set = ", np.c_[x, y]

    x_means = XMeans(k_init=2).fit(np.c_[x, y])
    et = time.time() - st

    print(x_means.labels_)
    print(x_means.cluster_centers_)

    colors = ["g", "b", "c", "m", "y", "b", "w"]
    # for label in range(x_means.labels_.max() + 1):
        # plt.scatter(x[x_means.labels_ == label], y[x_means.labels_ == label], c=colors[label], label="sample")
    tl.drawDataSet(image_data_set, 'g+')
    plt.scatter(x_means.cluster_centers_[:, 0], x_means.cluster_centers_[:, 1], c="r", marker="+", label="center",
                s=100)
    print "find {} number of clusters".format(x_means.cluster_centers_)
    # plt.xlim(0, 3)
    # plt.ylim(0, 3)
    plt.title("x-means_test")
    plt.legend()
    plt.grid()
    plt.show()
    return et



# test_1(True)
test_on_image(imageString=imageString)


# test = np.array([[1, 2], [3, 4], [5, 6]])
# print test
#
# print test[:, 0]
# print test[:, 1]