import numpy as np
import matplotlib.pyplot as plt
import mylib



d1 = np.random.randn(100) + 2
d2 = np.random.randn(100) + 8

# plt.plot(d1, 'g+')
# plt.plot(d2, 'bx')

data = np.concatenate((d1, d2), axis=0)
data = np.random.permutation(data)

# data[80] = 200

nrow= len(data)

# kMeans cluster
# clusteringResult = mylib.kMeans(data=data, num_of_clusters=2, num_of_loop=20)
#
# centroid1 = clusteringResult[0][0]
# centroid2 = clusteringResult[0][1]
#
# cluster1 = clusteringResult[1][0]
# cluster2 = clusteringResult[1][1]
#
# clusterData1 = []
# clusterData2 = []
#
# for i in cluster1:
#     clusterData1.append(data[i])
#
# for i in cluster2:
#     clusterData1.append(data[i])

# plt.plot(d1, 'g+')
# plt.plot(d2, 'bx')
# plt.plot(nrow/4, centroid1,'r^')
# plt.plot(nrow/4, centroid2,'r^')
# plt.show()

# cMeans cluster
cMean = mylib.CMeans(dataSet=data, numberOfClusters=2, numOfLoop=20)
centroids = cMean.getCentroids()
print centroids

kMeans = mylib.KMeans(dataSet=data, numOfClusters=2, numOfLoop=20)
print kMeans.getCentroids()
