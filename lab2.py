import numpy as np
import matplotlib.pyplot as plt
from FuzzyCMeans import CMeans

d1 = np.random.randn(100)+2
d2 = np.random.randn(100)+8
d3 = np.random.randn(100)+14

data=np.concatenate((d1,d2,d3),axis=0)
data=np.random.permutation(data)

cMean = CMeans(dataSet=data, numberOfClusters=3, iterationCount=20)
centroids = cMean.getCentroids()
plt.plot(d1,'g+')
plt.plot(d2,'bx')
plt.plot(d3,'g+')

for centroid in centroids:
    plt.plot(len(cMean.dataSet) / 4, centroid, 'r^')
plt.show()