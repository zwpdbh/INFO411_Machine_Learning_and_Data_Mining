import numpy as np
import matplotlib.pyplot as pl
from MySelfOrganizingMap import MySelfOrganizingMap

from PIL import Image
from sklearn.cluster import KMeans
from sklearn import metrics

# img=Image.open("africa.jpg")
# pix=np.array(img)[:,:,0:3]

#
# # reshape the pix array for clustering
# pix=pix.reshape((-1,3)).astype(float)
# print 'Plain k-means - k : silh. score'
# # loops through a few k's - takes a while to finish
# for k in range(2,10):
#     kmeans = KMeans(init='random', n_clusters=k, copy_x=False, n_init=10).fit(pix)
#     labels=kmeans.labels_
#     print k,':', metrics.silhouette_score(pix, labels, sample_size=3000, metric='euclidean')

data = np.zeros([6, 1, 3])
pl.boxplot(data)
print data











# # first the random pixel array, and displaying it as an image
# ar=np.random.randint(1,256, size=(64, 64,3))
# # pl.imshow(ar.astype('uint8'),interpolation='nearest')
#
# # reshape it into a long array colour pixels (each as a 3-D RGB vectors) - this is the input data to be clustered
# data=ar.reshape(64*64,3)
#
# # randomly init SOM weight vectors by copying random input vectors
# mapsize=16
# ninput=data.shape[0]
# w=data[np.random.randint(ninput)].copy()
#
# # randomly picks more data vectors, and vertically stack them under
# for i in range(mapsize*mapsize-1):
#     w=np.vstack((w,data[np.random.randint(ninput)].copy()))
#
#
# # reshape weight vectors so they map onto a 16x16 grid (now they are referred to as "nodes"). Check on the shape.
# nodes=w.reshape(mapsize,mapsize,-1).astype(float)
#
# # Now the main program - train the map
# niters=6000  # number of total iterations
# neigh=4      # initial neighbour radius
# gamma=0.05   # initial learning rate
#
# sof = MySelfOrganizingMap(size=mapsize, data=data, iterations=niters, initialNeighborRadius=neigh, learningRate=gamma)
# sof.train()
#
# pl.imshow(sof.nodes.astype('uint8'))
# pl.show()