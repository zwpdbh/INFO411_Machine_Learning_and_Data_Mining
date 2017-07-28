# Your code here
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)
X = StandardScaler().fit_transform(X)

for i in [0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
    db = DBSCAN(eps=i, min_samples=10).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    score = metrics.silhouette_score(X, labels, metric='euclidean')
    print 'Eps = %.2f, Estimated number of clusters: %d, Silhouette Score = %.4f' % (i, n_clusters_, score)




for k in [2, 3, 4, 5]:
    kmeans1 = KMeans(n_clusters=k, init='random').fit(X)
    labels1 = kmeans1.labels_
    score1 = metrics.silhouette_score(X, labels1, metric='euclidean')

    kmeans2 = KMeans(n_clusters=k, init='k-means++').fit(X)
    labels2 = kmeans2.labels_
    score2 = metrics.silhouette_score(X, labels2, metric='euclidean')
    print 'k = %d, Silhouette Score: random_KMean = %.4f; k-means++ = %.4f.' % (k, score1, score2)