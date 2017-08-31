import numpy as np
import matplotlib.pylab as pl
import math


# Decentralize data
# Get covariance matrix
# Initialize w
# Do power method iterations (e.g., 30 times) and get w (converged)
# Assign transform matrix W=w
# for other eigenvectors: repeat steps 3 and 4; hstack w to W
# return X*W

class MYPCA:
    def __init__(self, data):
        self.X = np.matrix(data)
        self.num_of_instances = self.X.shape[0]
        self.dim_of_data = self.X.shape[1]

        # make it zero meaned:
        self.m = np.mean(self.X, axis=0)
        self.X -= self.m

        # create covariance matrix
        self.covariance_matrix = (self.X.T * self.X) / self.num_of_instances

        self.eigens = []

    def get_covariance_matrix(self):
        return self.covariance_matrix

    def findOneEigenvectorWithInitialVector(self, w):
        e = self.covariance_matrix * w
        e = e / np.sqrt(e.T * e)

        if np.dot(e.T, w) == 1:
            return e
        else:
            return self.findOneEigenvectorWithInitialVector(e)

    def findEigenVectors(self):
        w = np.matrix(np.ones(self.dim_of_data)).T
        w = w / np.sqrt(w.T * w)
        e1 = self.findOneEigenvectorWithInitialVector(w)
        self.eigens.append(e1)

        self.covariance_matrix = self.covariance_matrix - self.covariance_matrix * e1 * e1.T
        w = np.matrix(np.ones(self.dim_of_data)).T
        w = w / np.sqrt(w.T * w)
        e2 = self.findOneEigenvectorWithInitialVector(w)

        self.eigens.append(e2)

        return np.hstack(self.eigens)


if __name__ == '__main__':
    # Part A. PCA and Visualization

    bank = np.loadtxt('bank_new.txt')
    bank = np.random.permutation(bank)

    # data is the bank data without the last column
    data = bank[:, 0:-1]
    # the last column of bank data is the label
    label = bank[:, -1]

    # print bank.shape, data.shape, label.shape

    # get the subset for class '0' and '1' respectively
    sub0 = label[:] == 0
    sub1 = label[:] == 1

    # plot the 1 the 5 column of attribute of data
    # pl.plot(data[sub0, 0], data[sub0, 4], 'cx')
    # pl.plot(data[sub1, 0], data[sub1, 4], 'mo')
    # pl.show()

    # use PCA from python library
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    x = pca.fit_transform(data)
    # # the variance they have explained
    # print x.shape, pca.explained_variance_ratio_

    # re-plot the plot using the two PCs
    # pl.plot(x[sub0, 0], x[sub0, 1], 'bx')
    # pl.plot(x[sub1, 0], x[sub1, 1], 'g+')
    # pl.show()

    # Part B. Classification
    # For this part we will experiment with the k-NN classifier implemented in the Sklearn package.

    from sklearn.neighbors import KNeighborsClassifier
    # neigh = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    # neigh.fit(data[:1000], label[:1000])
    #
    # print neigh.score(data[1000:], label[1000:])

    """
    Task 1a. Taking the same split of the 'data' array (first 1000 vs the rest) for training and testing, 
    carry out k-NN classification using different  k(n_neighbors) values. 
    Use a "for" loop to collect the performance data into a list; print out or display the list.
    """

    """
    Add another list to collect performance data using distance-based weighting for k-NN. From the two accuracy lists, 
    generate two performance curves for comparison.
    """
    def demoB():
        def calculateScoreForKNN(cluster_num, dataSet, weightsMethod):
            neigh = KNeighborsClassifier(n_neighbors=cluster_num, weights=weightsMethod)
            neigh.fit(dataSet[:1000], label[:1000])
            return neigh.score(dataSet[1000:], label[1000:])

        eachKs = []
        eachScore1 = []
        eachScore2 = []
        eachScore3 = []
        eachScore4 = []
        for i in range(30):
            numberOfCuster = i + 1
            eachKs.append(numberOfCuster)
            score1 = calculateScoreForKNN(numberOfCuster, data, 'uniform')
            eachScore1.append(score1)
            score2 = calculateScoreForKNN(numberOfCuster, data, 'distance')
            eachScore2.append(score2)
            score3 = calculateScoreForKNN(numberOfCuster, x, 'uniform')
            eachScore3.append(score3)
            score4 = calculateScoreForKNN(numberOfCuster, x, 'distance')
            eachScore4.append(score4)
            print 'when nearest neighbours = {}, the score1 is: {}, score2 is {}'.format(numberOfCuster, score1, score2)

        pl.plot(eachKs, eachScore1)
        pl.plot(eachKs, eachScore2)
        pl.plot(eachKs, eachScore3)
        pl.plot(eachKs, eachScore4)
        pl.legend(
            ['weight function = uniform', 'weight function = distance', 'PCA using distance', 'PCA using uniform'])
        pl.show()


    def demoPCA():
        bank = np.loadtxt('bank_new.txt')
        bank = np.random.permutation(bank)
        data = bank[:, 0:-1]

        pca = MYPCA(data=data)
        print pca.findEigenVectors()


    demoPCA()